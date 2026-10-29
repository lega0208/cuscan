// Every case fits and restores on the explicit CPU backend.

use umap::{
    BackendChoice, FittedUMAP, NeighborSearch, UMAP, UMAPConfig, standard_umap::ModelState,
};

fn estimator(search: NeighborSearch, unique: bool, n_epochs: Option<usize>) -> UMAP {
    UMAP::new(
        UMAPConfig {
            n_neighbors: 3,
            init: umap::Init::Random,
            random_state: Some(7),
            unique,
            n_epochs,
            ..Default::default()
        },
        BackendChoice::Cpu,
        search,
    )
}

fn rows() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![2.0, 1.0],
        vec![3.0, 0.0],
    ]
}

#[test]
fn state_roundtrip_retains_exact_fit_and_transform() {
    let fitted = estimator(NeighborSearch::Exact, false, Some(8))
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    let restored = FittedUMAP::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
    assert_eq!(
        restored.transform(&[vec![0.5, 0.5]]).unwrap(),
        fitted.transform(&[vec![0.5, 0.5]]).unwrap()
    );
    assert_eq!(restored.export_state().unwrap(), state);
}

#[test]
fn unique_roundtrip_restores_original_order() {
    let data = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![1.0, 2.0],
        vec![5.0, 6.0],
    ];
    let fitted = estimator(NeighborSearch::Exact, true, Some(5))
        .fit(&data)
        .unwrap();
    let restored =
        FittedUMAP::from_state(fitted.export_state().unwrap(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
}

#[test]
fn auto_and_singleton_roundtrip() {
    let fitted = estimator(NeighborSearch::Auto, false, None)
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!(state.requested_search, NeighborSearch::Auto);
    assert_eq!(state.resolved_search, NeighborSearch::Exact);
    FittedUMAP::from_state(state, BackendChoice::Cpu).unwrap();
    let singleton = UMAP::new(
        UMAPConfig {
            n_neighbors: 2,
            init: umap::Init::Random,
            ..Default::default()
        },
        BackendChoice::Cpu,
        NeighborSearch::Auto,
    )
    .fit(&[vec![1.0, 2.0]])
    .unwrap();
    let restored =
        FittedUMAP::from_state(singleton.export_state().unwrap(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), &[vec![0.0, 0.0]]);
}

#[test]
fn approximate_search_and_automatic_epochs_roundtrip() {
    let fitted = estimator(NeighborSearch::Approximate, false, None)
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!(state.requested_search, NeighborSearch::Approximate);
    assert_eq!(state.resolved_search, NeighborSearch::Approximate);
    let restored = FittedUMAP::from_state(state, BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
    let query = [vec![0.5, 0.5], vec![2.5, 0.5]];
    assert_eq!(
        restored.transform(&query).unwrap(),
        fitted.transform(&query).unwrap()
    );
}

fn invalid(mut state: ModelState) {
    assert!(FittedUMAP::from_state(state.clone(), BackendChoice::Cpu).is_err());
    state.config.n_neighbors = 0;
    assert!(FittedUMAP::from_state(state, BackendChoice::Cpu).is_err());
}

#[test]
fn rejects_malformed_arrays_graph_mapping_and_search() {
    let valid = estimator(NeighborSearch::Exact, false, Some(4))
        .fit(&rows())
        .unwrap()
        .export_state()
        .unwrap();
    let mut state = valid.clone();
    state.n_features += 1;
    invalid(state);
    let mut state = valid.clone();
    state.embedding[0] = f32::NAN;
    invalid(state);
    let mut state = valid.clone();
    state.original_to_training = Some(vec![0]);
    invalid(state);
    let mut state = estimator(NeighborSearch::Approximate, false, Some(4))
        .fit(&rows())
        .unwrap()
        .export_state()
        .unwrap();
    state.query_graph.as_mut().unwrap()[0] = -1;
    invalid(state);
    let mut state = valid;
    state.resolved_search = NeighborSearch::Approximate;
    invalid(state);
}

#[test]
fn rejects_nonpositive_curve_for_exact_and_approximate_models() {
    for search in [NeighborSearch::Exact, NeighborSearch::Approximate] {
        let state = estimator(search, false, Some(4))
            .fit(&rows())
            .unwrap()
            .export_state()
            .unwrap();
        for (a, b) in [
            (0.0, state.b),
            (state.a, 0.0),
            (-1.0, state.b),
            (state.a, -1.0),
        ] {
            let mut malformed = state.clone();
            malformed.a = a;
            malformed.b = b;
            assert!(FittedUMAP::from_state(malformed, BackendChoice::Cpu).is_err());
        }
    }
}

#[test]
fn rejects_two_row_approximate_state_with_zero_graph_width() {
    let mut state = estimator(NeighborSearch::Approximate, false, Some(4))
        .fit(&rows())
        .unwrap()
        .export_state()
        .unwrap();
    state.config.n_neighbors = 2;
    state.training_rows = 2;
    state.embedding_rows = 2;
    state.training_features.truncate(4);
    state.embedding.truncate(2 * state.n_components);
    state.query_graph = Some(Vec::new());
    state.query_graph_width = 0;
    assert!(FittedUMAP::from_state(state, BackendChoice::Cpu).is_err());
}
