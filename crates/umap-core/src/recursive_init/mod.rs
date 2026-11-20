//! Host primitives for recursive UMAP initialization.

mod geometry;
pub mod graph;
#[cfg(test)]
mod oracle_tests;

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{RecursiveInitConfig, Result, UmapError};
use graph::HostCsr;

macro_rules! stage {
    ($name:literal, $graph:expr, $depth:expr, $work:expr) => {{
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!(
            $name,
            rows = $graph.n,
            edges = $graph.indices.len(),
            depth = $depth
        )
        .entered();
        $work
    }};
}

/// Apply the configured final scaling to recursive coordinates before upload.
pub fn finish(
    values: &mut [f32],
    rows: usize,
    heads: &[usize],
    tails: &[usize],
    weights: &[f32],
    config: &crate::UMAPConfig,
) -> Result<()> {
    crate::init_rescale::rescale(
        values,
        rows,
        config.n_components,
        heads,
        tails,
        weights,
        config,
    )
}

/// Run the host recursion; the caller owns device placement for coarse layouts.
#[allow(clippy::too_many_arguments)]
pub fn run(
    graph: &HostCsr,
    data: &[f32],
    features: usize,
    dimensions: usize,
    curve: (f32, f32),
    config: &RecursiveInitConfig,
    seed: u64,
    mut admit: impl FnMut(&'static str, usize) -> Result<()>,
    mut layout: impl FnMut(&HostCsr, Vec<f32>, LevelParams, u64, usize) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    config.validate()?;
    graph.validate()?;
    let params = LevelParams::first(graph.n, curve.0, curve.1, config)?;
    let anchor_bytes = geometry::anchor_allocation_bytes(
        graph.n,
        features,
        dimensions,
        config.anchor_sample_size,
    )?;
    let anchor_owned = geometry_bytes(graph.n, dimensions, size_of::<f32>())?
        .checked_add(geometry_bytes(graph.n, 1, size_of::<bool>())?)
        .ok_or(UmapError::Overflow("recursive anchor ownership"))?;
    let later_bytes = match params {
        Some(params) => level_one_bound(graph, dimensions, params.parts)?,
        None => base_allocation_bytes(graph.n, dimensions)?,
    };
    let upfront = anchor_bytes.max(
        anchor_owned
            .checked_add(later_bytes)
            .ok_or(UmapError::Overflow("recursive upfront allocation"))?,
    );
    #[cfg(feature = "tracing")]
    tracing::info!(
        anchor_bytes,
        later_bytes,
        upfront,
        "recursive estimated host upfront peak excluding existing owners and coarse layout"
    );
    admit("recursive anchor", upfront)?;
    let mut rng = StdRng::seed_from_u64(stream_seed(seed, 1, 0));
    let anchor = stage!(
        "recursive_anchor",
        graph,
        1,
        geometry::anchor_reference(
            data,
            graph.n,
            features,
            dimensions,
            config.anchor_sample_size,
            &mut rng,
        )
    )?;
    let mut admit_stage = |name, bytes| {
        #[cfg(feature = "tracing")]
        tracing::info!(
            stage = name,
            incremental_host_bytes = bytes,
            "recursive stage admission estimate (excludes live owners and device allocations)"
        );
        admit(name, bytes)
    };
    descend(
        graph,
        None,
        &anchor,
        params,
        config,
        seed,
        1,
        0,
        &mut admit_stage,
        &mut layout,
    )
}

#[allow(clippy::too_many_arguments)]
fn descend(
    graph: &HostCsr,
    summed: Option<graph::SummedCsr>,
    anchor: &geometry::Anchor,
    params: Option<LevelParams>,
    config: &RecursiveInitConfig,
    seed: u64,
    depth: usize,
    ancestors: usize,
    admit: &mut impl FnMut(&'static str, usize) -> Result<()>,
    layout: &mut impl FnMut(&HostCsr, Vec<f32>, LevelParams, u64, usize) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let Some(params) = params else {
        let bytes = base_allocation_bytes(graph.n, anchor.dimensions)?;
        admit("recursive base", bytes)?;
        return stage!(
            "recursive_base",
            graph,
            depth,
            geometry::base_case(graph, anchor, depth)
        );
    };
    let max_degree = graph
        .indptr
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0);
    let hub_bytes = geometry_bytes(graph.n, 1, 8 + size_of::<usize>() + 2 * size_of::<bool>())?;
    let vote_bytes = geometry_bytes(
        max_degree,
        rayon::current_num_threads(),
        2 * size_of::<(usize, i32, f32)>(),
    )?
    .checked_add(geometry_bytes(graph.n, 1, 2 * size_of::<i32>())?)
    .ok_or(UmapError::Overflow("recursive propagation allocation"))?;
    admit("recursive labels", hub_bytes.max(vote_bytes))?;
    let mut labels = stage!(
        "recursive_hubs",
        graph,
        depth,
        graph::initialize_labels(
            graph,
            params.parts,
            config.hub_selection,
            stream_seed(seed, 2, depth),
        )
    )?;
    let _sweeps = stage!(
        "recursive_propagation",
        graph,
        depth,
        graph::propagate_labels(graph, &mut labels, depth > 1)
    )?;
    #[cfg(feature = "tracing")]
    tracing::info!(depth, sweeps = _sweeps, "recursive propagation completed");
    let stack_capacity = max_degree
        .checked_mul(100)
        .and_then(|n| n.checked_add(1))
        .ok_or(UmapError::Overflow("recursive outlier allocation"))?;
    let outlier_bytes = geometry_bytes(stack_capacity, 1, size_of::<usize>())?;
    admit("recursive outliers", outlier_bytes)?;
    stage!(
        "recursive_outliers",
        graph,
        depth,
        graph::label_outliers(graph, &mut labels, stream_seed(seed, 3, depth))
    )?;
    admit(
        "recursive remap",
        geometry_bytes(labels.len(), 1, size_of::<i32>())?,
    )?;
    let parts = graph::remap_labels(&mut labels)?;
    let edges = summed
        .as_ref()
        .map_or(graph.indices.len(), |g| g.indices.len());
    let part_offsets = parts
        .checked_add(1)
        .ok_or(UmapError::Overflow("recursive parts"))?;
    let grouping = geometry_bytes(graph.n, 1, size_of::<usize>())?
        .checked_add(geometry_bytes(part_offsets, 1, 3 * size_of::<usize>())?)
        .ok_or(UmapError::Overflow("recursive grouping"))?;
    admit(
        "recursive count",
        grouping
            .checked_add(geometry_bytes(edges.max(1), 2, size_of::<i32>())?)
            .ok_or(UmapError::Overflow("recursive count"))?,
    )?;
    let counts = stage!(
        "recursive_count",
        graph,
        depth,
        graph::count_coarse_edges(graph, summed.as_ref(), &labels, parts)
    )?;
    let coarse_edges = counts.iter().try_fold(0usize, |sum, &count| {
        sum.checked_add(count)
            .ok_or(UmapError::Overflow("recursive coarse edges"))
    })?;
    let aggregate = grouping
        .checked_add(geometry_bytes(
            edges.max(1),
            2,
            size_of::<(usize, usize, f64)>(),
        )?)
        .and_then(|bytes| bytes.checked_add(edges.checked_mul(size_of::<(usize, f64)>())?))
        .and_then(|bytes| bytes.checked_add(coarse_edges.checked_mul(size_of::<(usize, f64)>())?))
        .and_then(|bytes| {
            bytes.checked_add(
                parts.checked_mul(size_of::<Vec<(usize, f64)>>() + size_of::<usize>())?,
            )
        })
        .and_then(|bytes| bytes.checked_add(parts.checked_add(1)?.checked_mul(size_of::<usize>())?))
        .ok_or(UmapError::Overflow("recursive aggregation"))?;
    admit("recursive aggregate", aggregate)?;
    let coarse_summed = stage!(
        "recursive_aggregate",
        graph,
        depth,
        graph::aggregate_graph_counted(graph, summed.as_ref(), &labels, parts, &counts)
    )?;
    drop(counts);
    drop(summed);
    let transition = geometry_bytes(
        coarse_edges,
        2,
        size_of::<(usize, usize)>() + size_of::<usize>() + 2 * size_of::<f32>(),
    )?
    .checked_add(geometry_bytes(
        part_offsets,
        1,
        size_of::<usize>() + size_of::<f64>(),
    )?)
    .ok_or(UmapError::Overflow("recursive transition"))?;
    admit("recursive transition", transition)?;
    let coarse_graph = stage!(
        "recursive_transition",
        graph,
        depth,
        graph::transition_graph(&coarse_summed)
    )?;
    let bytes = geometry_bytes(parts, anchor.dimensions, 12)?
        .checked_add(geometry_bytes(
            parts,
            1,
            size_of::<usize>() + size_of::<bool>(),
        )?)
        .ok_or(UmapError::Overflow("recursive coarse anchor allocation"))?;
    admit("recursive coarse anchor", bytes)?;
    let coarse_anchor = geometry::coarse_anchor(anchor, &labels, parts)?;
    let next_params = params.next(parts, config)?;
    let retained = ancestors
        .checked_add(csr_owned_bytes(graph)?)
        .and_then(|bytes| {
            bytes.checked_add(anchor.values.capacity().checked_mul(size_of::<f32>())?)
        })
        .and_then(|bytes| bytes.checked_add(anchor.mask.capacity().checked_mul(size_of::<bool>())?))
        .and_then(|bytes| bytes.checked_add(labels.capacity().checked_mul(size_of::<i32>())?))
        .ok_or(UmapError::Overflow("recursive retained owners"))?;
    let coarse = descend(
        &coarse_graph,
        Some(coarse_summed),
        &coarse_anchor,
        next_params,
        config,
        seed,
        depth + 1,
        retained,
        admit,
        layout,
    )?;
    drop(coarse_anchor);
    let coarse = if coarse_graph.weights.is_empty() {
        coarse
    } else {
        let retained = retained
            .checked_add(csr_owned_bytes(&coarse_graph)?)
            .ok_or(UmapError::Overflow("recursive retained owners"))?;
        layout(
            &coarse_graph,
            coarse,
            params,
            stream_seed(seed, 4, depth),
            retained,
        )?
    };
    if coarse.len()
        != parts
            .checked_mul(anchor.dimensions)
            .ok_or(UmapError::Overflow("coarse layout shape"))?
        || !coarse.iter().all(|value| value.is_finite())
    {
        return Err(UmapError::Numerical(
            "invalid recursive coarse layout output".into(),
        ));
    }
    drop(coarse_graph);
    admit(
        "recursive prolongation",
        geometry_bytes(graph.n, anchor.dimensions, 12)?,
    )?;
    let mut result = stage!(
        "recursive_prolongation",
        graph,
        depth,
        geometry::prolongate(graph, &labels, &coarse, anchor.dimensions)
    )?;
    admit(
        "recursive Procrustes",
        geometry::procrustes_allocation_bytes(anchor.dimensions)?,
    )?;
    stage!(
        "recursive_procrustes",
        graph,
        depth,
        geometry::procrustes_align(&mut result, anchor)
    )?;
    Ok(result)
}

fn csr_owned_bytes(graph: &HostCsr) -> Result<usize> {
    graph
        .indptr
        .capacity()
        .checked_mul(size_of::<usize>())
        .and_then(|bytes| {
            bytes.checked_add(graph.indices.capacity().checked_mul(size_of::<usize>())?)
        })
        .and_then(|bytes| {
            bytes.checked_add(graph.weights.capacity().checked_mul(size_of::<f32>())?)
        })
        .ok_or(UmapError::Overflow("recursive retained graph"))
}

fn base_allocation_bytes(rows: usize, dimensions: usize) -> Result<usize> {
    geometry_bytes(rows, dimensions, 12)?
        .checked_add(geometry_bytes(1, rows, 8)?)
        .and_then(|bytes| bytes.checked_add(dimensions.checked_mul(8)?))
        .ok_or(UmapError::Overflow("recursive base allocation"))
}

// Bounds the existing level-one host estimates, including their live owners.
// Coarse layout and deeper levels are admitted separately at their actual sizes.
fn level_one_bound(graph: &HostCsr, dimensions: usize, parts: usize) -> Result<usize> {
    let n = graph.n;
    let edges = graph.indices.len();
    let max_degree = graph
        .indptr
        .windows(2)
        .map(|p| p[1] - p[0])
        .max()
        .unwrap_or(0);
    let labels = geometry_bytes(n, 1, size_of::<i32>())?;
    let hubs = geometry_bytes(n, 1, 8 + size_of::<usize>() + 2 * size_of::<bool>())?;
    let votes = geometry_bytes(
        max_degree,
        rayon::current_num_threads(),
        2 * size_of::<(usize, i32, f32)>(),
    )?
    .checked_add(geometry_bytes(n, 1, 2 * size_of::<i32>())?)
    .ok_or(UmapError::Overflow("recursive upfront votes"))?;
    let stack = max_degree
        .checked_mul(100)
        .and_then(|x| x.checked_add(1))
        .and_then(|x| x.checked_mul(size_of::<usize>()))
        .ok_or(UmapError::Overflow("recursive upfront outliers"))?;
    let offsets = parts
        .checked_add(1)
        .ok_or(UmapError::Overflow("recursive upfront parts"))?;
    let grouping = geometry_bytes(n, 1, size_of::<usize>())?
        .checked_add(geometry_bytes(offsets, 1, 3 * size_of::<usize>())?)
        .ok_or(UmapError::Overflow("recursive upfront grouping"))?;
    let count = grouping
        .checked_add(geometry_bytes(edges.max(1), 2, size_of::<i32>())?)
        .ok_or(UmapError::Overflow("recursive upfront count"))?;
    let aggregate = grouping
        .checked_add(geometry_bytes(
            edges.max(1),
            2,
            size_of::<(usize, usize, f64)>(),
        )?)
        .and_then(|x| x.checked_add(edges.checked_mul(2 * size_of::<(usize, f64)>())?))
        .and_then(|x| {
            x.checked_add(parts.checked_mul(size_of::<Vec<(usize, f64)>>() + size_of::<usize>())?)
        })
        .and_then(|x| x.checked_add(offsets.checked_mul(size_of::<usize>())?))
        .and_then(|x| x.checked_add(parts.checked_mul(size_of::<usize>())?))
        .ok_or(UmapError::Overflow("recursive upfront aggregation"))?;
    let summed = geometry_bytes(offsets, 1, size_of::<usize>())?
        .checked_add(geometry_bytes(
            edges,
            1,
            size_of::<usize>() + size_of::<f64>(),
        )?)
        .ok_or(UmapError::Overflow("recursive upfront summed graph"))?;
    let transition = geometry_bytes(
        edges,
        2,
        size_of::<(usize, usize)>() + size_of::<usize>() + 2 * size_of::<f32>(),
    )?
    .checked_add(geometry_bytes(
        offsets,
        1,
        size_of::<usize>() + size_of::<f64>(),
    )?)
    .ok_or(UmapError::Overflow("recursive upfront transition"))?;
    let coarse_graph = geometry_bytes(offsets, 1, size_of::<usize>())?
        .checked_add(geometry_bytes(
            edges,
            1,
            size_of::<usize>() + size_of::<f32>(),
        )?)
        .ok_or(UmapError::Overflow("recursive upfront coarse graph"))?;
    let coarse_anchor = geometry_bytes(parts, dimensions, 12)?
        .checked_add(geometry_bytes(
            parts,
            1,
            size_of::<usize>() + size_of::<bool>(),
        )?)
        .ok_or(UmapError::Overflow("recursive upfront coarse anchor"))?;
    let post_labels = stack
        .max(count)
        .max(aggregate)
        .max(
            summed
                .checked_add(transition)
                .ok_or(UmapError::Overflow("recursive upfront transition peak"))?,
        )
        .max(
            summed
                .checked_add(coarse_graph)
                .and_then(|x| x.checked_add(coarse_anchor))
                .ok_or(UmapError::Overflow("recursive upfront anchor peak"))?,
        )
        .max(geometry_bytes(n, 1, size_of::<i32>())?);
    Ok(hubs.max(votes).max(
        labels
            .checked_add(post_labels)
            .ok_or(UmapError::Overflow("recursive upfront labels"))?,
    ))
}

fn geometry_bytes(rows: usize, columns: usize, element: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .and_then(|count| count.checked_mul(element))
        .ok_or(UmapError::Overflow("recursive geometry allocation"))
}

fn stream_seed(seed: u64, domain: u64, depth: usize) -> u64 {
    let mut value = seed ^ (depth as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Requested coarsening and layout settings for one descent step.
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    pub parts: usize,
    pub epochs: u32,
    pub a: f32,
    pub b: f32,
    pub good_initialization: bool,
    curve: (f64, f64),
}

impl LevelParams {
    /// Return no layout schedule when the input is already a base case.
    pub fn first(n: usize, a: f32, b: f32, config: &RecursiveInitConfig) -> Result<Option<Self>> {
        if n <= config.base_init_threshold {
            return Ok(None);
        }
        let target = (n / config.coarsening_ratio).max(config.base_init_threshold);
        Ok(Some(Self::new(
            n,
            target,
            config.n_embedding_epochs,
            (a.into(), b.into()),
            config,
        )?))
    }

    /// Derive the next target from the requested parts, not the realized hubs.
    pub fn next(self, n: usize, config: &RecursiveInitConfig) -> Result<Option<Self>> {
        if n <= config.base_init_threshold {
            return Ok(None);
        }
        let epochs = (f64::from(self.epochs) * std::f64::consts::SQRT_2).floor();
        if epochs > f64::from(u32::MAX) {
            return Err(UmapError::Overflow("recursive layout epochs"));
        }
        Ok(Some(Self::new(
            n,
            self.parts / config.coarsening_ratio,
            epochs as usize,
            self.curve,
            config,
        )?))
    }

    fn new(
        n: usize,
        target: usize,
        epochs: usize,
        curve: (f64, f64),
        config: &RecursiveInitConfig,
    ) -> Result<Self> {
        let parts = target.min(n / 2).max(2);
        let curve = (curve.0.powf(0.25), curve.1.powf(0.25));
        Ok(Self {
            parts,
            epochs: u32::try_from(epochs)
                .map_err(|_| UmapError::Overflow("recursive layout epochs"))?,
            a: curve.0 as f32,
            b: curve.1 as f32,
            good_initialization: parts / config.coarsening_ratio > config.base_init_threshold,
            curve,
        })
    }
}

/// Half of the realized coarse size, rounded to nearest even and clamped.
pub fn coarse_negative_range(rows: usize) -> usize {
    let half = rows / 2;
    (half + (rows % 2) * (half % 2)).max(1).min(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_admission_rejection_precedes_layout() {
        let graph = HostCsr::from_coo(2, &[0, 1], &[1, 0], &[1.0, 1.0]).unwrap();
        let mut stages = Vec::new();
        let result = run(
            &graph,
            &[1.0, 2.0],
            1,
            2,
            (1.0, 1.0),
            &Default::default(),
            11,
            |stage, bytes| {
                assert!(bytes > 0);
                stages.push(stage);
                if stage == "recursive base" {
                    Err(UmapError::Capacity(stage.into()))
                } else {
                    Ok(())
                }
            },
            |_, _, _, _, _| panic!("base-only graph must not launch a layout"),
        );
        assert!(matches!(result, Err(UmapError::Capacity(stage)) if stage == "recursive base"));
        assert_eq!(stages, ["recursive anchor", "recursive base"]);
    }

    #[test]
    fn upfront_admission_includes_base_and_level_one_bounds() {
        let base = HostCsr::from_coo(2, &[0, 1], &[1, 0], &[1.0; 2]).unwrap();
        let config = RecursiveInitConfig {
            base_init_threshold: 2,
            ..Default::default()
        };
        for graph in [
            base,
            HostCsr::from_coo(3, &[0, 1, 1, 2], &[1, 0, 2, 1], &[1.0; 4]).unwrap(),
        ] {
            let params = LevelParams::first(graph.n, 1.0, 1.0, &config).unwrap();
            let owned = graph.n * (2 * size_of::<f32>() + size_of::<bool>());
            let later = match params {
                Some(params) => {
                    assert_eq!(params.parts, 2);
                    level_one_bound(&graph, 2, params.parts).unwrap()
                }
                None => base_allocation_bytes(graph.n, 2).unwrap(),
            };
            let mut calls = 0;
            let result = run(
                &graph,
                &[],
                1,
                2,
                (1.0, 1.0),
                &config,
                11,
                |stage, bytes| {
                    calls += 1;
                    assert_eq!(stage, "recursive anchor");
                    assert!(bytes >= owned + later);
                    Err(UmapError::Capacity(stage.into()))
                },
                |_, _, _, _, _| panic!("upfront rejection must precede layout"),
            );
            assert!(matches!(result, Err(UmapError::Capacity(_))));
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn recursive_work_stages_reject_before_later_stages() {
        let graph = HostCsr::from_coo(4, &[0, 1, 2, 3], &[1, 0, 3, 2], &[1.0; 4]).unwrap();
        let config = RecursiveInitConfig {
            base_init_threshold: 2,
            ..Default::default()
        };
        for rejected in [
            "recursive outliers",
            "recursive remap",
            "recursive count",
            "recursive aggregate",
        ] {
            let mut stages = Vec::new();
            let result = run(
                &graph,
                &[1.0, 2.0, 3.0, 4.0],
                1,
                2,
                (1.0, 1.0),
                &config,
                11,
                |stage, bytes| {
                    stages.push(stage);
                    if stage == "recursive outliers" {
                        // Labels are already live; admission charges only the new stack.
                        assert_eq!(bytes, 101 * size_of::<usize>());
                    }
                    if stage == rejected {
                        Err(UmapError::Capacity(stage.into()))
                    } else {
                        Ok(())
                    }
                },
                |_, _, _, _, _| panic!("rejected stage must precede layout"),
            );
            assert!(matches!(result, Err(UmapError::Capacity(stage)) if stage == rejected));
            assert_eq!(stages.last(), Some(&rejected));
        }
    }

    #[test]
    fn level_targets_use_requested_parts_and_realized_clamps() {
        let config = RecursiveInitConfig {
            base_init_threshold: 2,
            ..Default::default()
        };
        assert!(LevelParams::first(2, 16.0, 1.0, &config).unwrap().is_none());
        assert_eq!(
            LevelParams::first(3, 16.0, 1.0, &config)
                .unwrap()
                .unwrap()
                .parts,
            2
        );
        let first = LevelParams::first(400, 16.0, 1.0, &config)
            .unwrap()
            .unwrap();
        assert_eq!((first.parts, first.epochs, first.a), (100, 50, 2.0));
        let second = first.next(60, &config).unwrap().unwrap();
        assert_eq!((second.parts, second.epochs), (25, 70));
        assert_eq!((coarse_negative_range(5), coarse_negative_range(7)), (2, 4));
    }
}

/// Draw a pair of independent standard normal variates by Box–Muller.
pub fn gaussian_pair(rng: &mut StdRng) -> (f64, f64) {
    let radius = (-2.0 * (1.0 - rng.random::<f64>()).ln()).sqrt();
    let angle = std::f64::consts::TAU * rng.random::<f64>();
    (radius * angle.cos(), radius * angle.sin())
}
