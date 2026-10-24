use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use fast_hdbscan::{BackendChoice, HdbscanOps};
use fast_hdbscan_core::mst::ComponentReductionPlan;

fn oracle(labels: &[i32], weights: &[f32], indices: &[i32]) -> (Vec<u32>, Vec<i32>) {
    let n = labels.len();
    let mut roots = Vec::new();
    let mut winners: Vec<Option<usize>> = Vec::new();
    let mut bad: Vec<Option<usize>> = Vec::new();
    for source in 0..n {
        let slot = roots
            .iter()
            .position(|root| *root == labels[source])
            .unwrap_or_else(|| {
                roots.push(labels[source]);
                winners.push(None);
                bad.push(None);
                roots.len() - 1
            });
        if weights[source].is_nan() || weights[source] < 0.0 || indices[source] >= n as i32 {
            if bad[slot].is_none() {
                bad[slot] = Some(source);
            }
        } else if indices[source] >= 0
            && winners[slot].is_none_or(|old| weights[source] < weights[old])
        {
            winners[slot] = Some(source);
        }
    }
    let mut output = Vec::new();
    let mut metadata = Vec::new();
    for slot in 0..roots.len() {
        let (source, destination, bits) = winners[slot]
            .map_or((-1, -1, f32::INFINITY.to_bits()), |source| {
                (source as i32, indices[source], weights[source].to_bits())
            });
        let (bad_row, kind, bad_destination) = bad[slot].map_or((n as i32, 0, -1), |row| {
            if weights[row].is_nan() || weights[row] < 0.0 {
                (row as i32, 1, -1)
            } else {
                (row as i32, 2, indices[row])
            }
        });
        output.push(bits);
        metadata.extend([source, destination, bad_row, kind, bad_destination]);
    }
    (output, metadata)
}

fn check(device: &Device, labels: &[i32], values: &[f32], destinations: &[i32], width: usize) {
    let n = labels.len();
    let plan = ComponentReductionPlan::build(labels, width).unwrap();
    let c = plan.offsets.len() - 1;
    let int = |values: Vec<i32>| {
        let len = values.len();
        Tensor::<1, Int>::from_data(TensorData::new(values, [len]), device).into_dispatch()
    };
    // Dispatch wraps Cube in Fusion in this workspace. Cat creates an actual
    // pending dependency before the extension, instead of testing only uploads.
    let weights = if n > 1 {
        Tensor::<1>::cat(
            vec![
                Tensor::from_data(TensorData::new(values[..n / 2].to_vec(), [n / 2]), device),
                Tensor::from_data(
                    TensorData::new(values[n / 2..].to_vec(), [n - n / 2]),
                    device,
                ),
            ],
            0,
        )
    } else {
        Tensor::<1>::from_data(TensorData::new(values.to_vec(), [n]), device)
    };
    let (weights, metadata) = <Dispatch as HdbscanOps>::component_reduce(
        weights.into_dispatch(),
        int(destinations.to_vec()),
        int(plan.members),
        int(plan.tile_starts),
        int(plan.tile_ends),
        int(plan.partial_offsets),
        c as u32,
    );
    let weights = Tensor::<1>::from_dispatch(weights);
    let metadata = Tensor::<2, Int>::from_dispatch(metadata);
    assert_eq!(weights.dims(), [c]);
    assert_eq!(metadata.dims(), [c, 5]);
    let bits = weights
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let metadata = metadata.to_data().try_to_vec::<i32>().unwrap();
    assert_eq!(
        (bits, metadata),
        oracle(labels, values, destinations),
        "n={n}, tile_width={width}"
    );
}

fn suite(device: &Device) {
    check(device, &[], &[], &[], 512);
    check(device, &[0], &[f32::INFINITY], &[-7], 512);
    check(device, &[0], &[-0.0], &[0], 512);
    // Numeric errors precede index errors, even on absent/losing rows.
    check(
        device,
        &[3, 1, 3, 1],
        &[-0.0, 0.0, f32::INFINITY, 2.0],
        &[3, 2, 5, 0],
        512,
    );
    for weight in [
        f32::NAN,
        f32::from_bits(0xffc0_0001),
        -1.0,
        f32::NEG_INFINITY,
    ] {
        for destination in [-8, 0, 4] {
            check(
                device,
                &[3, 1, 3, 1],
                &[0.0, weight, f32::NAN, -2.0],
                &[3, destination, -1, 5],
                1,
            );
        }
    }
    for n in [2, 127, 128, 129, 511, 512, 513, 1539] {
        let palette = [
            -0.0,
            0.0,
            f32::from_bits(1),
            f32::from_bits(2),
            f32::MAX,
            f32::INFINITY,
            1.0,
        ];
        let mut weights = (0..n)
            .map(|row| palette[row % palette.len()])
            .collect::<Vec<_>>();
        let mut indices = (0..n)
            .map(|row| {
                if row % 11 == 0 {
                    -3
                } else {
                    ((row + 1) % n) as i32
                }
            })
            .collect::<Vec<_>>();
        // Stable nonmonotonic interleaved roots and a highly skewed giant segment.
        for labels in [
            (0..n)
                .map(|row| if row % 3 == 0 { (n - 1) as i32 } else { 0 })
                .collect::<Vec<_>>(),
            (0..n)
                .map(|row| if row + 1 == n { 0 } else { (n - 1) as i32 })
                .collect(),
            (0..n).map(|row| (n - row - 1) as i32).collect(),
            vec![0; n],
        ] {
            for width in [1, 127, 128, 129, 511, 512, 513] {
                check(device, &labels, &weights, &indices, width);
            }
        }
        // Same sizes/allocator pools, changed partitions, poisoned losing rows.
        weights[n - 1] = f32::NAN;
        indices[n / 2] = n as i32;
        check(device, &vec![0; n], &weights, &indices, 128);
        weights[n - 1] = -0.0;
        indices[n / 2] = -1;
        check(
            device,
            &(0..n).map(|row| (row % 2) as i32).collect::<Vec<_>>(),
            &weights,
            &indices,
            512,
        );
    }
}

#[test]
fn component_reduce_dispatch_fusion_cpu_oracle_matrix() {
    suite(&Device::cpu());
}

#[test]
fn component_reduce_dispatch_fusion_cuda_oracle_matrix() {
    match fast_cubecl_common::selected_device(BackendChoice::Cuda { device_index: 0 }) {
        Ok((device, _)) => suite(&device),
        Err(error) => {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable: {error}"
            );
            eprintln!("SKIPPED component reduction CUDA: {error}");
        }
    }
}
