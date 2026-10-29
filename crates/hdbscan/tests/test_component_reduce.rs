use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use hdbscan::{BackendChoice, HdbscanOps};
use hdbscan_core::mst::ComponentReductionPlan;

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
    // Short and ragged segments include starts that are not tile-aligned,
    // arbitrary root order, and singleton roots. Giant-versus-tiny shapes are
    // covered by the larger matrix below. Keep this matrix deliberately small
    // and deterministic.
    for len in 1..=7 {
        let labels = (0..len)
            .map(|i| [4, 1, 6, 1, 4, 6, 4][i] as usize % len)
            .map(|label| label as i32)
            .collect::<Vec<_>>();
        let values = (0..len)
            .map(|i| {
                [
                    f32::from_bits(1),
                    -0.0,
                    f32::INFINITY,
                    f32::MAX,
                    0.0,
                    2.0,
                    f32::from_bits(2),
                ][i]
            })
            .collect::<Vec<_>>();
        let destinations = (0..len)
            .map(|i| if i % 3 == 0 { -1 } else { (len - 1) as i32 })
            .collect::<Vec<_>>();
        for width in [1, 2, 3, 4, 5, 7, 8] {
            check(device, &labels, &values, &destinations, width);
        }
    }
    for len in [3, 4, 5, 7, 8, 9] {
        let mut labels = vec![0; len];
        labels[len - 1] = 1;
        labels[0] = 2;
        let mut values = vec![f32::MAX; len];
        values[0] = f32::INFINITY; // valid infinity must beat absence.
        values[len - 1] = f32::from_bits(1);
        let mut destinations = vec![1; len];
        destinations[0] = -9; // absent infinity, not a numeric error.
        destinations[len - 1] = (len - 1) as i32;
        if len > 3 {
            values[2] = f32::NAN;
            destinations[2] = len as i32;
            values[3] = -1.0;
            destinations[3] = -1;
        }
        for width in [3, 4, 5] {
            check(device, &labels, &values, &destinations, width);
        }
    }
    // Numeric errors precede destination errors on the same row, including
    // absent/losing rows; across rows the earliest bad source wins.
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
    // Multirow all-absent components, each spanning several tiles at both
    // widths. Interleaved roots group rows, so CSR positions differ from rows.
    let n = 1100;
    let labels = (0..n)
        .map(|row| if row % 2 == 0 { 0 } else { (n - 1) as i32 })
        .collect::<Vec<i32>>();
    let weights = (0..n)
        .map(|row| [-0.0, 0.0, f32::INFINITY, f32::MAX, f32::from_bits(1)][row % 5])
        .collect::<Vec<f32>>();
    let absent = vec![-1; n];
    for width in [512, 129] {
        check(device, &labels, &weights, &absent, width);
    }
    // Scattered invalid rows 1, 33, 129, and n-2 across interleaved roots.
    // Each root's earliest bad source is chosen by original row order; root
    // first-seen order is non-monotonic. Row n-2 is the only bad row of its
    // root and sits in a later width-129 tile.
    let n = 600;
    let roots: [i32; 3] = [599, 0, 317];
    let labels = (0..n).map(|row| roots[(row / 2) % 3]).collect::<Vec<i32>>();
    let mut weights = (0..n)
        .map(|row| [-0.0, 0.0, f32::INFINITY, f32::MAX, 1.0][row % 5])
        .collect::<Vec<f32>>();
    let mut indices = (0..n)
        .map(|row| {
            if row % 4 == 0 {
                -1
            } else {
                ((row * 7 + 3) % n) as i32
            }
        })
        .collect::<Vec<i32>>();
    weights[1] = 0.5;
    indices[1] = n as i32;
    weights[33] = f32::NAN;
    indices[33] = 4;
    weights[129] = -2.0;
    indices[129] = 7;
    weights[n - 2] = 0.25;
    indices[n - 2] = (n + 3) as i32;
    for width in [512, 129] {
        check(device, &labels, &weights, &indices, width);
    }
    // One component spanning several tiles. Other rows are 1.0. Signed zeros
    // compare equal, so the tie resolves to the lowest source: 530 must win
    // over the -0.0 at 690 and the +0.0 at 650 (a later width-129 tile).
    let n = 700;
    let labels = vec![0; n];
    let mut weights = vec![1.0_f32; n];
    weights[530] = -0.0;
    weights[650] = 0.0;
    weights[690] = -0.0;
    let indices = (0..n)
        .map(|row| ((row * 7 + 3) % n) as i32)
        .collect::<Vec<i32>>();
    for width in [512, 129] {
        check(device, &labels, &weights, &indices, width);
    }
}

#[test]
fn component_reduce_dispatch_fusion_cpu_oracle_matrix() {
    suite(&Device::cpu());
}

#[test]
fn component_reduce_dispatch_fusion_cuda_oracle_matrix() {
    match cuscan_common::selected_device(BackendChoice::Cuda { device_index: 0 }) {
        Ok((device, _)) => suite(&device),
        Err(error) => {
            assert_ne!(
                std::env::var("CUSCAN_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable: {error}"
            );
            eprintln!("SKIPPED component reduction CUDA: {error}");
        }
    }
}
