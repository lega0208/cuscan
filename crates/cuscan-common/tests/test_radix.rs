//! Stable radix argsort against an independent host stable numeric sort.

use cubecl::prelude::*;
use cuscan_common::cuda_backend;
use cuscan_common::radix::*;

fn oracle(values: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..values.len() as u32).collect();
    ids.sort_by(|&a, &b| values[a as usize].partial_cmp(&values[b as usize]).unwrap());
    ids
}

fn clients() -> Vec<(&'static str, Client)> {
    let mut out = vec![("cpu", cubecl::Device::cpu().expect("cpu device").client())];
    let required = std::env::var("CUSCAN_REQUIRE_CUDA").as_deref() == Ok("1");
    if cuda_backend::is_available() {
        out.push((
            "cuda",
            cubecl::Device::cuda(0).expect("cuda device").client(),
        ));
    } else {
        assert!(!required, "CUSCAN_REQUIRE_CUDA=1 but no CUDA device");
    }
    out
}

fn configs(client: &Client) -> Vec<RadixSortConfig> {
    let path = RadixDeviceLimits::from_client(client).path;
    let base = RadixSortConfig::conservative(path);
    match path {
        RadixPath::SerialCpu => vec![
            base,
            RadixSortConfig {
                digit_bits: 11,
                ..base
            },
            RadixSortConfig {
                digit_bits: 3,
                max_cubes: 7,
                cpu_min_chunk_rows: 5,
                ..base
            },
        ],
        RadixPath::CooperativePlane => vec![
            base,
            RadixSortConfig {
                digit_bits: 4,
                max_cubes: 3,
                cube_dim: 64,
                ..base
            },
            RadixSortConfig {
                digit_bits: 5,
                max_cubes: 1,
                cube_dim: 32,
                ..base
            },
        ],
    }
}

fn sort(
    client: &Client,
    values: &[f32],
    config: RadixSortConfig,
) -> Result<(Vec<u32>, Vec<f32>, RadixSortStats), RadixSortError> {
    let input = client.create_from_slice(f32::as_bytes(if values.is_empty() {
        &[0.0]
    } else {
        values
    }));
    let mut workspace = RadixSortWorkspace::new(client, values.len(), config)?;
    let output = workspace.sort(&input, values.len())?;
    let ids = u32::from_bytes(&client.read_one(output.ids).unwrap())[..values.len()].to_vec();
    let sorted =
        f32::from_bytes(&client.read_one(output.sorted_values).unwrap())[..values.len()].to_vec();
    Ok((ids, sorted, output.stats))
}

fn check(values: &[f32]) {
    let expected = oracle(values);
    for (name, client) in clients() {
        for config in configs(&client) {
            let (ids, sorted, _) = sort(&client, values, config).unwrap();
            assert_eq!(ids, expected, "{name} {config:?} n={}", values.len());
            let mut seen = vec![false; values.len()];
            for &id in &ids {
                assert!(!std::mem::replace(&mut seen[id as usize], true));
            }
            let bits: Vec<u32> = sorted.iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = ids.iter().map(|&i| values[i as usize].to_bits()).collect();
            assert_eq!(bits, want, "{name} gathered values keep original bits");
        }
    }
}

fn pseudo(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u32 ^ seed)
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453);
            x ^= x >> 16;
            match i % 11 {
                0 => -0.0,
                1 => 0.0,
                2 => (x % 7) as f32,
                _ => {
                    if x & 0x7f80_0000 == 0x7f80_0000 {
                        x ^= 0x0080_0000;
                    }
                    f32::from_bits(x)
                }
            }
        })
        .collect()
}

#[test]
fn empty_and_singleton() {
    check(&[]);
    check(&[-0.0]);
    check(&[3.5]);
}

#[test]
fn duplicates_signed_zero_and_extremes() {
    check(&[0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 1.0, -1.0]);
    check(&[
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        0.0,
        -0.0,
        f32::MAX,
        f32::MIN,
    ]);
    check(&vec![2.0; 1000]);
}

#[test]
fn ordered_reversed_random_and_tails() {
    for n in [
        2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1003, 4097, 65_537, 300_001,
    ] {
        let ordered: Vec<f32> = (0..n).map(|i| i as f32 - 7.0).collect();
        let reversed: Vec<f32> = ordered.iter().rev().copied().collect();
        check(&ordered);
        check(&reversed);
        check(&pseudo(n, n as u32));
    }
}

#[test]
fn skipped_passes_track_parity() {
    for (name, client) in clients() {
        let config = RadixSortConfig::conservative(RadixDeviceLimits::from_client(&client).path);
        // All equal: every pass skipped, result in slot a.
        let (_, _, stats) = sort(&client, &[5.0; 300], config).unwrap();
        assert_eq!((stats.passes_executed, stats.result_slot), (0, 0), "{name}");
        // Keys differing only in the lowest byte: one pass, odd parity.
        let base = 1.0f32.to_bits();
        let one: Vec<f32> = (0..300u32)
            .map(|i| f32::from_bits(base | (i * 7 % 256)))
            .collect();
        let (ids, _, stats) = sort(&client, &one, config).unwrap();
        assert_eq!(ids, oracle(&one));
        assert_eq!(
            (
                stats.passes_executed,
                stats.passes_skipped,
                stats.result_slot
            ),
            (1, 3, 1),
            "{name}"
        );
        // Lowest two bytes vary: two passes, even parity.
        let two: Vec<f32> = (0..300u32)
            .map(|i| f32::from_bits(base | (i * 7919 % 65_536)))
            .collect();
        let (ids, _, stats) = sort(&client, &two, config).unwrap();
        assert_eq!(ids, oracle(&two));
        assert_eq!((stats.passes_executed, stats.result_slot), (2, 0), "{name}");
    }
}

#[test]
fn non_finite_inputs_are_rejected_with_lowest_row() {
    for (name, client) in clients() {
        let config = RadixSortConfig::conservative(RadixDeviceLimits::from_client(&client).path);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = pseudo(5000, 3);
            values[4321] = bad;
            values[1234] = bad;
            assert_eq!(
                sort(&client, &values, config).unwrap_err(),
                RadixSortError::NonFinite { index: 1234 },
                "{name}"
            );
            assert_eq!(
                sort(&client, &[bad], config).unwrap_err(),
                RadixSortError::NonFinite { index: 0 }
            );
        }
    }
}

#[test]
fn backends_agree_for_identical_keys() {
    let values = pseudo(200_003, 9);
    let mut results = Vec::new();
    for (_, client) in clients() {
        for config in configs(&client) {
            results.push(sort(&client, &values, config).unwrap().0);
        }
    }
    assert!(results.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn workspace_is_reusable_and_bounded() {
    for (_, client) in clients() {
        let config = RadixSortConfig::conservative(RadixDeviceLimits::from_client(&client).path);
        let mut workspace = RadixSortWorkspace::new(&client, 70_000, config).unwrap();
        for n in [70_000, 1, 0, 33_333, 70_000] {
            let values = pseudo(n, n as u32 + 1);
            let input =
                client.create_from_slice(f32::as_bytes(if n == 0 { &[0.0] } else { &values }));
            let output = workspace.sort(&input, n).unwrap();
            let ids = u32::from_bytes(&client.read_one(output.ids).unwrap())[..n].to_vec();
            assert_eq!(ids, oracle(&values));
        }
        let input = client.create_from_slice(f32::as_bytes(&[0.0; 4]));
        assert!(matches!(
            workspace.sort(&input, 70_001),
            Err(RadixSortError::Workspace(_))
        ));
        // Raw handles expose physical allocation size, including device padding.
        let beyond_allocation = input.size() as usize / size_of::<f32>() + 1;
        assert!(beyond_allocation <= workspace.capacity());
        assert!(matches!(
            workspace.sort(&input, beyond_allocation),
            Err(RadixSortError::InvalidInput(_))
        ));
    }
}

#[test]
fn more_than_two_pow_24_rows() {
    let n = (1usize << 24) + 3;
    // Distinct f32 integers stop at 2^24, so ties beyond it exercise stability.
    let values: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    check(&values);
}

fn limits(path: RadixPath) -> RadixDeviceLimits {
    RadixDeviceLimits {
        path,
        alignment: 256,
        max_allocation_bytes: u64::MAX,
        max_cube_count_x: 65_535,
        max_units_per_cube: 1_024,
        max_cube_dim_x: 1_024,
        plane_ops: true,
        plane_size_min: 32,
        plane_size_max: 32,
        max_shared_memory_bytes: 48 * 1024,
    }
}

#[test]
fn plan_rejects_invalid_limits_and_overflow() {
    let cpu = RadixSortConfig::conservative(RadixPath::SerialCpu);
    let gpu = RadixSortConfig::conservative(RadixPath::CooperativePlane);
    let l = limits(RadixPath::CooperativePlane);
    assert!(matches!(
        radix_sort_plan_for_limits(u32::MAX as usize + 1, gpu, l),
        Err(RadixSortError::InvalidInput(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            RadixSortConfig {
                digit_bits: 0,
                ..gpu
            },
            l
        ),
        Err(RadixSortError::InvalidInput(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            RadixSortConfig {
                digit_bits: 12,
                ..gpu
            },
            l
        ),
        Err(RadixSortError::InvalidInput(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            RadixSortConfig {
                max_cubes: 0,
                ..gpu
            },
            l
        ),
        Err(RadixSortError::InvalidInput(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            gpu,
            RadixDeviceLimits {
                plane_ops: false,
                ..l
            }
        ),
        Err(RadixSortError::Capability(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            gpu,
            RadixDeviceLimits {
                plane_size_max: 64,
                ..l
            }
        ),
        Err(RadixSortError::Capability(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            RadixSortConfig {
                digit_bits: 11,
                ..gpu
            },
            l
        ),
        Err(RadixSortError::Capability(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(10, gpu, RadixDeviceLimits { alignment: 0, ..l }),
        Err(RadixSortError::Capability(_))
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(10, cpu, limits(RadixPath::SerialCpu)).map(|p| p.cube_dim),
        Ok(1)
    ));
    assert!(matches!(
        radix_sort_plan_for_limits(
            10,
            RadixSortConfig { cube_dim: 2, ..cpu },
            limits(RadixPath::SerialCpu)
        ),
        Err(RadixSortError::InvalidInput(_))
    ));
}

#[test]
fn plan_checks_each_aligned_allocation_independently() {
    let gpu = RadixSortConfig::conservative(RadixPath::CooperativePlane);
    let plan =
        radix_sort_plan_for_limits(1_000_000, gpu, limits(RadixPath::CooperativePlane)).unwrap();
    let names: Vec<_> = plan.allocations.iter().map(|a| a.buffer).collect();
    assert_eq!(
        &names[..6],
        &[
            "keys_a",
            "keys_b",
            "ids_a",
            "ids_b",
            "sorted_values",
            "histogram_table"
        ]
    );
    assert_eq!(*names.last().unwrap(), "status");
    assert_eq!(
        plan.total_aligned_bytes,
        plan.allocations
            .iter()
            .map(|a| a.aligned_bytes)
            .sum::<u64>()
    );
    assert!(
        plan.allocations
            .iter()
            .all(|a| a.aligned_bytes % 256 == 0 && a.aligned_bytes >= a.bytes)
    );
    // A limit just below the row buffers fails on the first one, naming it.
    let limit = plan.allocations[0].aligned_bytes - 1;
    let error = radix_sort_plan_for_limits(
        1_000_000,
        gpu,
        RadixDeviceLimits {
            max_allocation_bytes: limit,
            ..limits(RadixPath::CooperativePlane)
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        RadixSortError::Allocation {
            buffer: "keys_a",
            required_bytes: limit + 1,
            limit_bytes: limit
        }
    );
    // A limit admitting row buffers but not the larger histogram table fails on that table.
    let small = RadixSortConfig {
        max_cubes: 65_535,
        ..gpu
    };
    let plan =
        radix_sort_plan_for_limits(4_000, small, limits(RadixPath::CooperativePlane)).unwrap();
    let table = plan.allocations[5].aligned_bytes;
    assert!(table > plan.allocations[0].aligned_bytes);
    let error = radix_sort_plan_for_limits(
        4_000,
        small,
        RadixDeviceLimits {
            max_allocation_bytes: table - 1,
            ..limits(RadixPath::CooperativePlane)
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        RadixSortError::Allocation {
            buffer: "histogram_table",
            ..
        }
    ));
    // Cube-count limits bound chunk ownership.
    let plan = radix_sort_plan_for_limits(
        10_000_000,
        gpu,
        RadixDeviceLimits {
            max_cube_count_x: 5,
            ..limits(RadixPath::CooperativePlane)
        },
    )
    .unwrap();
    assert!(plan.cubes <= 5 && plan.cubes as usize * plan.chunk_len >= 10_000_000);
}
