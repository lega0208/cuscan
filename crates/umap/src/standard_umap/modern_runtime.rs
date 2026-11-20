//! Estimator orchestration for compact modern layouts.
use burn::{
    backend::{
        BackendTensor, Dispatch, DispatchTensorKind,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{Device, Int, Tensor, TensorData},
};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use umap_core::{
    Optimizer, Result, UMAPConfig, UmapError,
    modern_layout::{
        BufferLifetime, BufferSpace, CompactGraph, LayoutOperation, LiveBuffer, ModernLayoutParams,
        ModernPlanInput, NegativeScaleAdaptation, WeightedEdge, adam_schedule, adaptation_due,
        compact_graph, effective_negative_range, modern_buffer_plan, momentum_schedule,
        resolved_negative_scale,
    },
};

use super::modern_negatives::{ForceSamples, ProjectionOrder};
use super::modern_orders::ModernOrders;
use crate::backend::StandardUmapOps;

pub(super) fn allocation_bytes(a: usize, b: usize, c: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|v| v.checked_mul(c))
        .ok_or(UmapError::Overflow("modern retained allocation"))
}

#[cfg(test)]
thread_local! {
    static FAIL_ALLOCATION: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn inject_allocation_failure(name: Option<&'static str>) {
    FAIL_ALLOCATION.with(|slot| slot.set(name));
}

#[cfg(test)]
#[derive(Debug)]
struct EpochObservation {
    epoch: u32,
    projection_ready: bool,
    before: f32,
    ratio: Option<f64>,
    gradient_scale: f32,
    negative_order: Option<Vec<i32>>,
    embedding: Option<Vec<f32>>,
    moment_before: Option<Vec<f32>>,
    moment_after: Option<Vec<f32>>,
    alpha: f32,
    mu: f32,
}

#[cfg(test)]
thread_local! {
    static EPOCH_OBSERVER: std::cell::RefCell<Option<Vec<EpochObservation>>> = const { std::cell::RefCell::new(None) };
}

fn check_injected_allocation(buffer: &LiveBuffer, _aligned: usize) -> Result<()> {
    check_injected_name(buffer.name)
}

/// Test-only named failure point shared by layout buffers and recursive stages.
pub(super) fn check_injected_name(name: &'static str) -> Result<()> {
    #[cfg(test)]
    if FAIL_ALLOCATION.with(|slot| slot.get() == Some(name)) {
        return Err(UmapError::Capacity(format!(
            "injected allocation failure: {name}"
        )));
    }
    let _ = name;
    Ok(())
}

pub(super) fn validate_host_allocations(
    allocations: &[(&'static str, usize, usize)],
) -> Result<usize> {
    let mut total = 0usize;
    for &(name, bytes, count) in allocations {
        if bytes > isize::MAX as usize {
            return Err(UmapError::Capacity(format!(
                "{name} allocation requires {bytes} bytes, limit {}",
                isize::MAX
            )));
        }
        total = total
            .checked_add(
                bytes
                    .checked_mul(count)
                    .ok_or(UmapError::Overflow("modern repeated buffers"))?,
            )
            .ok_or(UmapError::Overflow("modern live-buffer total"))?;
        check_injected_allocation(
            &LiveBuffer {
                name,
                space: BufferSpace::Host,
                lifetime: BufferLifetime::RepeatedBuild(count),
                bytes,
            },
            bytes,
        )?;
    }
    Ok(total)
}

/// UMAP exact search retains block outputs until concatenation and makes
/// contiguous query/reference copies inside the common neighbor backend.
pub(super) fn exact_search_buffers(
    buffers: &mut Vec<LiveBuffer>,
    device: &Device,
    rows: usize,
    references: usize,
    features: usize,
    k: usize,
) -> Result<()> {
    let block = references.saturating_sub(1).clamp(1, 64);
    let blocks = rows.div_ceil(block);
    let block_rows = rows.min(block);
    for (name, bytes, count) in [
        (
            "exact block indices",
            allocation_bytes(block_rows, k, 4)?,
            blocks,
        ),
        (
            "exact block distances",
            allocation_bytes(block_rows, k, 4)?,
            blocks,
        ),
        (
            "exact contiguous query",
            allocation_bytes(block_rows, features, 4)?,
            1,
        ),
        (
            "exact contiguous reference",
            allocation_bytes(references, features, 4)?,
            1,
        ),
        (
            "exact concatenated indices",
            matrix_allocation_bytes(device, rows, k, 4)?,
            1,
        ),
        (
            "exact concatenated distances",
            matrix_allocation_bytes(device, rows, k, 4)?,
            1,
        ),
    ] {
        buffers.push(LiveBuffer {
            name,
            space: BufferSpace::Device,
            lifetime: BufferLifetime::RepeatedBuild(count),
            bytes,
        });
    }
    Ok(())
}

/// Mirrors CubeCL's public pitched layout policy, using its alignment heuristic.
/// CPU uses the contiguous policy; CUDA uses the pitched policy.
pub(super) fn matrix_allocation_bytes(
    device: &Device,
    rows: usize,
    columns: usize,
    element: usize,
) -> Result<usize> {
    let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch();
    let width = columns
        .checked_mul(element)
        .ok_or(UmapError::Overflow("modern matrix row bytes"))?;
    let pitch = match cube {
        burn_cubecl::CubeDevice::Cpu(_) => width,
        burn_cubecl::CubeDevice::Cuda(_) => {
            let client = cube.client();
            let alignment = usize::try_from(client.properties().memory.alignment)
                .map_err(|_| UmapError::Overflow("device alignment"))?;
            if alignment < 16 || element == 0 || width.checked_next_power_of_two().is_none() {
                return Err(UmapError::InvalidInput(
                    "invalid device matrix alignment".into(),
                ));
            }
            let pitch_align =
                cubecl_server::memory_management::optimal_align(columns, element, alignment);
            let rem = width % pitch_align;
            if rem == 0 {
                width
            } else {
                width
                    .checked_add(pitch_align - rem)
                    .ok_or(UmapError::Overflow("modern matrix pitch"))?
            }
        }
        _ => {
            return Err(UmapError::BackendUnavailable(
                "modern matrix sizing requires CPU or CUDA".into(),
            ));
        }
    };
    rows.checked_mul(pitch)
        .ok_or(UmapError::Overflow("modern physical matrix bytes"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LayoutAdmissionStage {
    ConstructionBoundsOnly,
    BeforeCompactGraph,
    BeforeCsrUpload,
    CompactInput,
}

/// Future modern-layout allocations only. Earlier search, initialization and
/// already-owned inputs are outside this admission boundary.
#[allow(clippy::too_many_arguments)]
fn future_layout_buffers(
    stage: LayoutAdmissionStage,
    h: usize,
    n: usize,
    d: usize,
    capacity: usize,
    edges: usize,
    embedding: usize,
    adam: bool,
    alias_orders: bool,
    cuda: bool,
    caller: &[LiveBuffer],
    negative: &[LiveBuffer],
) -> Result<Vec<LiveBuffer>> {
    use BufferLifetime::{Both, Build, Layout, RepeatedBuild, RepeatedLayout};
    use BufferSpace::{Device, Host};
    let mut buffers = Vec::new();
    buffers
        .try_reserve(
            negative
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(caller.len()))
                .and_then(|n| n.checked_add(32))
                .ok_or(UmapError::Overflow("future layout table"))?,
        )
        .map_err(|e| UmapError::Memory(format!("future layout table: {e}")))?;
    let offsets = allocation_bytes(
        h.checked_add(1).ok_or(UmapError::Overflow("CSR offsets"))?,
        1,
        4,
    )?;
    let index = allocation_bytes(edges, 1, 4)?;
    let clock = allocation_bytes(edges, 1, 8)?;
    let mut add = |name, space, lifetime, bytes| {
        buffers.push(LiveBuffer {
            name,
            space,
            lifetime,
            bytes,
        })
    };
    if stage == LayoutAdmissionStage::BeforeCompactGraph {
        add(
            "graph readbacks",
            Host,
            RepeatedBuild(3),
            allocation_bytes(capacity, 1, 4)?,
        );
        add(
            "compact COO construction",
            Host,
            RepeatedBuild(3),
            allocation_bytes(capacity, 1, std::mem::size_of::<WeightedEdge>())?,
        );
        add("host CSR offsets", Host, Build, offsets);
        add(
            "host CSR indices and weights",
            Host,
            RepeatedBuild(2),
            index,
        );
        add("host CSR periods and clocks", Host, RepeatedBuild(4), clock);
    }
    add("signed CSR offsets", Host, Build, offsets);
    add("signed CSR indices", Host, Build, index);
    // Backend upload may create pageable and pinned host copies even when
    // TensorData takes ownership of the source vector.
    for (bytes, count) in [(offsets, 2), (index, 2), (clock, 8)] {
        // Global-negative layouts need not synchronize before state/order
        // allocation, so asynchronous upload owners may still be live there.
        add(
            "CSR upload staging",
            Host,
            BufferLifetime::RepeatedBoth(count),
            bytes,
        );
    }
    add("device CSR offsets", Device, Both, offsets);
    add("device CSR indices", Device, Both, index);
    add(
        "device CSR periods and clocks",
        Device,
        umap_core::modern_layout::BufferLifetime::RepeatedBoth(4),
        clock,
    );
    add("updates or velocity", Device, Layout, embedding);
    if adam {
        add("Adam moments", Device, RepeatedLayout(2), embedding);
    }
    for count in [h, if alias_orders { 0 } else { n }] {
        if count != 0 {
            let bytes = allocation_bytes(count, 1, 4)?;
            add("order host slots", Host, RepeatedLayout(2), bytes);
            add("order device slots", Device, RepeatedLayout(2), bytes);
        }
    }
    if cuda {
        add(
            "order initial pageable staging",
            Host,
            Layout,
            allocation_bytes(h.max(n), 1, 4)?,
        );
    }
    // Burn's into_data first requests a contiguous aligned tensor. A strided
    // source can require a fresh device copy before logical host materialization.
    add("contiguous embedding readback", Device, Layout, embedding);
    add(
        "embedding materialization",
        Host,
        Layout,
        allocation_bytes(h, d, 4)?,
    );
    buffers.extend(
        caller
            .iter()
            .filter(|b| matches!(b.lifetime, BufferLifetime::FutureResult(_)))
            .copied(),
    );
    buffers.extend_from_slice(negative);
    // Upload owners can retain a producer vector and a separate staging slot.
    // Keep an additional host-sized bound for negative preparation staging.
    buffers.extend(negative.iter().filter(|b| b.space == Host).copied());
    Ok(buffers)
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use umap_core::modern_layout::aligned_buffer_peaks;

    fn table(
        stage: LayoutAdmissionStage,
        caller: &[LiveBuffer],
        negative: &[LiveBuffer],
    ) -> Vec<LiveBuffer> {
        future_layout_buffers(
            stage, 128, 128, 2, 256, 200, 1024, true, false, true, caller, negative,
        )
        .unwrap()
    }

    #[test]
    fn full_layout_is_admitted_even_without_negative_scratch() {
        let negative = [LiveBuffer {
            name: "negative",
            space: BufferSpace::Device,
            lifetime: BufferLifetime::Layout,
            bytes: 100,
        }];
        let buffers = table(LayoutAdmissionStage::BeforeCompactGraph, &[], &negative);
        let peaks = aligned_buffer_peaks(&buffers, 1, 1).unwrap();
        assert!(100 < peaks.layout_device);
        let budget = peaks.layout_device as u64 - 1;
        assert!(
            buffers
                .iter()
                .filter(|b| b.space == BufferSpace::Device)
                .all(|b| (b.bytes as u64) <= budget)
        );
        assert!(matches!(
            peaks.admit_future(u64::MAX, Some(budget)),
            Err(UmapError::Memory(_))
        ));
        let global = aligned_buffer_peaks(
            &table(LayoutAdmissionStage::BeforeCompactGraph, &[], &[]),
            1,
            1,
        )
        .unwrap();
        assert!(matches!(
            global.admit_future(u64::MAX, Some(100)),
            Err(UmapError::Memory(_))
        ));
        assert!(matches!(
            global.admit_future(0, Some(u64::MAX)),
            Err(UmapError::Memory(_))
        ));
        assert!(matches!(
            global.admit_future(global.layout_device as u64, None),
            Err(UmapError::Memory(_))
        ));
    }

    #[test]
    fn only_future_results_are_charged_and_spectral_bounds_are_not_admitted() {
        let caller = [
            LiveBuffer {
                name: "already owned",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Both,
                bytes: usize::MAX,
            },
            LiveBuffer {
                name: "centroid bound",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::RepeatedBuild(20000),
                bytes: 20000 * 8,
            },
            LiveBuffer {
                name: "future output",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::FutureResult(3),
                bytes: 10,
            },
        ];
        let base = aligned_buffer_peaks(
            &table(LayoutAdmissionStage::BeforeCsrUpload, &[], &[]),
            1,
            1,
        )
        .unwrap();
        let with = aligned_buffer_peaks(
            &table(LayoutAdmissionStage::BeforeCsrUpload, &caller, &[]),
            1,
            1,
        )
        .unwrap();
        assert_eq!(with.build_host, base.build_host);
        assert_eq!(with.layout_host, base.layout_host + 30);
        assert!(with.admit_future(10_000_000, Some(10_000_000)).is_ok());
        let before = aligned_buffer_peaks(
            &table(LayoutAdmissionStage::BeforeCompactGraph, &[], &[]),
            1,
            1,
        )
        .unwrap();
        assert!(before.build_host > base.build_host);
    }

    #[test]
    fn spectral_construction_bounds_do_not_reject_feasible_curve_layout() {
        let spectral = umap_core::init::spectral_solver_buffers(20_000, 2, 3, 600_000).unwrap();
        let future = future_layout_buffers(
            LayoutAdmissionStage::BeforeCompactGraph,
            20_000,
            20_000,
            2,
            600_000,
            600_000,
            160_000,
            true,
            false,
            false,
            &spectral,
            &[],
        )
        .unwrap();
        let peaks = aligned_buffer_peaks(&future, 1, 1).unwrap();
        assert!(peaks.admit_future(512 * 1024 * 1024, None).is_ok());
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn preflight(
    device: &Device,
    stage: LayoutAdmissionStage,
    operation: LayoutOperation,
    h: usize,
    n: usize,
    d: usize,
    features: usize,
    original_rows: usize,
    capacity: usize,
    edges: usize,
    config: &UMAPConfig,
    additional: &[LiveBuffer],
    ann: Option<&cuscan_common::AnnPlan>,
) -> Result<umap_core::modern_layout::ModernPeaks> {
    let coords = h
        .checked_mul(d)
        .ok_or(UmapError::Overflow("modern coordinates"))?;
    u32::try_from(coords).map_err(|_| UmapError::Overflow("modern coordinates"))?;
    let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch();
    let client = cube.client();
    let memory = &client.properties().memory;
    let mut complete = Vec::new();
    complete
        .try_reserve_exact(
            additional
                .len()
                .checked_add(8)
                .ok_or(UmapError::Overflow("modern plan entries"))?,
        )
        .map_err(|e| UmapError::Memory(format!("modern plan entries: {e}")))?;
    complete.extend_from_slice(additional);
    for (name, bytes) in [
        (
            "CSR signed offset conversion",
            allocation_bytes(
                h.checked_add(1).ok_or(UmapError::Overflow("CSR offsets"))?,
                1,
                4,
            )?,
        ),
        (
            "CSR signed index conversion",
            allocation_bytes(edges, 1, 4)?,
        ),
        ("graph head readback", allocation_bytes(capacity, 1, 4)?),
        ("graph tail readback", allocation_bytes(capacity, 1, 4)?),
        ("graph weight readback", allocation_bytes(capacity, 1, 4)?),
        (
            "input compact edges",
            allocation_bytes(capacity, 1, std::mem::size_of::<WeightedEdge>())?,
        ),
    ] {
        if stage == LayoutAdmissionStage::CompactInput
            && (name == "graph head readback"
                || name == "graph tail readback"
                || name == "graph weight readback"
                || name == "input compact edges")
        {
            continue;
        }
        if !additional.iter().any(|buffer| buffer.name == name) {
            complete.push(LiveBuffer {
                name,
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Build,
                bytes,
            });
        }
    }
    if !additional
        .iter()
        .any(|buffer| buffer.name == "embedding materialization")
    {
        complete.push(LiveBuffer {
            name: "embedding materialization",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Layout,
            bytes: allocation_bytes(h, d, 4)?,
        });
    }
    if matches!(cube, burn_cubecl::CubeDevice::Cuda(_)) {
        complete.push(LiveBuffer {
            name: "order initial pageable staging",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Layout,
            bytes: allocation_bytes(h.max(n), 1, 4)?,
        });
    }
    let negative_start = complete.len();
    super::modern_negatives::append_buffers(&mut complete, &client, operation, h, n, d, config)?;
    let alias_orders = operation == LayoutOperation::Fit
        && config.optimizer == Optimizer::Momentum
        && config.negative_selection_range >= n;
    let plan = modern_buffer_plan(ModernPlanInput {
        operation,
        source_rows: h,
        reference_rows: n,
        dimensions: d,
        features,
        original_rows,
        source_embedding_bytes: matrix_allocation_bytes(device, h, d, 4)?,
        reference_embedding_bytes: matrix_allocation_bytes(device, n, d, 4)?,
        graph_capacity: capacity,
        edges,
        adam: config.optimizer == Optimizer::Adam,
        alias_orders,
        additional: &complete,
    })?;
    let (host_orders, device_orders) = super::modern_orders::order_array_bytes(h, n, alias_orders)?;
    for (space, expected) in [
        (BufferSpace::Host, host_orders),
        (BufferSpace::Device, device_orders),
    ] {
        let actual = plan
            .buffers()
            .filter(|buffer| {
                buffer.space == space
                    && (buffer.name == "source order" || buffer.name == "negative order")
            })
            .try_fold(0usize, |sum, buffer| {
                sum.checked_add(buffer.bytes)
                    .ok_or(UmapError::Overflow("modern order accounting"))
            })?;
        if actual != expected {
            return Err(UmapError::Numerical(
                "modern order accounting mismatch".into(),
            ));
        }
    }
    plan.validate_allocations(
        1,
        usize::try_from(memory.alignment).map_err(|_| UmapError::Overflow("device alignment"))?,
        isize::MAX as usize,
        usize::try_from(memory.max_page_size)
            .map_err(|_| UmapError::Overflow("device allocation limit"))?,
        check_injected_allocation,
    )?;
    if stage != LayoutAdmissionStage::ConstructionBoundsOnly {
        let cpu = matches!(cube, burn_cubecl::CubeDevice::Cpu(_));
        let future = future_layout_buffers(
            stage,
            h,
            n,
            d,
            capacity,
            edges,
            matrix_allocation_bytes(device, h, d, 4)?,
            config.optimizer == Optimizer::Adam,
            alias_orders,
            !cpu,
            additional,
            &complete[negative_start..],
        )?;
        let future_peaks = umap_core::modern_layout::aligned_buffer_peaks(
            &future,
            1,
            usize::try_from(memory.alignment)
                .map_err(|_| UmapError::Overflow("device alignment"))?,
        )?;
        let device_budget = cuscan_common::resolve_memory_budget(device, None)
            .map_err(super::backend_choice::map_backend_error)?
            .usable_bytes(0);
        let (host_budget, separate_device) = if cpu {
            (device_budget, None)
        } else {
            (
                cuscan_common::host_usable_memory()
                    .map_err(super::backend_choice::map_backend_error)?,
                Some(device_budget),
            )
        };
        future_peaks.admit_future(host_budget, separate_device)?;
    }
    let mut peaks = plan.aligned_peaks(
        1,
        usize::try_from(memory.alignment).map_err(|_| UmapError::Overflow("device alignment"))?,
    )?;
    if let Some(ann) = ann {
        // ANN admission already checks every allocation against max_page_size.
        // Its fixed peak includes its input, so it is an alternative phase,
        // not another allocation to add to the later graph/layout phase.
        let ann_peak = ann
            .memory
            .fixed_peak_bytes
            .checked_add(ann.memory.staging_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(UmapError::Overflow("ANN live peak"))?;
        let logical_input = allocation_bytes(n, features, 4)?;
        let pitched_input = matrix_allocation_bytes(device, n, features, 4)?;
        let ann_peak = ann_peak
            .checked_add(pitched_input - logical_input)
            .ok_or(UmapError::Overflow("ANN live peak"))?;
        peaks.build_device = peaks.build_device.max(ann_peak);
        let host = cuscan_common::ann_host_estimate(ann.n, ann.features, ann.k)
            .map_err(super::estimator::map_ann_error)?;
        let retained_host = additional
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.name,
                    "original input rows"
                        | "original input row metadata"
                        | "distinct training rows"
                        | "distinct training row metadata"
                        | "unique inverse mapping"
                )
            })
            .try_fold(0usize, |sum, buffer| {
                let count = match buffer.lifetime {
                    BufferLifetime::RepeatedBoth(count) | BufferLifetime::RepeatedBuild(count) => {
                        count
                    }
                    _ => 1,
                };
                sum.checked_add(
                    buffer
                        .bytes
                        .checked_mul(count)
                        .ok_or(UmapError::Overflow("ANN retained host"))?,
                )
                .ok_or(UmapError::Overflow("ANN retained host"))
            })?;
        peaks.build_host = peaks.build_host.max(
            host.host_peak_bytes
                .checked_add(retained_host)
                .ok_or(UmapError::Overflow("ANN host live peak"))?,
        );
    }
    Ok(peaks)
}

fn raw_float(tensor: FloatTensor<Dispatch>) -> Result<CubeTensor> {
    let DispatchTensorKind::Cube(BackendTensor::Float(tensor)) = tensor.kind else {
        return Err(UmapError::BackendUnavailable(
            "modern layout requires CubeCL".into(),
        ));
    };
    Ok(tensor
        .client
        .clone()
        .resolve_tensor_float::<CubeBackend>(tensor))
}

fn raw_int(tensor: IntTensor<Dispatch>) -> Result<CubeTensor> {
    let DispatchTensorKind::Cube(BackendTensor::Int(tensor)) = tensor.kind else {
        return Err(UmapError::BackendUnavailable(
            "modern layout requires CubeCL".into(),
        ));
    };
    Ok(tensor
        .client
        .clone()
        .resolve_tensor_int::<CubeBackend>(tensor))
}

// Register each owner slot as a Fusion input without copying its device storage.
fn order_tensor(template: &FloatTensor<Dispatch>, raw: &CubeTensor) -> Result<IntTensor<Dispatch>> {
    use burn::backend::TensorMetadata;
    use burn::backend::fusion::custom::{
        CustomOpIr, HandleContainer, OperationFn, OperationIr, StreamId, TensorIr,
    };
    use burn::backend::fusion::{FusionBackend, FusionHandle};
    use burn::backend::{DispatchAutodiffContext, DispatchTensor};
    let DispatchTensorKind::Cube(BackendTensor::Float(template)) = &template.kind else {
        return Err(UmapError::BackendUnavailable(
            "modern layout requires CubeCL".into(),
        ));
    };
    let client = template.client.clone();
    let id = client.create_empty_handle();
    let output = TensorIr::uninit(id, raw.shape(), raw.dtype);
    let op = OperationIr::Custom(CustomOpIr::new("umap.modern_order", &[], &[output]));
    let raw = raw.clone();
    let mut outputs = client.register(
        StreamId::current(),
        op,
        OperationFn(
            move |handles: &mut HandleContainer<
                FusionHandle<<CubeBackend as FusionBackend>::FusionRuntime>,
            >| {
                handles.register_int_tensor::<CubeBackend>(&id, raw.clone());
                Ok(())
            },
        ),
    );
    let tensor = outputs.pop().expect("one order output");
    Ok(DispatchTensor {
        kind: DispatchTensorKind::Cube(BackendTensor::Int(tensor)),
        autodiff: DispatchAutodiffContext::Disabled,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize(
    head: Tensor<2>,
    tail: Tensor<2>,
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    epochs: u32,
    a: f32,
    b: f32,
    learning_rate: f32,
    config: &UMAPConfig,
    operation: LayoutOperation,
    seed: u64,
    features: usize,
    original_rows: usize,
    caller_buffers: &[LiveBuffer],
) -> Result<Tensor<2>> {
    let params = ModernLayoutParams::main(config, epochs, a, b, learning_rate, false, seed);
    optimize_device_graph(
        head,
        tail,
        heads,
        tails,
        weights,
        &params,
        operation,
        features,
        original_rows,
        caller_buffers,
    )
}

fn config_for_params(params: &ModernLayoutParams) -> UMAPConfig {
    UMAPConfig {
        optimizer: params.optimizer,
        negative_sample_rate: params.negative_sample_rate,
        repulsion_strength: params.repulsion,
        negative_selection_range: params.negative_selection_range,
        negative_sample_scale: params.negative_sample_scale,
        negative_sample_scale_adaptation_samples: params.adaptation_samples,
        exclude_graph_neighbors: params.exclude_graph_neighbors,
        ..UMAPConfig::default()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_device_graph(
    head: Tensor<2>,
    tail: Tensor<2>,
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    params: &ModernLayoutParams,
    operation: LayoutOperation,
    features: usize,
    original_rows: usize,
    caller_buffers: &[LiveBuffer],
) -> Result<Tensor<2>> {
    let epochs = params.epochs;
    let config = config_for_params(params);
    if epochs == 0 {
        return Ok(head);
    }
    let [h, d] = head.dims();
    let n = tail.dims()[0];
    let capacity = weights.dims()[0];
    let bytes = |count: usize, element: usize| {
        count
            .checked_mul(element)
            .ok_or(UmapError::Overflow("modern retained allocation"))
    };
    let graph_bytes = bytes(capacity, 4)?;
    let embedding_bytes = bytes(
        h.checked_mul(d)
            .ok_or(UmapError::Overflow("modern coordinates"))?,
        4,
    )?;
    let mut additional = Vec::new();
    additional
        .try_reserve_exact(
            caller_buffers
                .len()
                .checked_add(8)
                .ok_or(UmapError::Overflow("modern plan entries"))?,
        )
        .map_err(|e| UmapError::Memory(format!("modern plan entries: {e}")))?;
    additional.extend_from_slice(caller_buffers);
    additional.extend_from_slice(&[
        LiveBuffer {
            name: "graph head readback",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: graph_bytes,
        },
        LiveBuffer {
            name: "graph tail readback",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: graph_bytes,
        },
        LiveBuffer {
            name: "graph weight readback",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: graph_bytes,
        },
        LiveBuffer {
            name: "input compact edges",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: bytes(capacity, std::mem::size_of::<WeightedEdge>())?,
        },
        LiveBuffer {
            name: "embedding materialization",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Layout,
            bytes: embedding_bytes,
        },
        LiveBuffer {
            name: "CSR signed offset conversion",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: bytes(
                h.checked_add(1).ok_or(UmapError::Overflow("CSR offsets"))?,
                4,
            )?,
        },
        LiveBuffer {
            name: "CSR signed index conversion",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: graph_bytes,
        },
    ]);
    preflight(
        &head.device(),
        LayoutAdmissionStage::BeforeCompactGraph,
        operation,
        h,
        n,
        d,
        features,
        original_rows,
        capacity,
        capacity,
        &config,
        &additional,
        None,
    )?;
    let edges = host_positive_edges(heads, tails, weights, capacity)?;
    let graph = compact_graph(
        h,
        n,
        &edges,
        epochs as usize,
        config.negative_sample_rate as f32,
        operation,
    )?;
    drop(edges);
    optimize_compact_inner(
        head,
        tail,
        graph,
        params,
        operation,
        features,
        original_rows,
        capacity,
        &additional,
        LayoutAdmissionStage::BeforeCsrUpload,
    )
}

/// Read a device COO stream back and keep its positive entries in COO order.
fn host_positive_edges(
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    capacity: usize,
) -> Result<Vec<WeightedEdge>> {
    let heads = heads
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|e| UmapError::Numerical(format!("modern heads: {e}")))?;
    let tails = tails
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|e| UmapError::Numerical(format!("modern tails: {e}")))?;
    let weights = weights
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|e| UmapError::Numerical(format!("modern weights: {e}")))?;
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(capacity)
        .map_err(|e| UmapError::Memory(format!("modern graph allocation: {e}")))?;
    for ((row, column), weight) in heads.into_iter().zip(tails).zip(weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(UmapError::Numerical("invalid graph weight".into()));
        }
        if weight > 0.0 {
            edges.push(WeightedEdge {
                row: usize::try_from(row)
                    .map_err(|_| UmapError::Numerical("invalid graph row".into()))?,
                column: usize::try_from(column)
                    .map_err(|_| UmapError::Numerical("invalid graph column".into()))?,
                weight,
            });
        }
    }
    Ok(edges)
}

/// Measurement-only: the compact CSR graph, periods and initial clocks that
/// [`optimize`] builds and consumes for the same edge stream and epochs.
/// Returns `None` at zero epochs, where the layout returns its input unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn diagnostic_compact_schedule(
    rows: usize,
    columns: usize,
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    epochs: u32,
    config: &UMAPConfig,
    operation: LayoutOperation,
) -> Result<Option<umap_core::modern_layout::CompactGraph>> {
    if epochs == 0 {
        return Ok(None);
    }
    let capacity = weights.dims()[0];
    let edges = host_positive_edges(heads, tails, weights, capacity)?;
    compact_graph(
        rows,
        columns,
        &edges,
        epochs as usize,
        config.negative_sample_rate as f32,
        operation,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize_compact(
    head: Tensor<2>,
    tail: Tensor<2>,
    graph: CompactGraph,
    params: &ModernLayoutParams,
    operation: LayoutOperation,
    features: usize,
    original_rows: usize,
    caller_buffers: &[LiveBuffer],
) -> Result<Tensor<2>> {
    let capacity = graph.indices.len();
    optimize_compact_inner(
        head,
        tail,
        graph,
        params,
        operation,
        features,
        original_rows,
        capacity,
        caller_buffers,
        LayoutAdmissionStage::CompactInput,
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_compact_inner(
    head: Tensor<2>,
    tail: Tensor<2>,
    graph: CompactGraph,
    params: &ModernLayoutParams,
    operation: LayoutOperation,
    features: usize,
    original_rows: usize,
    capacity: usize,
    additional: &[LiveBuffer],
    stage: LayoutAdmissionStage,
) -> Result<Tensor<2>> {
    let epochs = params.epochs;
    if epochs == 0 {
        return Ok(head);
    }
    let config = config_for_params(params);
    let [h, d] = head.dims();
    let n = tail.dims()[0];
    let momentum = params.optimizer == Optimizer::Momentum;
    let shared = operation == LayoutOperation::Fit;
    let head_dispatch = head.clone().into_dispatch();
    let template = raw_float(head_dispatch.clone())?;
    let a = params.a;
    let b = params.b;
    let seed = params.seed;
    preflight(
        &head.device(),
        stage,
        operation,
        h,
        n,
        d,
        features,
        original_rows,
        capacity,
        graph.indices.len(),
        &config,
        additional,
        None,
    )?;
    let device = head.device();
    let int = |v: Vec<u32>| -> Result<_> {
        let len = v.len();
        let mut converted = Vec::new();
        converted
            .try_reserve_exact(len)
            .map_err(|e| UmapError::Memory(format!("CSR conversion: {e}")))?;
        for x in v {
            converted.push(i32::try_from(x).map_err(|_| UmapError::Overflow("CSR index"))?);
        }
        Ok(Tensor::<1, Int>::from_data(TensorData::new(converted, [len]), &device).into_dispatch())
    };
    let float = |v: Vec<f64>| {
        let len = v.len();
        Tensor::<1>::from_data(
            TensorData::new(v, [len]),
            (&device, burn::tensor::DType::F64),
        )
        .into_dispatch()
    };
    drop(graph.weights);
    let offsets = int(graph.offsets)?;
    let indices = int(graph.indices)?;
    let periods = float(graph.periods);
    let negative_periods = float(graph.negative_periods);
    let mut positive = float(graph.next_positive);
    let mut negative = float(graph.next_negative);
    let mut current_head = head_dispatch.clone();
    let fixed_tail = tail.into_dispatch();
    let fixed_tail_raw = raw_float(fixed_tail.clone())?;
    let offsets_raw = raw_int(offsets.clone())?;
    let indices_raw = raw_int(indices.clone())?;
    let mut updates = Tensor::<2>::zeros([h, d], &device).into_dispatch();
    let mut m = if momentum {
        updates.clone()
    } else {
        Tensor::<2>::zeros([h, d], &device).into_dispatch()
    };
    let mut v = if momentum {
        updates.clone()
    } else {
        Tensor::<2>::zeros([h, d], &device).into_dispatch()
    };
    let hard = config.negative_selection_range < n;
    let mut orders = ModernOrders::new(&template, h, n, shared && momentum && !hard, seed)?;
    let mut projection = if hard {
        Some(ProjectionOrder::new(&template, h, n, d, shared, seed)?)
    } else {
        None
    };
    let mut projection_ready = hard && h > 1024;
    if projection_ready {
        projection
            .as_mut()
            .expect("hard projection owner")
            .refresh(&template, &fixed_tail_raw)?;
        orders.retire_reference_order()?;
    }
    let mut samples =
        if hard && shared && !momentum && config.negative_sample_scale_adaptation_samples > 0 {
            Some(ForceSamples::new(
                &template,
                config.negative_sample_scale_adaptation_samples,
                n,
            )?)
        } else {
            None
        };
    let mut slots: [Option<(IntTensor<Dispatch>, IntTensor<Dispatch>)>; 2] = [None, None];
    let scale = resolved_negative_scale(
        config.negative_selection_range,
        n,
        config.negative_sample_scale,
        config.negative_sample_scale_adaptation_samples > 0,
    )?;
    let mut scale = NegativeScaleAdaptation::new(scale)?;
    let range = u32::try_from(effective_negative_range(
        config.negative_selection_range,
        n,
    )?)
    .map_err(|_| UmapError::Overflow("negative gradient range"))?;
    let mut adam = adam_schedule(
        epochs as usize,
        params.initial_alpha as f64,
        params.good_initialization,
    );
    let mut velocity = momentum_schedule(
        epochs as usize,
        params.initial_alpha as f64,
        params.good_initialization,
    );
    for epoch in 0..epochs {
        #[cfg(test)]
        let before = scale.scale();
        #[cfg(test)]
        let mut observed_ratio = None;
        let slot = orders.epoch() % 2;
        if slots[slot].is_none() {
            let (source, negatives) = orders.current();
            slots[slot] = Some((
                order_tensor(&head_dispatch, source)?,
                order_tensor(&head_dispatch, negatives)?,
            ));
        }
        let (source, negatives) = slots[slot].as_ref().expect("registered order slot");
        let (negative_order, ranks) = if projection_ready {
            let projection = projection.as_ref().expect("hard projection owner");
            (
                order_tensor(&head_dispatch, &projection.order)?,
                order_tensor(&head_dispatch, &projection.ranks)?,
            )
        } else {
            (negatives.clone(), source.clone())
        };
        if adaptation_due(
            operation,
            !momentum,
            config.negative_sample_scale_adaptation_samples,
            config.negative_selection_range,
            n,
            epoch as usize,
            epochs as usize,
        ) {
            let embedding = raw_float(current_head.clone())?;
            let order = if projection_ready {
                &projection.as_ref().expect("projection prepared").order
            } else {
                orders.current().1
            };
            let ratio = samples
                .as_mut()
                .expect("eligible adaptation samples")
                .ratio(
                    &embedding,
                    order,
                    &offsets_raw,
                    &indices_raw,
                    config.negative_selection_range,
                    a,
                    b,
                    config.repulsion_strength,
                    config.exclude_graph_neighbors,
                )?;
            #[cfg(test)]
            {
                observed_ratio = Some(ratio);
            }
            scale.observe_ratio(ratio)?;
        }
        #[cfg(test)]
        EPOCH_OBSERVER.with(|observer| {
            if let Some(events) = observer.borrow_mut().as_mut() {
                events.push(EpochObservation {
                    epoch,
                    projection_ready,
                    before,
                    ratio: observed_ratio,
                    gradient_scale: scale.scale(),
                    negative_order: if (operation == LayoutOperation::Transform && epoch < 2)
                        || observed_ratio.is_some()
                    {
                        Some(
                            Tensor::<1, Int>::from_dispatch(negative_order.clone())
                                .to_data()
                                .try_to_vec::<i32>()
                                .expect("observed negative order"),
                        )
                    } else {
                        None
                    },
                    embedding: if observed_ratio.is_some()
                        || (operation == LayoutOperation::Transform && epoch < 2)
                    {
                        Some(
                            Tensor::<2>::from_dispatch(current_head.clone())
                                .to_data()
                                .try_to_vec::<f32>()
                                .expect("observed embedding"),
                        )
                    } else {
                        None
                    },
                    moment_before: if operation == LayoutOperation::Transform && epoch < 2 {
                        Some(
                            Tensor::<2>::from_dispatch(if momentum {
                                updates.clone()
                            } else {
                                m.clone()
                            })
                            .to_data()
                            .try_to_vec::<f32>()
                            .expect("observed moment"),
                        )
                    } else {
                        None
                    },
                    moment_after: None,
                    alpha: 0.0,
                    mu: 0.0,
                });
            }
        });
        let schedule = if momentum {
            velocity.next().expect("schedule")
        } else {
            adam.next().expect("schedule")
        }
        .to_f32();
        let correction1 = (1.0 - (0.9_f32 as f64).powf(epoch as f64 + 1.0)) as f32;
        let correction2 = (1.0 - (0.99_f32 as f64).powf(epoch as f64 + 1.0)) as f32;
        #[cfg(test)]
        EPOCH_OBSERVER.with(|observer| {
            if let Some(event) = observer
                .borrow_mut()
                .as_mut()
                .and_then(|events| events.last_mut())
            {
                event.alpha = schedule.alpha;
                event.mu = schedule.momentum;
            }
        });
        (current_head, positive, negative, updates, m, v) =
            <Dispatch as StandardUmapOps>::modern_epoch(
                current_head,
                fixed_tail.clone(),
                offsets.clone(),
                indices.clone(),
                source.clone(),
                negative_order,
                ranks,
                periods.clone(),
                positive,
                negative_periods.clone(),
                negative,
                updates,
                m,
                v,
                epoch,
                a,
                b,
                config.repulsion_strength,
                scale.scale(),
                range,
                schedule.alpha,
                schedule.momentum,
                correction1,
                correction2,
                momentum,
                shared,
                config.negative_sample_rate != 0,
                config.exclude_graph_neighbors,
            );
        #[cfg(test)]
        EPOCH_OBSERVER.with(|observer| {
            if let Some(event) = observer
                .borrow_mut()
                .as_mut()
                .and_then(|events| events.last_mut())
                && event.moment_before.is_some()
            {
                event.moment_after = Some(
                    Tensor::<2>::from_dispatch(if momentum { updates.clone() } else { m.clone() })
                        .to_data()
                        .try_to_vec::<f32>()
                        .expect("observed post-gradient moment"),
                );
            }
        });
        // Resolving drains the Fusion stream and submits the device work, without
        // waiting for it. The producer then overlaps with the submitted epoch.
        let head_raw = raw_float(current_head.clone())?;
        if projection.is_some() && epoch == 0 {
            // The small initial epoch consumes the original reference order.
            // Retire it before the producer so its unused shuffle cannot consume RNG.
            burn::cubecl::future::block_on(head_raw.client.sync()).map_err(|e| {
                UmapError::BackendUnavailable(format!("negative order retirement: {e}"))
            })?;
            orders.retire_reference_order()?;
            slots[slot] = None;
        }
        orders.begin_prepare_next()?;
        let pos_raw = raw_float(positive.clone())?;
        let neg_raw = raw_float(negative.clone())?;
        let updates_raw = raw_float(updates.clone())?;
        let m_raw = raw_float(m.clone())?;
        let v_raw = raw_float(v.clone())?;
        orders.finish_next(&[
            &head_raw.handle,
            &pos_raw.handle,
            &neg_raw.handle,
            &updates_raw.handle,
            &m_raw.handle,
            &v_raw.handle,
        ])?;
        if let Some(projection) = &mut projection {
            projection.refresh(&head_raw, &fixed_tail_raw)?;
            projection_ready = true;
        }
    }
    Ok(Tensor::from_dispatch(current_head))
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    // Whole optimize calls, not a sum of isolated kernel timings. Inputs and
    // their uploads are prepared before each timed call; optimize's own
    // allocations and graph compaction remain inside the measured interval.
    fn modern_runtime_budget(device: Device, backend: &str) {
        use burn::cubecl::{MemoryScope, future::block_on};
        use std::time::Instant;

        let n: usize = std::env::var("CUSCAN_MODERN_PROFILE_ROWS")
            .expect("CUSCAN_MODERN_PROFILE_ROWS is required")
            .parse()
            .expect("invalid profile rows");
        assert!(
            [20_000, 200_000].contains(&n),
            "profile rows must be 20000 or 200000"
        );
        let seed: u64 = std::env::var("CUSCAN_MODERN_PROFILE_SEED")
            .unwrap_or_else(|_| "11".into())
            .parse()
            .expect("invalid profile seed");
        assert_eq!(seed, 11, "approved profile seed is 11");
        const EPOCHS: u32 = 42;
        for d in [2, 16, 48] {
            for operation in [LayoutOperation::Fit, LayoutOperation::Transform] {
                let h = if operation == LayoutOperation::Fit {
                    n
                } else {
                    300
                };
                let values = |rows: usize, salt: u64| -> Vec<f32> {
                    (0..rows * d)
                        .map(|i| {
                            let mut x = (i as u64).wrapping_add(seed).wrapping_add(salt);
                            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                            ((x ^ (x >> 31)) % 20001) as f32 / 10000.0 - 1.0
                        })
                        .collect()
                };
                let source = values(h, 456);
                let reference = if operation == LayoutOperation::Fit {
                    source.clone()
                } else {
                    values(n, 123)
                };
                let heads: Vec<i32> = (0..h)
                    .flat_map(|i| std::iter::repeat_n(i as i32, 15))
                    .collect();
                let tails: Vec<i32> = (0..h)
                    .flat_map(|i| (1..=15).map(move |j| ((i * 17 + j * 101) % n) as i32))
                    .collect();
                for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
                    let mut measurements = Vec::new();
                    for range in [n, 512] {
                        let config = UMAPConfig {
                            optimizer,
                            negative_selection_range: range,
                            init: umap_core::Init::Spectral,
                            ..Default::default()
                        };
                        let mut times = [0.0; 2];
                        let mut immutable = true;
                        let mut memory = String::new();
                        for time in &mut times {
                            let head = Tensor::<2>::from_data(
                                TensorData::new(source.clone(), [h, d]),
                                &device,
                            );
                            let tail = if operation == LayoutOperation::Fit {
                                head.clone()
                            } else {
                                Tensor::<2>::from_data(
                                    TensorData::new(reference.clone(), [n, d]),
                                    &device,
                                )
                            };
                            let graph_heads = Tensor::<1, Int>::from_data(
                                TensorData::new(heads.clone(), [heads.len()]),
                                &device,
                            );
                            let graph_tails = Tensor::<1, Int>::from_data(
                                TensorData::new(tails.clone(), [tails.len()]),
                                &device,
                            );
                            let weights = Tensor::<1>::ones([heads.len()], &device);
                            let raw = raw_float(head.clone().into_dispatch()).unwrap();
                            let _ = raw_float(tail.clone().into_dispatch()).unwrap();
                            let _ = raw_int(graph_heads.clone().into_dispatch()).unwrap();
                            let _ = raw_int(graph_tails.clone().into_dispatch()).unwrap();
                            let _ = raw_float(weights.clone().into_dispatch()).unwrap();
                            let client = raw.client.clone();
                            block_on(client.sync()).expect("profile sync failed");
                            let start = Instant::now();
                            let result = optimize(
                                head,
                                tail.clone(),
                                graph_heads,
                                graph_tails,
                                weights,
                                EPOCHS,
                                1.3,
                                0.8,
                                0.1,
                                &config,
                                operation,
                                seed,
                                d,
                                n,
                                &[],
                            )
                            .unwrap();
                            let _ = raw_float(result.into_dispatch()).unwrap();
                            block_on(client.sync()).expect("profile sync failed");
                            *time = start.elapsed().as_secs_f64() * 1000.0;
                            if operation == LayoutOperation::Transform {
                                immutable &=
                                    tail.to_data().try_to_vec::<f32>().unwrap() == reference;
                            }
                            memory = format!("{:?}", client.memory_report(MemoryScope::Device));
                        }
                        assert!(immutable, "transform reference changed");
                        measurements.push((range, times, immutable, memory));
                    }
                    let added = (measurements[1].1[1] - measurements[0].1[1]) / f64::from(EPOCHS);
                    for (range, times, immutable, memory) in measurements {
                        println!(
                            "{}",
                            serde_json::json!({
                                "backend": backend, "n": n, "h": h, "d": d,
                                "optimizer": format!("{optimizer:?}"), "operation": format!("{operation:?}"),
                                "seed": seed, "range": range, "totalepochs": EPOCHS,
                                "first_call_ms": times[0], "warm_ms": times[1],
                                "addedwarmms_per_epoch": added,
                                "immutablecheck": if operation == LayoutOperation::Transform { Some(immutable) } else { None },
                                "memory_report_sample": memory,
                                "memory_scope": "client device memory_report sample, not peak",
                                "duration_scope": "synchronized complete optimize call including graph compaction and production workspace allocations; input construction excluded; first call may include JIT and prior cases may warm kernels",
                                "semantics": "global Q=N versus hard Q=512; selection and scale differ; small H=300 uses global unit-scale gradient despite hard-path projection"
                            })
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn modern_runtime_budget_cpu() {
        modern_runtime_budget(
            Device::new(burn::backend::DispatchDevice::Cube(
                burn_cubecl::CubeDevice::Cpu(Default::default()),
            )),
            "cpu",
        );
    }

    #[test]
    #[ignore]
    fn modern_runtime_budget_cuda() {
        assert!(crate::cuda_backend::device(0).is_some(), "CUDA is required");
        modern_runtime_budget(
            Device::new(burn::backend::DispatchDevice::Cube(
                burn_cubecl::CubeDevice::Cuda(Default::default()),
            )),
            "cuda",
        );
    }

    fn adaptation_runtime(device: Device) {
        use super::super::modern_negatives::FORCE_OBSERVER;
        for (samples, explicit) in [(4, None), (0, None), (4, Some(0.0))] {
            let n = 100;
            let coords: Vec<f32> = (0..n * 2)
                .map(|i| ((i * 31 % 113) as f32 - 56.0) * 0.017)
                .collect();
            let head = Tensor::<2>::from_data(TensorData::new(coords, [n, 2]), &device);
            let heads: Vec<i32> = (0..n as i32).collect();
            let tails: Vec<i32> = (0..n as i32).map(|i| (i + 1) % n as i32).collect();
            let config = UMAPConfig {
                optimizer: Optimizer::Adam,
                negative_selection_range: 7,
                negative_sample_scale_adaptation_samples: samples,
                negative_sample_scale: explicit,
                init: umap_core::Init::Spectral,
                ..Default::default()
            };
            EPOCH_OBSERVER.with(|observer| *observer.borrow_mut() = Some(Vec::new()));
            FORCE_OBSERVER.with(|observer| *observer.borrow_mut() = Some(Vec::new()));
            let result = optimize(
                head.clone(),
                head,
                Tensor::<1, Int>::from_data(TensorData::new(heads, [n]), &device),
                Tensor::<1, Int>::from_data(TensorData::new(tails, [n]), &device),
                Tensor::<1>::ones([n], &device),
                42,
                1.3,
                0.8,
                0.1,
                &config,
                LayoutOperation::Fit,
                11,
                2,
                n,
                &[],
            );
            let events = EPOCH_OBSERVER.with(|observer| observer.borrow_mut().take().unwrap());
            let force = FORCE_OBSERVER.with(|observer| observer.borrow_mut().take().unwrap());
            result.unwrap();
            assert_eq!(events.len(), 42);
            let due: Vec<_> = events
                .iter()
                .filter(|event| event.ratio.is_some())
                .map(|event| event.epoch)
                .collect();
            assert_eq!(due, if samples == 0 { vec![] } else { vec![10, 20] });
            assert_eq!(force.len(), due.len());
            for (epoch, snapshot) in due.iter().zip(&force) {
                let event = &events[*epoch as usize];
                let x = event.embedding.as_ref().unwrap();
                let order = event.negative_order.as_ref().unwrap();
                let count = snapshot.sources.len();
                let mut expected = Vec::new();
                for local in [false, true] {
                    for t in 0..count {
                        let rank = snapshot.sources[t] as usize;
                        let source = order[rank] as usize;
                        let first = rank.saturating_sub(7 / 2).min(n - 7);
                        let mut sum = [0.0_f64; 2];
                        for k in 0..5 {
                            let candidate_rank = if local {
                                first + snapshot.local[t * 5 + k] as usize
                            } else {
                                snapshot.global[t * 5 + k] as usize
                            };
                            let target = order[candidate_rank] as usize;
                            let delta = [
                                x[source * 2] as f64 - x[target * 2] as f64,
                                x[source * 2 + 1] as f64 - x[target * 2 + 1] as f64,
                            ];
                            let q = delta[0] * delta[0] + delta[1] * delta[1];
                            let gamma = config.repulsion_strength as f64;
                            let coefficient = 2.0 * gamma * (0.8_f32 as f64)
                                / ((0.001 + q) * ((1.3_f32 as f64) * q.powf(0.8_f32 as f64) + 1.0));
                            let norm = coefficient * q.sqrt();
                            let clip = if norm > 0.0 {
                                gamma * (norm / gamma).tanh() / norm
                            } else {
                                1.0
                            };
                            for c in 0..2 {
                                sum[c] += coefficient * delta[c] * clip;
                            }
                        }
                        expected.push((sum[0] * sum[0] + sum[1] * sum[1]).sqrt());
                    }
                }
                for (actual, oracle) in snapshot.resultants.iter().zip(&expected) {
                    assert!(
                        (actual - oracle).abs() <= 1e-8 * (1.0 + oracle.abs()),
                        "force {actual} != {oracle}"
                    );
                }
                let mut logs: Vec<f64> = expected[..count]
                    .iter()
                    .zip(&expected[count..])
                    .filter(|&(&g, &l)| g > 0.0 && l > 0.0)
                    .map(|(&g, &l)| g.ln() - l.ln())
                    .collect();
                logs.sort_by(f64::total_cmp);
                let ratio = if logs.is_empty() {
                    1.0
                } else {
                    let quantile = |q: f64| {
                        let p = (logs.len() - 1) as f64 * q;
                        let lo = p.floor() as usize;
                        let hi = p.ceil() as usize;
                        logs[lo] + (logs[hi] - logs[lo]) * (p - lo as f64)
                    };
                    let low = quantile(0.1);
                    let high = quantile(0.9);
                    (logs.iter().map(|v| v.clamp(low, high)).sum::<f64>() / logs.len() as f64).exp()
                };
                assert!((event.ratio.unwrap() - ratio).abs() <= 1e-8 * (1.0 + ratio));
            }
            assert!(!events[0].projection_ready);
            assert!(events[1..].iter().all(|event| event.projection_ready));
            for event in &events {
                if event.epoch != 20 || samples == 0 || explicit == Some(0.0) {
                    assert_eq!(event.before, event.gradient_scale);
                }
            }
            if samples > 0 && explicit.is_none() {
                let first = events[10].ratio.unwrap();
                let second = events[20].ratio.unwrap();
                let prior = events[10].before as f64;
                let target = (prior / first * second).clamp(prior * 0.5, prior * 2.0);
                let change = (target / prior).ln().clamp(0.9_f64.ln(), 1.1_f64.ln());
                let expected = (prior * (0.1 * change).exp()) as f32;
                assert!((events[20].gradient_scale - expected).abs() <= 1e-6);
            }
            if explicit == Some(0.0) {
                assert!(events.iter().all(|event| event.gradient_scale == 0.0));
            }
        }
    }

    #[test]
    fn adaptation_runtime_cpu() {
        adaptation_runtime(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cpu(Default::default()),
        )));
    }

    #[test]
    fn adaptation_runtime_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        adaptation_runtime(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cuda(Default::default()),
        )));
    }

    fn transform_orders(device: Device) {
        for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
            for n in [100, 300] {
                let h = 17;
                let source: Vec<f32> = (0..h * 2).map(|i| (i * 13 % 37) as f32 * 0.03).collect();
                let reference: Vec<f32> =
                    (0..n * 2).map(|i| (i * 29 % 103) as f32 * 0.02).collect();
                let head = Tensor::<2>::from_data(TensorData::new(source, [h, 2]), &device);
                let tail =
                    Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 2]), &device);
                let heads: Vec<i32> = (0..h as i32).collect();
                let tails: Vec<i32> = (0..h as i32).map(|i| i + 1).collect();
                let config = UMAPConfig {
                    optimizer,
                    negative_selection_range: 7,
                    init: umap_core::Init::Spectral,
                    ..Default::default()
                };
                EPOCH_OBSERVER.with(|observer| *observer.borrow_mut() = Some(Vec::new()));
                let result = optimize(
                    head,
                    tail.clone(),
                    Tensor::<1, Int>::from_data(TensorData::new(heads, [h]), &device),
                    Tensor::<1, Int>::from_data(TensorData::new(tails, [h]), &device),
                    Tensor::<1>::ones([h], &device),
                    2,
                    1.3,
                    0.8,
                    0.1,
                    &config,
                    LayoutOperation::Transform,
                    11,
                    2,
                    n,
                    &[],
                );
                let events = EPOCH_OBSERVER.with(|observer| observer.borrow_mut().take().unwrap());
                result.unwrap();
                assert!(!events[0].projection_ready && events[1].projection_ready);
                let initial = events[0].negative_order.as_ref().unwrap();
                let refreshed = events[1].negative_order.as_ref().unwrap();
                for order in [initial, refreshed] {
                    let mut sorted = order.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, (0..n as i32).collect::<Vec<_>>());
                }
                assert_ne!(initial, refreshed);
                // Every edge has period one, so epoch zero visits no edge.
                // At epoch one independently reconstruct the small-layout
                // global candidate hashes and compare the resulting device
                // optimizer state, rather than merely comparing host IDs.
                assert_eq!(events[0].moment_before, events[0].moment_after);
                let event = &events[1];
                let x = event.embedding.as_ref().unwrap();
                let before = event.moment_before.as_ref().unwrap();
                let after = event.moment_after.as_ref().unwrap();
                let negative_period = 1.5 / config.negative_sample_rate as f64;
                let count = ((1.0 - negative_period) / negative_period) as usize;
                assert!(count > 0);
                for i in 0..h {
                    let mut gradient = [0.0_f32; 2];
                    let mut accumulate = |j: usize, attractive: bool| {
                        let delta = [
                            x[2 * i] - reference[2 * j],
                            x[2 * i + 1] - reference[2 * j + 1],
                        ];
                        let q = delta[0] * delta[0] + delta[1] * delta[1];
                        if q == 0.0 {
                            return;
                        }
                        let coeff = if attractive {
                            0.5 * (-2.0 * 1.3 * 0.8 * q.powf(0.8 - 1.0)) / (1.0 + 1.3 * q.powf(0.8))
                        } else {
                            let gamma = config.repulsion_strength;
                            let c = 2.0 * gamma * 0.8 / ((0.001 + q) * (1.0 + 1.3 * q.powf(0.8)));
                            let norm = (c * c * q).sqrt();
                            if norm == 0.0 {
                                0.0
                            } else {
                                c * gamma.min(16.0) * (norm / gamma).tanh() / norm
                            }
                        };
                        for c in 0..2 {
                            gradient[c] += coeff * delta[c];
                        }
                    };
                    accumulate(i + 1, true);
                    for p in 0..count {
                        let rank = (i * (2 + p)) % n;
                        accumulate(refreshed[rank] as usize, false);
                    }
                    for (c, &component) in gradient.iter().enumerate() {
                        let k = i * 2 + c;
                        let expected = if optimizer == Optimizer::Momentum {
                            (before[k] + event.alpha * component) * event.mu
                        } else {
                            0.9 * before[k] + (1.0 - 0.9_f32) * component
                        };
                        assert!(
                            (after[k] - expected).abs() < 2e-4 * (1.0 + expected.abs()),
                            "optimizer={optimizer:?} N={n} row={i} component={c}: {} != {expected}",
                            after[k]
                        );
                    }
                }
                assert_eq!(tail.to_data().try_to_vec::<f32>().unwrap(), reference);
            }
        }
    }

    #[test]
    fn transform_orders_cpu() {
        transform_orders(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cpu(Default::default()),
        )));
    }

    #[test]
    fn transform_orders_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        transform_orders(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cuda(Default::default()),
        )));
    }

    fn hard_runtime(device: Device) {
        for optimizer in [Optimizer::Adam, Optimizer::Momentum] {
            for (operation, h, n) in [
                (LayoutOperation::Fit, 1025, 1025),
                (LayoutOperation::Fit, 100, 100),
                (LayoutOperation::Transform, 8, 1100),
                (LayoutOperation::Transform, 1100, 1100),
                (LayoutOperation::Transform, 8, 100),
                (LayoutOperation::Transform, 17, 300),
            ] {
                let reference: Vec<f32> = (0..n * 2).map(|i| (i % 71) as f32 * 0.013).collect();
                let source: Vec<f32> = (0..h * 2).map(|i| (i % 53) as f32 * 0.017).collect();
                let config = UMAPConfig {
                    optimizer,
                    negative_selection_range: 7,
                    negative_sample_scale_adaptation_samples: 4,
                    exclude_graph_neighbors: true,
                    init: umap_core::Init::Spectral,
                    ..Default::default()
                };
                let run = || {
                    super::super::modern_negatives::PROJECTION_COUNTS
                        .with(|counts| counts.set((0, 0)));
                    let head =
                        Tensor::<2>::from_data(TensorData::new(source.clone(), [h, 2]), &device);
                    let tail = if operation == LayoutOperation::Fit {
                        head.clone()
                    } else {
                        Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 2]), &device)
                    };
                    let before = tail.to_data().try_to_vec::<f32>().unwrap();
                    let heads: Vec<i32> = (0..h).map(|i| i as i32).collect();
                    let tails: Vec<i32> = (0..h).map(|i| ((i + 1) % n) as i32).collect();
                    let result = optimize(
                        head,
                        tail.clone(),
                        Tensor::<1, Int>::from_data(TensorData::new(heads, [h]), &device),
                        Tensor::<1, Int>::from_data(TensorData::new(tails, [h]), &device),
                        Tensor::<1>::ones([h], &device),
                        22,
                        1.3,
                        0.8,
                        0.1,
                        &config,
                        operation,
                        11,
                        2,
                        h,
                        &[],
                    )
                    .unwrap();
                    let values = result.to_data().try_to_vec::<f32>().unwrap();
                    let expected = (1, 22 + usize::from(h > 1024));
                    assert_eq!(
                        super::super::modern_negatives::PROJECTION_COUNTS
                            .with(|counts| counts.get()),
                        expected,
                        "observed refresh cadence optimizer={optimizer:?} operation={operation:?} H={h} N={n}"
                    );
                    assert!(values.iter().all(|value| value.is_finite()));
                    if operation == LayoutOperation::Transform {
                        assert_eq!(
                            tail.to_data().try_to_vec::<f32>().unwrap(),
                            before,
                            "immutable reference optimizer={optimizer:?} H={h} N={n}"
                        );
                    }
                    values
                };
                assert_eq!(
                    run(),
                    run(),
                    "seeded replay optimizer={optimizer:?} operation={operation:?} H={h} N={n}"
                );
            }
        }
    }

    #[test]
    fn hard_negative_runtime_cpu() {
        hard_runtime(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cpu(Default::default()),
        )));
    }

    #[test]
    fn hard_negative_runtime_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        hard_runtime(Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cuda(Default::default()),
        )));
    }

    #[test]
    fn exact_search_table_checks_each_block_before_search() {
        let device = Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cpu(Default::default()),
        ));
        let mut extra = Vec::new();
        exact_search_buffers(&mut extra, &device, 100, 100, 3, 8).unwrap();
        let plan = modern_buffer_plan(ModernPlanInput {
            operation: LayoutOperation::Fit,
            source_rows: 100,
            reference_rows: 100,
            dimensions: 2,
            features: 3,
            original_rows: 100,
            source_embedding_bytes: 800,
            reference_embedding_bytes: 800,
            graph_capacity: 800,
            edges: 800,
            adam: false,
            alias_orders: true,
            additional: &extra,
        })
        .unwrap();
        let block = plan
            .buffers()
            .find(|b| b.name == "exact block indices")
            .unwrap();
        assert_eq!(block.bytes, 64 * 8 * 4);
        assert_eq!(block.lifetime, BufferLifetime::RepeatedBuild(2));
        assert!(matches!(
            plan.validate_allocations(1, 1, usize::MAX, 1024, |_, _| Ok(())),
            Err(UmapError::Capacity(_))
        ));
        assert!(exact_search_buffers(&mut Vec::new(), &device, usize::MAX, 100, 3, 8).is_err());
    }

    #[test]
    fn cpu_layout_is_contiguous_and_overflow_is_rejected() {
        let device = Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cpu(Default::default()),
        ));
        assert_eq!(
            matrix_allocation_bytes(&device, 7, 48, 4).unwrap(),
            7 * 48 * 4
        );
        assert!(matrix_allocation_bytes(&device, usize::MAX, 48, 4).is_err());
    }

    #[test]
    fn cuda_layout_matches_policy_and_limit() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        let device = Device::new(burn::backend::DispatchDevice::Cube(
            burn_cubecl::CubeDevice::Cuda(Default::default()),
        ));
        let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch();
        let align = cube.client().properties().memory.alignment as usize;
        let row_align = cubecl_server::memory_management::optimal_align(48, 4, align);
        let physical = matrix_allocation_bytes(&device, 7, 48, 4).unwrap();
        assert_eq!(physical, 7 * (48usize * 4).next_multiple_of(row_align));
        assert!(physical > 7 * 48 * 4);
        assert!(matrix_allocation_bytes(&device, usize::MAX / 192, 48, 4).is_err());
    }
}
