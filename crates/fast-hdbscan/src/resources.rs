//! Pure fit resource estimates. Graph, width and source resolution are shared with fit.
//!
//! Lifetimes are modeled per backend and per phase. Phase peaks are maxima over phases,
//! never sums across phases. CPU reports one overlapped host+runtime peak. CUDA reports
//! host and device peaks separately. Exact and ANN graph routes are modeled as they
//! execute, not as a flat `n * width` allocation.
//!
//! Excluded: the caller-owned raw input buffer (`raw_n * features * 4`) and the
//! outlier-compaction copy taken when non-finite rows are dropped. The estimate receives
//! only the finite-row count, which cannot tell whether `raw_n` exceeds it, so neither
//! term is modeled. These exclusions prevent interpreting estimates as total process usage
//! or guaranteed upper or lower bounds on observed memory.

use fast_cubecl_common::{
    BackendChoice,
    neighbors::{AnnEstimate, AnnPhase, ann_host_estimate, bounded_neighbor_scratch},
};
use fast_hdbscan_core::{
    HdbscanConfig, HdbscanError, MstGraph, error::Result, finite::validate_dims,
    linkage::SortedMstEdge,
};

use crate::estimator::{
    KnnSource, NEIGHBOR_BATCH_ROWS, graph_width, knn_uses_ann, map_ann_error, resolve_graph,
};

/// Documented UX advisory threshold (1 GiB, binary). Not a safety or support boundary.
pub const ADVISORY_THRESHOLD_BYTES: usize = 1 << 30;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const I32_BYTES: usize = std::mem::size_of::<i32>();
const F64_BYTES: usize = std::mem::size_of::<f64>();
/// Packed `(id: i32, distance: f32)` per kNN slot.
const KNN_SLOT_BYTES: usize = I32_BYTES + F32_BYTES;
/// Candidate tuple `(weight, src, dst, index)` used by host Kruskal (conservative).
const KNN_TUPLE_BYTES: usize = std::mem::size_of::<(f64, usize, usize, usize)>();
/// Host DSU row: one `usize` parent word plus one `u32` rank word.
const DSU_ROW_BYTES: usize = std::mem::size_of::<usize>() + std::mem::size_of::<u32>();
const EDGE_BYTES: usize = std::mem::size_of::<SortedMstEdge>();
/// CubeCL CUDA allocation alignment for pitched 2-D tensors (`mem_alignment`).
const CUDA_PITCH_MAX_ALIGN: usize = 512;
/// Smallest pitch alignment CubeCL applies to pitched rows (`optimal_align` lower clamp).
const CUDA_PITCH_MIN_ALIGN: usize = 16;

/// Backend side of a scenario.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceBackend {
    Cpu,
    Cuda,
}

/// Resolved kNN neighbor source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnResourceSource {
    Exact,
    Ann,
}

/// Fit lifetime phases, in execution order. Exactly one of `ExactGraph` or the four
/// ANN phases appears per kNN scenario.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourcePhase {
    /// Finite host copy used to build the upload, and the uploaded input tensor.
    Input,
    AnnInput,
    /// Exact brute-force graph: query blocks, readback and core accumulation.
    ExactGraph,
    AnnForestInit,
    AnnRefinement,
    AnnOutput,
    /// Host Kruskal over the kNN graph; the input tensor is still live.
    Kruskal,
    /// Retained diagnostics, public neighbors, core values and MST edges.
    Output,
}

/// Host and runtime bytes live during one phase. On the CPU backend runtime bytes are
/// host RAM; on CUDA they are device bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourcePhaseBytes {
    pub phase: ResourcePhase,
    pub host_bytes: usize,
    pub runtime_bytes: usize,
}

/// One conditional backend scenario with its per-phase lifetimes.
/// CPU: `host_peak_bytes` is the overlapped host+runtime peak; `device_peak_bytes` is 0.
/// CUDA: `host_peak_bytes` and `device_peak_bytes` are separate peaks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceScenario {
    pub backend: ResourceBackend,
    pub host_peak_bytes: usize,
    pub device_peak_bytes: usize,
    pub phases: Vec<ResourcePhaseBytes>,
}

/// Advisory emitted when any applicable scenario reaches `threshold_bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceAdvisory {
    pub threshold_bytes: usize,
    /// Scenarios whose host or device peak reached the threshold.
    pub triggers: Vec<ResourceBackend>,
    /// True when an explicit `knn_k` is below `min_samples`; the width cannot drop below it.
    pub width_floor_applies: bool,
}

/// kNN graph estimate. Advisory applies only on this route.
#[derive(Clone, Debug, PartialEq)]
pub struct KnnResourceEstimate {
    pub source: KnnResourceSource,
    pub finite_rows: usize,
    pub features: usize,
    pub min_samples: usize,
    /// Production graph width after `min_samples` and row-count clamping (`knn_k` policy).
    pub effective_width: usize,
    /// Candidate width actually retained by the source: `effective_width` for exact,
    /// `min(effective_width + 4, n - 1)` for ANN.
    pub retained_width: usize,
    /// `finite_rows * effective_width` directed kNN graph edges used by Kruskal.
    pub directed_edges: usize,
    /// `effective_width / (finite_rows - 1)`; context only, not a trigger.
    pub density: f64,
    pub scenarios: Vec<ResourceScenario>,
    pub advisory: Option<ResourceAdvisory>,
}

/// Advisory estimate for one fit shape. `knn` is `None` on the complete route and for
/// fewer than two finite rows, where no kNN graph is allocated.
#[derive(Clone, Debug, PartialEq)]
pub struct FitResourceEstimate {
    pub resolved_graph: MstGraph,
    pub knn: Option<KnnResourceEstimate>,
}

/// Shared fit plan. `estimate` is consumed by the fit warning path.
pub(crate) struct FitPlan {
    pub(crate) graph: MstGraph,
    pub(crate) width: usize,
    pub(crate) approximate: bool,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) estimate: FitResourceEstimate,
}

fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(HdbscanError::Overflow("fit resource estimate"))
}

fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(HdbscanError::Overflow("fit resource estimate"))
}

fn checked_sum(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(0usize, |sum, &value| checked_add(sum, value))
}

/// Pure advisory estimate. Validates config and shape, reads no data, and never touches
/// a device, so it does not initialize CUDA or run device discovery. `n_finite` is the
/// finite-row count; zero finite rows yields no kNN estimate, matching fit for
/// all-nonfinite input. Empty raw input is rejected by fit before planning.
pub fn estimate_fit_resources(
    n_finite: usize,
    features: usize,
    config: &HdbscanConfig,
    backend_choice: BackendChoice,
) -> Result<FitResourceEstimate> {
    config.validate()?;
    validate_dims(n_finite.max(1), features)?;
    Ok(plan_fit(n_finite, features, config, backend_choice, KnnSource::Auto)?.estimate)
}

/// Shared resolution used by both the public estimate and `fit`. `backend` is the
/// concrete backend in fit and selects scenarios for the public estimate; `source`
/// carries fit's forced comparator policy.
pub(crate) fn plan_fit(
    n: usize,
    features: usize,
    config: &HdbscanConfig,
    backend: BackendChoice,
    source: KnnSource,
) -> Result<FitPlan> {
    let min_samples = config.resolved_min_samples();
    let graph = resolve_graph(config.mst_graph, features);
    let width = if graph == MstGraph::KnnGraph {
        graph_width(n, min_samples, config.knn_k)?
    } else {
        min_samples.min(n.saturating_sub(1))
    };
    if graph == MstGraph::Complete || n < 2 {
        return Ok(FitPlan {
            graph,
            width,
            approximate: false,
            estimate: FitResourceEstimate {
                resolved_graph: graph,
                knn: None,
            },
        });
    }
    let approximate = knn_uses_ann(source, n);
    let knn = knn_estimate(n, features, config, backend, approximate, width)?;
    Ok(FitPlan {
        graph,
        width,
        approximate,
        estimate: FitResourceEstimate {
            resolved_graph: graph,
            knn: Some(knn),
        },
    })
}

fn knn_estimate(
    n: usize,
    features: usize,
    config: &HdbscanConfig,
    backend: BackendChoice,
    approximate: bool,
    width: usize,
) -> Result<KnnResourceEstimate> {
    let min_samples = config.resolved_min_samples();
    let ann = if approximate {
        Some(ann_host_estimate(n, features, width).map_err(map_ann_error)?)
    } else {
        None
    };
    let backends: &[ResourceBackend] = match backend {
        BackendChoice::Cpu => &[ResourceBackend::Cpu],
        BackendChoice::Cuda { .. } => &[ResourceBackend::Cuda],
        BackendChoice::Auto => &[ResourceBackend::Cpu, ResourceBackend::Cuda],
    };
    let scenarios = backends
        .iter()
        .map(|&backend| scenario(backend, n, features, min_samples, width, ann))
        .collect::<Result<Vec<_>>>()?;
    let triggers: Vec<ResourceBackend> = scenarios
        .iter()
        .filter(|s| scenario_triggers(s, ADVISORY_THRESHOLD_BYTES))
        .map(|s| s.backend)
        .collect();
    let advisory = (!triggers.is_empty()).then(|| ResourceAdvisory {
        threshold_bytes: ADVISORY_THRESHOLD_BYTES,
        triggers,
        width_floor_applies: config.knn_k.is_some_and(|k| k < min_samples),
    });
    Ok(KnnResourceEstimate {
        source: if approximate {
            KnnResourceSource::Ann
        } else {
            KnnResourceSource::Exact
        },
        finite_rows: n,
        features,
        min_samples,
        effective_width: width,
        retained_width: ann.map_or(width, |estimate| estimate.retained_width),
        directed_edges: checked_mul(n, width)?,
        density: width as f64 / (n - 1) as f64,
        scenarios,
        advisory,
    })
}

/// CUDA pitch of one row-major `[n, features]` f32 tensor, mirroring CubeCL's pitched
/// layout: the row width rounded up to `optimal_align(features, 4, 512)`. Returns the
/// pitch in bytes. Single-feature tensors use element-size pitch.
fn cuda_pitch_bytes(features: usize) -> Result<usize> {
    let row = checked_mul(features, F32_BYTES)?;
    let align = if features == 1 {
        F32_BYTES
    } else {
        row.next_power_of_two()
            .clamp(CUDA_PITCH_MIN_ALIGN, CUDA_PITCH_MAX_ALIGN)
    };
    row.checked_next_multiple_of(align)
        .ok_or(HdbscanError::Overflow("CUDA pitched row"))
}

/// Build one backend scenario. Phases are disjoint lifetimes; peaks are per-phase maxima.
///
/// Input accounting:
/// - Host: the finite copy used to build the upload, counted once in `Input`.
/// - CPU runtime: a single contiguous tensor (`finite`), which is also the ANN input.
/// - CUDA runtime: the pitched upload tensor (`n * pitch`), live through the whole fit.
///   The ANN and exact paths also allocate contiguous copies of that tensor; ANN's own
///   input term already includes its copy, and exact adds the reference copy explicitly.
fn scenario(
    backend: ResourceBackend,
    n: usize,
    features: usize,
    min_samples: usize,
    width: usize,
    ann: Option<AnnEstimate>,
) -> Result<ResourceScenario> {
    let finite = checked_mul(checked_mul(n, features)?, F32_BYTES)?;
    let (input_device, pitched_is_padded) = match backend {
        ResourceBackend::Cpu => (finite, false),
        ResourceBackend::Cuda => {
            let pitch = cuda_pitch_bytes(features)?;
            (
                checked_mul(n, pitch)?,
                pitch != checked_mul(features, F32_BYTES)?,
            )
        }
    };
    // CUDA exact reads a contiguous reference copy only when the pitched tensor is padded.
    let reference_copy = match backend {
        ResourceBackend::Cuda if pitched_is_padded => finite,
        _ => 0,
    };
    // ANN carries its own contiguous input term. On CPU that term is the same buffer as
    // the facade tensor, so nothing is added. On CUDA the facade pitched tensor persists
    // beside ANN's contiguous copy for every ANN phase.
    let ann_facade_extra = match backend {
        ResourceBackend::Cuda if pitched_is_padded => input_device,
        _ => 0,
    };

    let mut phases = Vec::with_capacity(8);
    phases.push(ResourcePhaseBytes {
        phase: ResourcePhase::Input,
        host_bytes: finite,
        runtime_bytes: input_device,
    });
    match ann {
        Some(ann) => {
            for phase in ann.phases {
                // ANN host bytes are passed through for every phase. The facade `Input` phase
                // counts the finite copy, and ANN's `Input` host term (decode plus read-back of
                // the same data) is a separate lifetime that is still live during ANN input.
                phases.push(ResourcePhaseBytes {
                    phase: ann_resource_phase(phase.phase),
                    host_bytes: phase.host_bytes,
                    runtime_bytes: checked_add(phase.runtime_bytes, ann_facade_extra)?,
                });
            }
        }
        None => phases.push(exact_graph_phase(
            backend,
            n,
            features,
            width,
            input_device,
            reference_copy,
        )?),
    }
    phases.push(kruskal_phase(n, width, input_device)?);
    phases.push(output_phase(n, width, min_samples)?);

    let (host_peak_bytes, device_peak_bytes) = match backend {
        ResourceBackend::Cpu => {
            let mut combined = 0usize;
            for phase in &phases {
                combined = combined.max(checked_add(phase.host_bytes, phase.runtime_bytes)?);
            }
            (combined, 0)
        }
        ResourceBackend::Cuda => phases.iter().fold((0usize, 0usize), |(host, device), p| {
            (host.max(p.host_bytes), device.max(p.runtime_bytes))
        }),
    };
    Ok(ResourceScenario {
        backend,
        host_peak_bytes,
        device_peak_bytes,
        phases,
    })
}

fn ann_resource_phase(phase: AnnPhase) -> ResourcePhase {
    match phase {
        AnnPhase::Input => ResourcePhase::AnnInput,
        AnnPhase::ForestInit => ResourcePhase::AnnForestInit,
        AnnPhase::Refinement => ResourcePhase::AnnRefinement,
        AnnPhase::Output => ResourcePhase::AnnOutput,
    }
}

/// Exact brute-force graph, mirroring `core_distances`. The fit loops over query blocks
/// of `min(n - 1, 256)` rows. Each block calls the common exact op, which allocates the
/// block's full id and distance outputs on every route. Blocks are batched, so only one
/// block's outputs are live at a time, not the whole `n * width` graph.
///
/// Per-block runtime, on top of the input tensor:
/// - block id/distance outputs (`block * width` slots);
/// - query copy (`block * features * 4`);
/// - core accumulation for all blocks (`n * 4`) and the final `cat` output (`n * 4`);
/// - CUDA only: contiguous reference copy when the pitched tensor is padded
///   (`reference_copy`), and bounded squared-distance scratch from the common plan.
///
/// Host: the full id/distance vectors reserved up front (`n * width` slots), plus one
/// block's readback copy.
fn exact_graph_phase(
    backend: ResourceBackend,
    n: usize,
    features: usize,
    width: usize,
    input_device: usize,
    reference_copy: usize,
) -> Result<ResourcePhaseBytes> {
    let block_rows = n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS);
    let block_slots = checked_mul(checked_mul(block_rows, width)?, KNN_SLOT_BYTES)?;
    let query_copy = checked_mul(checked_mul(block_rows, features)?, F32_BYTES)?;
    let core = checked_mul(n, F32_BYTES)?;
    let host = checked_add(
        checked_mul(checked_mul(n, width)?, KNN_SLOT_BYTES)?,
        block_slots,
    )?;
    let scratch = match backend {
        ResourceBackend::Cpu => 0,
        // No device discovery: page and alignment are unbounded here, so this is the
        // 32 MiB live-scratch cap. Real device page limits can only shrink query groups.
        ResourceBackend::Cuda => {
            bounded_neighbor_scratch(block_rows, n, width, F32_BYTES, usize::MAX, 1)
                .map(|plan| plan.live_bytes)
                .ok_or(HdbscanError::Overflow("bounded exact neighbor scratch"))?
        }
    };
    let reference = match backend {
        ResourceBackend::Cpu => 0,
        ResourceBackend::Cuda => reference_copy,
    };
    Ok(ResourcePhaseBytes {
        phase: ResourcePhase::ExactGraph,
        host_bytes: host,
        runtime_bytes: checked_sum(&[
            input_device,
            reference,
            core,
            core,
            block_slots,
            query_copy,
            scratch,
        ])?,
    })
}

/// Host Kruskal. `seen[n]` is dropped before candidate tuples are allocated, so it is
/// omitted. Graph ids/distances, candidate tuples plus sort workspace, core radii, DSU
/// rows and MST edges coexist. The input tensor is still live in fit.
fn kruskal_phase(n: usize, width: usize, input_device: usize) -> Result<ResourcePhaseBytes> {
    let slots = checked_mul(n, width)?;
    let host = checked_sum(&[
        checked_mul(slots, KNN_SLOT_BYTES)?,
        checked_mul(checked_mul(slots, KNN_TUPLE_BYTES)?, 2)?,
        checked_mul(n, F64_BYTES)?,
        checked_mul(n, DSU_ROW_BYTES)?,
        checked_mul(n.saturating_sub(1), EDGE_BYTES)?,
    ])?;
    Ok(ResourcePhaseBytes {
        phase: ResourcePhase::Kruskal,
        host_bytes: host,
        runtime_bytes: input_device,
    })
}

/// Output. `KnnDiagnostics` is always retained by fit, so the id/distance graph remains
/// live alongside public neighbors, the f64 MST core values, the f32 core conversion and
/// MST edges. No device runtime remains.
fn output_phase(n: usize, width: usize, min_samples: usize) -> Result<ResourcePhaseBytes> {
    let slots = checked_mul(n, width)?;
    let host = checked_sum(&[
        checked_mul(slots, KNN_SLOT_BYTES)?,
        checked_mul(n, F64_BYTES)?,
        checked_mul(n, F32_BYTES)?,
        checked_mul(checked_mul(n, min_samples)?, I32_BYTES)?,
        checked_mul(n.saturating_sub(1), EDGE_BYTES)?,
    ])?;
    Ok(ResourcePhaseBytes {
        phase: ResourcePhase::Output,
        host_bytes: host,
        runtime_bytes: 0,
    })
}

fn scenario_triggers(scenario: &ResourceScenario, threshold: usize) -> bool {
    scenario.host_peak_bytes >= threshold || scenario.device_peak_bytes >= threshold
}

/// Emit the single optional advisory for a fit. Only compiled with the `tracing` feature.
#[cfg(feature = "tracing")]
pub(crate) fn emit_advisory(estimate: &FitResourceEstimate) {
    let Some(knn) = estimate.knn.as_ref() else {
        return;
    };
    let Some(advisory) = knn.advisory.as_ref() else {
        return;
    };
    tracing::warn!(
        finite_rows = knn.finite_rows,
        effective_width = knn.effective_width,
        retained_width = knn.retained_width,
        min_samples = knn.min_samples,
        directed_edges = knn.directed_edges,
        threshold_bytes = advisory.threshold_bytes,
        width_floor_applies = advisory.width_floor_applies,
        scenarios = ?knn.scenarios,
        "kNN graph estimate reaches 1 GiB (estimates, not process usage); execution continues with your settings"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(min_cluster_size: usize, knn_k: Option<usize>) -> HdbscanConfig {
        HdbscanConfig {
            min_cluster_size,
            mst_graph: MstGraph::KnnGraph,
            knn_k,
            ..HdbscanConfig::default()
        }
    }

    fn knn_of(plan: &FitPlan) -> &KnnResourceEstimate {
        plan.estimate.knn.as_ref().unwrap()
    }

    fn scenario_of(knn: &KnnResourceEstimate, backend: ResourceBackend) -> &ResourceScenario {
        knn.scenarios.iter().find(|s| s.backend == backend).unwrap()
    }

    fn phase_of(scenario: &ResourceScenario, phase: ResourcePhase) -> ResourcePhaseBytes {
        *scenario.phases.iter().find(|p| p.phase == phase).unwrap()
    }

    #[test]
    fn threshold_boundary_is_one_gib_inclusive() {
        let scenario = |backend, host, device| ResourceScenario {
            backend,
            host_peak_bytes: host,
            device_peak_bytes: device,
            phases: Vec::new(),
        };
        assert!(!scenario_triggers(
            &scenario(ResourceBackend::Cpu, ADVISORY_THRESHOLD_BYTES - 1, 0),
            ADVISORY_THRESHOLD_BYTES
        ));
        assert!(scenario_triggers(
            &scenario(ResourceBackend::Cpu, ADVISORY_THRESHOLD_BYTES, 0),
            ADVISORY_THRESHOLD_BYTES
        ));
        assert!(scenario_triggers(
            &scenario(ResourceBackend::Cuda, 0, ADVISORY_THRESHOLD_BYTES),
            ADVISORY_THRESHOLD_BYTES
        ));
    }

    /// Hand-computed for n=10 finite rows, 3 features, knn_k=4, min_samples=2.
    /// width = min(max(4, 2), 9) = 4; finite copy = 10*3*4 = 120 bytes; block = 9 rows.
    /// Expected values below are derived from the lifetimes, not from the struct fields.
    #[test]
    fn cpu_exact_matches_hand_computed_lifetimes() {
        let plan = plan_fit(
            10,
            3,
            &config(2, Some(4)),
            BackendChoice::Cpu,
            KnnSource::Exact,
        )
        .unwrap();
        let cpu = scenario_of(knn_of(&plan), ResourceBackend::Cpu);
        // Input: host 120 (finite), runtime 120 (contiguous tensor).
        assert_eq!(
            phase_of(cpu, ResourcePhase::Input),
            ResourcePhaseBytes {
                phase: ResourcePhase::Input,
                host_bytes: 120,
                runtime_bytes: 120,
            }
        );
        // Exact: host 10*4*8=320 full vectors + 9*4*8=288 readback = 608.
        // Runtime: input 120 + core 40 + cat 40 + block outputs 288 + query 108 = 596.
        assert_eq!(
            phase_of(cpu, ResourcePhase::ExactGraph),
            ResourcePhaseBytes {
                phase: ResourcePhase::ExactGraph,
                host_bytes: 608,
                runtime_bytes: 596,
            }
        );
        // Kruskal host: 320 slots + 2*40*32 tuples + 10*8 f64 core + 10*12 DSU + 9*16 edges = 3224.
        assert_eq!(
            phase_of(cpu, ResourcePhase::Kruskal),
            ResourcePhaseBytes {
                phase: ResourcePhase::Kruskal,
                host_bytes: 3224,
                runtime_bytes: 120,
            }
        );
        // Output host: 320 slots + 10*8 f64 + 10*4 f32 + 10*2*4 labels + 9*16 edges = 664.
        assert_eq!(
            phase_of(cpu, ResourcePhase::Output),
            ResourcePhaseBytes {
                phase: ResourcePhase::Output,
                host_bytes: 664,
                runtime_bytes: 0,
            }
        );
        // CPU peak is the largest overlapped phase: Kruskal 3224 + 120.
        assert_eq!(cpu.host_peak_bytes, 3344);
        assert_eq!(cpu.device_peak_bytes, 0);
    }

    /// Same shape on CUDA. Pitch for 3 f32 features is next_multiple(12, 16) = 16, so the
    /// pitched input is 10*16 = 160 and padding forces a 120-byte contiguous reference copy.
    /// Bounded scratch for 9 query rows over 10 references with k=4, one reference group:
    /// partial index 144 + partial distance 144 + running and merged (index+distance) 2*288 = 864.
    #[test]
    fn cuda_exact_matches_hand_computed_lifetimes_with_padding() {
        let plan = plan_fit(
            10,
            3,
            &config(2, Some(4)),
            BackendChoice::Cuda { device_index: 0 },
            KnnSource::Exact,
        )
        .unwrap();
        let cuda = scenario_of(knn_of(&plan), ResourceBackend::Cuda);
        assert_eq!(phase_of(cuda, ResourcePhase::Input).runtime_bytes, 160);
        // Exact runtime: pitched 160 + reference 120 + core 40 + cat 40 + outputs 288 + query 108 + scratch 864.
        assert_eq!(
            phase_of(cuda, ResourcePhase::ExactGraph),
            ResourcePhaseBytes {
                phase: ResourcePhase::ExactGraph,
                host_bytes: 608,
                runtime_bytes: 1620,
            }
        );
        assert_eq!(cuda.host_peak_bytes, 3224);
        assert_eq!(cuda.device_peak_bytes, 1620);
    }

    #[test]
    fn cuda_unpadded_pitch_has_no_reference_copy() {
        // features = 4 gives row 16 bytes, which is already a 16-byte pitch multiple.
        let plan = plan_fit(
            10,
            4,
            &config(2, Some(4)),
            BackendChoice::Cuda { device_index: 0 },
            KnnSource::Exact,
        )
        .unwrap();
        let exact = phase_of(
            scenario_of(knn_of(&plan), ResourceBackend::Cuda),
            ResourcePhase::ExactGraph,
        );
        // Pitched input 10*16 = 160; no reference copy; core 40+40; outputs 9*4*8=288;
        // query 9*4*4=144; scratch 864 (same k, same rows).
        assert_eq!(exact.runtime_bytes, 160 + 80 + 288 + 144 + 864);
    }

    #[test]
    fn cpu_ann_phases_are_common_estimate_without_double_counting_input() {
        let plan = plan_fit(
            1000,
            3,
            &config(5, Some(75)),
            BackendChoice::Cpu,
            KnnSource::Ann,
        )
        .unwrap();
        let cpu = scenario_of(knn_of(&plan), ResourceBackend::Cpu);
        let ann = ann_host_estimate(1000, 3, 75).unwrap();
        let expected = [
            (
                ResourcePhase::AnnInput,
                ann.phases[0].host_bytes,
                ann.phases[0].runtime_bytes,
            ),
            (
                ResourcePhase::AnnForestInit,
                ann.phases[1].host_bytes,
                ann.phases[1].runtime_bytes,
            ),
            (
                ResourcePhase::AnnRefinement,
                ann.phases[2].host_bytes,
                ann.phases[2].runtime_bytes,
            ),
            (
                ResourcePhase::AnnOutput,
                ann.phases[3].host_bytes,
                ann.phases[3].runtime_bytes,
            ),
        ];
        for (phase, host, runtime) in expected {
            let got = phase_of(cpu, phase);
            assert_eq!(got.host_bytes, host);
            assert_eq!(got.runtime_bytes, runtime);
        }
        assert!(
            cpu.phases
                .iter()
                .all(|p| p.phase != ResourcePhase::ExactGraph)
        );
    }

    #[test]
    fn cuda_ann_adds_pitched_facade_tensor_to_every_ann_phase() {
        // 1000 rows, 3 features: pitch 16, pitched tensor 16000 bytes.
        let plan = plan_fit(
            1000,
            3,
            &config(5, Some(75)),
            BackendChoice::Cuda { device_index: 0 },
            KnnSource::Ann,
        )
        .unwrap();
        let cuda = scenario_of(knn_of(&plan), ResourceBackend::Cuda);
        let ann = ann_host_estimate(1000, 3, 75).unwrap();
        for (idx, phase) in [
            ResourcePhase::AnnInput,
            ResourcePhase::AnnForestInit,
            ResourcePhase::AnnRefinement,
            ResourcePhase::AnnOutput,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                phase_of(cuda, phase).runtime_bytes,
                ann.phases[idx].runtime_bytes + 16_000
            );
        }
    }

    #[test]
    fn cuda_ann_unpadded_f32_adds_no_facade_input() {
        // features = 4: pitch equals row bytes, so the facade tensor is unpadded and
        // no separate facade copy is added to ANN phase runtime.
        let plan = plan_fit(
            1000,
            4,
            &config(5, Some(75)),
            BackendChoice::Cuda { device_index: 0 },
            KnnSource::Ann,
        )
        .unwrap();
        let cuda = scenario_of(knn_of(&plan), ResourceBackend::Cuda);
        let ann = ann_host_estimate(1000, 4, 75).unwrap();
        for (idx, phase) in [
            ResourcePhase::AnnInput,
            ResourcePhase::AnnForestInit,
            ResourcePhase::AnnRefinement,
            ResourcePhase::AnnOutput,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                phase_of(cuda, phase).runtime_bytes,
                ann.phases[idx].runtime_bytes
            );
        }
    }

    #[test]
    fn high_dim_ann_host_peak_covers_double_precision_input_copies() {
        // 1000 rows, 64 features: ANN input host (2x finite f32 copy) must be part of the
        // host peak on both CPU and CUDA scenarios.
        let (n, f) = (1000usize, 64usize);
        let finite = n * f * 4;
        for (backend, resource) in [
            (BackendChoice::Cpu, ResourceBackend::Cpu),
            (
                BackendChoice::Cuda { device_index: 0 },
                ResourceBackend::Cuda,
            ),
        ] {
            let plan = plan_fit(n, f, &config(5, Some(75)), backend, KnnSource::Ann).unwrap();
            let scenario = scenario_of(knn_of(&plan), resource);
            assert!(scenario.host_peak_bytes >= 2 * finite);
        }
    }

    #[test]
    fn forced_ann_retains_wider_than_effective_while_auto_stays_exact_up_to_4096() {
        let forced = plan_fit(
            1000,
            3,
            &config(5, Some(75)),
            BackendChoice::Cpu,
            KnnSource::Ann,
        )
        .unwrap();
        let knn = knn_of(&forced);
        assert_eq!(knn.source, KnnResourceSource::Ann);
        assert_eq!(knn.effective_width, 75);
        assert_eq!(knn.retained_width, 79);
        assert_eq!(knn.directed_edges, 1000 * 75);

        let auto = plan_fit(
            1000,
            3,
            &config(5, Some(75)),
            BackendChoice::Cpu,
            KnnSource::Auto,
        )
        .unwrap();
        assert_eq!(knn_of(&auto).source, KnnResourceSource::Exact);
        assert_eq!(knn_of(&auto).retained_width, 75);
        assert!(knn_uses_ann(KnnSource::Auto, 4097));
        assert!(!knn_uses_ann(KnnSource::Auto, 4096));
    }

    #[test]
    fn auto_backend_yields_both_conditional_scenarios() {
        let estimate =
            estimate_fit_resources(100, 3, &config(5, None), BackendChoice::Auto).unwrap();
        let knn = estimate.knn.unwrap();
        let backends: Vec<_> = knn.scenarios.iter().map(|s| s.backend).collect();
        assert_eq!(backends, [ResourceBackend::Cpu, ResourceBackend::Cuda]);
        assert!(knn.advisory.is_none());
    }

    #[test]
    fn default_width_is_raised_by_min_samples_and_capped_by_rows() {
        let wide = plan_fit(
            1000,
            3,
            &config(25, None),
            BackendChoice::Cpu,
            KnnSource::Auto,
        )
        .unwrap();
        assert_eq!(wide.width, 75);
        let capped = plan_fit(
            50,
            3,
            &config(25, None),
            BackendChoice::Cpu,
            KnnSource::Auto,
        )
        .unwrap();
        assert_eq!(capped.width, 49);
    }

    #[test]
    fn explicit_knn_k_below_min_samples_is_raised_and_flagged() {
        let cfg = config(25, Some(10));
        let plan = plan_fit(1000, 3, &cfg, BackendChoice::Cpu, KnnSource::Auto).unwrap();
        assert_eq!(plan.width, 25);
        assert!(cfg.knn_k.is_some_and(|k| k < cfg.resolved_min_samples()));
    }

    #[test]
    fn complete_route_empty_and_singleton_shapes_have_no_knn_estimate() {
        let complete = HdbscanConfig {
            mst_graph: MstGraph::Complete,
            ..config(5, None)
        };
        let estimate = estimate_fit_resources(100, 3, &complete, BackendChoice::Cpu).unwrap();
        assert_eq!(estimate.resolved_graph, MstGraph::Complete);
        assert!(estimate.knn.is_none());

        let singleton = estimate_fit_resources(1, 3, &config(5, None), BackendChoice::Cpu).unwrap();
        assert!(singleton.knn.is_none());

        let no_finite = estimate_fit_resources(0, 3, &config(5, None), BackendChoice::Cpu).unwrap();
        assert!(no_finite.knn.is_none());
    }

    #[test]
    fn zero_features_and_invalid_config_are_rejected() {
        assert!(estimate_fit_resources(100, 0, &config(5, None), BackendChoice::Cpu).is_err());
        assert!(estimate_fit_resources(100, 3, &config(0, None), BackendChoice::Cpu).is_err());
    }
}
