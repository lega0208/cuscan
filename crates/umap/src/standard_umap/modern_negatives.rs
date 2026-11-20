//! Coordinate-dependent negative orders and bounded Adam force sampling.
//! All mutable buffers are reused only after the preceding epoch has completed.
use burn::tensor::{DType, Shape};
use burn_cubecl::tensor::CubeTensor;
use cuscan_common::radix::{
    RadixDeviceLimits, RadixSortConfig, RadixSortError, RadixSortWorkspace,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use umap_core::{Result, UmapError, modern_layout::negative_force_ratio};

use super::modern_orders::ModernUpload;
use crate::kernels::{
    modern_project, modern_projection_ranks, modern_query_projection_status,
    modern_sample_resultants,
};

#[cfg(test)]
thread_local! {
    pub(super) static PROJECTION_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_buffers(
    buffers: &mut Vec<umap_core::modern_layout::LiveBuffer>,
    client: &cubecl::client::Client,
    operation: umap_core::modern_layout::LayoutOperation,
    h: usize,
    n: usize,
    d: usize,
    config: &umap_core::UMAPConfig,
) -> Result<()> {
    use umap_core::modern_layout::{BufferLifetime, BufferSpace, LayoutOperation, LiveBuffer};
    let projection = config.negative_selection_range < n;
    let adaptation = projection
        && operation == LayoutOperation::Fit
        && config.optimizer == umap_core::Optimizer::Adam
        && config.negative_sample_scale_adaptation_samples > 0;
    let alignment = usize::try_from(client.properties().memory.alignment)
        .map_err(|_| UmapError::Overflow("negative readback alignment"))?;
    let readback_bytes = |bytes: usize| -> Result<usize> {
        if alignment == 0 {
            return Err(UmapError::InvalidConfig(
                "device allocation alignment must be positive".into(),
            ));
        }
        bytes
            .checked_add(alignment - 1)
            .map(|bytes| bytes / alignment * alignment)
            .ok_or(UmapError::Overflow("negative readback alignment"))
    };
    buffers
        .try_reserve(64)
        .map_err(|e| UmapError::Memory(format!("negative buffer table: {e}")))?;
    let mut add = |name, space, count: usize, element: usize| -> Result<()> {
        buffers.push(LiveBuffer {
            name,
            space,
            lifetime: BufferLifetime::Layout,
            bytes: count
                .checked_mul(element)
                .ok_or(UmapError::Overflow("negative buffer plan"))?,
        });
        Ok(())
    };
    if projection {
        for (name, count) in [
            ("projection direction", d),
            ("reference projections", n),
            ("source ranks", h),
        ] {
            add(name, BufferSpace::Device, count, 4)?;
        }
        add("direction values", BufferSpace::Host, d, 4)?;
        add("direction upload staging", BufferSpace::Host, d, 4)?;
        if operation == LayoutOperation::Transform {
            add("query projections", BufferSpace::Device, h, 4)?;
            add("query projection status", BufferSpace::Device, 1, 4)?;
            add(
                "query projection readback",
                BufferSpace::Host,
                readback_bytes(4)?,
                1,
            )?;
        }
        let sort_config =
            RadixSortConfig::conservative(RadixDeviceLimits::from_client(client).path);
        let plan =
            cuscan_common::radix::radix_sort_plan(client, n, sort_config).map_err(radix_error)?;
        for allocation in &plan.allocations {
            add(
                allocation.buffer,
                BufferSpace::Device,
                usize::try_from(allocation.aligned_bytes)
                    .map_err(|_| UmapError::Overflow("radix allocation"))?,
                1,
            )?;
        }
        add(
            "radix status readback",
            BufferSpace::Host,
            usize::try_from(plan.host_status_bytes)
                .map_err(|_| UmapError::Overflow("radix readback"))?,
            1,
        )?;
        add(
            "radix allocation metadata",
            BufferSpace::Host,
            plan.allocations.len(),
            size_of::<cuscan_common::radix::RadixAllocation>(),
        )?;
        add(
            "radix scan levels",
            BufferSpace::Host,
            plan.scan_levels.len(),
            size_of::<usize>(),
        )?;
        add(
            "radix scan capacity",
            BufferSpace::Host,
            plan.scan_capacity.len(),
            size_of::<usize>(),
        )?;
    }
    if adaptation {
        let samples = config.negative_sample_scale_adaptation_samples.min(n);
        let candidates = samples
            .checked_mul(5)
            .ok_or(UmapError::Overflow("adaptation sample plan"))?;
        i32::try_from(candidates)
            .map_err(|_| UmapError::Overflow("adaptation candidate indices"))?;
        let resultants = samples
            .checked_mul(2)
            .ok_or(UmapError::Overflow("adaptation result plan"))?;
        for (name, count) in [
            ("adaptation sources", samples),
            ("adaptation local ranks", candidates),
            ("adaptation global ranks", candidates),
        ] {
            add(name, BufferSpace::Device, count, 4)?;
            add(name, BufferSpace::Host, count, 4)?;
        }
        add("adaptation resultants", BufferSpace::Device, resultants, 8)?;
        add(
            "adaptation readback",
            BufferSpace::Host,
            readback_bytes(
                resultants
                    .checked_mul(8)
                    .ok_or(UmapError::Overflow("adaptation readback"))?,
            )?,
            1,
        )?;
        add(
            "adaptation host resultants",
            BufferSpace::Host,
            resultants,
            8,
        )?;
        add("adaptation log ratios", BufferSpace::Host, samples, 8)?;
    }
    Ok(())
}

pub(super) fn radix_error(error: RadixSortError) -> UmapError {
    match error {
        RadixSortError::NonFinite { index } => {
            UmapError::Numerical(format!("non-finite reference projection at row {index}"))
        }
        RadixSortError::Overflow(_) => UmapError::Overflow("modern radix scratch"),
        RadixSortError::Allocation { .. } => UmapError::Memory(error.to_string()),
        RadixSortError::Capability(_) | RadixSortError::Device(_) => {
            UmapError::BackendUnavailable(error.to_string())
        }
        _ => UmapError::InvalidInput(error.to_string()),
    }
}

fn allocate(
    template: &CubeTensor,
    rows: usize,
    dtype: DType,
    element_bytes: usize,
) -> Result<CubeTensor> {
    let bytes = rows
        .checked_mul(element_bytes)
        .ok_or(UmapError::Overflow("negative preparation buffer"))?;
    Ok(CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        Shape::new([rows]),
        template.client.empty(bytes),
        dtype,
    ))
}

fn zeroed_floats(rows: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(rows)
        .map_err(|e| UmapError::Memory(format!("projection direction: {e}")))?;
    values.resize(rows, 0.0);
    Ok(values)
}

// A degenerate random direction deterministically selects the first unit axis;
// this is part of the projection definition, not a backend fallback. The slice
// is nonempty because the embedding dimension is validated before layout.
fn normalize_direction(values: &mut [f32], squared_norm: f64) {
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        values.fill(0.0);
        values[0] = 1.0;
    } else {
        for value in values {
            *value = (*value as f64 / norm) as f32;
        }
    }
}

pub(super) struct ProjectionOrder {
    direction: ModernUpload,
    direction_values: Vec<f32>,
    reference_projection: CubeTensor,
    query_projection: CubeTensor,
    pub ranks: CubeTensor,
    status: Option<CubeTensor>,
    workspace: RadixSortWorkspace,
    pub order: CubeTensor,
    shared: bool,
    rng: StdRng,
}

impl ProjectionOrder {
    pub fn new(
        template: &CubeTensor,
        h: usize,
        n: usize,
        d: usize,
        shared: bool,
        seed: u64,
    ) -> Result<Self> {
        let direction = ModernUpload::new(template, d, DType::F32)?;
        let reference_projection = allocate(template, n, DType::F32, 4)?;
        let query_projection = if shared {
            reference_projection.clone()
        } else {
            allocate(template, h, DType::F32, 4)?
        };
        let ranks = allocate(template, h, DType::I32, 4)?;
        let config =
            RadixSortConfig::conservative(RadixDeviceLimits::from_client(&template.client).path);
        let device =
            burn::tensor::Device::new(burn::backend::DispatchDevice::Cube(template.device.clone()));
        let budget = cuscan_common::resolve_memory_budget(&device, None)
            .map_err(super::backend_choice::map_backend_error)?;
        let workspace = RadixSortWorkspace::with_budget(&template.client, n, config, &budget, 0)
            .map_err(radix_error)?;
        #[cfg(test)]
        PROJECTION_COUNTS.with(|counts| {
            let (owners, refreshes) = counts.get();
            counts.set((owners + 1, refreshes));
        });
        // Replaced by workspace-owned IDs before the first consumer.
        let order = ranks.clone();
        Ok(Self {
            direction,
            direction_values: zeroed_floats(d)?,
            reference_projection,
            query_projection,
            ranks,
            status: if shared {
                None
            } else {
                Some(allocate(template, 1, DType::I32, 4)?)
            },
            workspace,
            order,
            shared,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn refresh(&mut self, head: &CubeTensor, fixed_tail: &CubeTensor) -> Result<()> {
        let mut norm = 0.0_f64;
        for pair in self.direction_values.chunks_mut(2) {
            let (first, second) = umap_core::recursive_init::gaussian_pair(&mut self.rng);
            pair[0] = first as f32;
            if pair.len() == 2 {
                pair[1] = second as f32;
            }
            for value in pair {
                norm += (*value as f64).powi(2);
            }
        }
        normalize_direction(&mut self.direction_values, norm);
        self.direction
            .update(self.direction_values.iter().map(|value| value.to_bits()))?;
        let reference = if self.shared { head } else { fixed_tail };
        modern_project(
            reference,
            head,
            &self.direction.tensor,
            &self.reference_projection,
            &self.query_projection,
            self.shared,
        );
        let sorted = self
            .workspace
            .sort(&self.reference_projection.handle, reference.meta.shape[0])
            .map_err(radix_error)?;
        if let Some(status) = &self.status {
            modern_query_projection_status(&self.query_projection, status);
            let bytes = status
                .client
                .read_one(status.handle.clone())
                .map_err(|e| UmapError::BackendUnavailable(format!("projection status: {e}")))?;
            let row = u32::from_ne_bytes(bytes[..4].try_into().expect("one status word"));
            if row != u32::MAX {
                return Err(UmapError::Numerical(format!(
                    "non-finite query projection at row {row}"
                )));
            }
        }
        self.order = CubeTensor::new_contiguous(
            reference.client.clone(),
            reference.device.clone(),
            Shape::new([reference.meta.shape[0]]),
            sorted.ids,
            DType::I32,
        );
        let values = CubeTensor::new_contiguous(
            reference.client.clone(),
            reference.device.clone(),
            Shape::new([reference.meta.shape[0]]),
            sorted.sorted_values,
            DType::F32,
        );
        modern_projection_ranks(
            &self.order,
            &values,
            &self.query_projection,
            &self.ranks,
            self.shared,
        );
        #[cfg(test)]
        PROJECTION_COUNTS.with(|counts| {
            let (owners, refreshes) = counts.get();
            counts.set((owners, refreshes + 1));
        });
        Ok(())
    }
}

pub(super) struct ForceSamples {
    sources: ModernUpload,
    local: ModernUpload,
    global: ModernUpload,
    output: CubeTensor,
    host: Vec<f64>,
    count: usize,
    rng: StdRng,
}

#[cfg(test)]
pub(super) struct ForceSnapshot {
    pub sources: Vec<i32>,
    pub local: Vec<i32>,
    pub global: Vec<i32>,
    pub resultants: Vec<f64>,
}

#[cfg(test)]
thread_local! {
    pub(super) static FORCE_OBSERVER: std::cell::RefCell<Option<Vec<ForceSnapshot>>> = const { std::cell::RefCell::new(None) };
}

impl ForceSamples {
    pub fn new(template: &CubeTensor, count: usize, n: usize) -> Result<Self> {
        let count = count.min(n);
        let candidates = count
            .checked_mul(5)
            .ok_or(UmapError::Overflow("adaptation candidates"))?;
        let outputs = count
            .checked_mul(2)
            .ok_or(UmapError::Overflow("adaptation resultants"))?;
        let mut host = Vec::new();
        host.try_reserve_exact(outputs)
            .map_err(|e| UmapError::Memory(format!("adaptation readback: {e}")))?;
        host.resize(outputs, 0.0);
        Ok(Self {
            sources: ModernUpload::new(template, count, DType::I32)?,
            local: ModernUpload::new(template, candidates, DType::I32)?,
            global: ModernUpload::new(template, candidates, DType::I32)?,
            output: allocate(template, outputs, DType::F64, 8)?,
            host,
            count,
            rng: StdRng::seed_from_u64(42),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ratio(
        &mut self,
        embedding: &CubeTensor,
        order: &CubeTensor,
        offsets: &CubeTensor,
        indices: &CubeTensor,
        raw_range: usize,
        a: f32,
        b: f32,
        gamma: f32,
        exclude: bool,
    ) -> Result<f64> {
        let n = embedding.meta.shape[0] as u32;
        let q = u32::try_from(raw_range).map_err(|_| UmapError::Overflow("adaptation range"))?;
        let rng = &mut self.rng;
        self.sources
            .update((0..self.count).map(|_| rng.random_range(0..n)))?;
        self.local
            .update((0..self.count * 5).map(|_| rng.random_range(0..q)))?;
        self.global
            .update((0..self.count * 5).map(|_| rng.random_range(0..n)))?;
        modern_sample_resultants(
            embedding,
            order,
            offsets,
            indices,
            &self.sources.tensor,
            &self.local.tensor,
            &self.global.tensor,
            &self.output,
            q,
            a,
            b,
            gamma,
            exclude,
        );
        let bytes = self
            .output
            .client
            .read_one(self.output.handle.clone())
            .map_err(|e| UmapError::BackendUnavailable(format!("adaptation readback: {e}")))?;
        for (value, chunk) in self.host.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *value = f64::from_ne_bytes(*chunk);
        }
        #[cfg(test)]
        FORCE_OBSERVER.with(|observer| {
            if let Some(events) = observer.borrow_mut().as_mut() {
                let read = |tensor: &CubeTensor| {
                    let bytes = tensor
                        .client
                        .read_one(tensor.handle.clone())
                        .expect("sample snapshot readback");
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| i32::from_ne_bytes(*word))
                        .collect()
                };
                events.push(ForceSnapshot {
                    sources: read(&self.sources.tensor),
                    local: read(&self.local.tensor),
                    global: read(&self.global.tensor),
                    resultants: self.host.clone(),
                });
            }
        });
        negative_force_ratio(&self.host[..self.count], &self.host[self.count..])
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_direction;

    #[test]
    fn raw_readback_plans_cover_physical_handle_extents() {
        let mut devices = vec![cubecl::Device::cpu().expect("CPU device")];
        if crate::cuda_backend::device(0).is_some() {
            devices.push(cubecl::Device::cuda(0).expect("CUDA device"));
        } else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA required"
            );
        }
        let config = umap_core::UMAPConfig {
            optimizer: umap_core::Optimizer::Adam,
            init: umap_core::Init::Spectral,
            negative_selection_range: 1,
            negative_sample_scale_adaptation_samples: 1,
            ..Default::default()
        };
        for device in devices {
            let client = device.client();
            for (operation, name, logical_bytes) in [
                (
                    umap_core::modern_layout::LayoutOperation::Fit,
                    "adaptation readback",
                    16,
                ),
                (
                    umap_core::modern_layout::LayoutOperation::Transform,
                    "query projection readback",
                    4,
                ),
            ] {
                let mut buffers = Vec::new();
                super::append_buffers(&mut buffers, &client, operation, 2, 2, 2, &config).unwrap();
                let planned = buffers
                    .iter()
                    .find(|buffer| buffer.name == name)
                    .unwrap()
                    .bytes;
                let handle = client.empty(logical_bytes);
                assert!(
                    planned >= handle.size() as usize,
                    "{device:?} {name}: {planned} < {}",
                    handle.size()
                );
            }
        }
    }

    #[test]
    fn projection_direction_normalization_is_deterministic() {
        for squared_norm in [0.0, f64::INFINITY, f64::NAN] {
            let mut direction = [0.0, -2.0, 3.0];
            normalize_direction(&mut direction, squared_norm);
            assert_eq!(direction, [1.0, 0.0, 0.0]);
        }

        let mut direction = [3.0, 4.0];
        normalize_direction(&mut direction, 25.0);
        assert_eq!(direction, [0.6, 0.8]);
    }
}
