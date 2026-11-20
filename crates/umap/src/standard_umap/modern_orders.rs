//! CPU mapped and CUDA pinned order double buffering.
//!
//! Orders are kernel inputs only. Callers must not retain or submit a consumer of
//! an inactive slot, and must pass every written output of the current epoch to
//! `finish_next`. A failed epoch poisons this owner; it cannot be reused.

use std::thread::{self, JoinHandle};
#[cfg(test)]
use std::time::Instant;

use burn::tensor::{DType, Shape};
use burn_cubecl::tensor::CubeTensor;
use cubecl::server::Handle;
use cubecl_cpu::compute::server::CpuServer;
use cubecl_environment::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes, Reader, Writer,
};
use cubecl_environment::stream::StreamId;
use cubecl_server::storage::{BytesResource, ManagedResource};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::mem::MaybeUninit;
use std::sync::mpsc::{self, SyncSender};
use umap_core::{Result, UmapError};

type Mapped = ManagedResource<BytesResource>;

enum Slot {
    Cpu {
        host: Vec<i32>,
        mapped: Mapped,
    },
    Cuda {
        host: Option<Bytes>,
        handle: Handle,
        client: cubecl::client::Client,
        stream: StreamId,
        #[cfg(test)]
        profile: Option<Option<cubecl::logging::ProfileDuration>>,
    },
}

struct ReturnPinned {
    bytes: Option<Bytes>,
    sender: SyncSender<Bytes>,
}

impl AllocationController for ReturnPinned {
    fn alloc_align(&self) -> usize {
        self.bytes.as_ref().unwrap().align()
    }
    fn property(&self) -> AllocationProperty {
        self.bytes.as_ref().unwrap().property()
    }
    fn capacity(&self) -> usize {
        self.bytes.as_ref().unwrap().len()
    }
    fn memory(&self, policy: AccessPolicy) -> std::result::Result<&[MaybeUninit<u8>], AccessError> {
        let reader = if policy.copy_allowed() {
            Reader::new()
        } else {
            Reader::new().no_copy()
        };
        let bytes = self.bytes.as_ref().unwrap().read(reader)?;
        // Every byte exposed by Bytes::read is initialized and MaybeUninit has identical layout.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) })
    }
    unsafe fn memory_mut(
        &mut self,
        policy: AccessPolicy,
    ) -> std::result::Result<&mut [MaybeUninit<u8>], AccessError> {
        let writer = if policy.copy_allowed() {
            Writer::new()
        } else {
            Writer::new().no_copy()
        };
        let bytes = self.bytes.as_mut().unwrap().write(writer)?;
        // The inner allocation is exclusively borrowed and all bytes are initialized.
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len()) })
    }
}

impl Drop for ReturnPinned {
    fn drop(&mut self) {
        if let Some(bytes) = self.bytes.take() {
            let _ = self.sender.try_send(bytes);
        }
    }
}

struct Order {
    tensors: [CubeTensor; 2],
    slots: [Option<Slot>; 2],
}

/// One reusable upload allocation for projection directions and sampled ranks.
/// Callers must complete every preceding device consumer before updating it.
pub(super) struct ModernUpload {
    pub tensor: CubeTensor,
    slot: Slot,
    words: usize,
}

impl ModernUpload {
    pub fn new(template: &CubeTensor, words: usize, dtype: DType) -> Result<Self> {
        if words == 0 || words > i32::MAX as usize {
            return Err(UmapError::InvalidInput(
                "modern upload length must fit positive i32".into(),
            ));
        }
        let bytes = words
            .checked_mul(4)
            .ok_or(UmapError::Overflow("modern upload"))?;
        let stream = StreamId::current();
        let tensor = CubeTensor::new_contiguous(
            template.client.clone(),
            template.device.clone(),
            Shape::new([words]),
            stream.executes(|| template.client.empty(bytes)),
            dtype,
        );
        let slot = if matches!(&template.device, burn_cubecl::CubeDevice::Cpu(_)) {
            let mut host = Vec::new();
            host.try_reserve_exact(words)
                .map_err(|e| UmapError::Memory(format!("modern upload staging: {e}")))?;
            host.resize(words, 0);
            let mapped = template
                .client
                .get_resource::<CpuServer>(tensor.handle.clone())
                .map_err(|e| error(format!("modern upload mapping: {e}")))?;
            Slot::Cpu { host, mapped }
        } else {
            let mut initial = Vec::new();
            initial
                .try_reserve_exact(bytes)
                .map_err(|e| UmapError::Memory(format!("modern upload staging: {e}")))?;
            initial.resize(bytes, 0u8);
            let mut host = Bytes::from_elems(initial);
            template.client.staging(std::iter::once(&mut host), false);
            if host.property() != AllocationProperty::Pinned {
                return Err(error("CUDA modern upload staging is not pinned"));
            }
            Slot::Cuda {
                host: Some(host),
                handle: tensor.handle.clone(),
                client: template.client.clone(),
                stream,
                #[cfg(test)]
                profile: None,
            }
        };
        Ok(Self {
            tensor,
            slot,
            words,
        })
    }

    pub fn update(&mut self, words: impl ExactSizeIterator<Item = u32>) -> Result<()> {
        if words.len() != self.words {
            return Err(UmapError::InvalidInput(
                "modern upload length mismatch".into(),
            ));
        }
        match &mut self.slot {
            Slot::Cpu { host, .. } => {
                for (target, word) in host.iter_mut().zip(words) {
                    *target = word as i32;
                }
            }
            Slot::Cuda { host, .. } => {
                let bytes = host
                    .as_mut()
                    .ok_or_else(|| error("modern upload is in flight"))?
                    .write(Writer::new().no_copy())
                    .map_err(|e| error(format!("modern upload staging write: {e}")))?;
                for (target, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                    *target = word.to_ne_bytes();
                }
            }
        }
        upload(&mut self.slot)
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(super) struct PrepareInterval {
    pub shuffle_start: Instant,
    pub shuffle_end: Instant,
    pub copy_end: Instant,
}

struct Prepared {
    source: [Slot; 2],
    negative: Option<[Slot; 2]>,
    rng: StdRng,
    #[cfg(test)]
    interval: PrepareInterval,
    #[cfg(test)]
    witness: Option<[(u32, u32); 3]>,
}

#[cfg(test)]
struct ExecutionMarker {
    // Binding pins the mapped allocation for as long as either host thread reads it.
    _mapping: Mapped,
    ptr: *const std::sync::atomic::AtomicU32,
}

// The only accesses to the two aligned words are atomic, and the mapping is
// retained for the lifetime of every reference. The test kernel writes atomically.
#[cfg(test)]
unsafe impl Send for ExecutionMarker {}
#[cfg(test)]
unsafe impl Sync for ExecutionMarker {}
#[cfg(test)]
impl ExecutionMarker {
    fn sample(&self) -> (u32, u32) {
        use std::sync::atomic::Ordering;
        // The mapped region has two aligned AtomicU32 words, initialized before
        // launch, and remains alive through the owned ManagedResource.
        unsafe {
            (
                (*self.ptr).load(Ordering::SeqCst),
                (*self.ptr.add(1)).load(Ordering::SeqCst),
            )
        }
    }
}

/// Owns two host arrays and two device arrays per order. CPU device arrays are
/// mapped once; CUDA host arrays are pinned and retained until upload completion.
pub(super) struct ModernOrders {
    source: Order,
    negative: Option<Order>,
    active: usize,
    epoch: usize,
    rng: Option<StdRng>,
    worker: Option<JoinHandle<Result<Prepared>>>,
    poisoned: bool,
    #[cfg(test)]
    pub(super) last_interval: Option<PrepareInterval>,
    pub(super) preparations: usize,
    #[cfg(test)]
    gate: Option<std::sync::mpsc::Receiver<()>>,
    #[cfg(test)]
    fail_upload: bool,
    #[cfg(test)]
    marker: Option<std::sync::Arc<ExecutionMarker>>,
    #[cfg(test)]
    witness: Option<[(u32, u32); 3]>,
}

fn error(message: impl Into<String>) -> UmapError {
    UmapError::BackendUnavailable(message.into())
}

/// Byte counts for the fixed arrays. On CUDA the two host arrays per order
/// must be the pinned staging allocations themselves, not additional Vecs.
/// Runtime transfer envelopes and channel bookkeeping are not included.
pub(super) fn order_array_bytes(
    h: usize,
    n: usize,
    shared_fit_momentum: bool,
) -> Result<(usize, usize)> {
    if h == 0 || n == 0 || h > i32::MAX as usize || n > i32::MAX as usize {
        return Err(UmapError::InvalidInput(
            "modern order lengths must fit positive i32".into(),
        ));
    }
    if shared_fit_momentum && h != n {
        return Err(UmapError::InvalidInput(
            "shared fit order requires H=N".into(),
        ));
    }
    let elements = if shared_fit_momentum {
        h
    } else {
        h.checked_add(n)
            .ok_or(UmapError::Overflow("modern order elements"))?
    };
    let bytes = elements
        .checked_mul(2)
        .and_then(|x| x.checked_mul(std::mem::size_of::<i32>()))
        .ok_or(UmapError::Overflow("modern order bytes"))?;
    Ok((bytes, bytes))
}

fn create_order(template: &CubeTensor, len: usize, stream: StreamId) -> Result<Order> {
    if len == 0 || len > i32::MAX as usize {
        return Err(UmapError::InvalidInput(
            "modern order length must fit positive i32".into(),
        ));
    }
    let bytes = len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(UmapError::Overflow("modern order bytes"))?;
    let make = || -> Result<(CubeTensor, Slot)> {
        let tensor = CubeTensor::new_contiguous(
            template.client.clone(),
            template.device.clone(),
            Shape::new([len]),
            stream.executes(|| template.client.empty(bytes)),
            DType::I32,
        );
        let mut slot = if matches!(&template.device, burn_cubecl::CubeDevice::Cpu(_)) {
            let mut host = Vec::new();
            host.try_reserve_exact(len)
                .map_err(|e| UmapError::Memory(format!("modern order allocation: {e}")))?;
            host.extend((0..len).map(|n| n as i32));
            let mapped = template
                .client
                .get_resource::<CpuServer>(tensor.handle.clone())
                .map_err(|e| error(format!("modern CPU order mapping: {e}")))?;
            Slot::Cpu { host, mapped }
        } else {
            let mut initial = Vec::new();
            initial
                .try_reserve_exact(bytes)
                .map_err(|e| UmapError::Memory(format!("modern order staging: {e}")))?;
            for n in 0..len {
                initial.extend_from_slice(&(n as i32).to_ne_bytes());
            }
            let mut host = Bytes::from_elems(initial);
            template.client.staging(std::iter::once(&mut host), false);
            if host.property() != AllocationProperty::Pinned {
                return Err(error("CUDA modern order staging is not pinned"));
            }
            Slot::Cuda {
                host: Some(host),
                handle: tensor.handle.clone(),
                client: template.client.clone(),
                stream,
                #[cfg(test)]
                profile: None,
            }
        };
        upload(&mut slot)?;
        Ok((tensor, slot))
    };
    let (a, a_slot) = make()?;
    let (b, b_slot) = make()?;
    Ok(Order {
        tensors: [a, b],
        slots: [Some(a_slot), Some(b_slot)],
    })
}

fn copy_order(host: &[i32], mapped: &mut Mapped) {
    // The bytes are written individually to avoid alignment and lifetime assumptions.
    // The ManagedResource pins the allocation. The state machine gives this writer
    // exclusive access to the inactive slot, with no active device consumer.
    let target = mapped.resource_mut().write();
    for (chunk, value) in target.as_chunks_mut::<4>().0.iter_mut().zip(host) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn prepare_order(slots: &mut [Slot; 2], active: usize, rng: &mut StdRng) -> Result<()> {
    let (previous, next) = if active == 0 {
        let (a, b) = slots.split_at_mut(1);
        (&a[0], &mut b[0])
    } else {
        let (a, b) = slots.split_at_mut(1);
        (&b[0], &mut a[0])
    };
    match (previous, next) {
        (Slot::Cpu { host: previous, .. }, Slot::Cpu { host: next, .. }) => {
            next.copy_from_slice(previous);
            next.shuffle(rng);
        }
        (
            Slot::Cuda {
                host: Some(previous),
                ..
            },
            Slot::Cuda {
                host: Some(next), ..
            },
        ) => {
            let previous = previous
                .read(Reader::new().no_copy())
                .map_err(|e| error(format!("pinned order read: {e}")))?;
            let next = next
                .write(Writer::new().no_copy())
                .map_err(|e| error(format!("pinned order write: {e}")))?;
            next.copy_from_slice(previous);
            // [u8; 4] has alignment one, and the buffer holds initialized four-byte indices.
            let indices = unsafe {
                std::slice::from_raw_parts_mut(next.as_mut_ptr().cast::<[u8; 4]>(), next.len() / 4)
            };
            indices.shuffle(rng);
        }
        _ => return Err(error("inconsistent modern order slots")),
    }
    Ok(())
}

fn upload(slot: &mut Slot) -> Result<()> {
    match slot {
        Slot::Cpu { host, mapped } => {
            copy_order(host, mapped);
            Ok(())
        }
        Slot::Cuda {
            host,
            handle,
            client,
            stream,
            #[cfg(test)]
            profile,
        } => {
            let bytes = host
                .take()
                .ok_or_else(|| error("pinned order already in flight"))?;
            let len = bytes.len();
            let (sender, receiver) = mpsc::sync_channel(1);
            // The envelope retains initialized pinned storage until the backend releases it.
            let envelope = unsafe {
                Bytes::from_controller(
                    Box::new(ReturnPinned {
                        bytes: Some(bytes),
                        sender,
                    }),
                    len,
                )
            };
            stream.executes(|| {
                #[cfg(test)]
                let window = if profile.is_some() {
                    Some(
                        client
                            .profile_start()
                            .map_err(|e| error(format!("upload profile start: {e}")))?,
                    )
                } else {
                    None
                };
                client.write(handle, envelope);
                #[cfg(test)]
                if let Some(window) = window {
                    *profile = Some(Some(
                        client
                            .profile_end(window)
                            .map_err(|e| error(format!("upload profile end: {e}")))?,
                    ));
                }
                burn::cubecl::future::block_on(client.sync_buffers([&*handle]))
                    .map_err(|e| error(format!("modern order upload: {e}")))?;
                client
                    .flush()
                    .map_err(|e| error(format!("modern order upload flush: {e}")))?;
                client
                    .flush()
                    .map_err(|e| error(format!("modern order upload flush: {e}")))?;
                Ok::<(), UmapError>(())
            })?;
            *host = Some(receiver.try_recv().map_err(|e| {
                error(format!(
                    "pinned order buffer not returned after completion: {e}"
                ))
            })?);
            Ok(())
        }
    }
}

fn take_slots(order: &mut Order) -> [Slot; 2] {
    [
        order.slots[0].take().expect("idle order slot"),
        order.slots[1].take().expect("idle order slot"),
    ]
}

impl ModernOrders {
    /// `shared_fit_momentum` is true only for fit momentum, never transform.
    pub(super) fn new(
        template: &CubeTensor,
        h: usize,
        n: usize,
        shared_fit_momentum: bool,
        seed: u64,
    ) -> Result<Self> {
        if !matches!(
            &template.device,
            burn_cubecl::CubeDevice::Cpu(_) | burn_cubecl::CubeDevice::Cuda(_)
        ) {
            return Err(error("modern orders require CPU or CUDA"));
        }
        order_array_bytes(h, n, shared_fit_momentum)?;
        let stream = StreamId::allocate();
        let source = create_order(template, h, stream)?;
        let negative = if shared_fit_momentum {
            None
        } else {
            Some(create_order(template, n, stream)?)
        };
        Ok(Self {
            source,
            negative,
            active: 0,
            epoch: 0,
            rng: Some(StdRng::seed_from_u64(seed)),
            worker: None,
            poisoned: false,
            preparations: 0,
            #[cfg(test)]
            last_interval: None,
            #[cfg(test)]
            gate: None,
            #[cfg(test)]
            fail_upload: false,
            #[cfg(test)]
            marker: None,
            #[cfg(test)]
            witness: None,
        })
    }

    pub(super) fn current(&self) -> (&CubeTensor, &CubeTensor) {
        let source = &self.source.tensors[self.active];
        let negative = self
            .negative
            .as_ref()
            .map_or(source, |o| &o.tensors[self.active]);
        (source, negative)
    }

    pub(super) fn epoch(&self) -> usize {
        self.epoch
    }

    /// Retire random reference orders once projection owns all future negatives.
    /// Every consumer of the old reference buffers must already be complete.
    pub(super) fn retire_reference_order(&mut self) -> Result<()> {
        if self.poisoned || self.worker.is_some() {
            return Err(error("cannot retire reference order during preparation"));
        }
        self.negative = None;
        Ok(())
    }

    /// Start before waiting on the current epoch. Exactly one preparation may be in flight.
    pub(super) fn begin_prepare_next(&mut self) -> Result<()> {
        if self.poisoned || self.worker.is_some() {
            return Err(error("modern order owner is poisoned or already preparing"));
        }
        let mut source = take_slots(&mut self.source);
        let mut negative = self.negative.as_mut().map(take_slots);
        let mut rng = self.rng.take().expect("idle order RNG");
        let active = self.active;
        #[cfg(test)]
        let gate = self.gate.take();
        #[cfg(test)]
        let fail_upload = std::mem::take(&mut self.fail_upload);
        #[cfg(test)]
        let marker = self.marker.take();
        #[cfg(feature = "tracing")]
        let parent = tracing::Span::current();
        let worker = thread::Builder::new()
            .name("umap-orders".into())
            .spawn(move || {
                #[cfg(test)]
                if let Some(gate) = gate {
                    let _ = gate.recv();
                }
                #[cfg(test)]
                let shuffle_start = Instant::now();
                #[cfg(test)]
                let before = marker.as_ref().map(|m| m.sample());
                {
                    #[cfg(feature = "tracing")]
                    let _span = tracing::info_span!(parent: &parent, "modern_source_shuffle_copy")
                        .entered();
                    prepare_order(&mut source, active, &mut rng)?;
                }
                if let Some(slots) = &mut negative {
                    #[cfg(feature = "tracing")]
                    let _span =
                        tracing::info_span!(parent: &parent, "modern_reference_shuffle_copy")
                            .entered();
                    prepare_order(slots, active, &mut rng)?;
                }
                #[cfg(test)]
                let shuffle_end = Instant::now();
                #[cfg(test)]
                let after_shuffle = marker.as_ref().map(|m| m.sample());
                let next = 1 - active;
                #[cfg(test)]
                if fail_upload {
                    return Err(error("injected order upload failure"));
                }
                {
                    #[cfg(feature = "tracing")]
                    let _span =
                        tracing::info_span!(parent: &parent, "modern_source_upload_completion")
                            .entered();
                    upload(&mut source[next])?;
                }
                if let Some(slots) = &mut negative {
                    #[cfg(feature = "tracing")]
                    let _span =
                        tracing::info_span!(parent: &parent, "modern_reference_upload_completion")
                            .entered();
                    upload(&mut slots[next])?;
                }
                Ok(Prepared {
                    source,
                    negative,
                    rng,
                    #[cfg(test)]
                    interval: PrepareInterval {
                        shuffle_start,
                        shuffle_end,
                        copy_end: Instant::now(),
                    },
                    #[cfg(test)]
                    witness: marker
                        .as_ref()
                        .map(|m| [before.unwrap(), after_shuffle.unwrap(), m.sample()]),
                })
            });
        match worker {
            Ok(worker) => self.worker = Some(worker),
            Err(e) => {
                self.poisoned = true;
                return Err(error(format!("modern order producer spawn: {e}")));
            }
        }
        Ok(())
    }

    /// Wait for every current-epoch written output and the producer before switching slots.
    /// The caller must include every gradient/step output and submit no more consumers
    /// of the old slot after this call. Empty output sets are rejected.
    pub(super) fn finish_next(&mut self, outputs: &[&Handle]) -> Result<()> {
        if self.poisoned || self.worker.is_none() || outputs.is_empty() {
            return Err(error(
                "modern order finish requires a producer and epoch outputs",
            ));
        }
        let sync = burn::cubecl::future::block_on(
            self.source.tensors[self.active]
                .client
                .sync_buffers(outputs.iter().copied()),
        );
        let worker = self.worker.take().expect("checked producer");
        let prepared = match worker.join() {
            Ok(Ok(value)) => value,
            Ok(Err(e)) => {
                self.poisoned = true;
                return Err(e);
            }
            Err(_) => {
                self.poisoned = true;
                return Err(error("modern order producer panicked"));
            }
        };
        self.source.slots = prepared.source.map(Some);
        if let (Some(order), Some(slots)) = (&mut self.negative, prepared.negative) {
            order.slots = slots.map(Some);
        }
        self.rng = Some(prepared.rng);
        if let Err(e) = sync {
            self.poisoned = true;
            return Err(error(format!("modern epoch completion: {e}")));
        }
        #[cfg(test)]
        {
            self.last_interval = Some(prepared.interval);
        }
        #[cfg(test)]
        {
            self.witness = prepared.witness;
        }
        self.preparations += 1;
        self.active = 1 - self.active;
        self.epoch += 1;
        Ok(())
    }
}

impl Drop for ModernOrders {
    fn drop(&mut self) {
        // The worker owns mapped allocations; join before releasing owner state.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ops::FloatTensorOps;
    use burn::cubecl::prelude::*;
    use burn::tensor::TensorData;
    use burn_cubecl::{CubeBackend, CubeDevice};
    use std::sync::mpsc;

    fn template() -> CubeTensor {
        CubeBackend::float_from_data(
            TensorData::new(vec![0.0_f32], [1]),
            &CubeDevice::Cpu(Default::default()),
        )
    }

    #[cube(launch)]
    fn overlap_work(order: &Tensor<i32>, output: &mut Tensor<f32>, rounds: u32) {
        let i = ABSOLUTE_POS_X as usize;
        if i < order.shape(0) {
            let mut x = order[i] as f32 * 0.000001 + 0.125;
            for _ in 0..rounds {
                x = x * 0.99991 + 0.000017;
                x = x * x + 0.000001;
            }
            output[i] = x;
        }
    }

    fn launch_work(order: &CubeTensor, output: &CubeTensor, rounds: u32, n: u32) {
        overlap_work::launch(
            &order.client,
            CubeCount::Static(n.div_ceil(128), 1, 1),
            CubeDim::new_1d(128),
            order.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            rounds,
        );
    }

    #[cube(launch)]
    fn cpu_execution_witness(
        order: &Tensor<i32>,
        output: &mut Tensor<f32>,
        marker: &mut Tensor<Atomic<u32>>,
        rounds: u32,
    ) {
        if ABSOLUTE_POS_X == 0 {
            // The previous atomic value feeds the computation; the completion
            // marker depends on the computed result, rather than on submission.
            let initial = marker[0].fetch_add(1u32);
            let mut x = order[0] as f32 + initial as f32 + 0.125f32;
            for _ in 0..rounds {
                x = x * 0.99991 + 0.000017;
                x = x * x + 0.000001;
            }
            output[0] = x;
            marker[1].store(if x > 0.0f32 { 1u32 } else { 2u32 });
        }
    }

    fn launch_witness(order: &CubeTensor, output: &CubeTensor, marker: &CubeTensor, rounds: u32) {
        cpu_execution_witness::launch(
            &order.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            order.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            marker.clone().into_tensor_arg(),
            rounds,
        );
    }

    #[test]
    fn cpu_shuffle_and_mapped_copy_overlap_executing_kernel() {
        use std::sync::atomic::{AtomicU32, Ordering};
        const N: usize = 262_144;
        let template = template();
        let mut owner = ModernOrders::new(&template, N, N, true, 7).unwrap();
        let output = CubeTensor::new_contiguous(
            template.client.clone(),
            template.device.clone(),
            Shape::new([1]),
            template.client.empty(4),
            DType::F32,
        );
        let marker_tensor = CubeTensor::new_contiguous(
            template.client.clone(),
            template.device.clone(),
            Shape::new([2]),
            template.client.empty(8),
            DType::U32,
        );
        let mut mapping = template
            .client
            .get_resource::<CpuServer>(marker_tensor.handle.clone())
            .unwrap();
        let (ptr, len) = mapping.resource().get_write_ptr_and_length();
        assert!(len >= 8 && ptr.align_offset(std::mem::align_of::<AtomicU32>()) == 0);
        // Initialize before concurrent access. All later host/kernel accesses are atomic.
        mapping.resource_mut().write()[..8].fill(0);
        let marker = std::sync::Arc::new(ExecutionMarker {
            _mapping: mapping,
            ptr: ptr.cast(),
        });
        launch_witness(owner.current().0, &output, &marker_tensor, 1);
        burn::cubecl::future::block_on(
            template
                .client
                .sync_buffers([&output.handle, &marker_tensor.handle]),
        )
        .unwrap();
        unsafe {
            (*marker.ptr).store(0, Ordering::SeqCst);
            (*marker.ptr.add(1)).store(0, Ordering::SeqCst);
        }
        launch_witness(owner.current().0, &output, &marker_tensor, 100_000_000);
        let client = template.client.clone();
        let output_handle = output.handle.clone();
        let marker_handle = marker_tensor.handle.clone();
        let compute_stream = StreamId::current();
        let keep_mapping = marker.clone();
        let waiter = thread::spawn(move || {
            let _keep_mapping = keep_mapping;
            compute_stream.executes(|| {
                burn::cubecl::future::block_on(
                    client.sync_buffers([&output_handle, &marker_handle]),
                )
            })
        });
        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        while marker.sample().0 == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(
            marker.sample(),
            (1, 0),
            "kernel did not enter or finished before preparation"
        );
        owner.marker = Some(marker.clone());
        owner.begin_prepare_next().unwrap();
        owner
            .finish_next(&[&output.handle, &marker_tensor.handle])
            .unwrap();
        waiter.join().unwrap().unwrap();
        assert_eq!(
            owner.witness.unwrap(),
            [(1, 0); 3],
            "kernel must remain executing across shuffle and mapped copy"
        );
        assert_eq!(marker.sample().1, 1);
        let interval = owner.last_interval.unwrap();
        assert!(
            interval.shuffle_start <= interval.shuffle_end
                && interval.shuffle_end <= interval.copy_end
        );
    }

    fn cpu_slot(slot: &Option<Slot>) -> (&Vec<i32>, &Mapped) {
        match slot.as_ref().unwrap() {
            Slot::Cpu { host, mapped } => (host, mapped),
            Slot::Cuda { .. } => panic!("expected CPU slot"),
        }
    }

    #[test]
    fn slots_are_reused_and_rng_is_sequential() {
        let template = template();
        let mut owner = ModernOrders::new(&template, 8, 7, false, 19).unwrap();
        let mut expected_source: Vec<i32> = (0..8).collect();
        let mut expected_negative: Vec<i32> = (0..7).collect();
        let mut rng = StdRng::seed_from_u64(19);
        let host_pointers = owner
            .source
            .slots
            .each_ref()
            .map(|slot| cpu_slot(slot).0.as_ptr());
        let device_pointers = owner
            .source
            .slots
            .each_ref()
            .map(|slot| cpu_slot(slot).1.resource().get_write_ptr_and_length().0);
        for epoch in 1..=4 {
            expected_source.shuffle(&mut rng);
            expected_negative.shuffle(&mut rng);
            owner.begin_prepare_next().unwrap();
            let output = owner.current().0.handle.clone();
            owner.finish_next(&[&output]).unwrap();
            assert_eq!(owner.epoch(), epoch);
            assert_eq!(
                cpu_slot(&owner.source.slots[owner.active]).0,
                &expected_source
            );
            assert_eq!(
                cpu_slot(&owner.negative.as_ref().unwrap().slots[owner.active]).0,
                &expected_negative
            );
        }
        assert_eq!(
            owner
                .source
                .slots
                .each_ref()
                .map(|slot| cpu_slot(slot).0.as_ptr()),
            host_pointers
        );
        assert_eq!(
            owner.source.slots.each_ref().map(|slot| cpu_slot(slot)
                .1
                .resource()
                .get_write_ptr_and_length()
                .0),
            device_pointers
        );
        assert_eq!(owner.preparations, 4);
    }

    #[test]
    fn delayed_producer_cannot_reuse_or_publish_inactive_slot() {
        let template = template();
        let mut owner = ModernOrders::new(&template, 8, 8, true, 7).unwrap();
        assert!(std::ptr::eq(owner.current().0, owner.current().1));
        let (release, gate) = mpsc::channel();
        owner.gate = Some(gate);
        owner.begin_prepare_next().unwrap();
        assert_eq!(owner.epoch(), 0);
        assert!(owner.begin_prepare_next().is_err());
        assert!(owner.finish_next(&[]).is_err());
        let output = owner.current().0.handle.clone();
        release.send(()).unwrap();
        owner.finish_next(&[&output]).unwrap();
        assert_eq!(owner.epoch(), 1);
        assert!(std::ptr::eq(owner.current().0, owner.current().1));
    }

    #[test]
    fn equal_rows_transform_stays_distinct_and_upload_failure_poisons() {
        let template = template();
        let mut owner = ModernOrders::new(&template, 8, 8, false, 7).unwrap();
        assert!(!std::ptr::eq(owner.current().0, owner.current().1));
        owner.fail_upload = true;
        owner.begin_prepare_next().unwrap();
        let output = owner.current().0.handle.clone();
        assert!(owner.finish_next(&[&output]).is_err());
        assert_eq!(owner.epoch(), 0);
        assert!(owner.begin_prepare_next().is_err());
        assert!(owner.finish_next(&[&output]).is_err());
    }

    #[test]
    #[ignore = "non-gating multi-million-row order cost probe; requires CPU and CUDA"]
    fn multi_million_order_cost_probe() {
        const N: usize = 2_000_003;
        const EPOCHS: usize = 4;
        assert!(
            crate::cuda_backend::device(0).is_some(),
            "CUDA is required for this probe"
        );
        let (host_bytes, device_bytes) = order_array_bytes(N, N, false).unwrap();
        for (backend, device) in [
            ("cpu", CubeDevice::Cpu(Default::default())),
            ("cuda", CubeDevice::Cuda(Default::default())),
        ] {
            let template =
                CubeBackend::float_from_data(TensorData::new(vec![0.0_f32], [1]), &device);
            let mut owner = ModernOrders::new(&template, N, N, false, 42).unwrap();
            for epoch in 1..=EPOCHS {
                owner.begin_prepare_next().unwrap();
                let output = owner.current().0.handle.clone();
                owner.finish_next(&[&output]).unwrap();
                let interval = owner.last_interval.unwrap();
                let record = serde_json::json!({
                    "probe": "modern_order_cost",
                    "backend": backend,
                    "device": format!("{:?}", device),
                    "client": template.client.name(),
                    "epoch": epoch,
                    "source_rows": N,
                    "reference_rows": N,
                    "distinct_orders": true,
                    "fixed_host_bytes": host_bytes,
                    "fixed_device_bytes": device_bytes,
                    "host_memory": if backend == "cuda" { "pinned" } else { "vec" },
                    "source_reference_shuffle_ns": interval.shuffle_end.duration_since(interval.shuffle_start).as_nanos(),
                    "upload_host_elapsed_ns": interval.copy_end.duration_since(interval.shuffle_end).as_nanos(),
                    "upload_host_elapsed_scope": if backend == "cuda" {
                        "two pinned writes including synchronization, flush and ownership return; not pure DMA"
                    } else { "two mapped CPU copies" },
                    "device_dma_ns": null,
                    "measurement": "isolated preparation cost; no fit, range guard, or overlap claim"
                });
                println!("{record}");
            }
        }
    }

    #[test]
    fn cuda_pinned_orders_reuse_storage_and_follow_seeded_stream() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        let template = CubeBackend::float_from_data(
            TensorData::new(vec![0.0_f32], [1]),
            &CubeDevice::Cuda(Default::default()),
        );
        let mut owner = ModernOrders::new(&template, 9, 7, false, 27).unwrap();
        let addresses = owner
            .source
            .slots
            .each_ref()
            .map(|slot| match slot.as_ref().unwrap() {
                Slot::Cuda {
                    host: Some(host), ..
                } => host.read(Reader::new().no_copy()).unwrap().as_ptr(),
                _ => panic!("expected pinned CUDA slot"),
            });
        let mut expected: Vec<i32> = (0..9).collect();
        let mut negative: Vec<i32> = (0..7).collect();
        let mut rng = StdRng::seed_from_u64(27);
        for _ in 0..4 {
            expected.shuffle(&mut rng);
            negative.shuffle(&mut rng);
            owner.begin_prepare_next().unwrap();
            let output = owner.current().0.handle.clone();
            owner.finish_next(&[&output]).unwrap();
            let slot = owner.source.slots[owner.active].as_ref().unwrap();
            let Slot::Cuda {
                host: Some(host), ..
            } = slot
            else {
                panic!("expected CUDA slot")
            };
            let values: Vec<i32> = host
                .read(Reader::new().no_copy())
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| i32::from_ne_bytes(*chunk))
                .collect();
            assert_eq!(values, expected);
        }
        assert_eq!(
            owner
                .source
                .slots
                .each_ref()
                .map(|slot| match slot.as_ref().unwrap() {
                    Slot::Cuda {
                        host: Some(host), ..
                    } => host.read(Reader::new().no_copy()).unwrap().as_ptr(),
                    _ => panic!("expected pinned CUDA slot"),
                }),
            addresses
        );
    }

    #[test]
    fn cuda_shuffle_and_pinned_dma_overlap_executing_kernel() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        const N: usize = 1_048_576;
        let device = CubeDevice::Cuda(Default::default());
        let template = CubeBackend::float_from_data(TensorData::new(vec![0.0_f32], [1]), &device);
        let mut owner = ModernOrders::new(&template, N, N, true, 7).unwrap();
        let output = CubeTensor::new_contiguous(
            template.client.clone(),
            device,
            Shape::new([N]),
            template.client.empty(N * 4),
            DType::F32,
        );
        // Warm compilation and the compute stream before opening the measurement.
        launch_work(owner.current().0, &output, 1, N as u32);
        burn::cubecl::future::block_on(template.client.sync_buffers([&output.handle])).unwrap();
        let Slot::Cuda { profile, .. } = owner.source.slots[1].as_mut().unwrap() else {
            panic!("expected CUDA")
        };
        *profile = Some(None);
        let epoch = cubecl_environment::time::Instant::now();
        let compute_window = template.client.profile_start().unwrap();
        launch_work(owner.current().0, &output, 2_097_152, N as u32);
        let compute_profile = template.client.profile_end(compute_window).unwrap();
        assert_eq!(
            compute_profile.timing_method(),
            cubecl::logging::TimingMethod::Device
        );
        owner.begin_prepare_next().unwrap();
        owner.finish_next(&[&output.handle]).unwrap();
        let host = owner.last_interval.unwrap();
        let Slot::Cuda { profile, .. } = owner.source.slots[owner.active].as_mut().unwrap() else {
            panic!("expected CUDA")
        };
        let upload_profile = profile.take().unwrap().unwrap();
        assert_eq!(
            upload_profile.timing_method(),
            cubecl::logging::TimingMethod::Device
        );
        let compute = burn::cubecl::future::block_on(compute_profile.resolve()).unwrap();
        let transfer = burn::cubecl::future::block_on(upload_profile.resolve()).unwrap();
        assert!(compute.duration() < std::time::Duration::from_secs(60));
        assert!(transfer.duration() < std::time::Duration::from_secs(60));
        let (cs, ce) = (
            compute.start_duration_since(epoch),
            compute.end_duration_since(epoch),
        );
        let (us, ue) = (
            transfer.start_duration_since(epoch),
            transfer.end_duration_since(epoch),
        );
        let (hs, he) = (
            host.shuffle_start.duration_since(epoch),
            host.shuffle_end.duration_since(epoch),
        );
        assert!(
            cs < he && hs < ce,
            "kernel {cs:?}..{ce:?}, shuffle {hs:?}..{he:?}"
        );
        assert!(
            cs < ue && us < ce,
            "kernel {cs:?}..{ce:?}, DMA {us:?}..{ue:?}"
        );
    }
}
