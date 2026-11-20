//! Host-side building blocks for the modern Euclidean layouts.
//!
//! These helpers do not select or execute a layout backend. Edge positions are
//! always positions in the compact, sorted CSR, never positions in padded COO.

use crate::recursive_init::{LevelParams, coarse_negative_range, graph::HostCsr};
use crate::{Optimizer, Result, UMAPConfig, UmapError};

/// Fully resolved, operation-specific settings for a modern layout.
#[derive(Clone, Copy, Debug)]
pub struct ModernLayoutParams {
    pub optimizer: Optimizer,
    pub epochs: u32,
    pub a: f32,
    pub b: f32,
    pub repulsion: f32,
    pub initial_alpha: f32,
    pub negative_sample_rate: usize,
    pub negative_selection_range: usize,
    pub negative_sample_scale: Option<f32>,
    pub adaptation_samples: usize,
    pub exclude_graph_neighbors: bool,
    pub good_initialization: bool,
    pub seed: u64,
}

impl ModernLayoutParams {
    pub fn main(
        config: &UMAPConfig,
        epochs: u32,
        a: f32,
        b: f32,
        initial_alpha: f32,
        good_initialization: bool,
        seed: u64,
    ) -> Self {
        Self {
            optimizer: config.optimizer,
            epochs,
            a,
            b,
            repulsion: config.repulsion_strength,
            initial_alpha,
            negative_sample_rate: config.negative_sample_rate,
            negative_selection_range: config.negative_selection_range,
            negative_sample_scale: config.negative_sample_scale,
            adaptation_samples: config.negative_sample_scale_adaptation_samples,
            exclude_graph_neighbors: config.exclude_graph_neighbors,
            good_initialization,
            seed,
        }
    }

    pub fn recursive_level(level: LevelParams, realized_rows: usize, seed: u64) -> Self {
        Self {
            optimizer: Optimizer::Adam,
            epochs: level.epochs,
            a: level.a,
            b: level.b,
            repulsion: 4.0,
            initial_alpha: 0.5,
            negative_sample_rate: 1,
            negative_selection_range: coarse_negative_range(realized_rows),
            negative_sample_scale: None,
            adaptation_samples: 128,
            exclude_graph_neighbors: false,
            good_initialization: level.good_initialization,
            seed,
        }
    }
}

/// Explicit operation identity: equal fit and transform row counts are not equivalent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutOperation {
    Fit,
    Transform,
}

/// Schedule values before the single conversion to launch precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule64 {
    pub alpha: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
}

/// Per-epoch launch parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schedule32 {
    pub alpha: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl Schedule64 {
    pub fn to_f32(self) -> Schedule32 {
        Schedule32 {
            alpha: self.alpha as f32,
            momentum: self.momentum as f32,
            beta1: self.beta1 as f32,
            beta2: self.beta2 as f32,
        }
    }
}

/// Construct upstream's Adam schedule in f64, including the empty schedule.
pub fn adam_schedule(
    epochs: usize,
    initial_alpha: f64,
    good_initialization: bool,
) -> impl ExactSizeIterator<Item = Schedule64> {
    let warmup = if good_initialization {
        (epochs / 8).min(100)
    } else {
        (epochs / 2).min(100)
    };
    (0..epochs).map(move |n| {
        let alpha = if n < warmup {
            let k = if good_initialization { 0.5 } else { 2.0 };
            (k * initial_alpha - 0.1) * (1.0 - n as f64 / warmup as f64).powi(2) + 0.1
        } else {
            0.15 * (1.0 - (n - warmup) as f64 / (epochs - warmup) as f64) + 0.05
        };
        Schedule64 {
            alpha,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.99,
        }
    })
}

/// Construct upstream's momentum schedule in f64.
pub fn momentum_schedule(
    epochs: usize,
    initial_alpha: f64,
    good_initialization: bool,
) -> impl ExactSizeIterator<Item = Schedule64> {
    let warmup = epochs.min(200.max(epochs / 8));
    (0..epochs).map(move |n| {
        let r = if good_initialization {
            if n < warmup {
                (1.0 - n as f64 / warmup as f64).powi(2)
            } else {
                0.0
            }
        } else {
            0.25 * (1.0 - n as f64 / epochs as f64).powi(2)
        };
        let alpha = if good_initialization {
            0.25 * initial_alpha * r + 0.005
        } else {
            initial_alpha * r
        };
        Schedule64 {
            alpha,
            momentum: 0.5 * (1.0 - r),
            beta1: 0.0,
            beta2: 0.0,
        }
    })
}

/// A post-pruning COO entry. Zero-weight entries are ignored even if their
/// column is a padding sentinel. Duplicate positive entries are summed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedEdge {
    pub row: usize,
    pub column: usize,
    pub weight: f32,
}

/// CSR arrays and clocks are aligned at each compact edge position.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactGraph {
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
    pub periods: Vec<f64>,
    pub negative_periods: Vec<f64>,
    pub next_positive: Vec<f64>,
    pub next_negative: Vec<f64>,
}

fn edge_periods(
    weight: f32,
    max_weight: f32,
    epochs: usize,
    negative_sample_rate: f32,
) -> Result<(f64, f64)> {
    let samples = epochs as f32 * (weight / max_weight);
    let period = epochs as f64 / f64::from(samples);
    let negative_period = if negative_sample_rate == 0.0 {
        f64::INFINITY
    } else {
        (period / f64::from(negative_sample_rate)) * 1.5
    };
    if !period.is_finite()
        || period <= 0.0
        || (negative_sample_rate > 0.0 && (!negative_period.is_finite() || negative_period <= 0.0))
    {
        return Err(UmapError::Numerical("invalid layout period".into()));
    }
    Ok((period, negative_period))
}

/// Build layout clocks directly from a canonical square host CSR.
pub fn compact_graph_from_csr(
    csr: &HostCsr,
    epochs: usize,
    negative_sample_rate: f32,
) -> Result<CompactGraph> {
    let offset_len = csr
        .n
        .checked_add(1)
        .ok_or(UmapError::Overflow("CSR offsets"))?;
    u32::try_from(csr.n).map_err(|_| UmapError::Overflow("CSR rows"))?;
    u32::try_from(csr.indices.len()).map_err(|_| UmapError::Overflow("CSR edges"))?;
    if epochs == 0 || !negative_sample_rate.is_finite() || negative_sample_rate < 0.0 {
        return Err(UmapError::InvalidConfig(
            "epochs must be positive and negative sample rate finite and nonnegative".into(),
        ));
    }
    if csr.indptr.len() != offset_len
        || csr.indptr.first() != Some(&0)
        || csr.indptr.last() != Some(&csr.indices.len())
        || csr.indices.len() != csr.weights.len()
        || csr.indices.is_empty()
    {
        return Err(UmapError::InvalidInput("invalid or empty host CSR".into()));
    }
    for row in 0..csr.n {
        let (start, end) = (csr.indptr[row], csr.indptr[row + 1]);
        if start > end
            || end > csr.indices.len()
            || csr.indices[start..end]
                .iter()
                .any(|&column| column >= csr.n)
            || !csr.indices[start..end].windows(2).all(|w| w[0] < w[1])
        {
            return Err(UmapError::InvalidInput("invalid host CSR row".into()));
        }
    }
    if csr.weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return Err(UmapError::InvalidInput(
            "edge weights must be finite and positive".into(),
        ));
    }
    let max_weight = csr.weights.iter().copied().fold(0.0_f32, f32::max);
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(offset_len)
        .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    for &offset in &csr.indptr {
        offsets.push(u32::try_from(offset).map_err(|_| UmapError::Overflow("CSR edges"))?);
    }
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut periods = Vec::new();
    let mut negative_periods = Vec::new();
    let mut next_positive = Vec::new();
    let mut next_negative = Vec::new();
    indices
        .try_reserve_exact(csr.indices.len())
        .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    weights
        .try_reserve_exact(csr.weights.len())
        .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    for v in [
        &mut periods,
        &mut negative_periods,
        &mut next_positive,
        &mut next_negative,
    ] {
        v.try_reserve_exact(csr.indices.len())
            .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    }
    for (&index, &weight) in csr.indices.iter().zip(&csr.weights) {
        let (period, negative_period) =
            edge_periods(weight, max_weight, epochs, negative_sample_rate)?;
        indices.push(index as u32);
        weights.push(weight);
        periods.push(period);
        negative_periods.push(negative_period);
        next_positive.push(period);
        next_negative.push(negative_period);
    }
    Ok(CompactGraph {
        offsets,
        indices,
        weights,
        periods,
        negative_periods,
        next_positive,
        next_negative,
    })
}

/// Build compact CSR from post-prune entries; duplicates follow CSR sum-duplicates semantics.
pub fn compact_graph(
    rows: usize,
    columns: usize,
    edges: &[WeightedEdge],
    epochs: usize,
    negative_sample_rate: f32,
    operation: LayoutOperation,
) -> Result<CompactGraph> {
    let offset_len = rows
        .checked_add(1)
        .ok_or(UmapError::Overflow("CSR offsets"))?;
    u32::try_from(rows).map_err(|_| UmapError::Overflow("CSR rows"))?;
    u32::try_from(columns).map_err(|_| UmapError::Overflow("CSR columns"))?;
    u32::try_from(edges.len()).map_err(|_| UmapError::Overflow("CSR edges"))?;
    if !negative_sample_rate.is_finite() || negative_sample_rate < 0.0 || epochs == 0 {
        return Err(UmapError::InvalidConfig(
            "epochs must be positive and negative sample rate finite and nonnegative".into(),
        ));
    }
    let mut sorted = Vec::new();
    sorted
        .try_reserve_exact(edges.len())
        .map_err(|_| UmapError::Memory("CSR scratch allocation failed".into()))?;
    for &edge in edges {
        if !edge.weight.is_finite() || edge.weight < 0.0 {
            return Err(UmapError::InvalidInput(
                "edge weights must be finite and nonnegative".into(),
            ));
        }
        if edge.weight == 0.0 {
            continue;
        }
        if edge.row >= rows || edge.column >= columns {
            return Err(UmapError::InvalidInput(
                "edge index outside CSR shape".into(),
            ));
        }
        sorted.push(edge);
    }
    sorted.sort_unstable_by_key(|e| (e.row, e.column));
    let mut merged: Vec<WeightedEdge> = Vec::new();
    merged
        .try_reserve_exact(sorted.len())
        .map_err(|_| UmapError::Memory("CSR scratch allocation failed".into()))?;
    for edge in sorted {
        if let Some(last) = merged.last_mut()
            && (last.row, last.column) == (edge.row, edge.column)
        {
            last.weight += edge.weight;
            if !last.weight.is_finite() {
                return Err(UmapError::Numerical(
                    "duplicate edge weight overflow".into(),
                ));
            }
            continue;
        }
        merged.push(edge);
    }
    if merged.is_empty() {
        let message = match operation {
            LayoutOperation::Fit => "fuzzy graph is empty after positive-edge pruning",
            LayoutOperation::Transform => {
                "transform fuzzy graph is empty after positive-edge pruning"
            }
        };
        return Err(UmapError::InvalidInput(message.into()));
    }
    let max_weight = merged
        .iter()
        .map(|edge| edge.weight)
        .fold(0.0_f32, f32::max);
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(offset_len)
        .map_err(|_| UmapError::Memory("CSR offsets allocation failed".into()))?;
    offsets.resize(offset_len, 0u32);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut periods = Vec::new();
    let mut negative_periods = Vec::new();
    indices
        .try_reserve_exact(merged.len())
        .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    weights
        .try_reserve_exact(merged.len())
        .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    let mut next_positive = Vec::new();
    let mut next_negative = Vec::new();
    for v in [
        &mut periods,
        &mut negative_periods,
        &mut next_positive,
        &mut next_negative,
    ] {
        v.try_reserve_exact(merged.len())
            .map_err(|_| UmapError::Memory("CSR allocation failed".into()))?;
    }
    for edge in merged {
        offsets[edge.row + 1] += 1;
        let (period, negative_period) =
            edge_periods(edge.weight, max_weight, epochs, negative_sample_rate)?;
        indices.push(edge.column as u32);
        weights.push(edge.weight);
        periods.push(period);
        negative_periods.push(negative_period);
        next_positive.push(period);
        next_negative.push(negative_period);
    }
    for row in 0..rows {
        offsets[row + 1] = offsets[row + 1]
            .checked_add(offsets[row])
            .ok_or(UmapError::Overflow("CSR edges"))?;
    }
    debug_assert!((0..rows).all(|r| {
        indices[offsets[r] as usize..offsets[r + 1] as usize]
            .windows(2)
            .all(|w| w[0] < w[1])
    }));
    Ok(CompactGraph {
        offsets,
        indices,
        weights,
        next_positive,
        next_negative,
        periods,
        negative_periods,
    })
}

/// Global negative order index, without overflowing even at usize limits.
pub fn negative_index(
    edge: usize,
    epoch: usize,
    sample: usize,
    reference_count: usize,
) -> Result<usize> {
    if reference_count == 0 {
        return Err(UmapError::InvalidInput(
            "reference count must be positive".into(),
        ));
    }
    let m = reference_count as u128;
    let factor = ((epoch as u128 % m) + (sample as u128 % m) + 1) % m;
    Ok(((edge as u128 % m) * factor % m) as usize)
}

/// Effective gradient window; adaptation deliberately uses the raw range instead.
pub fn effective_negative_range(raw_range: usize, reference_count: usize) -> Result<usize> {
    if raw_range == 0 || reference_count == 0 {
        return Err(UmapError::InvalidInput(
            "negative range and reference count must be positive".into(),
        ));
    }
    Ok(reference_count.min(raw_range.max(200)))
}

/// Select a rank in a centered, endpoint-clamped negative window.
/// `range` may be either the effective gradient range or raw adaptation range.
pub fn window_negative_index(
    edge: usize,
    epoch: usize,
    sample: usize,
    source_rank: usize,
    reference_count: usize,
    range: usize,
) -> Result<usize> {
    if range == 0 || range > reference_count || source_rank >= reference_count {
        return Err(UmapError::InvalidInput(
            "negative window requires a valid rank and nonempty bounded range".into(),
        ));
    }
    let start = source_rank
        .saturating_sub(range / 2)
        .min(reference_count - range);
    Ok(start + negative_index(edge, epoch, sample, range)?)
}

/// Small-layout retry rank when graph-neighbor exclusion is enabled.
pub fn small_retry_index(
    edge: usize,
    epoch: usize,
    attempt: usize,
    reference_count: usize,
) -> Result<usize> {
    let base = negative_index(edge, epoch, 0, reference_count)?;
    Ok(((base as u128 + attempt as u128) % reference_count as u128) as usize)
}

/// Bound the small-layout rejection loop, including a fully excluded row.
pub fn small_retry_limit(requested: usize, reference_count: usize) -> Result<usize> {
    if reference_count == 0 {
        return Err(UmapError::InvalidInput(
            "reference count must be positive".into(),
        ));
    }
    requested
        .max(1)
        .checked_mul(reference_count)
        .ok_or(UmapError::Overflow("small negative retry limit"))
}

/// Search an already sorted compact CSR row without inferring transform self edges.
pub fn negative_candidate_excluded(
    operation: LayoutOperation,
    source: usize,
    candidate: usize,
    sorted_neighbors: &[u32],
) -> bool {
    (operation == LayoutOperation::Fit && source == candidate)
        || u32::try_from(candidate)
            .is_ok_and(|candidate| sorted_neighbors.binary_search(&candidate).is_ok())
}

/// Adapt before the gradient only for Adam fit, using the raw range trigger.
#[allow(clippy::too_many_arguments)]
pub fn adaptation_due(
    operation: LayoutOperation,
    adam: bool,
    samples: usize,
    raw_range: usize,
    reference_count: usize,
    epoch: usize,
    total_epochs: usize,
) -> bool {
    operation == LayoutOperation::Fit
        && adam
        && samples > 0
        && raw_range < reference_count
        && epoch >= 10
        && epoch <= total_epochs / 2
        && epoch.is_multiple_of(10)
}

/// Resolve the upstream range correction before selecting a kernel.
pub fn resolved_negative_scale(
    raw_range: usize,
    reference_count: usize,
    explicit: Option<f32>,
    adaptation_enabled: bool,
) -> Result<f32> {
    if reference_count == 0 {
        return Err(UmapError::InvalidInput(
            "reference count must be positive".into(),
        ));
    }
    if let Some(scale) = explicit {
        if !scale.is_finite() || scale < 0.0 {
            return Err(UmapError::InvalidConfig(
                "negative sample scale must be finite and nonnegative".into(),
            ));
        }
        return Ok(scale);
    }
    if adaptation_enabled {
        Ok(((raw_range.min(reference_count) as f64 / reference_count as f64).powf(0.75)) as f32)
    } else {
        Ok(effective_negative_range(raw_range, reference_count)? as f32 / reference_count as f32)
    }
}

/// Winsorized global/local resultant-force ratio with NumPy's linear quantiles.
/// Zero resultants are omitted; a sample set with no valid pairs has ratio one.
pub fn negative_force_ratio(global: &[f64], local: &[f64]) -> Result<f64> {
    if global.len() != local.len() {
        return Err(UmapError::InvalidInput(
            "force sample lengths differ".into(),
        ));
    }
    let mut logs = Vec::new();
    logs.try_reserve_exact(global.len())
        .map_err(|error| UmapError::Memory(format!("force ratio samples: {error}")))?;
    for (&global, &local) in global.iter().zip(local) {
        if !global.is_finite() || !local.is_finite() || global < 0.0 || local < 0.0 {
            return Err(UmapError::Numerical("invalid force resultant".into()));
        }
        if global > 0.0 && local > 0.0 {
            logs.push(global.ln() - local.ln());
        }
    }
    if logs.is_empty() {
        return Ok(1.0);
    }
    logs.sort_unstable_by(f64::total_cmp);
    let quantile = |q: f64| {
        let position = (logs.len() - 1) as f64 * q;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        logs[lower] + (logs[upper] - logs[lower]) * (position - lower as f64)
    };
    let low = quantile(0.1);
    let high = quantile(0.9);
    let ratio =
        (logs.iter().map(|value| value.clamp(low, high)).sum::<f64>() / logs.len() as f64).exp();
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(UmapError::Numerical("invalid negative force ratio".into()));
    }
    Ok(ratio)
}

/// Stateful anchored update used only by eligible Adam fit epochs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeScaleAdaptation {
    prior: f64,
    current: f64,
    anchor: Option<f64>,
}

impl NegativeScaleAdaptation {
    pub fn new(scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(UmapError::InvalidConfig(
                "negative sample scale must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            prior: scale as f64,
            current: scale as f64,
            anchor: None,
        })
    }

    pub fn scale(&self) -> f32 {
        self.current as f32
    }

    /// The first ratio establishes the anchor without changing the scale.
    /// Explicit zero is absorbing, including at the first adaptation call.
    pub fn observe_ratio(&mut self, ratio: f64) -> Result<f32> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(UmapError::Numerical("invalid negative force ratio".into()));
        }
        if self.prior == 0.0 {
            self.anchor = Some(0.0);
            return Ok(0.0);
        }
        if let Some(anchor) = self.anchor {
            let target = (anchor * ratio).clamp(0.5 * self.prior, 2.0 * self.prior);
            let change = (target / self.current)
                .ln()
                .clamp(0.9_f64.ln(), 1.1_f64.ln());
            self.current *= (0.1 * change).exp();
        } else {
            let anchor = self.current / ratio;
            if !anchor.is_finite() || anchor <= 0.0 {
                return Err(UmapError::Numerical("invalid negative force anchor".into()));
            }
            self.anchor = Some(anchor);
        }
        Ok(self.scale())
    }
}

/// Small-layout force semantics depend on the source row count, including transforms.
pub fn is_small_layout(source_rows: usize) -> bool {
    source_rows <= 1024
}

/// Advance a visited edge's clocks and return the number of negative events.
/// A non-firing positive edge does not advance either clock.
pub fn advance_clock(graph: &mut CompactGraph, edge: usize, epoch: usize) -> Result<Option<usize>> {
    if edge >= graph.periods.len() {
        return Err(UmapError::InvalidInput("edge index outside CSR".into()));
    }
    let now = epoch as f64;
    if graph.next_positive[edge] > now {
        return Ok(None);
    }
    let neg_period = graph.negative_periods[edge];
    if neg_period == f64::INFINITY {
        let next_pos = graph.next_positive[edge] + graph.periods[edge];
        if !next_pos.is_finite() {
            return Err(UmapError::Numerical("layout clock overflow".into()));
        }
        graph.next_positive[edge] = next_pos;
        return Ok(Some(0));
    }
    let k = ((now - graph.next_negative[edge]) / neg_period).trunc();
    if !k.is_finite() || k >= usize::MAX as f64 {
        return Err(UmapError::Overflow("negative event count"));
    }
    let next_pos = graph.next_positive[edge] + graph.periods[edge];
    let next_neg = graph.next_negative[edge] + k * neg_period;
    if !next_pos.is_finite() || !next_neg.is_finite() {
        return Err(UmapError::Numerical("layout clock overflow".into()));
    }
    graph.next_positive[edge] = next_pos;
    graph.next_negative[edge] = next_neg;
    Ok(Some(if k > 0.0 { k as usize } else { 0 }))
}

/// Add a source-only attraction or soft-clipped repulsion to `gradient`.
/// Small layouts halve attraction and ignore explicit negative scale.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_force(
    gradient: &mut [f32],
    source: &[f32],
    target: &[f32],
    a: f32,
    b: f32,
    gamma: f32,
    negative_scale: f32,
    small: bool,
    negative: bool,
) -> Result<()> {
    if gradient.len() != source.len() || source.len() != target.len() {
        return Err(UmapError::InvalidInput(
            "gradient dimension mismatch".into(),
        ));
    }
    let q: f32 = source
        .iter()
        .zip(target)
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    if q <= 0.0 {
        return Ok(());
    }
    let coefficient = if negative {
        if gamma <= 0.0 {
            return Ok(());
        }
        let scale = if small { 1.0 } else { negative_scale };
        let c = scale * 2.0 * gamma * b / ((0.001 + q) * (1.0 + a * q.powf(b)));
        if c <= 0.0 {
            return Ok(());
        }
        let norm = (c * c * q).sqrt();
        if norm == 0.0 {
            return Ok(());
        }
        c * gamma.min(16.0) * (norm / gamma).tanh() / norm
    } else {
        let c = -2.0 * a * b * q.powf(b - 1.0) / (1.0 + a * q.powf(b));
        c * if small { 0.5 } else { 1.0 }
    };
    for ((g, x), y) in gradient.iter_mut().zip(source).zip(target) {
        *g += coefficient * (x - y);
    }
    Ok(())
}

/// Apply one lazy Adam coordinate step and clear its gradient.
pub fn adam_step(
    position: &mut f32,
    gradient: &mut f32,
    first: &mut f32,
    second: &mut f32,
    epoch: usize,
    schedule: Schedule32,
) {
    let g = *gradient;
    if g != 0.0 {
        *first = schedule.beta1 * *first + (1.0 - schedule.beta1) * g;
        *second = schedule.beta2 * *second + (1.0 - schedule.beta2) * g * g;
        let n = epoch as f64 + 1.0;
        let m = *first / (1.0 - (schedule.beta1 as f64).powf(n)) as f32;
        let v = *second / (1.0 - (schedule.beta2 as f64).powf(n)) as f32;
        *position += schedule.alpha * m / (v.sqrt() + 1e-4);
    }
    *gradient = 0.0;
}

/// Number of source positions per sequential momentum block.
pub fn momentum_block_size(rows: usize) -> usize {
    if rows <= 1024 {
        rows.max(1)
    } else {
        (rows / 2).max(1)
    }
}

/// Accumulate an alpha-scaled gradient into persistent momentum velocity.
pub fn momentum_accumulate(velocity: &mut f32, gradient: f32, alpha: f32) {
    *velocity += alpha * gradient;
}
/// Apply a block's velocity before processing the next block.
pub fn momentum_apply(position: &mut f32, velocity: f32) {
    *position += velocity;
}
/// Decay the complete velocity buffer once after all blocks.
pub fn momentum_finalize(velocity: &mut [f32], momentum: f32) {
    for value in velocity {
        *value *= momentum;
    }
}

/// Address space of an individually allocated live buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferSpace {
    Host,
    Device,
}

/// A buffer's explicitly declared lifetime. Caller-owned buffers must use the
/// real lifetime, not an assumed early release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLifetime {
    Build,
    Layout,
    Both,
    /// Equally sized independent allocations present during construction.
    RepeatedBuild(usize),
    /// Equally sized independent allocations present in both phases.
    RepeatedBoth(usize),
    /// Equally sized independent allocations present during layout.
    RepeatedLayout(usize),
    /// Future caller result allocations present in the layout phase.
    FutureResult(usize),
}

/// A single allocation, including caller-owned retained inputs or transfer staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveBuffer {
    pub name: &'static str,
    pub space: BufferSpace,
    pub lifetime: BufferLifetime,
    pub bytes: usize,
}

/// Dimensions and ownership choices for an individual modern operation.
/// `additional` lists actual retained input, unique mapping, query/search,
/// initialization, backend staging and materialization allocations not owned
/// by this helper, separately per allocation and with their real lifetimes.
#[derive(Clone, Copy, Debug)]
pub struct ModernPlanInput<'a> {
    pub operation: LayoutOperation,
    pub source_rows: usize,
    pub reference_rows: usize,
    pub dimensions: usize,
    /// Actual backend allocation sizes for the source and fixed reference matrices.
    /// These include any backend-specific row pitch, but not allocator rounding.
    pub source_embedding_bytes: usize,
    pub reference_embedding_bytes: usize,
    pub features: usize,
    pub original_rows: usize,
    pub graph_capacity: usize,
    pub edges: usize,
    pub adam: bool,
    pub alias_orders: bool,
    pub additional: &'a [LiveBuffer],
}

/// Live byte totals by phase and address space, with independent maxima.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModernPeaks {
    pub build_host: usize,
    pub build_device: usize,
    pub layout_host: usize,
    pub layout_device: usize,
}

impl ModernPeaks {
    /// Admit future concurrent allocations against free-memory budgets.
    /// Already-owned storage must be excluded by the caller before summing peaks.
    /// CPU host and runtime allocations share one address-space budget.
    pub fn admit_future(self, host_budget: u64, device_budget: Option<u64>) -> Result<()> {
        for (phase, host, device) in [
            ("build", self.build_host, self.build_device),
            ("layout", self.layout_host, self.layout_device),
        ] {
            if let Some(device_budget) = device_budget {
                if host as u128 > host_budget as u128 || device as u128 > device_budget as u128 {
                    return Err(UmapError::Memory(format!(
                        "modern {phase} requires {host} host and {device} device bytes; available {host_budget} host and {device_budget} device bytes"
                    )));
                }
            } else {
                let total = host
                    .checked_add(device)
                    .ok_or(UmapError::Overflow("modern unified memory"))?;
                if total as u128 > host_budget as u128 {
                    return Err(UmapError::Memory(format!(
                        "modern {phase} requires {total} unified bytes; available {host_budget} bytes"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn future_peaks_use_combined_cpu_and_separate_cuda_budgets() {
        let peaks = ModernPeaks {
            build_host: 20,
            build_device: 30,
            layout_host: 45,
            layout_device: 60,
        };
        assert!(peaks.admit_future(104, None).is_err());
        assert!(peaks.admit_future(105, None).is_ok());
        assert!(peaks.admit_future(44, Some(60)).is_err());
        assert!(peaks.admit_future(45, Some(59)).is_err());
        assert!(peaks.admit_future(45, Some(60)).is_ok());
    }

    #[test]
    fn phases_are_alternatives_and_unified_overflow_is_checked() {
        let peaks = ModernPeaks {
            build_host: 80,
            build_device: 0,
            layout_host: 0,
            layout_device: 80,
        };
        assert!(peaks.admit_future(80, None).is_ok());
        let overflow = ModernPeaks {
            layout_host: usize::MAX,
            layout_device: 1,
            ..ModernPeaks::default()
        };
        assert!(matches!(
            overflow.admit_future(u64::MAX, None),
            Err(UmapError::Overflow(_))
        ));
    }
}

/// Fixed-capacity planner: no heap allocation is needed to build the table.
#[derive(Clone, Debug)]
pub struct ModernBufferPlan {
    buffers: [Option<LiveBuffer>; 192],
    len: usize,
    pub peaks: ModernPeaks,
}

impl ModernBufferPlan {
    pub fn buffers(&self) -> impl ExactSizeIterator<Item = &LiveBuffer> {
        self.buffers[..self.len]
            .iter()
            .map(|entry| entry.as_ref().expect("occupied plan entry"))
    }

    /// Concurrent physical peaks, rounding each individual allocation separately.
    pub fn aligned_peaks(
        &self,
        host_alignment: usize,
        device_alignment: usize,
    ) -> Result<ModernPeaks> {
        aligned_buffer_peaks(self.buffers(), host_alignment, device_alignment)
    }

    /// Check each allocation separately. Alignment is the allocator's actual
    /// allocation-size granularity, not a blanket whole-operation budget.
    /// The callback permits caller-specific allocation checks and failure
    /// injection; no allocation is attempted by this method.
    pub fn validate_allocations<F>(
        &self,
        host_alignment: usize,
        device_alignment: usize,
        host_limit: usize,
        device_limit: usize,
        mut check: F,
    ) -> Result<()>
    where
        F: FnMut(&LiveBuffer, usize) -> Result<()>,
    {
        for buffer in self.buffers() {
            let (alignment, limit) = match buffer.space {
                BufferSpace::Host => (host_alignment, host_limit),
                BufferSpace::Device => (device_alignment, device_limit),
            };
            if alignment == 0 {
                return Err(UmapError::InvalidConfig(
                    "allocation alignment must be positive".into(),
                ));
            }
            let rem = buffer.bytes % alignment;
            let aligned = if rem == 0 {
                buffer.bytes
            } else {
                buffer
                    .bytes
                    .checked_add(alignment - rem)
                    .ok_or(UmapError::Overflow("modern allocation alignment"))?
            };
            if aligned > limit {
                return Err(UmapError::Capacity(format!(
                    "{} allocation requires {} bytes, limit {}",
                    buffer.name, aligned, limit
                )));
            }
            check(buffer, aligned)?;
        }
        Ok(())
    }
}

/// Sum selected allocations with the same checked alignment as the full plan.
pub fn aligned_buffer_peaks<'a>(
    buffers: impl IntoIterator<Item = &'a LiveBuffer>,
    host_alignment: usize,
    device_alignment: usize,
) -> Result<ModernPeaks> {
    let mut peaks = ModernPeaks::default();
    for buffer in buffers {
        let alignment = match buffer.space {
            BufferSpace::Host => host_alignment,
            BufferSpace::Device => device_alignment,
        };
        if alignment == 0 {
            return Err(UmapError::InvalidConfig(
                "allocation alignment must be positive".into(),
            ));
        }
        let bytes = buffer
            .bytes
            .checked_add(alignment - 1)
            .map(|bytes| bytes / alignment * alignment)
            .ok_or(UmapError::Overflow("modern allocation alignment"))?;
        let count = match buffer.lifetime {
            BufferLifetime::RepeatedBoth(count)
            | BufferLifetime::RepeatedBuild(count)
            | BufferLifetime::RepeatedLayout(count)
            | BufferLifetime::FutureResult(count) => count,
            _ => 1,
        };
        let bytes = bytes
            .checked_mul(count)
            .ok_or(UmapError::Overflow("modern aligned peak"))?;
        let (build, layout) = match buffer.space {
            BufferSpace::Host => (&mut peaks.build_host, &mut peaks.layout_host),
            BufferSpace::Device => (&mut peaks.build_device, &mut peaks.layout_device),
        };
        let (live_build, live_layout) = match buffer.lifetime {
            BufferLifetime::Build | BufferLifetime::RepeatedBuild(_) => (true, false),
            BufferLifetime::Layout
            | BufferLifetime::RepeatedLayout(_)
            | BufferLifetime::FutureResult(_) => (false, true),
            BufferLifetime::Both | BufferLifetime::RepeatedBoth(_) => (true, true),
        };
        if live_build {
            *build = build
                .checked_add(bytes)
                .ok_or(UmapError::Overflow("modern aligned build peak"))?;
        }
        if live_layout {
            *layout = layout
                .checked_add(bytes)
                .ok_or(UmapError::Overflow("modern aligned layout peak"))?;
        }
    }
    Ok(peaks)
}

/// Construct a named table with explicit build/layout overlap. Graph staging
/// and compact CSR host arrays are conservatively retained through layout;
/// only the temporary host sort copies are released after construction.
/// Device padded graph stays live through layout unless the caller proves and
/// implements an earlier release; no ANN memory limit is applied here.
pub fn modern_buffer_plan(input: ModernPlanInput<'_>) -> Result<ModernBufferPlan> {
    let h = input.source_rows;
    let n = input.reference_rows;
    let d = input.dimensions;
    if h == 0 || n == 0 || d == 0 || input.features == 0 || input.original_rows == 0 {
        return Err(UmapError::InvalidInput(
            "modern layout dimensions must be positive".into(),
        ));
    }
    if input.edges > input.graph_capacity {
        return Err(UmapError::InvalidInput(
            "compact edges exceed graph capacity".into(),
        ));
    }
    if input.operation == LayoutOperation::Fit && h != n {
        return Err(UmapError::InvalidInput("modern fit requires H=N".into()));
    }
    if input.alias_orders && (input.operation != LayoutOperation::Fit || input.adam || h != n) {
        return Err(UmapError::InvalidInput(
            "only fit momentum may alias orders".into(),
        ));
    }
    for value in [
        h,
        n,
        d,
        input.features,
        input.original_rows,
        input.graph_capacity,
        input.edges,
    ] {
        i32::try_from(value).map_err(|_| UmapError::Overflow("modern layout i32 index"))?;
    }
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or(UmapError::Overflow("modern layout buffer bytes"))
    };
    let logical_embedding = mul(mul(h, d)?, 4)?;
    let logical_reference = mul(mul(n, d)?, 4)?;
    let embedding = input.source_embedding_bytes;
    let reference = input.reference_embedding_bytes;
    if embedding < logical_embedding || reference < logical_reference {
        return Err(UmapError::InvalidInput(
            "physical embedding allocation is smaller than its logical shape".into(),
        ));
    }
    // Validate feature and unique shapes even when caller owns the buffers.
    mul(mul(input.original_rows, input.features)?, 4)?;
    let cap = input.graph_capacity;
    let e = input.edges;
    let offsets = mul(
        h.checked_add(1)
            .ok_or(UmapError::Overflow("modern CSR offsets"))?,
        4,
    )?;
    let mut plan = ModernBufferPlan {
        buffers: [None; 192],
        len: 0,
        peaks: ModernPeaks::default(),
    };
    let mut add = |name, space, lifetime, bytes: usize| -> Result<()> {
        if plan.len == plan.buffers.len() {
            return Err(UmapError::Capacity(
                "modern live-buffer table is full".into(),
            ));
        }
        let b = LiveBuffer {
            name,
            space,
            lifetime,
            bytes,
        };
        let total_bytes = match lifetime {
            BufferLifetime::RepeatedBoth(count)
            | BufferLifetime::RepeatedBuild(count)
            | BufferLifetime::RepeatedLayout(count)
            | BufferLifetime::FutureResult(count) => bytes
                .checked_mul(count)
                .ok_or(UmapError::Overflow("modern repeated buffers"))?,
            _ => bytes,
        };
        let add_peak = |total: &mut usize| -> Result<()> {
            *total = total
                .checked_add(total_bytes)
                .ok_or(UmapError::Overflow("modern live-buffer total"))?;
            Ok(())
        };
        match (space, lifetime) {
            (BufferSpace::Host, BufferLifetime::Build) => add_peak(&mut plan.peaks.build_host)?,
            (BufferSpace::Host, BufferLifetime::Layout) => add_peak(&mut plan.peaks.layout_host)?,
            (BufferSpace::Host, BufferLifetime::Both) => {
                add_peak(&mut plan.peaks.build_host)?;
                add_peak(&mut plan.peaks.layout_host)?;
            }
            (BufferSpace::Device, BufferLifetime::Build) => add_peak(&mut plan.peaks.build_device)?,
            (BufferSpace::Device, BufferLifetime::Layout) => {
                add_peak(&mut plan.peaks.layout_device)?
            }
            (BufferSpace::Device, BufferLifetime::Both) => {
                add_peak(&mut plan.peaks.build_device)?;
                add_peak(&mut plan.peaks.layout_device)?;
            }
            (BufferSpace::Host, BufferLifetime::RepeatedBuild(_)) => {
                add_peak(&mut plan.peaks.build_host)?
            }
            (BufferSpace::Device, BufferLifetime::RepeatedBuild(_)) => {
                add_peak(&mut plan.peaks.build_device)?
            }
            (BufferSpace::Host, BufferLifetime::RepeatedBoth(_)) => {
                add_peak(&mut plan.peaks.build_host)?;
                add_peak(&mut plan.peaks.layout_host)?;
            }
            (BufferSpace::Device, BufferLifetime::RepeatedBoth(_)) => {
                add_peak(&mut plan.peaks.build_device)?;
                add_peak(&mut plan.peaks.layout_device)?;
            }
            (BufferSpace::Host, BufferLifetime::FutureResult(_)) => {
                add_peak(&mut plan.peaks.layout_host)?
            }
            (BufferSpace::Device, BufferLifetime::FutureResult(_)) => {
                add_peak(&mut plan.peaks.layout_device)?
            }
            (BufferSpace::Host, BufferLifetime::RepeatedLayout(_)) => {
                add_peak(&mut plan.peaks.layout_host)?
            }
            (BufferSpace::Device, BufferLifetime::RepeatedLayout(_)) => {
                add_peak(&mut plan.peaks.layout_device)?
            }
        }
        plan.buffers[plan.len] = Some(b);
        plan.len += 1;
        Ok(())
    };
    use BufferLifetime::{Both, Build, Layout};
    use BufferSpace::{Device, Host};
    add("source embedding", Device, Both, embedding)?;
    if input.operation == LayoutOperation::Transform {
        add("fixed reference embedding", Device, Both, reference)?;
    }
    for name in ["padded heads", "padded tails", "padded weights"] {
        add(name, Device, Both, mul(cap, 4)?)?;
    }
    for name in ["sorted COO scratch", "merged COO scratch"] {
        add(
            name,
            Host,
            Build,
            mul(cap, std::mem::size_of::<WeightedEdge>())?,
        )?;
    }
    for space in [Host, Device] {
        add("CSR offsets", space, Both, offsets)?;
        add("CSR indices", space, Both, mul(e, 4)?)?;
        add("CSR weights", space, Both, mul(e, 4)?)?;
        for name in [
            "positive periods",
            "negative periods",
            "positive clocks",
            "negative clocks",
        ] {
            add(name, space, Both, mul(e, 8)?)?;
        }
    }
    add("updates or velocity", Device, Layout, embedding)?;
    if input.adam {
        add("Adam first moment", Device, Layout, embedding)?;
        add("Adam second moment", Device, Layout, embedding)?;
    }
    for (name, count) in [
        ("source order", h),
        ("negative order", if input.alias_orders { 0 } else { n }),
    ] {
        if count == 0 {
            continue;
        }
        for space in [Host, Device] {
            for _ in 0..2 {
                add(name, space, Layout, mul(count, 4)?)?;
            }
        }
    }
    for extra in input.additional {
        add(extra.name, extra.space, extra.lifetime, extra.bytes)?;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_csr_matches_coo_clocks() {
        let csr = HostCsr {
            n: 3,
            indptr: vec![0, 2, 3, 4],
            indices: vec![1, 2, 0, 1],
            weights: vec![0.25, 1.0, 0.5, 0.125],
        };
        let edges = [
            WeightedEdge {
                row: 0,
                column: 1,
                weight: 0.25,
            },
            WeightedEdge {
                row: 0,
                column: 2,
                weight: 1.0,
            },
            WeightedEdge {
                row: 1,
                column: 0,
                weight: 0.5,
            },
            WeightedEdge {
                row: 2,
                column: 1,
                weight: 0.125,
            },
        ];
        for rate in [0.0, 1.0, 5.0] {
            assert_eq!(
                compact_graph_from_csr(&csr, 37, rate).unwrap(),
                compact_graph(3, 3, &edges, 37, rate, LayoutOperation::Fit).unwrap()
            );
        }
        let mut bad = csr.clone();
        bad.indices[1] = 1;
        assert!(compact_graph_from_csr(&bad, 37, 1.0).is_err());
        bad = csr;
        bad.indptr[1] = 5;
        assert!(compact_graph_from_csr(&bad, 37, 1.0).is_err());
    }

    #[test]
    fn exclusion_retries_are_bounded_and_operation_aware() {
        assert!(negative_candidate_excluded(LayoutOperation::Fit, 3, 3, &[]));
        assert!(!negative_candidate_excluded(
            LayoutOperation::Transform,
            3,
            3,
            &[]
        ));
        for operation in [LayoutOperation::Fit, LayoutOperation::Transform] {
            assert!(negative_candidate_excluded(operation, 3, 9, &[1, 5, 9]));
            assert!(!negative_candidate_excluded(operation, 3, 8, &[1, 5, 9]));
        }
        let ranks = (0..5)
            .map(|attempt| small_retry_index(2, 1, attempt, 5).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranks, [4, 0, 1, 2, 3]);
        assert_eq!(small_retry_limit(0, 5).unwrap(), 5);
        assert_eq!(small_retry_limit(3, 5).unwrap(), 15);
        assert!(small_retry_limit(usize::MAX, 2).is_err());
        assert!(small_retry_index(0, 0, 0, 0).is_err());
        assert!(small_retry_limit(1, 0).is_err());
        assert!(small_retry_index(usize::MAX, usize::MAX, usize::MAX, 5).unwrap() < 5);
    }

    #[test]
    fn adaptation_cadence_uses_raw_range_and_fit_identity() {
        for epoch in [9, 11, 21, 30] {
            assert!(!adaptation_due(
                LayoutOperation::Fit,
                true,
                128,
                7,
                100,
                epoch,
                40
            ));
        }
        for epoch in [10, 20] {
            assert!(adaptation_due(
                LayoutOperation::Fit,
                true,
                128,
                7,
                100,
                epoch,
                40
            ));
        }
        assert!(!adaptation_due(
            LayoutOperation::Transform,
            true,
            128,
            7,
            100,
            10,
            40
        ));
        assert!(!adaptation_due(
            LayoutOperation::Fit,
            false,
            128,
            7,
            100,
            10,
            40
        ));
        assert!(!adaptation_due(
            LayoutOperation::Fit,
            true,
            0,
            7,
            100,
            10,
            40
        ));
        assert!(!adaptation_due(
            LayoutOperation::Fit,
            true,
            128,
            100,
            100,
            10,
            40
        ));
    }

    #[test]
    fn force_ratio_uses_valid_pairs_and_linear_quantiles() {
        assert_eq!(negative_force_ratio(&[], &[]).unwrap(), 1.0);
        assert_eq!(negative_force_ratio(&[0.0, 1.0], &[1.0, 0.0]).unwrap(), 1.0);
        assert!((negative_force_ratio(&[2.0, 2.0, 2.0], &[1.0; 3]).unwrap() - 2.0).abs() < 1e-12);
        // Log ratios [0, 0, 0, 4] have interpolated q90=2.8 and mean=0.7.
        let actual = negative_force_ratio(&[1.0, 1.0, 1.0, 4.0_f64.exp()], &[1.0; 4]).unwrap();
        assert!((actual - 0.7_f64.exp()).abs() < 1e-12);
        assert!(negative_force_ratio(&[1.0], &[]).is_err());
        assert!(negative_force_ratio(&[f64::NAN], &[1.0]).is_err());
        assert!(negative_force_ratio(&[1.0], &[-1.0]).is_err());
    }

    #[test]
    fn adaptation_preserves_anchor_and_explicit_zero() {
        let mut state = NegativeScaleAdaptation::new(0.5).unwrap();
        assert_eq!(state.observe_ratio(2.0).unwrap(), 0.5);
        close(
            state.observe_ratio(100.0).unwrap(),
            (0.5 * 1.1_f64.powf(0.1)) as f32,
        );
        let before = state.scale();
        close(
            state.observe_ratio(0.001).unwrap(),
            before * 0.9_f32.powf(0.1),
        );
        assert!(state.observe_ratio(f64::NAN).is_err());
        assert!(state.observe_ratio(0.0).is_err());
        let mut zero = NegativeScaleAdaptation::new(0.0).unwrap();
        for ratio in [1.0, 2.0, 0.5, f64::MIN_POSITIVE, f64::MAX] {
            assert_eq!(zero.observe_ratio(ratio).unwrap(), 0.0);
        }
        assert!(NegativeScaleAdaptation::new(-1.0).is_err());
        assert!(NegativeScaleAdaptation::new(f32::INFINITY).is_err());
    }

    #[test]
    fn negative_windows_distinguish_raw_and_effective_ranges() {
        assert_eq!(effective_negative_range(7, 100).unwrap(), 100);
        assert_eq!(effective_negative_range(7, 300).unwrap(), 200);
        assert_eq!(effective_negative_range(299, 300).unwrap(), 299);
        assert_eq!(effective_negative_range(300, 300).unwrap(), 300);
        for (rank, start) in [(0, 0), (150, 50), (299, 100)] {
            assert_eq!(
                window_negative_index(0, 0, 0, rank, 300, 200).unwrap(),
                start
            );
            assert_eq!(
                window_negative_index(199, 0, 0, rank, 300, 200).unwrap(),
                start + 199
            );
        }
        assert_eq!(window_negative_index(0, 0, 0, 50, 100, 7).unwrap(), 47);
        for rank in [0, 50, 99] {
            assert_eq!(window_negative_index(99, 0, 0, rank, 100, 100).unwrap(), 99);
        }
        let expected = ((usize::MAX as u128 % 199) * ((2 * (usize::MAX as u128 % 199) + 1) % 199)
            % 199) as usize;
        assert_eq!(
            window_negative_index(usize::MAX, usize::MAX, usize::MAX, 0, 300, 199).unwrap(),
            expected
        );
        assert!(effective_negative_range(0, 300).is_err());
        assert!(effective_negative_range(1, 0).is_err());
        assert!(window_negative_index(0, 0, 0, 300, 300, 200).is_err());
        assert!(window_negative_index(0, 0, 0, 0, 300, 301).is_err());
        assert!(window_negative_index(0, 0, 0, 0, 300, 0).is_err());
    }

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }

    fn edge(row: usize, column: usize, weight: f32) -> WeightedEdge {
        WeightedEdge {
            row,
            column,
            weight,
        }
    }

    fn plan_input<'a>(
        operation: LayoutOperation,
        additional: &'a [LiveBuffer],
    ) -> ModernPlanInput<'a> {
        ModernPlanInput {
            operation,
            source_rows: 4,
            reference_rows: 4,
            dimensions: 2,
            source_embedding_bytes: 32,
            reference_embedding_bytes: 32,
            features: 3,
            original_rows: 5,
            graph_capacity: 8,
            edges: 3,
            adam: false,
            alias_orders: operation == LayoutOperation::Fit,
            additional,
        }
    }

    #[test]
    fn live_table_accounts_aliases_phases_and_caller_staging() {
        let extras = [
            LiveBuffer {
                name: "retained fitted state",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Both,
                bytes: 123,
            },
            LiveBuffer {
                name: "transfer staging",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Build,
                bytes: 29,
            },
            LiveBuffer {
                name: "query search",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Layout,
                bytes: 11,
            },
        ];
        let fit = modern_buffer_plan(plan_input(LayoutOperation::Fit, &extras)).unwrap();
        let orders = |plan: &ModernBufferPlan, space| {
            plan.buffers()
                .filter(|b| b.space == space && b.name.contains("order"))
                .map(|b| b.bytes)
                .sum::<usize>()
        };
        assert_eq!(orders(&fit, BufferSpace::Host), 32);
        assert_eq!(orders(&fit, BufferSpace::Device), 32);
        assert_eq!(
            fit.buffers()
                .find(|b| b.name == "sorted COO scratch")
                .unwrap()
                .bytes,
            8 * std::mem::size_of::<WeightedEdge>()
        );
        assert_eq!(
            fit.buffers()
                .find(|b| b.name == "positive periods" && b.space == BufferSpace::Device)
                .unwrap()
                .bytes,
            24
        );
        assert!(fit.peaks.build_host > fit.peaks.layout_host);
        assert!(
            fit.buffers()
                .any(|b| b.name == "transfer staging" && b.lifetime == BufferLifetime::Build)
        );
        let mut transform_input = plan_input(LayoutOperation::Transform, &extras);
        transform_input.adam = true;
        let transform = modern_buffer_plan(transform_input).unwrap();
        assert_eq!(orders(&transform, BufferSpace::Host), 64);
        assert_eq!(orders(&transform, BufferSpace::Device), 64);
        assert_eq!(
            transform
                .buffers()
                .find(|b| b.name == "fixed reference embedding")
                .unwrap()
                .bytes,
            32
        );
        assert!(transform.buffers().any(|b| b.name == "Adam first moment"));
        assert_eq!(
            transform.peaks.layout_device - fit.peaks.layout_device,
            32 + 64 + 32
        );
    }

    #[test]
    fn live_table_checks_limits_overflow_and_injected_failure() {
        let plan = modern_buffer_plan(plan_input(LayoutOperation::Fit, &[])).unwrap();
        assert!(
            plan.validate_allocations(16, 16, usize::MAX, usize::MAX, |_, _| Ok(()))
                .is_ok()
        );
        assert!(matches!(
            plan.validate_allocations(16, 16, usize::MAX, 31, |_, _| Ok(())),
            Err(UmapError::Capacity(_))
        ));
        assert!(matches!(
            plan.validate_allocations(0, 1, usize::MAX, usize::MAX, |_, _| Ok(())),
            Err(UmapError::InvalidConfig(_))
        ));
        assert_eq!(
            plan.validate_allocations(1, 1, usize::MAX, usize::MAX, |buffer, _| {
                if buffer.name == "CSR indices" {
                    Err(UmapError::Memory("injected allocation failure".into()))
                } else {
                    Ok(())
                }
            })
            .unwrap_err(),
            UmapError::Memory("injected allocation failure".into())
        );
        let mut invalid = plan_input(LayoutOperation::Fit, &[]);
        invalid.graph_capacity = usize::MAX;
        assert!(matches!(
            modern_buffer_plan(invalid),
            Err(UmapError::Overflow(_))
        ));
        invalid = plan_input(LayoutOperation::Transform, &[]);
        invalid.alias_orders = true;
        assert!(modern_buffer_plan(invalid).is_err());
        let extra = [LiveBuffer {
            name: "oversized",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: usize::MAX,
        }];
        assert!(matches!(
            modern_buffer_plan(plan_input(LayoutOperation::Fit, &extra)),
            Err(UmapError::Overflow(_))
        ));
        let many = [LiveBuffer {
            name: "extra",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::Build,
            bytes: 0,
        }; 192];
        assert!(matches!(
            modern_buffer_plan(plan_input(LayoutOperation::Fit, &many)),
            Err(UmapError::Capacity(_))
        ));
    }

    #[test]
    fn repeated_rows_use_individual_limit_and_total_peak() {
        let rows = [LiveBuffer {
            name: "nested rows",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::RepeatedBoth(100),
            bytes: 16,
        }];
        let baseline = modern_buffer_plan(plan_input(LayoutOperation::Fit, &[])).unwrap();
        let plan = modern_buffer_plan(plan_input(LayoutOperation::Fit, &rows)).unwrap();
        assert_eq!(plan.peaks.build_host - baseline.peaks.build_host, 1600);
        assert_eq!(plan.peaks.layout_host - baseline.peaks.layout_host, 1600);
        assert!(
            plan.validate_allocations(1, 1, 256, usize::MAX, |_, _| Ok(()))
                .is_ok()
        );
    }

    #[test]
    fn aligned_peaks_round_each_repeated_allocation() {
        let mut extra: Vec<_> = [1, 255, 256, 257]
            .into_iter()
            .map(|bytes| LiveBuffer {
                name: "aligned device boundary",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Layout,
                bytes,
            })
            .collect();
        extra.push(LiveBuffer {
            name: "repeated host boundary",
            space: BufferSpace::Host,
            lifetime: BufferLifetime::RepeatedBoth(5),
            bytes: 3,
        });
        let baseline = modern_buffer_plan(plan_input(LayoutOperation::Fit, &[]))
            .unwrap()
            .aligned_peaks(64, 256)
            .unwrap();
        let plan = modern_buffer_plan(plan_input(LayoutOperation::Fit, &extra)).unwrap();
        let aligned = plan.aligned_peaks(64, 256).unwrap();
        assert_eq!(aligned.layout_device - baseline.layout_device, 1280);
        assert_eq!(aligned.build_device, baseline.build_device);
        assert_eq!(aligned.build_host - baseline.build_host, 320);
        assert_eq!(aligned.layout_host - baseline.layout_host, 320);
        assert!(matches!(
            plan.aligned_peaks(0, 256),
            Err(UmapError::InvalidConfig(_))
        ));
        assert!(matches!(
            plan.aligned_peaks(1, usize::MAX),
            Err(UmapError::Overflow(_))
        ));
    }

    #[test]
    fn physical_matrix_sizes_drive_individual_limits() {
        let mut input = plan_input(LayoutOperation::Transform, &[]);
        input.dimensions = 48;
        let logical = 4 * 48 * 4;
        let physical = 4 * 64 * 4;
        input.source_embedding_bytes = physical;
        input.reference_embedding_bytes = physical;
        input.adam = true;
        let plan = modern_buffer_plan(input).unwrap();
        for name in [
            "source embedding",
            "fixed reference embedding",
            "updates or velocity",
            "Adam first moment",
            "Adam second moment",
        ] {
            assert_eq!(
                plan.buffers()
                    .find(|buffer| buffer.name == name)
                    .unwrap()
                    .bytes,
                physical
            );
        }
        assert!(matches!(
            plan.validate_allocations(1, 1, usize::MAX, logical + 1, |_, _| Ok(())),
            Err(UmapError::Capacity(_))
        ));
        input.source_embedding_bytes = logical - 1;
        assert!(modern_buffer_plan(input).is_err());
    }

    #[test]
    fn upstream_schedule_boundaries_and_casts() {
        assert_eq!(adam_schedule(0, 1.0, false).len(), 0);
        assert_eq!(momentum_schedule(0, 1.0, true).len(), 0);
        close(
            adam_schedule(1, 4.0, false).next().unwrap().to_f32().alpha,
            0.2,
        );
        let adam: Vec<_> = adam_schedule(400, 1.0, false).collect();
        for (n, value) in [(0, 2.0), (99, 0.10019), (100, 0.2), (399, 0.0505)] {
            assert!((adam[n].alpha - value).abs() < 1e-12);
            assert_eq!(adam[n].to_f32().alpha, value as f32);
        }
        let good: Vec<_> = adam_schedule(400, 2.0, true).collect();
        for (n, value) in [(0, 1.0), (49, 0.10036), (50, 0.2)] {
            assert!((good[n].alpha - value).abs() < 1e-12);
        }
        let m: Vec<_> = momentum_schedule(400, 2.0, true).collect();
        for (n, alpha, mu) in [
            (0, 0.505, 0.0),
            (199, 0.0050125, 0.4999875),
            (200, 0.005, 0.5),
            (399, 0.005, 0.5),
        ] {
            assert!((m[n].alpha - alpha).abs() < 1e-12);
            assert!((m[n].momentum - mu).abs() < 1e-12);
            assert_eq!(m[n].to_f32().alpha, alpha as f32);
        }
        let m: Vec<_> = momentum_schedule(4, 2.0, false).collect();
        assert_eq!(m[0].alpha, 0.5);
        assert_eq!(m[0].momentum, 0.375);
        assert_eq!(m[3].alpha, 0.03125);
        assert_eq!(m[3].momentum, 0.4921875);
        assert_eq!(adam[0].to_f32().beta1, 0.9_f32);
        assert_eq!(adam[0].to_f32().beta2, 0.99_f32);
    }

    #[test]
    fn compact_padding_duplicates_clocks_and_hashes() {
        let padded = [
            edge(2, 3, 0.5),
            edge(0, usize::MAX, 0.0),
            edge(0, 2, 0.25),
            edge(0, 1, 1.0),
            edge(0, 2, 0.25),
            edge(2, 0, 0.0),
        ];
        let compact = [edge(0, 1, 1.0), edge(0, 2, 0.5), edge(2, 3, 0.5)];
        let mut a = compact_graph(4, 4, &padded, 100, 3.0, LayoutOperation::Fit).unwrap();
        let mut b = compact_graph(4, 4, &compact, 100, 3.0, LayoutOperation::Fit).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.offsets, [0, 2, 2, 3, 3]);
        assert_eq!(a.indices, [1, 2, 3]);
        assert_eq!(a.periods, [1.0, 2.0, 2.0]);
        assert_eq!(a.negative_periods, [0.5, 1.0, 1.0]);
        for n in 0..6 {
            for e in 0..3 {
                let left = advance_clock(&mut a, e, n).unwrap();
                let right = advance_clock(&mut b, e, n).unwrap();
                assert_eq!(left, right);
                for p in 0..left.unwrap_or(0) {
                    assert_eq!(negative_index(e, n, p, 4).unwrap(), (e * (n + p + 1)) % 4);
                }
            }
        }
        assert_eq!(a, b);
        let mut c =
            compact_graph(1, 1, &[edge(0, 0, 1.0)], 100, 3.0, LayoutOperation::Fit).unwrap();
        assert_eq!(advance_clock(&mut c, 0, 0).unwrap(), None);
        assert_eq!(advance_clock(&mut c, 0, 1).unwrap(), Some(1));
        assert_eq!(advance_clock(&mut c, 0, 5).unwrap(), Some(8));
    }

    #[test]
    fn checked_inputs_and_hash_overflow() {
        for (op, message) in [
            (
                LayoutOperation::Fit,
                "fuzzy graph is empty after positive-edge pruning",
            ),
            (
                LayoutOperation::Transform,
                "transform fuzzy graph is empty after positive-edge pruning",
            ),
        ] {
            assert_eq!(
                compact_graph(1, 1, &[], 100, 5.0, op).unwrap_err(),
                UmapError::InvalidInput(message.into())
            );
        }
        assert!(matches!(
            compact_graph(usize::MAX, 1, &[], 100, 5.0, LayoutOperation::Fit),
            Err(UmapError::Overflow(_))
        ));
        assert!(compact_graph(1, 1, &[edge(0, 1, 1.0)], 100, 5.0, LayoutOperation::Fit).is_err());
        assert!(
            compact_graph(
                1,
                1,
                &[edge(0, 0, f32::NAN)],
                100,
                5.0,
                LayoutOperation::Fit
            )
            .is_err()
        );
        assert!(compact_graph(1, 1, &[edge(0, 0, 1.0)], 100, -1.0, LayoutOperation::Fit).is_err());
        assert!(negative_index(0, 0, 0, 0).is_err());
        let x = usize::MAX;
        assert_eq!(
            negative_index(x, x, x, 17).unwrap(),
            (((x as u128 % 17) * ((x as u128 % 17 + x as u128 % 17 + 1) % 17)) % 17) as usize
        );
    }

    #[test]
    fn fractional_clocks_and_zero_negative_rate() {
        let edges = [edge(0, 0, 1.0), edge(0, 1, 0.3)];
        let mut graph = compact_graph(1, 2, &edges, 7, 5.0, LayoutOperation::Transform).unwrap();
        let samples = 7.0_f32 * (0.3_f32 / 1.0_f32);
        let expected = 7.0_f64 / f64::from(samples);
        assert_eq!(graph.periods[1], expected);
        assert_ne!(graph.periods[1], f64::from(1.0_f32 / 0.3_f32));
        graph.next_positive[1] = 2.0 + 1e-10;
        assert_eq!(advance_clock(&mut graph, 1, 2).unwrap(), None);
        assert_eq!(advance_clock(&mut graph, 1, 3).unwrap(), Some(2));
        let mut no_negatives =
            compact_graph(1, 2, &edges, 7, 0.0, LayoutOperation::Transform).unwrap();
        assert_eq!(no_negatives.negative_periods, [f64::INFINITY; 2]);
        assert_eq!(advance_clock(&mut no_negatives, 0, 0).unwrap(), None);
        assert_eq!(advance_clock(&mut no_negatives, 0, 1).unwrap(), Some(0));
        assert_eq!(advance_clock(&mut no_negatives, 0, 100).unwrap(), Some(0));
        assert_eq!(no_negatives.next_negative[0], f64::INFINITY);
        assert_eq!(no_negatives.next_positive[0], 3.0);
    }

    #[test]
    fn force_matches_upstream_small_large_and_duplicate_rules() {
        assert!(is_small_layout(1024));
        assert!(!is_small_layout(1025));
        assert_eq!(
            resolved_negative_scale(200_000, 1025, None, true).unwrap(),
            1.0
        );
        assert_eq!(
            resolved_negative_scale(100, 200, None, true).unwrap(),
            0.5_f64.powf(0.75) as f32
        );
        assert_eq!(
            resolved_negative_scale(200_000, 1025, Some(3.0), true).unwrap(),
            3.0
        );
        assert_eq!(resolved_negative_scale(7, 100, None, false).unwrap(), 1.0);
        assert_eq!(resolved_negative_scale(7, 400, None, false).unwrap(), 0.5);
        assert_eq!(
            resolved_negative_scale(7, 400, Some(0.0), false).unwrap(),
            0.0
        );
        let source = [1.0, 0.0];
        let target = [0.0, 0.0];
        let mut small = [0.0; 2];
        let mut large = [0.0; 2];
        accumulate_force(
            &mut small, &source, &target, 1.0, 1.0, 1.0, 3.0, true, false,
        )
        .unwrap();
        accumulate_force(
            &mut large, &source, &target, 1.0, 1.0, 1.0, 3.0, false, false,
        )
        .unwrap();
        close(small[0], -0.5);
        close(large[0], -1.0);
        small = [0.0; 2];
        large = [0.0; 2];
        accumulate_force(&mut small, &source, &target, 1.0, 1.0, 1.0, 3.0, true, true).unwrap();
        accumulate_force(
            &mut large, &source, &target, 1.0, 1.0, 1.0, 3.0, false, true,
        )
        .unwrap();
        close(small[0], (1.0_f32 / 1.001).tanh());
        close(large[0], (3.0_f32 / 1.001).tanh());
        let before = small;
        accumulate_force(&mut small, &source, &source, 1.0, 1.0, 1.0, 1.0, true, true).unwrap();
        accumulate_force(&mut small, &source, &target, 1.0, 1.0, 0.0, 1.0, true, true).unwrap();
        assert_eq!(small, before);
        assert_eq!(source, [1.0, 0.0]);
        assert_eq!(target, [0.0, 0.0]);
    }

    #[test]
    fn lazy_adam_and_momentum_blocks() {
        let s = adam_schedule(10, 1.0, false).next().unwrap().to_f32();
        let (mut x, mut g, mut m, mut v) = (0.0, 2.0, 0.0, 0.0);
        adam_step(&mut x, &mut g, &mut m, &mut v, 0, s);
        close(x, s.alpha * 2.0 / 2.0001);
        assert_eq!(g, 0.0);
        let saved = (x, m, v);
        adam_step(&mut x, &mut g, &mut m, &mut v, 1, s);
        assert_eq!((x, m, v), saved);
        g = 2.0;
        adam_step(&mut x, &mut g, &mut m, &mut v, 2, s);
        assert!(x > saved.0);
        assert_eq!(g, 0.0);
        for (rows, block, count) in [
            (1_usize, 1, 1),
            (1024, 1024, 1),
            (1025, 512, 3),
            (1027, 513, 3),
        ] {
            assert_eq!(momentum_block_size(rows), block);
            assert_eq!(rows.div_ceil(block), count);
        }
        let mut positions = [0.0_f32, 2.0];
        let mut velocity = [0.0_f32; 2];
        momentum_accumulate(&mut velocity[0], positions[1] - positions[0], 0.5);
        momentum_apply(&mut positions[0], velocity[0]);
        momentum_accumulate(&mut velocity[1], positions[0] - positions[1], 0.5);
        momentum_apply(&mut positions[1], velocity[1]);
        assert_eq!(positions, [1.0, 1.5]);
        momentum_finalize(&mut velocity, 0.5);
        assert_eq!(velocity, [0.5, -0.25]);
        momentum_accumulate(&mut velocity[0], 0.0, 0.5);
        momentum_apply(&mut positions[0], velocity[0]);
        assert_eq!(positions[0], 1.5);
    }
}
