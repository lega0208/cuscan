//! Sparse-loss UMAP training — O(n·k) per epoch instead of O(n²).
//!
//! Performance techniques:
//!
//! 1. **Global KNN computed once** — brute-force on GPU, CPU sort, stored as edges.
//! 2. **Edge subsampling** — cap edges per epoch so cost is O(E_max), not O(n·k).
//! 3. **Negative sampling** — `neg_rate` random non-neighbor pairs per sampled edge.
//! 4. **In-memory checkpointing** — best model weights kept in RAM, no disk I/O.
//! 5. **Weighted edge draws** — fresh, bounded integer-count samples each epoch.
//! 6. **Fused row selection** — positive + negative indices merged into 2 selects (not 4).
//! 7. **Async loss readback** — GPU→CPU sync only every N epochs for progress display.
//! 8. **Lazy loss plotting** — charts rendered only every N epochs, not every epoch.

#[cfg(feature = "plotters")]
use crate::chart::{self, plot_loss, ChartConfigBuilder};
use crate::{format_duration, model::UMAPModel, utils::convert_vector_to_tensor};
use burn::{
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams},
    tensor::{Device, Int, Tensor, TensorData},
};

use super::config::*;
use super::get_distance_by_metric::*;

use crossbeam_channel::Receiver;
use indicatif::{ProgressBar, ProgressStyle};
use num::{Float, FromPrimitive, ToPrimitive};
use rand::{RngExt, SeedableRng};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
use std::{thread, time::Instant};

/// How often to read loss back from GPU (every N epochs).
const LOSS_READBACK_INTERVAL: usize = 5;

/// How often to render loss plots when verbose (every N epochs).
#[cfg(feature = "plotters")]
const PLOT_INTERVAL: usize = 25;

/// Maximum number of positive edges sampled per epoch.
/// Caps the per-epoch cost regardless of dataset size.
/// At 20K samples with k=15 there are 300K edges — using all of them
/// makes each epoch O(300K). Capping at 50K makes epochs ~6× cheaper
/// while still covering the full graph over multiple epochs.
const MAX_POS_EDGES_PER_EPOCH: usize = 50_000;

/// Train the UMAP model using sparse edge-based loss with negative sampling.
///
/// Per-epoch cost is O(min(n·k, MAX_EDGES) + neg_samples).
/// Progress info passed to the callback each epoch (or every N epochs).
#[derive(Debug, Clone)]
pub struct EpochProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f64,
    pub best_loss: f64,
    pub elapsed_secs: f64,
    pub epoch_ms: f64,
}

const NN_DESCENT_THRESHOLD: usize = 10_000;

#[derive(Debug, Clone, Copy)]
struct WeightedEdge {
    head: usize,
    tail: usize,
    weight: f32,
}

/// Reference smooth-kNN bandwidths. Rows include self in slot zero.
fn smooth_knn_dist(
    distances: &[f32],
    n: usize,
    k: usize,
    local_connectivity: f64,
    bandwidth: f64,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(distances.len(), n * k);
    assert!(n > 0 && k >= 2);
    assert!(local_connectivity.is_finite() && local_connectivity >= 0.0);
    assert!(bandwidth.is_finite() && bandwidth > 0.0);
    assert!(distances.iter().all(|d| d.is_finite() && *d >= 0.0));

    let target = (k as f64).log2() * bandwidth;
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    let mut sigmas = Vec::with_capacity(n);
    let mut rhos = Vec::with_capacity(n);

    for row in distances.chunks_exact(k) {
        let positive: Vec<f32> = row.iter().copied().filter(|d| *d > 0.0).collect();
        let mut rho = 0.0f32;

        if !positive.is_empty() {
            if positive.len() as f64 >= local_connectivity {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - index as f64;
                if index > 0 {
                    rho = positive[index - 1];
                    if interpolation > 1e-5 {
                        rho +=
                            (interpolation * (positive[index] - positive[index - 1]) as f64) as f32;
                    }
                } else {
                    rho = (interpolation * positive[0] as f64) as f32;
                }
            } else {
                rho = positive.iter().copied().fold(0.0f32, f32::max);
            }
        }

        let mut lo = 0.0f64;
        let mut hi = f32::MAX as f64;
        let mut mid = 1.0f64;

        for _ in 0..64 {
            let sum: f64 = row[1..]
                .iter()
                .map(|distance| {
                    let d = *distance - rho;
                    if d > 0.0 {
                        (-(d as f64 / mid)).exp()
                    } else {
                        1.0
                    }
                })
                .sum();

            if (sum - target).abs() < 1e-5 {
                break;
            }

            if sum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                mid = if hi >= f32::MAX as f64 {
                    mid * 2.0
                } else {
                    (lo + hi) / 2.0
                };
            }
        }

        let floor_mean = if rho > 0.0 {
            row.iter().sum::<f32>() / k as f32
        } else {
            mean
        };

        let sigma = (mid as f32).max(1e-3 * floor_mean);

        sigmas.push(sigma);
        rhos.push(rho);
    }

    (sigmas, rhos)
}

fn directed_memberships(
    indices: &[i32],
    distances: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    k: usize,
) -> BTreeMap<(usize, usize), f32> {
    let n = sigmas.len();
    assert_eq!(rhos.len(), n);
    assert_eq!(indices.len(), n * k);
    assert_eq!(distances.len(), n * k);
    let mut edges = BTreeMap::new();

    for head in 0..n {
        for slot in 0..k {
            let offset = head * k + slot;
            let tail = indices[offset];
            if tail == -1 || tail == head as i32 {
                continue;
            }
            assert!(tail >= 0 && (tail as usize) < n, "invalid neighbor index");
            let d = distances[offset] - rhos[head];
            let weight = if d <= 0.0 || sigmas[head] == 0.0 {
                1.0
            } else {
                (-d / sigmas[head]).exp()
            };
            if weight > 0.0 {
                *edges.entry((head, tail as usize)).or_insert(0.0) += weight;
            }
        }
    }

    edges
}

fn symmetrize(directed: &BTreeMap<(usize, usize), f32>, mix: f64) -> Vec<WeightedEdge> {
    assert!(mix.is_finite() && (0.0..=1.0).contains(&mix));
    let mix = mix as f32;
    let mut pairs = BTreeMap::new();

    for &(head, tail) in directed.keys() {
        pairs.insert((head.min(tail), head.max(tail)), ());
    }

    let mut edges = Vec::with_capacity(pairs.len() * 2);

    for ((head, tail), ()) in pairs {
        let a = directed.get(&(head, tail)).copied().unwrap_or(0.0);
        let b = directed.get(&(tail, head)).copied().unwrap_or(0.0);
        let intersection = a * b;
        let weight = mix * (a + b - intersection) + (1.0 - mix) * intersection;
        if weight > 0.0 {
            edges.push(WeightedEdge { head, tail, weight });
            edges.push(WeightedEdge {
                head: tail,
                tail: head,
                weight,
            });
        }
    }

    edges.sort_by_key(|edge| (edge.head, edge.tail));

    edges
}

#[derive(Debug, PartialEq, Eq)]
enum EdgePreparationError {
    InvalidEpochs,
    InvalidWeight,
    CountOverflow,
    EmptySupport,
}

struct EdgeSampler {
    edges: Vec<(usize, usize)>,
    prefixes: Vec<u64>,
    total: u64,
}

fn repetition_counts(weights: &[f32], epochs: usize) -> Result<Vec<u64>, EdgePreparationError> {
    if epochs == 0 {
        return Err(EdgePreparationError::InvalidEpochs);
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(EdgePreparationError::InvalidWeight);
    }

    let max = weights.iter().copied().fold(0.0f32, f32::max);
    let threshold = max / epochs as f32;

    weights
        .iter()
        .map(|weight| {
            if *weight < threshold {
                return Ok(0);
            }
            // NumPy's float32 array product rounds before integer conversion.
            // Widening the weight first would turn 0.7 * 10 into count 6, not 7.
            let count = (*weight * epochs as f32).floor();
            if !count.is_finite() || count >= u64::MAX as f32 {
                Err(EdgePreparationError::CountOverflow)
            } else {
                Ok(count as u64)
            }
        })
        .collect()
}

impl EdgeSampler {
    fn new(edges: &[WeightedEdge], epochs: usize) -> Result<Self, EdgePreparationError> {
        let counts = repetition_counts(
            &edges.iter().map(|edge| edge.weight).collect::<Vec<_>>(),
            epochs,
        )?;
        Self::from_counts(edges, &counts)
    }

    fn from_counts(edges: &[WeightedEdge], counts: &[u64]) -> Result<Self, EdgePreparationError> {
        assert_eq!(edges.len(), counts.len());
        let mut support = Vec::new();
        let mut prefixes = Vec::new();
        let mut total = 0u64;

        for (edge, &count) in edges.iter().zip(counts) {
            if count == 0 {
                continue;
            }
            total = total
                .checked_add(count)
                .ok_or(EdgePreparationError::CountOverflow)?;
            support.push((edge.head, edge.tail));
            prefixes.push(total);
        }

        if total == 0 {
            return Err(EdgePreparationError::EmptySupport);
        }

        Ok(Self {
            edges: support,
            prefixes,
            total,
        })
    }

    fn draw_count(&self) -> usize {
        self.total.min(MAX_POS_EDGES_PER_EPOCH as u64) as usize
    }

    fn edge_for_ticket(&self, ticket: u64) -> (usize, usize) {
        assert!(ticket < self.total);
        self.edges[self.prefixes.partition_point(|prefix| *prefix <= ticket)]
    }

    fn sample_batch(
        &self,
        num_samples: usize,
        neg_rate: usize,
        rng: &mut impl RngExt,
    ) -> (Vec<i64>, Vec<i64>) {
        let n_pos = self.draw_count();
        let n_neg = n_pos
            .checked_mul(neg_rate)
            .expect("negative sample count overflow")
            .max(num_samples);
        let total = n_pos.checked_add(n_neg).expect("edge batch size overflow");
        let mut heads = Vec::with_capacity(total);
        let mut tails = Vec::with_capacity(total);

        for _ in 0..n_pos {
            let (head, tail) = self.edge_for_ticket(rng.random_range(0..self.total));
            heads.push(head as i64);
            tails.push(tail as i64);
        }

        for _ in 0..n_neg {
            let head = rng.random_range(0..num_samples);
            let mut tail = rng.random_range(0..num_samples - 1);
            if tail >= head {
                tail += 1;
            }
            heads.push(head as i64);
            tails.push(tail as i64);
        }

        (heads, tails)
    }
}

fn with_self_slots(
    indices: Vec<i32>,
    distances: Vec<f32>,
    n: usize,
    k: usize,
) -> (Vec<i32>, Vec<f32>) {
    assert!(k >= 2);
    assert_eq!(indices.len(), n * (k - 1));
    assert_eq!(distances.len(), indices.len());
    let mut out_indices = Vec::with_capacity(n * k);
    let mut out_distances = Vec::with_capacity(n * k);

    for head in 0..n {
        out_indices.push(head as i32);
        out_distances.push(0.0);
        let mut seen = std::collections::BTreeSet::new();
        let start = head * (k - 1);
        for slot in start..start + k - 1 {
            let tail = indices[slot];
            if tail == head as i32 {
                continue;
            }
            assert!(
                tail >= 0 && (tail as usize) < n && seen.insert(tail),
                "invalid or duplicate neighbor row"
            );
            assert!(
                distances[slot].is_finite() && distances[slot] >= 0.0,
                "invalid neighbor distance"
            );
            out_indices.push(tail);
            out_distances.push(distances[slot]);
        }
        assert_eq!(
            out_indices.len(),
            (head + 1) * k,
            "short neighbor row after self removal"
        );
    }

    (out_indices, out_distances)
}

fn sparse_inputs<F: Float + ToPrimitive + burn::tensor::Element>(
    data: Vec<F>,
    num_samples: usize,
    num_features: usize,
    k: usize,
    device: &Device,
) -> (Tensor<2>, Vec<i32>, Vec<f32>) {
    let all_data_tensor = convert_vector_to_tensor(data.clone(), num_samples, num_features, device);

    #[cfg(feature = "nn-descent")]
    let use_nn_descent = num_samples >= NN_DESCENT_THRESHOLD;

    #[cfg(feature = "nn-descent")]
    if use_nn_descent {
        let data_f32: Vec<f32> = data
            .iter()
            .map(|x| num::ToPrimitive::to_f32(x).unwrap())
            .collect();

        let (indices, distances) =
            crate::nn_descent::nn_descent(&data_f32, num_samples, num_features, k - 1);

        let (indices, distances) = with_self_slots(indices, distances, num_samples, k);

        return (all_data_tensor, indices, distances);
    }

    let pairwise = pairwise_distances(all_data_tensor.clone());
    let flat: Vec<f32> = pairwise.to_data().try_to_vec::<f32>().unwrap();
    let (indices, distances) = knn_from_pairwise_cpu(&flat, num_samples, k - 1);
    let (indices, distances) = with_self_slots(indices, distances, num_samples, k);

    (all_data_tensor, indices, distances)
}

pub fn train_sparse<F: Float>(
    name: &str,
    mut model: UMAPModel,
    num_samples: usize,
    num_features: usize,
    data: Vec<F>,
    config: &TrainingConfig,
    device: Device,
    exit_rx: Receiver<()>,
    labels: Option<Vec<String>>,
    on_progress: Option<Box<dyn Fn(EpochProgress) + Send>>,
) -> (UMAPModel, Vec<F>, F)
where
    F: FromPrimitive + Send + Sync + burn::tensor::Element,
{
    // Resolve and create the figures output directory.
    // Gracefully degrade: if the directory cannot be created (e.g. read-only
    // filesystem) we warn once and disable all plot output for this run
    // rather than panicking.
    #[cfg(feature = "plotters")]
    let figures_dir: PathBuf = config
        .figures_dir
        .clone()
        .unwrap_or_else(std::env::temp_dir);

    #[cfg(feature = "plotters")]
    let can_plot = match std::fs::create_dir_all(&figures_dir) {
        Ok(()) => true,
        Err(e) => {
            eprintln!(
                "[fast-umap] Warning: could not create figures directory '{}': {}. \
                 Plot output will be disabled for this run. \
                 Set `figures_dir` in your config to a writable path.",
                figures_dir.display(),
                e
            );
            false
        }
    };

    let k = config.k_neighbors;
    let repulsion_strength = config.repulsion_strength;
    let kernel_a = config.kernel_a;
    let kernel_b = config.kernel_b;
    let neg_rate = config.neg_sample_rate;
    let verbose = config.verbose;

    assert!(
        k >= 2,
        "k_neighbors must include self and at least one other sample"
    );
    assert!(config.epochs > 0, "epochs must be positive");
    if num_samples <= k {
        panic!("num_samples ({num_samples}) must be > k_neighbors ({k}).");
    }

    // ── Log configuration ────────────────────────────────────────────────────
    if verbose {
        println!("[fast-umap] Configuration:");
        println!("[fast-umap]   samples={num_samples}  features={num_features}  k_neighbors={k}");
        println!(
            "[fast-umap]   epochs={}  lr={:.0e}  repulsion_strength={repulsion_strength}",
            config.epochs, config.learning_rate
        );
        println!(
            "[fast-umap]   kernel: a={kernel_a:.4}  b={kernel_b:.4}  (q = 1 / (1 + a·d^(2b)))"
        );
    }

    // ── Precompute global KNN once ───────────────────────────────────────────
    #[cfg(feature = "nn-descent")]
    let use_nn_descent = num_samples >= NN_DESCENT_THRESHOLD;
    #[cfg(not(feature = "nn-descent"))]
    let use_nn_descent = false;

    if verbose {
        if use_nn_descent {
            println!("[fast-umap] Computing approximate k-NN graph via NN-Descent (k={k}, n={num_samples}) …");
        } else {
            println!("[fast-umap] Computing global k-NN graph (k={k}) …");
        }
    }
    let knn_start = Instant::now();

    // PCA also needs the caller's unmodified feature values.
    #[cfg(feature = "pca")]
    let pca_data = data.clone();
    let (all_data_tensor, knn_indices, knn_distances) =
        sparse_inputs(data, num_samples, num_features, k, &device);

    let (sigmas, rhos) = smooth_knn_dist(
        &knn_distances,
        num_samples,
        k,
        config.local_connectivity,
        config.bandwidth,
    );

    let directed = directed_memberships(&knn_indices, &knn_distances, &sigmas, &rhos, k);
    let weighted_edges = symmetrize(&directed, config.set_op_mix_ratio);
    let maximum_weight = weighted_edges
        .iter()
        .map(|edge| edge.weight)
        .fold(0.0f32, f32::max);

    let prune_threshold = maximum_weight / config.epochs as f32;

    let retained_edges = weighted_edges
        .iter()
        .filter(|edge| edge.weight >= prune_threshold)
        .count();

    let sampler = EdgeSampler::new(&weighted_edges, config.epochs)
        .unwrap_or_else(|error| panic!("fuzzy edge preparation failed: {error:?}"));

    let n_pos = sampler.draw_count();

    let n_neg = n_pos
        .checked_mul(neg_rate)
        .expect("negative sample count overflow")
        .max(num_samples);

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    let knn_elapsed = knn_start.elapsed();
    if verbose {
        println!(
            "[fast-umap] Fuzzy graph done in {:.2}s — {} directed, {} symmetric, {} positive-count edges; integer mass={}",
            knn_elapsed.as_secs_f64(),
            directed.len(), weighted_edges.len(), sampler.edges.len(), sampler.total,
        );
        println!("[fast-umap]   local_connectivity={} bandwidth={} set_op_mix_ratio={}; fresh {n_pos} positive + {n_neg} negative draws per epoch", config.local_connectivity, config.bandwidth, config.set_op_mix_ratio);
        println!("[fast-umap]   coalesced={} symmetric={} retained_after_pruning={retained_edges} positive_count_support={} max_weight={maximum_weight} prune_threshold={prune_threshold}; NN-descent threshold={NN_DESCENT_THRESHOLD}", directed.len(), weighted_edges.len(), sampler.edges.len());
        println!("[fast-umap] Training started …");
    }

    // ── PCA warm-start (optional) ───────────────────────────────────────────
    #[cfg(feature = "pca")]
    {
        let pca_epochs = 10;
        let pca_lr = 1e-2;

        let data_f32: Vec<f32> = pca_data
            .iter()
            .map(|x| num::ToPrimitive::to_f32(x).unwrap())
            .collect();
        let n_components = model.forward(all_data_tensor.clone()).dims()[1];
        let (projected, _components, _mean) = crate::pca::pca(
            &data_f32,
            num_samples,
            num_features,
            n_components,
            &device.clone().into(),
        );

        let pca_target: Tensor<2> = Tensor::from_data(
            TensorData::new(projected, [num_samples, n_components]),
            &device,
        );

        if verbose {
            println!("[fast-umap] PCA warm-start ({pca_epochs} epochs) …");
        }

        let mut pca_optim = AdamConfig::new()
            .with_beta_1(0.9f32)
            .with_beta_2(0.999f32)
            .init();

        for _e in 0..pca_epochs {
            let output = model.forward(all_data_tensor.clone());
            let diff = output - pca_target.clone();
            let mse_vals = (diff.clone() * diff)
                .reshape([num_samples * n_components])
                .mean_dim(0);
            let grads = mse_vals.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = pca_optim.step(pca_lr, model, grads);
        }

        if verbose {
            println!("[fast-umap] PCA warm-start complete");
        }
    }

    // ── Optimizer ─────────────────────────────────────────────────────────────
    let config_optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.penalty)))
        .with_beta_1(config.beta1 as f32)
        .with_beta_2(config.beta2 as f32);
    let mut optim = config_optimizer.init();

    let start_time = Instant::now();

    let pb = if verbose {
        let pb = ProgressBar::new(config.epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{bar:40} | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let mut epoch = 0usize;
    let mut losses: Vec<F> = vec![];
    let mut best_loss = F::infinity();
    let mut last_read_loss = F::infinity();
    let mut epochs_without_improvement = 0i32;

    // ── In-memory best-model checkpoint ──────────────────────────────────────
    let mut best_record = model.clone().into_record();

    'main: loop {
        if exit_rx.try_recv().is_ok() {
            if verbose {
                eprintln!("[fast-umap] Interrupted — restoring best model (epoch {epoch}, loss {best_loss:.6})");
            }
            break 'main;
        }

        // ── Forward: embed ALL points at once ────────────────────────────────
        let embeddings = model.forward(all_data_tensor.clone()); // [n, output_dim]

        // ── Fused row selection ─────────────────────────────────────────────
        let (heads, tails) = sampler.sample_batch(num_samples, neg_rate, &mut rng);
        let batch_n_pos = n_pos;
        let batch_n_total = heads.len();
        let fused_h: Tensor<1, Int> =
            Tensor::from_data(TensorData::new(heads, [batch_n_total]), &device);
        let fused_t: Tensor<1, Int> =
            Tensor::from_data(TensorData::new(tails, [batch_n_total]), &device);

        let (all_head_emb, all_tail_emb) = (
            embeddings.clone().select(0, fused_h),
            embeddings.select(0, fused_t),
        );

        let diff = all_head_emb - all_tail_emb;
        let dist_sq = (diff.clone() * diff).sum_dim(1); // [batch_n_total, 1]

        // ── Split into positive (attraction) and negative (repulsion) ────────
        let dist_sq_pos = dist_sq.clone().slice([0..batch_n_pos]);
        let dist_sq_neg = dist_sq.slice([batch_n_pos..batch_n_total]);

        // UMAP kernel: q = 1 / (1 + a * d^(2b))
        // When b != 1 we need d^(2b) = (d²)^b = dist_sq.powf(b).
        // Attraction: -log(q)
        let dist_pow_pos = dist_sq_pos.clamp_min(1e-8f32).powf_scalar(kernel_b);
        let q_pos = (dist_pow_pos.clone() * kernel_a + 1.0f32).recip();
        let attract_vals = q_pos.clamp_min(1e-6f32).log().neg();
        let attraction = attract_vals
            .clone()
            .reshape([attract_vals.dims().iter().product::<usize>()])
            .mean_dim(0);

        // Repulsion: -log(1-q) where 1-q = a*d^(2b) / (1 + a*d^(2b))
        let dist_pow_neg = dist_sq_neg.clamp_min(1e-8f32).powf_scalar(kernel_b);
        let a_dpow_neg = dist_pow_neg.clone() * kernel_a;
        let one_minus_q_neg = a_dpow_neg.clone() / (a_dpow_neg + 1.0f32);
        let repulse_vals = one_minus_q_neg.clamp_min(1e-6f32).log().neg();
        let repulsion = repulse_vals
            .clone()
            .reshape([repulse_vals.dims().iter().product::<usize>()])
            .mean_dim(0);

        // ── UMAP cross-entropy loss ──────────────────────────────────────────
        let loss = attraction + repulsion_strength * repulsion;

        // ── Read loss from GPU only every N epochs ───────────────────────────
        let should_read =
            epoch % LOSS_READBACK_INTERVAL == 0 || epoch >= config.epochs || epoch == 0;

        let current_loss = if should_read {
            let v = F::from(loss.clone().into_scalar::<f32>() as f64).unwrap();
            last_read_loss = v;
            v
        } else {
            last_read_loss
        };

        if should_read && (current_loss.is_nan() || current_loss.is_infinite()) {
            eprintln!(
                "[fast-umap] WARNING: loss became {current_loss:.4} at epoch {epoch} — stopping early."
            );
            break 'main;
        }

        // ── Backward + optimizer step ────────────────────────────────────────
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learning_rate, model, grads);

        // ── Epoch bookkeeping ────────────────────────────────────────────────
        losses.push(current_loss);

        let elapsed = start_time.elapsed();
        let epoch_ms = if epoch > 0 {
            elapsed.as_secs_f64() * 1000.0 / epoch as f64
        } else {
            0.0
        };

        if let Some(pb) = &pb {
            pb.inc(1);
            if should_read {
                pb.set_message(format!(
                    "Elapsed: {} | Epoch: {epoch}/{} | Loss: {current_loss:.6} | Best: {best_loss:.6}",
                    format_duration(elapsed),
                    config.epochs,
                ));
            }
        }

        // Fire progress callback every LOSS_READBACK_INTERVAL epochs
        if should_read {
            if let Some(ref cb) = on_progress {
                cb(EpochProgress {
                    epoch,
                    total_epochs: config.epochs,
                    loss: ToPrimitive::to_f64(&current_loss).unwrap_or(0.0),
                    best_loss: ToPrimitive::to_f64(&best_loss).unwrap_or(0.0),
                    elapsed_secs: elapsed.as_secs_f64(),
                    epoch_ms,
                });
            }
        }

        if let Some(timeout) = config.timeout {
            if elapsed >= Duration::from_secs(timeout) {
                if verbose {
                    println!(
                        "[fast-umap] Timeout ({timeout}s) reached at epoch {epoch} — stopping."
                    );
                }
                break;
            }
        }

        if should_read && current_loss <= best_loss {
            best_loss = current_loss;
            epochs_without_improvement = 0;
            best_record = model.clone().into_record();
        } else if should_read {
            epochs_without_improvement += LOSS_READBACK_INTERVAL as i32;
        }

        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience {
                if verbose {
                    println!(
                        "[fast-umap] Early stopping — no improvement for {patience} epochs (best loss: {best_loss:.6})."
                    );
                }
                break;
            }
        }

        if let Some(min_desired_loss) = config.min_desired_loss {
            if should_read && current_loss < F::from(min_desired_loss).unwrap() {
                if verbose {
                    println!(
                        "[fast-umap] Desired loss {min_desired_loss:.6} reached at epoch {epoch} (loss: {current_loss:.6})."
                    );
                }
                break;
            }
        }

        if epoch >= config.epochs {
            break;
        }

        // ── Periodic coloured snapshot (verbose feature flag) ─────────────────
        #[cfg(feature = "plotters")]
        let loss_plot_path = figures_dir
            .join(format!("losses_{name}.png"))
            .to_string_lossy()
            .into_owned();

        #[cfg(all(feature = "plotters", feature = "verbose"))]
        if can_plot {
            const STEP: usize = 100;
            if epoch > 0 && epoch % STEP == 0 {
                let losses_snap = losses.clone();
                let model_snap = model.valid();
                let tensor_data =
                    convert_vector_to_tensor(data.clone(), num_samples, num_features, &device);
                let embeddings = model_snap.forward(tensor_data);
                let lpath = loss_plot_path.clone();
                let caption = format!("{name}_{epoch}");
                let fig_path = figures_dir
                    .join(format!("{name}_{epoch}.png"))
                    .to_string_lossy()
                    .into_owned();
                let snap_labels = labels.clone();

                thread::spawn(move || {
                    let chart_config = ChartConfigBuilder::default()
                        .caption(&caption)
                        .path(&fig_path)
                        .build();
                    chart::chart_tensor(embeddings, snap_labels, Some(chart_config));
                    if losses_snap.len() > STEP {
                        plot_loss(losses_snap[STEP..].to_vec(), &lpath).unwrap();
                    }
                });
            }
        }

        #[cfg(feature = "plotters")]
        if can_plot && verbose && epoch % PLOT_INTERVAL == 0 {
            plot_loss(losses.clone(), &loss_plot_path).unwrap();
        }

        // ── Per-epoch GPU cooldown ────────────────────────────────────────────
        // Sleeping here (between the optimizer step and the next forward pass)
        // gives the GPU scheduler breathing room and prevents 100 % utilisation
        // for the full run.  The default is 0 ms (no pause).
        if config.cooldown_ms > 0 {
            thread::sleep(Duration::from_millis(config.cooldown_ms));
        }

        epoch += 1;
    }

    #[cfg(all(feature = "plotters", feature = "verbose"))]
    if can_plot {
        let path = figures_dir
            .join(format!("losses_{name}.png"))
            .to_string_lossy()
            .into_owned();
        plot_loss(losses.clone(), &path).unwrap();
    }

    #[cfg(feature = "plotters")]
    if can_plot && verbose {
        let path = figures_dir
            .join(format!("losses_{name}.png"))
            .to_string_lossy()
            .into_owned();
        plot_loss(losses.clone(), &path).unwrap();
    }

    if let Some(pb) = pb {
        pb.finish();
    }

    // ── Restore best model from in-memory record ─────────────────────────────
    model = model.load_record(best_record);

    let total_elapsed = start_time.elapsed();
    if verbose {
        println!(
            "[fast-umap] Training complete — {epoch} epochs in {}, best loss: {best_loss:.6}",
            format_duration(total_elapsed),
        );
    }

    (model, losses, best_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn floats(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn close(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (*a - *b).abs() <= 1e-5 + 1e-4 * b.abs(),
                "{context}[{index}]: {a} != {b}"
            );
        }
    }

    #[test]
    fn reference_graph_fixtures_match_memberships_and_symmetrization() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/umap_learn_0.5.12_fuzzy_graph.json"
        ))
        .unwrap();
        assert_eq!(fixture["reference_version"], "0.5.12");
        for case in fixture["cases"].as_array().unwrap() {
            let n = case["n"].as_u64().unwrap() as usize;
            let k = case["k"].as_u64().unwrap() as usize;
            let name = case["name"].as_str().unwrap();
            let indices: Vec<i32> = case["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let distances = floats(&case["distances"]);
            let (sigmas, rhos) = smooth_knn_dist(
                &distances,
                n,
                k,
                case["local_connectivity"].as_f64().unwrap(),
                case["bandwidth"].as_f64().unwrap(),
            );
            close(&sigmas, &floats(&case["sigma"]), name);
            close(&rhos, &floats(&case["rho"]), name);
            let directed = directed_memberships(&indices, &distances, &sigmas, &rhos, k);
            let mut dense = vec![0.0; n * n];
            for (&(head, tail), &weight) in &directed {
                dense[head * n + tail] = weight;
            }
            close(&dense, &floats(&case["directed"]), name);
            let edges = symmetrize(&directed, case["set_op_mix_ratio"].as_f64().unwrap());
            let mut symmetric = vec![0.0; n * n];
            for edge in &edges {
                symmetric[edge.head * n + edge.tail] = edge.weight;
                assert_eq!(
                    edge.weight,
                    edges
                        .iter()
                        .find(|other| other.head == edge.tail && other.tail == edge.head)
                        .unwrap()
                        .weight
                );
            }
            close(&symmetric, &floats(&case["symmetric"]), name);
            let expected_counts: Vec<u64> = case["counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            assert_eq!(
                repetition_counts(&symmetric, case["epochs"].as_u64().unwrap() as usize).unwrap(),
                expected_counts,
                "{name} counts"
            );
            for i in 0..n {
                assert_eq!(symmetric[i * n + i], 0.0);
            }
        }
        for case in fixture["count_cases"].as_array().unwrap() {
            let expected: Vec<u64> = case["counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            assert_eq!(
                repetition_counts(
                    &floats(&case["weights"]),
                    case["epochs"].as_u64().unwrap() as usize
                )
                .unwrap(),
                expected,
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn duplicate_memberships_coalesce_without_confusing_self() {
        let directed =
            directed_memberships(&[0, 1, 1, 1, 0, -1], &[0.0; 6], &[0.0; 2], &[0.0; 2], 3);
        assert_eq!(directed.len(), 2);
        assert_eq!(directed[&(0, 1)], 2.0);
        assert_eq!(directed[&(1, 0)], 1.0);
        let asymmetric = BTreeMap::from([((0, 1), 0.5), ((1, 0), 0.2), ((1, 2), 0.8)]);
        let union = symmetrize(&asymmetric, 1.0);
        close(
            &union.iter().map(|edge| edge.weight).collect::<Vec<_>>(),
            &[0.6, 0.6, 0.8, 0.8],
            "union",
        );
        let intersection = symmetrize(&asymmetric, 0.0);
        assert_eq!(intersection.len(), 2);
        close(
            &intersection
                .iter()
                .map(|edge| edge.weight)
                .collect::<Vec<_>>(),
            &[0.1, 0.1],
            "intersection",
        );
    }

    #[test]
    fn integer_counts_pin_f32_boundaries_and_explicit_errors() {
        assert_eq!(
            repetition_counts(&[1.0, 0.2, 0.001], 10).unwrap(),
            vec![10, 2, 0]
        );
        assert_eq!(repetition_counts(&[0.7], 10).unwrap(), vec![7]);
        let boundary = 0.1f32;
        assert_eq!(
            repetition_counts(
                &[
                    1.0,
                    f32::from_bits(boundary.to_bits() - 1),
                    boundary,
                    f32::from_bits(boundary.to_bits() + 1)
                ],
                10
            )
            .unwrap(),
            vec![10, 0, 1, 1]
        );
        assert_eq!(
            repetition_counts(&[1.0], 0),
            Err(EdgePreparationError::InvalidEpochs)
        );
        assert_eq!(
            repetition_counts(&[f32::NAN], 10),
            Err(EdgePreparationError::InvalidWeight)
        );
        assert_eq!(
            repetition_counts(&[-1.0], 10),
            Err(EdgePreparationError::InvalidWeight)
        );
        assert_eq!(
            repetition_counts(&[f32::MAX], 10),
            Err(EdgePreparationError::CountOverflow)
        );
        let edges = vec![
            WeightedEdge {
                head: 0,
                tail: 1,
                weight: 1.0
            };
            2
        ];
        assert!(matches!(
            EdgeSampler::from_counts(&edges, &[u64::MAX, 1]),
            Err(EdgePreparationError::CountOverflow)
        ));
        assert!(matches!(
            EdgeSampler::from_counts(&edges, &[0, 0]),
            Err(EdgePreparationError::EmptySupport)
        ));
        assert!(matches!(
            EdgeSampler::new(&[], 10),
            Err(EdgePreparationError::EmptySupport)
        ));
        let intersection = symmetrize(&BTreeMap::from([((0, 1), 1.0)]), 0.0);
        assert!(matches!(
            EdgeSampler::new(&intersection, 10),
            Err(EdgePreparationError::EmptySupport)
        ));
    }

    #[test]
    fn sampler_uses_exact_integer_tickets_and_fresh_seeded_epoch_draws() {
        let edges = [
            WeightedEdge {
                head: 0,
                tail: 1,
                weight: 1.0,
            },
            WeightedEdge {
                head: 1,
                tail: 2,
                weight: 1.0,
            },
            WeightedEdge {
                head: 2,
                tail: 0,
                weight: 0.0,
            },
        ];
        let small = EdgeSampler::from_counts(&edges, &[10, 2, 0]).unwrap();
        assert_eq!(small.draw_count(), 12);
        for ticket in 0..12 {
            assert_eq!(
                small.edge_for_ticket(ticket),
                if ticket < 10 { (0, 1) } else { (1, 2) }
            );
        }
        let sampler = EdgeSampler::from_counts(&edges, &[70_000, 30_000, 0]).unwrap();
        assert_eq!(sampler.draw_count(), 50_000);
        assert_eq!(
            EdgeSampler::from_counts(&edges, &[50_000, 0, 0])
                .unwrap()
                .draw_count(),
            50_000
        );
        assert_eq!(
            EdgeSampler::from_counts(&edges, &[50_001, 0, 0])
                .unwrap()
                .draw_count(),
            50_000
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut repeat = rand::rngs::StdRng::seed_from_u64(42);
        let first = sampler.sample_batch(3, 2, &mut rng);
        assert_eq!(first, sampler.sample_batch(3, 2, &mut repeat));
        let second = sampler.sample_batch(3, 2, &mut rng);
        assert_eq!(second, sampler.sample_batch(3, 2, &mut repeat));
        assert_ne!(first.0[..50_000], second.0[..50_000]); // Fixed deterministic nondegenerate fixture.
        assert_ne!(first.1[50_000..], second.1[50_000..]);
        assert_eq!(sampler.total, 100_000); // Draws/early completion cannot change configured-E mass.
        assert_eq!(first.0.len(), 150_000);
        let first_edge_count = first.0[..50_000].iter().filter(|head| **head == 0).count();
        assert!((first_edge_count as f64 / 50_000.0 - 0.7).abs() < 0.01);
        for (head, tail) in first.0.iter().zip(&first.1) {
            assert_ne!(head, tail);
        }
        assert!(first.0[..50_000]
            .iter()
            .all(|head| *head == 0 || *head == 1));
        let (heads, tails) = small.sample_batch(100, 0, &mut rng);
        assert_eq!(heads.len(), 12 + 100);
        assert_eq!(heads.len(), tails.len());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| small.sample_batch(
                3,
                usize::MAX,
                &mut rng
            )))
            .is_err()
        );
    }

    #[test]
    fn self_slot_preserves_duplicate_zero_neighbors_and_rejects_short_rows() {
        let (indices, distances) = with_self_slots(vec![1, 0], vec![0.0, 0.0], 2, 2);
        assert_eq!(indices, vec![0, 1, 1, 0]);
        assert_eq!(distances, vec![0.0; 4]);
        assert!(
            std::panic::catch_unwind(|| with_self_slots(vec![0, 1], vec![0.0; 2], 2, 2)).is_err()
        );
    }

    #[test]
    fn sparse_preparation_uses_raw_values_for_knn_and_encoder_input() {
        let data = vec![
            0.0f32, 0.0, 7.0, // A
            0.0, 100.0, 7.0, // B
            10.0, 1.0, 7.0, // C
            0.0, 10_000.0, 7.0, // scale outlier
        ];
        let (encoder_input, knn_indices, distances) = sparse_inputs(data, 4, 3, 2, &Device::flex());

        // In raw space A is closest to C; z-scoring would make B closer.
        assert_eq!(knn_indices[..2], [0, 2]);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 101.0f32.sqrt()).abs() < 1e-5);
        let encoder_values = encoder_input.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(
            encoder_values,
            vec![0.0, 0.0, 7.0, 0.0, 100.0, 7.0, 10.0, 1.0, 7.0, 0.0, 10_000.0, 7.0,]
        );
    }

    #[cfg(feature = "nn-descent")]
    #[test]
    fn approximate_sparse_preparation_preserves_raw_values_at_threshold() {
        let num_samples = super::NN_DESCENT_THRESHOLD;
        let data: Vec<f32> = (0..num_samples)
            .flat_map(|index| match index {
                0 => [0.0, 0.0, 7.0],
                _ => {
                    let jitter = (index / 3) as f32 * 0.000_001;
                    match index % 3 {
                        0 => [0.0, 100.0 + jitter, 7.0],
                        1 => [10.0, 1.0 + jitter, 7.0],
                        _ => [0.0, 10_000.0 + jitter, 7.0],
                    }
                }
            })
            .collect();
        let k = 15;
        let (encoder_input, indices, distances) =
            sparse_inputs(data.clone(), num_samples, 3, k, &Device::flex());

        // Raw query neighbors come from [10, ~1, 7]; z-scoring the
        // NN-Descent-only input instead would select [0, ~100, 7].
        for &neighbor in &indices[1..k] {
            assert_eq!(data[neighbor as usize * 3], 10.0);
        }
        assert_eq!(encoder_input.to_data().try_to_vec::<f32>().unwrap(), data);
        assert_eq!(indices.len(), num_samples * k);
        for (sample, neighbors) in indices.chunks_exact(k).enumerate() {
            assert_eq!(neighbors[0] as usize, sample);
            assert_eq!(distances[sample * k], 0.0);
            for &neighbor in &neighbors[1..] {
                assert!(neighbor >= 0 && (neighbor as usize) < num_samples);
                assert_ne!(neighbor as usize, sample);
            }
        }
    }
}
