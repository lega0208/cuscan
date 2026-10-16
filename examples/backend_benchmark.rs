#![cfg(feature = "parametric")]

/// Benchmark the CubeCL CPU backend across a range of workloads.
///
/// Usage:
///     cargo run --release --example backend_benchmark
///
/// Output:
///     figures/backend_benchmark.md — Markdown summary table
///
/// Each run uses the same fast-umap configuration on its generated dataset.
/// Timing covers end-to-end fit (including KNN precomputation and training).
use std::{io::Write as _, time::Instant};

#[cfg(feature = "gpu")]
use burn::tensor::{Device, DeviceKind};
use fast_umap::{
    GraphParams, ParametricUMAP, ParametricUMAPConfig, ParametricUMAPOptimizationParams,
    utils::generate_test_data,
};

// ── Configuration ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct Scenario {
    label: String,
    n_samples: usize,
    n_features: usize,
    n_neighbors: usize,
    n_components: usize,
    hidden_sizes: Vec<usize>,
    n_epochs: usize,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        // ── Varying dataset size (fixed model) ─────────────────────────────
        Scenario {
            label: "500x20".into(),
            n_samples: 500,
            n_features: 20,
            n_neighbors: 10,
            n_components: 2,
            hidden_sizes: vec![64],
            n_epochs: 50,
        },
        Scenario {
            label: "1Kx50".into(),
            n_samples: 1_000,
            n_features: 50,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
        Scenario {
            label: "5Kx100".into(),
            n_samples: 5_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
        Scenario {
            label: "10Kx100".into(),
            n_samples: 10_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
        Scenario {
            label: "20Kx100".into(),
            n_samples: 20_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
        // ── High dimensionality ────────────────────────────────────────────
        Scenario {
            label: "2Kx500".into(),
            n_samples: 2_000,
            n_features: 500,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![256],
            n_epochs: 50,
        },
        Scenario {
            label: "5Kx784".into(),
            n_samples: 5_000,
            n_features: 784,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![256],
            n_epochs: 50,
        },
        // ── Deep network ───────────────────────────────────────────────────
        Scenario {
            label: "5Kx100 deep".into(),
            n_samples: 5_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![256, 128, 64],
            n_epochs: 50,
        },
        // ── 3-D output ─────────────────────────────────────────────────────
        Scenario {
            label: "5Kx100 3D".into(),
            n_samples: 5_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 3,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
        // ── Large scale (NN-Descent) ───────────────────────────────────────
        Scenario {
            label: "50Kx100".into(),
            n_samples: 50_000,
            n_features: 100,
            n_neighbors: 15,
            n_components: 2,
            hidden_sizes: vec![128],
            n_epochs: 50,
        },
    ]
}

fn gen_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    let flat: Vec<f32> = generate_test_data(n_samples, n_features);
    flat.chunks(n_features)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect()
}

fn make_config(sc: &Scenario) -> ParametricUMAPConfig {
    ParametricUMAPConfig {
        n_components: sc.n_components,
        hidden_sizes: sc.hidden_sizes.clone(),
        graph: GraphParams {
            n_neighbors: sc.n_neighbors,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            n_epochs: sc.n_epochs,
            batch_size: sc.n_samples,
            learning_rate: 1e-3,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

// ── Timing ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct TimingResult {
    scenario: String,
    backend: String,
    total_secs: f64,
    fit_secs: f64,
}

// ── Runners ─────────────────────────────────────────────────────────────────

fn run_cpu(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();
    let data = gen_data(sc.n_samples, sc.n_features);
    let fit_start = Instant::now();
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let fitted = ParametricUMAP::new(make_config(sc)).fit_with_signal(data, None, exit_rx);
    let fit_secs = fit_start.elapsed().as_secs_f64();
    let _ = fitted.into_embedding();
    TimingResult {
        scenario: sc.label.clone(),
        backend: "CubeCL CPU".into(),
        total_secs: total_start.elapsed().as_secs_f64(),
        fit_secs,
    }
}

#[cfg(feature = "gpu")]
fn run_wgpu(sc: &Scenario) -> TimingResult {
    let total_start = Instant::now();
    let data = gen_data(sc.n_samples, sc.n_features);
    let fit_start = Instant::now();
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let fitted =
        ParametricUMAP::with_device(make_config(sc), device).fit_with_signal(data, None, exit_rx);
    let fit_secs = fit_start.elapsed().as_secs_f64();
    let _ = fitted.into_embedding();
    TimingResult {
        scenario: sc.label.clone(),
        backend: "WGPU".into(),
        total_secs: total_start.elapsed().as_secs_f64(),
        fit_secs,
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("===========================================================");
    println!("       Backend Benchmark");
    println!("===========================================================");
    println!();
    println!("  Epochs: 50");

    println!(
        "  Backends: CubeCL CPU{}",
        if cfg!(feature = "gpu") { ", WGPU" } else { "" }
    );
    println!();

    let scenarios = scenarios();
    let mut results: Vec<TimingResult> = Vec::new();

    // ── Warmup ──────────────────────────────────────────────────────────────
    let warmup_sc = Scenario {
        label: "warmup".into(),
        n_samples: 100,
        n_features: 10,
        n_neighbors: 5,
        n_components: 2,
        hidden_sizes: vec![16],
        n_epochs: 5,
    };

    {
        print!("  CubeCL CPU warmup ... ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let _ = run_cpu(&warmup_sc);
        println!("done ({:.2}s)", t.elapsed().as_secs_f64());
    }
    #[cfg(feature = "gpu")]
    {
        print!("  WGPU warmup ... ");
        std::io::stdout().flush().unwrap();
        let t = Instant::now();
        let _ = run_wgpu(&warmup_sc);
        println!("done ({:.2}s)", t.elapsed().as_secs_f64());
    }
    println!();

    // ── Run scenarios ───────────────────────────────────────────────────────
    for sc in &scenarios {
        let hidden_str = sc
            .hidden_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "---  {} ({}x{}, hidden=[{}], k={}, out={})  ---",
            sc.label, sc.n_samples, sc.n_features, hidden_str, sc.n_neighbors, sc.n_components
        );

        print!("  CubeCL CPU ... ");
        std::io::stdout().flush().unwrap();
        let r = run_cpu(sc);
        println!("total={:.3}s  fit={:.3}s", r.total_secs, r.fit_secs);
        results.push(r);
        #[cfg(feature = "gpu")]
        if sc.n_samples <= 20_000 {
            print!("  WGPU ... ");
            std::io::stdout().flush().unwrap();
            let r = run_wgpu(sc);
            println!("total={:.3}s  fit={:.3}s", r.total_secs, r.fit_secs);
            results.push(r);
        } else if cfg!(feature = "gpu") {
            println!("  WGPU ... skipped (dataset too large for O(n^2) pairwise)");
        }
        println!();
    }

    // ── Write markdown results ──────────────────────────────────────────────
    println!("---  Results  ---\n");

    let mut md = String::new();
    md.push_str("# Backend Benchmark\n\n");
    md.push_str("50 epochs per scenario\n\n");
    md.push_str(
        "| Scenario | Samples | Features | Hidden | Out | CubeCL CPU | WGPU | CPU speedup |\n",
    );
    md.push_str(
        "|----------|---------|----------|--------|-----|----------|------|--------------|\n",
    );

    for sc in &scenarios {
        let cpu = results
            .iter()
            .find(|r| r.backend == "CubeCL CPU" && r.scenario == sc.label);
        let wgpu = results
            .iter()
            .find(|r| r.backend == "WGPU" && r.scenario == sc.label);
        let cpu_time = cpu
            .map(|r| format!("{:.2}s", r.total_secs))
            .unwrap_or_else(|| "--".into());
        let wgpu_time = wgpu
            .map(|r| format!("{:.2}s", r.total_secs))
            .unwrap_or_else(|| "--".into());
        let speedup = match (cpu, wgpu) {
            (Some(cpu), Some(wgpu)) => format!("{:.1}x", wgpu.total_secs / cpu.total_secs),
            _ => "--".into(),
        };
        let hidden_str = sc
            .hidden_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");

        md.push_str(&format!(
            "| {} | {} | {} | [{}] | {}D | {} | {} | {} |\n",
            sc.label,
            sc.n_samples,
            sc.n_features,
            hidden_str,
            sc.n_components,
            cpu_time,
            wgpu_time,
            speedup
        ));
    }

    print!("{}", md);

    std::fs::create_dir_all("figures").unwrap();
    std::fs::write("figures/backend_benchmark.md", &md).unwrap();
    println!("\nWrote figures/backend_benchmark.md");
}
