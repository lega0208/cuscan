/// fast-umap benchmark reporter.
///
/// Detects available hardware (CPU model, GPU model via WGPU/Metal/Vulkan),
/// runs timing loops for every benchmark group on both CPU (Flex) and GPU
/// (WGPU) backends, then writes hardware-tagged Markdown + SVG files to
/// `benches/results/`.
///
/// Usage:
///     cargo run --release --bin bench_report
///
/// Output files:
///     benches/results/cpu_<cpu_name>.md   + .svg
///     benches/results/gpu_<gpu_name>.md   + .svg   (when a GPU is found)
///
/// The files are also copied to the generic
///     benches/results/benchmark_results.md + .svg
/// so the README always links to the latest run.
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use fast_umap::{
    normalizer::normalize as layer_normalize,
    parametric_umap::model::{ParametricUMAPModel, ParametricUMAPModelConfigBuilder},
    utils::{
        convert_tensor_to_vector, convert_vector_to_tensor, generate_test_data, normalize_data,
        normalize_tensor,
    },
};
use std::{
    fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

// ─── Hardware detection ───────────────────────────────────────────────────────

/// CPU model string via platform-specific system commands.
fn detect_cpu() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        // x86: has "model name" field
        if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if line.starts_with("model name") {
                    if let Some(v) = line.split(':').nth(1) {
                        return v.trim().to_string();
                    }
                }
            }
            // ARM: use "Hardware" field, then implementer + architecture
            let mut hw = String::new();
            let mut implementer = String::new();
            for line in text.lines() {
                if line.starts_with("Hardware") {
                    if let Some(v) = line.split(':').nth(1) {
                        let s = v.trim().to_string();
                        if !s.is_empty() {
                            hw = s;
                        }
                    }
                }
                if implementer.is_empty() && line.starts_with("CPU implementer") {
                    if let Some(v) = line.split(':').nth(1) {
                        implementer = v.trim().to_string();
                    }
                }
            }
            if !hw.is_empty() {
                return hw;
            }
            if !implementer.is_empty() {
                let arch = std::process::Command::new("uname")
                    .arg("-m")
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|_| "aarch64".into());
                let vendor = match implementer.as_str() {
                    "0x41" => "ARM Cortex",
                    "0x51" => "Qualcomm",
                    "0x61" => "Apple Silicon",
                    "0x53" => "Samsung",
                    "0xc0" => "Ampere",
                    other => return format!("ARM64 (implementer {other})"),
                };
                return format!("{vendor} ({arch})");
            }
        }
        // Fallback: lscpu (available on most modern Linux distros)
        if let Ok(out) = std::process::Command::new("lscpu").output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if line.starts_with("Model name") {
                    if let Some(v) = line.split(':').nth(1) {
                        let s = v.trim().to_string();
                        if !s.is_empty() {
                            return s;
                        }
                    }
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(out) = std::process::Command::new("wmic")
            .args(["cpu", "get", "Name", "/value"])
            .output()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if let Some(v) = line.strip_prefix("Name=") {
                    return v.trim().to_string();
                }
            }
        }
    }
    "Unknown CPU".to_string()
}

/// GPU model string via platform-specific system commands.
/// Returns `None` when no GPU info can be found.
fn detect_gpu_name() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        // system_profiler lists all display adapters
        if let Ok(out) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let line = line.trim();
                if let Some(v) = line.strip_prefix("Chipset Model:") {
                    return Some(v.trim().to_string());
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        // Prefer nvidia-smi; fall back to lspci
        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() && out.status.success() {
                return Some(s.lines().next().unwrap_or(&s).trim().to_string());
            }
        }
        if let Ok(out) = std::process::Command::new("lspci").output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let l = line.to_lowercase();
                if l.contains("vga") || l.contains("3d") || l.contains("display") {
                    if let Some(idx) = line.find(':') {
                        if let Some(idx2) = line[idx + 1..].find(':') {
                            return Some(line[idx + 1 + idx2 + 1..].trim().to_string());
                        }
                    }
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(out) = std::process::Command::new("wmic")
            .args(["path", "win32_VideoController", "get", "Name", "/value"])
            .output()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if let Some(v) = line.strip_prefix("Name=") {
                    let name = v.trim().to_string();
                    if !name.is_empty() {
                        return Some(name);
                    }
                }
            }
        }
    }
    None
}

/// Check if the WGPU GPU backend can be initialised and execute a tensor op.
/// Uses `catch_unwind` so the binary never crashes if there is no usable GPU.
fn gpu_available() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = Device::wgpu(DeviceKind::DefaultDevice);
        let t: Tensor<1> = Tensor::from_data(TensorData::new(vec![1.0f32], [1]), &device);
        // Force sync: read back the value
        let _ = t.to_data().try_to_vec::<f32>().unwrap();
    }))
    .is_ok()
}

/// Sanitise a hardware name into a safe filename component.
///
/// "Apple M3 Pro"  →  "apple_m3_pro"
/// "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"  →  "intel_core_i9-9980hk_cpu"
fn sanitize(s: &str) -> String {
    let mut out = String::new();
    let mut prev_us = true; // collapse consecutive underscores / leading ones
    for c in s.chars() {
        let mapped = match c {
            'a'..='z' | '0'..='9' | '-' => c,
            'A'..='Z' => c.to_ascii_lowercase(),
            _ => '_',
        };
        if mapped == '_' {
            if !prev_us {
                out.push('_');
                prev_us = true;
            }
        } else {
            out.push(mapped);
            prev_us = false;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out.chars().take(50).collect()
}

// ─── Timing primitives ────────────────────────────────────────────────────────

/// Run `warmup` un-timed calls then `iters` timed calls.
/// Returns `(mean_µs, min_µs, max_µs)`.
fn measure<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> (f64, f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    times.sort();
    let mean = times.iter().sum::<Duration>() / times.len() as u32;
    let min = times[0];
    let max = *times.last().unwrap();
    (
        mean.as_nanos() as f64 / 1_000.0,
        min.as_nanos() as f64 / 1_000.0,
        max.as_nanos() as f64 / 1_000.0,
    )
}

/// A single measured data point.
#[derive(Clone)]
struct Entry {
    label: String,
    mean_us: f64,
    lo_us: f64,
    hi_us: f64,
}

/// A named group of entries (one per benchmark sub-function).
#[derive(Clone)]
struct Group {
    name: String,
    /// Short canonical identifier used to match CPU↔GPU pairs for comparison.
    /// E.g. both "model_forward (CPU / Flex)" and "model_forward (GPU / WGPU)"
    /// share the key "model_forward".
    key: String,
    entries: Vec<Entry>,
}

// ─── Model helpers ────────────────────────────────────────────────────────────

fn build_model(
    input: usize,
    hidden: &[usize],
    output: usize,
    device: &Device,
) -> ParametricUMAPModel {
    let cfg = ParametricUMAPModelConfigBuilder::default()
        .input_size(input)
        .hidden_sizes(hidden.to_vec())
        .output_size(output)
        .build()
        .unwrap();
    ParametricUMAPModel::new(&cfg, device)
}

// ─── CPU benchmark suite ──────────────────────────────────────────────────────

fn run_cpu_benchmarks() -> Vec<Group> {
    let device = Device::flex();
    let mut groups = Vec::new();

    // ── normalize_data ──
    {
        let mut entries = Vec::new();
        for (s, f) in [(100, 10), (500, 30), (1_000, 50), (5_000, 100)] {
            let iters = if s * f > 100_000 { 10 } else { 30 };
            let (mean, lo, hi) = measure(
                || {
                    let mut data: Vec<f64> = generate_test_data(s, f);
                    normalize_data(&mut data, s, f);
                },
                3,
                iters,
            );
            entries.push(Entry {
                label: format!("{s}×{f}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "normalize_data".into(),
            key: "normalize_data".into(),
            entries,
        });
    }

    // ── generate_test_data ──
    {
        let mut entries = Vec::new();
        for (s, f) in [(100, 10), (500, 30), (1_000, 50), (5_000, 100)] {
            let (mean, lo, hi) = measure(
                || {
                    let _: Vec<f32> = generate_test_data(s, f);
                },
                5,
                50,
            );
            entries.push(Entry {
                label: format!("{s}×{f}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "generate_test_data".into(),
            key: "generate_test_data".into(),
            entries,
        });
    }

    // ── tensor_convert  (Vec ↔ Tensor round-trip) ──
    {
        let mut entries = Vec::new();
        for (s, f) in [(100, 10), (500, 30), (1_000, 50)] {
            let data: Vec<f32> = generate_test_data(s, f);
            let (mean, lo, hi) = measure(
                || {
                    let t: Tensor<2> = convert_vector_to_tensor(data.clone(), s, f, &device);
                    let _: Vec<Vec<f64>> = convert_tensor_to_vector(t);
                },
                3,
                30,
            );
            entries.push(Entry {
                label: format!("{s}×{f}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "tensor_convert (Vec↔Tensor)".into(),
            key: "tensor_convert".into(),
            entries,
        });
    }

    // ── model_forward ──
    {
        let configs: &[(&str, usize, &[usize], usize, usize)] = &[
            ("16s×10f [32]→2", 16, &[32], 10, 2),
            ("64s×50f [64]→2", 64, &[64], 50, 2),
            ("128s×50f [128]→2", 128, &[128], 50, 2),
            ("64s×100f [128,64]→3", 64, &[128, 64], 100, 3),
            ("256s×100f [256,128]→2", 256, &[256, 128], 100, 2),
        ];
        let mut entries = Vec::new();
        for &(label, s, hidden, f, out) in configs {
            let model = build_model(f, hidden, out, &device);
            let data: Vec<f32> = generate_test_data(s, f);
            let input: Tensor<2> = Tensor::from_data(TensorData::new(data, [s, f]), &device);
            let iters = if s > 100 { 20 } else { 50 };
            let (mean, lo, hi) = measure(
                || {
                    let _ = model.forward(input.clone());
                },
                3,
                iters,
            );
            entries.push(Entry {
                label: label.into(),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "model_forward (CPU / Flex)".into(),
            key: "model_forward".into(),
            entries,
        });
    }

    // ── normalize_tensor (1-D min-max) ──
    {
        let mut entries = Vec::new();
        for n in [64usize, 512, 4_096, 32_768] {
            let data: Vec<f32> = generate_test_data(n, 1);
            let (mean, lo, hi) = measure(
                || {
                    let t: Tensor<1> =
                        Tensor::from_data(TensorData::new(data.clone(), [n]), &device);
                    let _ = normalize_tensor(t);
                },
                5,
                50,
            );
            entries.push(Entry {
                label: format!("n={n}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "normalize_tensor (1-D min-max)".into(),
            key: "normalize_tensor".into(),
            entries,
        });
    }

    // ── layer_normalize (2-D) ──
    {
        let mut entries = Vec::new();
        for (s, f) in [(32, 16), (128, 64), (512, 128), (1_000, 256)] {
            let data: Vec<f32> = generate_test_data(s, f);
            let t: Tensor<2> = Tensor::from_data(TensorData::new(data, [s, f]), &device);
            let (mean, lo, hi) = measure(
                || {
                    let _ = layer_normalize(t.clone());
                },
                3,
                30,
            );
            entries.push(Entry {
                label: format!("{s}×{f}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "layer_normalize (2-D)".into(),
            key: "layer_normalize".into(),
            entries,
        });
    }

    groups
}

// ─── GPU benchmark suite ──────────────────────────────────────────────────────

fn run_gpu_benchmarks() -> Vec<Group> {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    let mut groups = Vec::new();

    // ── model_forward on GPU ──
    {
        let configs: &[(&str, usize, &[usize], usize, usize)] = &[
            ("16s×10f [32]→2", 16, &[32], 10, 2),
            ("64s×50f [64]→2", 64, &[64], 50, 2),
            ("128s×50f [128]→2", 128, &[128], 50, 2),
            ("64s×100f [128,64]→3", 64, &[128, 64], 100, 3),
            ("256s×100f [256,128]→2", 256, &[256, 128], 100, 2),
            ("512s×100f [256,128]→2", 512, &[256, 128], 100, 2),
        ];
        let mut entries = Vec::new();
        for &(label, s, hidden, f, out) in configs {
            let model = build_model(f, hidden, out, &device);
            let data: Vec<f32> = generate_test_data(s, f);
            let input: Tensor<2> = Tensor::from_data(TensorData::new(data, [s, f]), &device);
            // force GPU init / JIT warmup
            let iters = 30;
            let (mean, lo, hi) = measure(
                || {
                    let out = model.forward(input.clone());
                    // read back to synchronise GPU before stopping the clock
                    let _ = out.to_data().try_to_vec::<f32>().unwrap();
                },
                5,
                iters,
            );
            entries.push(Entry {
                label: label.into(),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "model_forward (GPU / WGPU)".into(),
            key: "model_forward".into(),
            entries,
        });
    }

    // ── normalize_tensor on GPU ──
    {
        let mut entries = Vec::new();
        for n in [512usize, 4_096, 32_768, 262_144] {
            let data: Vec<f32> = generate_test_data(n, 1);
            let (mean, lo, hi) = measure(
                || {
                    let t: Tensor<1> =
                        Tensor::from_data(TensorData::new(data.clone(), [n]), &device);
                    let out = normalize_tensor(t);
                    let _ = out.to_data().try_to_vec::<f32>().unwrap();
                },
                5,
                30,
            );
            entries.push(Entry {
                label: format!("n={n}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "normalize_tensor (GPU / WGPU)".into(),
            key: "normalize_tensor".into(),
            entries,
        });
    }

    // ── layer_normalize on GPU ──
    {
        let mut entries = Vec::new();
        for (s, f) in [(128, 64), (512, 128), (1_000, 256), (4_000, 512)] {
            let data: Vec<f32> = generate_test_data(s, f);
            let (mean, lo, hi) = measure(
                || {
                    let t: Tensor<2> =
                        Tensor::from_data(TensorData::new(data.clone(), [s, f]), &device);
                    let out = layer_normalize(t);
                    let _ = out.to_data().try_to_vec::<f32>().unwrap();
                },
                5,
                20,
            );
            entries.push(Entry {
                label: format!("{s}×{f}"),
                mean_us: mean,
                lo_us: lo,
                hi_us: hi,
            });
        }
        groups.push(Group {
            name: "layer_normalize (GPU / WGPU)".into(),
            key: "layer_normalize".into(),
            entries,
        });
    }

    groups
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

fn fmt_us(us: f64) -> String {
    if us < 1.0 {
        format!("{:.2} ns", us * 1_000.0)
    } else if us < 1_000.0 {
        format!("{:.2} µs", us)
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1_000.0)
    } else {
        format!("{:.3} s", us / 1_000_000.0)
    }
}

// ─── Markdown generation ──────────────────────────────────────────────────────

fn write_markdown(path: &str, hw_label: &str, hw_detail: &str, groups: &[Group]) {
    let mut md = String::new();
    md.push_str(&format!("# fast-umap — Benchmark Results\n\n"));
    md.push_str(&format!("> **Hardware:** {hw_label}  \n"));
    md.push_str(&format!("> **Detail:** {hw_detail}  \n"));
    md.push_str("> **Backend:** burn 0.22.0-pre.4 · Flex / WGPU  \n");
    md.push_str("> **Reproduce:** `cargo run --release --bin bench_report`\n\n");
    md.push_str("![chart](benchmark_results.svg)\n\n");
    md.push_str("All times are **[min, mean, max]** over multiple timed iterations.\n\n");
    md.push_str("---\n\n");

    for g in groups {
        md.push_str(&format!("## `{}`\n\n", g.name));
        md.push_str("| Input | Min | **Mean** | Max |\n");
        md.push_str("|-------|-----|----------|-----|\n");
        for e in &g.entries {
            md.push_str(&format!(
                "| `{}` | {} | **{}** | {} |\n",
                e.label,
                fmt_us(e.lo_us),
                fmt_us(e.mean_us),
                fmt_us(e.hi_us)
            ));
        }
        md.push('\n');
    }

    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &md).unwrap();
    println!("  Wrote {path}");
}

// ─── SVG generation ───────────────────────────────────────────────────────────

fn write_svg(path: &str, hw_label: &str, groups: &[Group]) {
    const COLS: usize = 2;
    const SVG_W: f64 = 960.0;
    const TITLE_H: f64 = 48.0;
    const PANEL_W: f64 = SVG_W / COLS as f64; // 480
    const PANEL_H: f64 = 210.0;
    const CHART_LEFT: f64 = 170.0;
    const CHART_RIGHT: f64 = 30.0;
    const CHART_TOP: f64 = 36.0;
    const CHART_BOTTOM: f64 = 34.0;

    let rows = (groups.len() + COLS - 1) / COLS;
    let svg_h = TITLE_H + rows as f64 * PANEL_H + 20.0;

    // Palette: avoid raw strings with hex — just plain &str literals.
    let colours = [
        "4C78A8", "F58518", "E45756", "72B7B2", "54A24B", "B279A2", "FF9DA7", "9D755D",
    ];

    let log_ticks = |lo: f64, hi: f64| -> Vec<f64> {
        let mut ticks = Vec::new();
        let mut p = lo.log10().floor() as i32;
        while 10f64.powi(p) <= hi * 1.1 {
            for &m in &[1.0, 2.0, 5.0] {
                let v = m * 10f64.powi(p);
                if v >= lo * 0.8 && v <= hi * 1.2 {
                    ticks.push(v);
                }
            }
            p += 1;
        }
        ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ticks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        ticks
    };

    // Use a Vec<u8> writer to avoid any raw-string literal quoting issues.
    let mut out: Vec<u8> = Vec::new();
    let w = &mut out;

    // SVG root + background
    writeln!(w, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{SVG_W}\" height=\"{svg_h}\" font-family=\"monospace,Arial,sans-serif\">").unwrap();
    writeln!(
        w,
        "<rect width=\"{SVG_W}\" height=\"{svg_h}\" fill=\"#FAFAFA\"/>"
    )
    .unwrap();
    writeln!(w, "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-size=\"15\" font-weight=\"bold\" fill=\"#222\">{hw_label} \u{2014} fast-umap Benchmark Results</text>",
        SVG_W / 2.0).unwrap();

    for (idx, g) in groups.iter().enumerate() {
        let col = (idx % COLS) as f64;
        let row = (idx / COLS) as f64;
        let px = col * PANEL_W;
        let py = TITLE_H + row * PANEL_H;
        let hex = colours[idx % colours.len()]; // e.g. "4C78A8"
        let colour = format!("#{hex}");

        let vals: Vec<f64> = g.entries.iter().map(|e| e.mean_us).collect();
        let lo_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if lo_val <= 0.0 || hi_val <= 0.0 {
            continue;
        }

        let chart_w = PANEL_W - CHART_LEFT - CHART_RIGHT;
        let chart_h = PANEL_H - CHART_TOP - CHART_BOTTOM;
        let bar_count = g.entries.len() as f64;
        let bar_gap = 5.0_f64;
        let bar_h = ((chart_h - bar_gap * (bar_count - 1.0)) / bar_count).max(10.0);

        let lo_log = (lo_val * 0.55).log10();
        let hi_log = (hi_val * 1.45).log10();
        let x_of = |v: f64| -> f64 { (v.log10() - lo_log) / (hi_log - lo_log) * chart_w };

        let cx = px + CHART_LEFT;
        let cy = py + CHART_TOP;

        // panel background
        writeln!(w,
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"6\" fill=\"white\" stroke=\"#DDD\" stroke-width=\"1\"/>",
            px + 4.0, py + 2.0, PANEL_W - 8.0, PANEL_H - 4.0
        ).unwrap();

        // panel title
        writeln!(w,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"10.5\" font-weight=\"bold\" fill=\"#333\">{}</text>",
            px + PANEL_W / 2.0, py + 20.0, g.name
        ).unwrap();

        // x-axis ticks + grid lines
        let ticks = log_ticks(lo_val * 0.55, hi_val * 1.45);
        for t in &ticks {
            let tx = cx + x_of(*t);
            writeln!(w,
                "<line x1=\"{tx:.1}\" y1=\"{cy:.1}\" x2=\"{tx:.1}\" y2=\"{:.1}\" stroke=\"#EEE\" stroke-width=\"1\"/>",
                cy + chart_h
            ).unwrap();
            writeln!(w,
                "<text x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"8\" fill=\"#888\">{}</text>",
                cy + chart_h + 12.0, fmt_us(*t)
            ).unwrap();
        }

        // bars + whiskers + labels
        for (i, e) in g.entries.iter().enumerate() {
            let by = cy + i as f64 * (bar_h + bar_gap);
            let bw = x_of(e.mean_us);
            let lo_x = cx + x_of(e.lo_us.max(lo_val * 0.56));
            let hi_x = cx + x_of(e.hi_us);
            let mid_y = by + bar_h / 2.0;
            let lbl_y = mid_y + 3.5;

            // filled bar
            writeln!(w,
                "<rect x=\"{cx:.1}\" y=\"{by:.1}\" width=\"{bw:.1}\" height=\"{bar_h:.1}\" rx=\"2\" fill=\"{colour}\" opacity=\"0.80\"/>",
            ).unwrap();

            // error whisker (horizontal span)
            writeln!(w,
                "<line x1=\"{lo_x:.1}\" y1=\"{mid_y:.1}\" x2=\"{hi_x:.1}\" y2=\"{mid_y:.1}\" stroke=\"#555\" stroke-width=\"1.2\"/>",
            ).unwrap();
            // left cap
            writeln!(w,
                "<line x1=\"{lo_x:.1}\" y1=\"{:.1}\" x2=\"{lo_x:.1}\" y2=\"{:.1}\" stroke=\"#555\" stroke-width=\"1.2\"/>",
                mid_y - 3.0, mid_y + 3.0
            ).unwrap();
            // right cap
            writeln!(w,
                "<line x1=\"{hi_x:.1}\" y1=\"{:.1}\" x2=\"{hi_x:.1}\" y2=\"{:.1}\" stroke=\"#555\" stroke-width=\"1.2\"/>",
                mid_y - 3.0, mid_y + 3.0
            ).unwrap();

            // y-axis label (left of bar)
            writeln!(w,
                "<text x=\"{:.1}\" y=\"{lbl_y:.1}\" text-anchor=\"end\" font-size=\"8.5\" fill=\"#333\">{}</text>",
                cx - 4.0, e.label
            ).unwrap();

            // value label (right of bar)
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{lbl_y:.1}\" font-size=\"8\" fill=\"#555\">{}</text>",
                cx + bw + 4.0,
                fmt_us(e.mean_us)
            )
            .unwrap();
        }

        // x-axis baseline
        writeln!(w,
            "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#BBB\" stroke-width=\"1\"/>",
            cy + chart_h, cx + chart_w, cy + chart_h
        ).unwrap();
    }

    writeln!(w, "</svg>").unwrap();

    let s = String::from_utf8(out).expect("SVG is valid UTF-8");
    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &s).unwrap();
    println!("  Wrote {path}");
}

// ─── CPU vs GPU comparison ────────────────────────────────────────────────────

/// A matched (cpu_entry, gpu_entry) pair for the same input size.
#[derive(Clone)]
struct Pair {
    label: String,
    cpu_us: f64,
    gpu_us: f64,
}

impl Pair {
    fn speedup(&self) -> f64 {
        self.cpu_us / self.gpu_us
    }
}

/// All matched pairs within one benchmark group.
#[derive(Clone)]
struct CompGroup {
    key: String,
    cpu_name: String,
    gpu_name: String,
    pairs: Vec<Pair>,
}

/// Align CPU and GPU groups by key, then align entries by label.
fn build_comparison(cpu: &[Group], gpu: &[Group]) -> Vec<CompGroup> {
    let mut result = Vec::new();
    for gpu_g in gpu {
        if let Some(cpu_g) = cpu.iter().find(|g| g.key == gpu_g.key) {
            let pairs: Vec<Pair> = gpu_g
                .entries
                .iter()
                .filter_map(|ge| {
                    cpu_g
                        .entries
                        .iter()
                        .find(|ce| ce.label == ge.label)
                        .map(|ce| Pair {
                            label: ge.label.clone(),
                            cpu_us: ce.mean_us,
                            gpu_us: ge.mean_us,
                        })
                })
                .collect();
            if !pairs.is_empty() {
                result.push(CompGroup {
                    key: gpu_g.key.clone(),
                    cpu_name: cpu_g.name.clone(),
                    gpu_name: gpu_g.name.clone(),
                    pairs,
                });
            }
        }
    }
    result
}

// ── Comparison Markdown ───────────────────────────────────────────────────────

fn write_comparison_markdown(path: &str, cpu_hw: &str, gpu_hw: &str, comp: &[CompGroup]) {
    let mut md = String::new();
    md.push_str("# fast-umap — CPU vs GPU Comparison\n\n");
    md.push_str(&format!("> **CPU:** {cpu_hw}  \n"));
    md.push_str(&format!("> **GPU:** {gpu_hw} (WGPU / Metal)  \n"));
    md.push_str("> **Backend:** burn 0.22.0-pre.4 · Flex vs WGPU  \n");
    md.push_str("> **Reproduce:** `cargo run --release --bin bench_report`\n\n");
    md.push_str("![chart](comparison.svg)\n\n");
    md.push_str("Speedup = CPU mean ÷ GPU mean.  \n");
    md.push_str("> 1× = GPU faster, < 1× = CPU faster (e.g. small tensors where dispatch overhead dominates).\n\n");
    md.push_str("---\n\n");

    for cg in comp {
        md.push_str(&format!("## `{}`\n\n", cg.key));
        md.push_str(&format!(
            "| Input | {} | {} | Speedup |\n",
            cg.cpu_name, cg.gpu_name
        ));
        md.push_str("|-------|");
        md.push_str(&"-".repeat(cg.cpu_name.len() + 2));
        md.push_str("|");
        md.push_str(&"-".repeat(cg.gpu_name.len() + 2));
        md.push_str("|----------|\n");
        for p in &cg.pairs {
            let s = p.speedup();
            let speedup_cell = if s >= 1.5 {
                format!("**{s:.2}×** 🚀")
            } else if s >= 1.0 {
                format!("**{s:.2}×**")
            } else {
                format!("{s:.2}× *(CPU faster)*")
            };
            md.push_str(&format!(
                "| `{}` | {} | {} | {} |\n",
                p.label,
                fmt_us(p.cpu_us),
                fmt_us(p.gpu_us),
                speedup_cell
            ));
        }
        md.push('\n');
    }

    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &md).unwrap();
    println!("  Wrote {path}");
}

// ── Comparison SVG ────────────────────────────────────────────────────────────

fn write_comparison_svg(path: &str, cpu_hw: &str, gpu_hw: &str, comp: &[CompGroup]) {
    if comp.is_empty() {
        return;
    }

    // Layout constants
    const SVG_W: f64 = 960.0;
    const TITLE_H: f64 = 52.0;
    const LEGEND_H: f64 = 22.0;
    const CHART_LEFT: f64 = 185.0;
    const CHART_RIGHT: f64 = 72.0; // room for speedup label
    const CHART_TOP: f64 = 36.0; // panel title
    const CHART_BOTTOM: f64 = 32.0; // x-axis ticks
    const BAR_H: f64 = 11.0; // height of a single bar
    const PAIR_GAP: f64 = 2.0; // gap between CPU and GPU bar in same pair
    const ENTRY_GAP: f64 = 10.0; // gap between different input entries
    const PANEL_PAD: f64 = 8.0; // padding inside panel border

    let cpu_colour = "4C78A8"; // blue  — CPU
    let gpu_colour = "F58518"; // orange — GPU

    // Compute per-group panel heights
    let chart_w = SVG_W - CHART_LEFT - CHART_RIGHT;
    struct Panel<'a> {
        cg: &'a CompGroup,
        chart_h: f64,
        panel_h: f64,
    }
    let panels: Vec<Panel> = comp
        .iter()
        .map(|cg| {
            let n = cg.pairs.len() as f64;
            let ch = n * (2.0 * BAR_H + PAIR_GAP) + (n - 1.0) * ENTRY_GAP;
            let ph = CHART_TOP + LEGEND_H + ch + CHART_BOTTOM + 2.0 * PANEL_PAD;
            Panel {
                cg,
                chart_h: ch,
                panel_h: ph,
            }
        })
        .collect();

    let total_h: f64 = TITLE_H + panels.iter().map(|p| p.panel_h + 10.0).sum::<f64>() + 10.0;

    let log_ticks = |lo: f64, hi: f64| -> Vec<f64> {
        let mut ticks = Vec::new();
        let mut p = lo.log10().floor() as i32;
        while 10f64.powi(p) <= hi * 1.1 {
            for &m in &[1.0, 2.0, 5.0] {
                let v = m * 10f64.powi(p);
                if v >= lo * 0.8 && v <= hi * 1.2 {
                    ticks.push(v);
                }
            }
            p += 1;
        }
        ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ticks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        ticks
    };

    let mut out: Vec<u8> = Vec::new();
    let w = &mut out;

    // SVG root + background
    writeln!(w, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{SVG_W}\" height=\"{total_h:.1}\" font-family=\"monospace,Arial,sans-serif\">").unwrap();
    writeln!(
        w,
        "<rect width=\"{SVG_W}\" height=\"{total_h:.1}\" fill=\"#FAFAFA\"/>"
    )
    .unwrap();

    // Main title
    writeln!(w,
        "<text x=\"{:.1}\" y=\"22\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\" fill=\"#222\">{cpu_hw} vs {gpu_hw} \u{2014} CPU/GPU Comparison</text>",
        SVG_W / 2.0
    ).unwrap();

    // Global legend
    let lx = SVG_W / 2.0 - 100.0;
    writeln!(
        w,
        "<rect x=\"{lx:.1}\" y=\"30\" width=\"12\" height=\"10\" fill=\"#{cpu_colour}\" rx=\"2\"/>"
    )
    .unwrap();
    writeln!(
        w,
        "<text x=\"{:.1}\" y=\"39\" font-size=\"9\" fill=\"#333\">CPU (Flex)</text>",
        lx + 15.0
    )
    .unwrap();
    writeln!(
        w,
        "<rect x=\"{:.1}\" y=\"30\" width=\"12\" height=\"10\" fill=\"#{gpu_colour}\" rx=\"2\"/>",
        lx + 105.0
    )
    .unwrap();
    writeln!(
        w,
        "<text x=\"{:.1}\" y=\"39\" font-size=\"9\" fill=\"#333\">GPU (WGPU/Metal)</text>",
        lx + 120.0
    )
    .unwrap();

    let mut cur_y = TITLE_H;

    for panel in &panels {
        let cg = panel.cg;
        let py = cur_y;
        let ph = panel.panel_h;

        // All values for shared log scale
        let all_vals: Vec<f64> = cg.pairs.iter().flat_map(|p| [p.cpu_us, p.gpu_us]).collect();
        let lo_val = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi_val = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let lo_log = (lo_val * 0.45).log10();
        let hi_log = (hi_val * 1.55).log10();
        let x_of = |v: f64| -> f64 { (v.log10() - lo_log) / (hi_log - lo_log) * chart_w };

        let cx = CHART_LEFT;
        let cy = py + PANEL_PAD + CHART_TOP + LEGEND_H;

        // Panel background
        writeln!(w,
            "<rect x=\"4\" y=\"{py:.1}\" width=\"{:.1}\" height=\"{ph:.1}\" rx=\"6\" fill=\"white\" stroke=\"#DDD\" stroke-width=\"1\"/>",
            SVG_W - 8.0
        ).unwrap();

        // Panel title
        writeln!(w,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"11\" font-weight=\"bold\" fill=\"#333\">{}</text>",
            SVG_W / 2.0,
            py + PANEL_PAD + 20.0,
            cg.key
        ).unwrap();

        // X-axis ticks + grid lines
        let ticks = log_ticks(lo_val * 0.45, hi_val * 1.55);
        for t in &ticks {
            let tx = cx + x_of(*t);
            writeln!(w,
                "<line x1=\"{tx:.1}\" y1=\"{cy:.1}\" x2=\"{tx:.1}\" y2=\"{:.1}\" stroke=\"#EBEBEB\" stroke-width=\"1\"/>",
                cy + panel.chart_h
            ).unwrap();
            writeln!(w,
                "<text x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"8\" fill=\"#999\">{}</text>",
                cy + panel.chart_h + 12.0, fmt_us(*t)
            ).unwrap();
        }

        // Paired bars
        for (i, p) in cg.pairs.iter().enumerate() {
            let pair_h = 2.0 * BAR_H + PAIR_GAP;
            let by = cy + i as f64 * (pair_h + ENTRY_GAP);

            // ── CPU bar ──
            let cpu_bw = x_of(p.cpu_us);
            let cpu_lbl_y = by + BAR_H / 2.0 + 3.5;
            writeln!(w,
                "<rect x=\"{cx:.1}\" y=\"{by:.1}\" width=\"{cpu_bw:.1}\" height=\"{BAR_H:.1}\" rx=\"2\" fill=\"#{cpu_colour}\" opacity=\"0.85\"/>",
            ).unwrap();
            // value at right end
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{cpu_lbl_y:.1}\" font-size=\"7.5\" fill=\"#555\">{}</text>",
                cx + cpu_bw + 4.0,
                fmt_us(p.cpu_us)
            )
            .unwrap();

            // ── GPU bar ──
            let gpu_by = by + BAR_H + PAIR_GAP;
            let gpu_bw = x_of(p.gpu_us);
            let gpu_lbl_y = gpu_by + BAR_H / 2.0 + 3.5;
            writeln!(w,
                "<rect x=\"{cx:.1}\" y=\"{gpu_by:.1}\" width=\"{gpu_bw:.1}\" height=\"{BAR_H:.1}\" rx=\"2\" fill=\"#{gpu_colour}\" opacity=\"0.85\"/>",
            ).unwrap();
            writeln!(
                w,
                "<text x=\"{:.1}\" y=\"{gpu_lbl_y:.1}\" font-size=\"7.5\" fill=\"#555\">{}</text>",
                cx + gpu_bw + 4.0,
                fmt_us(p.gpu_us)
            )
            .unwrap();

            // ── Speedup label (rightmost edge) ──
            let speedup = p.speedup();
            let (sp_text, sp_fill) = if speedup >= 2.0 {
                (format!("{speedup:.1}\u{00D7} \u{1F680}"), "#157a1a")
            } else if speedup >= 1.0 {
                (format!("{speedup:.2}\u{00D7}"), "#2a7ec7")
            } else {
                (format!("{speedup:.2}\u{00D7}"), "#888")
            };
            let sp_x = SVG_W - CHART_RIGHT + 4.0;
            let sp_y = by + pair_h / 2.0 + 3.5;
            writeln!(w,
                "<text x=\"{sp_x:.1}\" y=\"{sp_y:.1}\" font-size=\"8.5\" font-weight=\"bold\" fill=\"{sp_fill}\">{sp_text}</text>",
            ).unwrap();

            // ── Input label (left of chart) ──
            writeln!(w,
                "<text x=\"{:.1}\" y=\"{sp_y:.1}\" text-anchor=\"end\" font-size=\"8.5\" fill=\"#333\">{}</text>",
                cx - 4.0, p.label
            ).unwrap();
        }

        // X-axis baseline
        writeln!(w,
            "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#BBB\" stroke-width=\"1\"/>",
            cy + panel.chart_h, cx + chart_w, cy + panel.chart_h
        ).unwrap();

        cur_y += ph + 10.0;
    }

    writeln!(w, "</svg>").unwrap();

    let svg_str = String::from_utf8(out).expect("SVG is valid UTF-8");
    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
    fs::write(path, &svg_str).unwrap();
    println!("  Wrote {path}");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── detect hardware ──────────────────────────────────────────────────────
    let cpu_name = detect_cpu();
    let cpu_safe = sanitize(&cpu_name);

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║           fast-umap benchmark reporter                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    println!("CPU : {cpu_name}");

    let has_gpu = gpu_available();
    let gpu_name = if has_gpu {
        detect_gpu_name().unwrap_or_else(|| "GPU (WGPU)".to_string())
    } else {
        String::new()
    };

    if has_gpu {
        println!("GPU : {gpu_name} (WGPU/Metal/Vulkan)\n");
    } else {
        println!("GPU : not available or not usable — skipping GPU benchmarks\n");
    }

    let results_dir = "benches/results";

    // ── CPU benchmarks ───────────────────────────────────────────────────────
    println!("━━━  Running CPU benchmarks (Flex backend)  ━━━");
    let cpu_groups = run_cpu_benchmarks();

    // Print a quick summary to stdout
    for g in &cpu_groups {
        println!("\n  [{}]", g.name);
        for e in &g.entries {
            println!("    {:40} {}", e.label, fmt_us(e.mean_us));
        }
    }

    println!("\n━━━  Saving CPU results  ━━━");

    // Hardware-tagged file
    let cpu_hw_label = format!("CPU: {cpu_name}");
    let cpu_md = format!("{results_dir}/cpu_{cpu_safe}.md");
    let cpu_svg = format!("{results_dir}/cpu_{cpu_safe}.svg");
    write_markdown(&cpu_md, &cpu_hw_label, &cpu_name, &cpu_groups);
    write_svg(&cpu_svg, &cpu_hw_label, &cpu_groups);

    // Generic "latest" symlinks (plain copy)
    let generic_md = format!("{results_dir}/benchmark_results.md");
    let generic_svg = format!("{results_dir}/benchmark_results.svg");
    fs::copy(&cpu_md, &generic_md).unwrap();
    println!("  Wrote {generic_md} (copy of latest CPU run)");
    // Overwrite the SVG so it matches the same run
    write_svg(&generic_svg, &cpu_hw_label, &cpu_groups);

    // ── GPU benchmarks + comparison ──────────────────────────────────────────
    if has_gpu {
        println!("\n━━━  Running GPU benchmarks (WGPU backend)  ━━━");
        let gpu_groups = run_gpu_benchmarks();

        for g in &gpu_groups {
            println!("\n  [{}]", g.name);
            for e in &g.entries {
                println!("    {:40} {}", e.label, fmt_us(e.mean_us));
            }
        }

        println!("\n━━━  Saving GPU results  ━━━");
        let gpu_safe = sanitize(&gpu_name);
        let gpu_hw_label = format!("GPU: {gpu_name}");
        let gpu_md = format!("{results_dir}/gpu_{gpu_safe}.md");
        let gpu_svg = format!("{results_dir}/gpu_{gpu_safe}.svg");
        write_markdown(&gpu_md, &gpu_hw_label, &gpu_name, &gpu_groups);
        write_svg(&gpu_svg, &gpu_hw_label, &gpu_groups);

        // ── CPU vs GPU comparison ────────────────────────────────────────────
        let comp = build_comparison(&cpu_groups, &gpu_groups);
        if !comp.is_empty() {
            println!("\n━━━  CPU vs GPU comparison  ━━━");

            // Print a quick speedup table to stdout
            for cg in &comp {
                println!("\n  [{}]", cg.key);
                println!(
                    "    {:38}  {:>10}  {:>10}  {:>8}",
                    "input", "CPU", "GPU", "speedup"
                );
                println!("    {}", "─".repeat(72));
                for p in &cg.pairs {
                    let s = p.speedup();
                    let icon = if s >= 2.0 {
                        " 🚀"
                    } else if s >= 1.0 {
                        "  ✓"
                    } else {
                        " ↓"
                    };
                    println!(
                        "    {:38}  {:>10}  {:>10}  {:>6.2}×{}",
                        p.label,
                        fmt_us(p.cpu_us),
                        fmt_us(p.gpu_us),
                        s,
                        icon,
                    );
                }
            }

            println!("\n━━━  Saving comparison results  ━━━");
            // Use the CPU chip name for the filename (both are the same chip on Apple Silicon)
            let comp_slug = cpu_safe.clone();
            let comp_md = format!("{results_dir}/comparison_{comp_slug}.md");
            let comp_svg = format!("{results_dir}/comparison_{comp_slug}.svg");
            // The SVG is written next to the md, but the md references comparison.svg
            // (relative, same dir) — write the SVG using the same base name.
            let comp_svg_rel = format!("{results_dir}/comparison.svg");

            write_comparison_markdown(&comp_md, &cpu_name, &gpu_name, &comp);
            write_comparison_svg(&comp_svg_rel, &cpu_name, &gpu_name, &comp);

            // Also write a hardware-tagged copy of the SVG
            fs::copy(&comp_svg_rel, &comp_svg).unwrap();
            println!("  Wrote {comp_svg} (copy)");
        }
    }

    println!("\n✓  Done.  Results are in {results_dir}/");
    println!();
    let rerun = "cargo run --release --bin bench_report";
    let rerun_full = "./bench.sh --criterion";
    // Box is 62 chars wide (between the ╔/╚ corners)
    let w = 62usize;
    let line = |s: &str| {
        let pad = w.saturating_sub(s.len() + 2);
        println!("║ {s}{} ║", " ".repeat(pad));
    };
    println!("╔{}╗", "═".repeat(w));
    line("To rerun:");
    line("");
    line("  # hardware report (CPU + GPU, ~1 min):");
    line(&format!("    {rerun}"));
    line("");
    line("  # + Criterion statistical suite (~5 min):");
    line(&format!("    {rerun_full}"));
    println!("╚{}╝", "═".repeat(w));
    println!();
}
