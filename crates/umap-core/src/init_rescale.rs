use crate::{InitRescale, Result, UMAPConfig, UmapError};

#[derive(Clone, Copy)]
struct EdgeDistance {
    distance: f32,
    weight: f32,
    ordinal: usize,
}

// Live host buffers: export has 12*C readback bytes plus compact directed edges;
// spectral retains its sparse graph, solver scratch and embedding (graph is dropped
// before rescaling); robust has 4*n*d embedding bytes, 8*n scratch and 8*d spans;
// graph-edge has embedding, compact edges and E * size_of::<EdgeDistance>() records;
// upload retains only the final embedding. All sums below are checked before allocation.
pub(crate) fn check_extents(n: usize, d: usize, edges: usize, mode: InitRescale) -> Result<()> {
    let shape = n
        .checked_mul(d)
        .ok_or(UmapError::Overflow("embedding shape"))?;
    let embedding = shape
        .checked_mul(size_of::<f32>())
        .ok_or(UmapError::Overflow("embedding bytes"))?;
    let scratch = match mode {
        InitRescale::GraphEdge => edges
            .checked_mul(size_of::<EdgeDistance>())
            .ok_or(UmapError::Overflow("edge distance scratch"))?,
        _ => n
            .checked_mul(size_of::<f64>())
            .and_then(|x| {
                d.checked_mul(size_of::<f64>())
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or(UmapError::Overflow("rescale scratch"))?,
    };
    let directed = edges
        .checked_mul(2 * size_of::<usize>() + size_of::<f32>())
        .ok_or(UmapError::Overflow("directed edge bytes"))?;
    embedding
        .checked_add(scratch)
        .and_then(|x| x.checked_add(directed))
        .ok_or(UmapError::Overflow("initialization live buffers"))?;
    Ok(())
}

fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|e| UmapError::Memory(e.to_string()))?;
    Ok(result)
}

fn numerical() -> UmapError {
    UmapError::Numerical("non-finite initialization rescaling intermediate or coordinate".into())
}

fn apply(values: &mut [f32], multiplier: f64) -> Result<()> {
    let multiplier = multiplier as f32;
    if !multiplier.is_finite() {
        return Err(numerical());
    }
    for value in values {
        *value *= multiplier;
        if !value.is_finite() {
            return Err(numerical());
        }
    }
    Ok(())
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64)
}

pub(crate) fn rescale(
    values: &mut [f32],
    n: usize,
    d: usize,
    heads: &[usize],
    tails: &[usize],
    weights: &[f32],
    config: &UMAPConfig,
) -> Result<()> {
    check_extents(n, d, heads.len(), config.init_rescale)?;
    if n == 0
        || d == 0
        || values.len()
            != n.checked_mul(d)
                .ok_or(UmapError::Overflow("embedding shape"))?
    {
        return Err(UmapError::InvalidInput(
            "embedding storage does not match its row-major shape".into(),
        ));
    }
    for axis in 0..d {
        let mut sum = 0.0_f32;
        for row in 0..n {
            let v = values[row * d + axis];
            if !v.is_finite() {
                return Err(numerical());
            }
            sum += v;
            if !sum.is_finite() {
                return Err(numerical());
            }
        }
        let mean = sum / n as f32;
        for row in 0..n {
            let v = &mut values[row * d + axis];
            *v -= mean;
            if !v.is_finite() {
                return Err(numerical());
            }
        }
    }
    if config.init_rescale == InitRescale::GraphEdge {
        if heads.is_empty() {
            return Ok(());
        }
        let mut records = reserve::<EdgeDistance>(heads.len())?;
        let mut total = 0.0_f64;
        for (ordinal, ((&head, &tail), &weight)) in heads.iter().zip(tails).zip(weights).enumerate()
        {
            let mut squared = 0.0_f32;
            for axis in 0..d {
                let difference = values[head * d + axis] - values[tail * d + axis];
                squared += difference * difference;
            }
            let distance = squared.sqrt();
            if !distance.is_finite() {
                return Err(numerical());
            }
            total += f64::from(weight);
            if !total.is_finite() {
                return Err(numerical());
            }
            records.push(EdgeDistance {
                distance,
                weight,
                ordinal,
            });
        }
        records.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.ordinal.cmp(&b.ordinal))
        });
        let mut cumulative = 0.0_f64;
        let mut median = 0.0_f32;
        for record in records {
            cumulative += f64::from(record.weight);
            if cumulative >= total / 2.0 {
                median = record.distance;
                break;
            }
        }
        let spread = if median == 0.0 {
            1.0
        } else {
            f64::from(median)
        };
        return apply(values, f64::from(config.init_scale_factor) / spread);
    }
    let mut scratch = reserve::<f64>(n)?;
    let mut spans = reserve::<f64>(d)?;
    for axis in 0..d {
        scratch.clear();
        scratch.extend((0..n).map(|row| f64::from(values[row * d + axis])));
        scratch.sort_unstable_by(f64::total_cmp);
        let span = quantile(&scratch, 0.95) - quantile(&scratch, 0.05);
        spans.push(if span == 0.0 { 1.0 } else { span });
    }
    let target = (n as f64).log10() * 3.0 * f64::from(config.repulsion_strength).ln_1p()
        / std::f64::consts::LN_2
        * f64::from(config.init_scale_factor);
    match config.init_rescale {
        InitRescale::Isotropic => {
            let geometric = (spans.iter().map(|span| span.ln()).sum::<f64>() / d as f64).exp();
            apply(values, target / geometric)?;
        }
        InitRescale::PerAxis => {
            for axis in 0..d {
                let multiplier = (target / spans[axis]) as f32;
                if !multiplier.is_finite() {
                    return Err(numerical());
                }
                for row in 0..n {
                    let value = &mut values[row * d + axis];
                    *value *= multiplier;
                    if !value.is_finite() {
                        return Err(numerical());
                    }
                }
            }
        }
        InitRescale::GraphEdge => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-5 + 1e-5 * b.abs(), "{a} != {b}");
    }
    fn cfg(mode: InitRescale, factor: f32) -> UMAPConfig {
        UMAPConfig {
            init: crate::Init::Spectral,
            optimizer: crate::Optimizer::Compatibility,
            init_rescale: mode,
            init_scale_factor: factor,
            ..UMAPConfig::default()
        }
    }
    fn run(values: &mut [f32], n: usize, d: usize, mode: InitRescale, factor: f32) {
        rescale(values, n, d, &[], &[], &[], &cfg(mode, factor)).unwrap();
    }

    #[test]
    fn linear_quantiles_and_row_order() {
        close(quantile(&[0.0, 10.0], 0.05) as f32, 0.5);
        close(quantile(&[0.0, 10.0, 30.0], 0.95) as f32, 28.0);
        let samples: Vec<f64> = (0..21).map(|x| (x / 2) as f64).collect();
        close(quantile(&samples, 0.05) as f32, 0.0);
        close(quantile(&samples, 0.95) as f32, 9.0);
        let mut values = [10.0, 8.0, 0.0, 8.0, 5.0, 8.0];
        run(&mut values, 3, 2, InitRescale::PerAxis, 1.0);
        let t = 3.0 * 3.0_f32.log10();
        close(values[0], t / 1.8);
        close(values[2], -t / 1.8);
        close(values[4], 0.0);
        close(values[1], 0.0);
        close(values[3], 0.0);
        close(values[5], 0.0);
    }

    #[test]
    fn anisotropy_translation_factor_and_gamma() {
        let original = [0.0, 0.0, 2.0, 20.0, 4.0, 40.0];
        let mut iso = original;
        run(&mut iso, 3, 2, InitRescale::Isotropic, 1.0);
        close(iso[1], iso[0] * 10.0);
        let t = 3.0 * 3.0_f32.log10();
        close(iso[0], -2.0 * t / (3.6_f32 * 36.0).sqrt());
        let mut axis = original;
        run(&mut axis, 3, 2, InitRescale::PerAxis, 1.0);
        close(axis[0], axis[1]);
        close(axis[4], axis[5]);
        let mut translated = [100.0, -3.0, 102.0, 17.0, 104.0, 37.0];
        run(&mut translated, 3, 2, InitRescale::Isotropic, 2.0);
        for (a, b) in translated.iter().zip(iso) {
            close(*a, 2.0 * b);
        }
        let mut zero = original;
        let mut config = cfg(InitRescale::Isotropic, 1.0);
        config.repulsion_strength = 0.0;
        rescale(&mut zero, 3, 2, &[], &[], &[], &config).unwrap();
        assert!(zero.iter().all(|x| *x == 0.0));
        config.repulsion_strength = 1e-6;
        let mut tiny = original;
        rescale(&mut tiny, 3, 2, &[], &[], &[], &config).unwrap();
        assert!(tiny[0] < 0.0 && tiny[0].abs() < 1e-5);
        let mut constant = [4.0; 6];
        run(&mut constant, 3, 2, InitRescale::PerAxis, 1.0);
        assert_eq!(constant, [0.0; 6]);
        let mut one = [9.0, -4.0];
        run(&mut one, 1, 2, InitRescale::Isotropic, 1.0);
        assert_eq!(one, [0.0; 2]);
    }

    #[test]
    fn directed_weighted_median_boundaries_and_zero() {
        let edges = [0, 0, 0, 1];
        let tails = [1, 2, 3, 0];
        let mut values = [0.0, 1.0, 3.0, 10.0];
        let mut config = cfg(InitRescale::GraphEdge, 2.0);
        config.repulsion_strength = 0.0;
        rescale(
            &mut values,
            4,
            1,
            &edges,
            &tails,
            &[1.0, 5.0, 1.0, 1.0],
            &config,
        )
        .unwrap();
        close(values[0], -3.5 * 2.0 / 3.0);
        let mut boundary = [0.0, 2.0, 5.0];
        rescale(&mut boundary, 3, 1, &[0, 0], &[1, 2], &[1.0, 1.0], &config).unwrap();
        close(boundary[0], -7.0 / 3.0);
        let mut zero = [2.0, 2.0, 5.0];
        rescale(&mut zero, 3, 1, &[0, 1], &[1, 0], &[1.0, 1.0], &config).unwrap();
        close(zero[0], -2.0);
        close(zero[2], 4.0);
        let mut empty = [1.0, 3.0];
        rescale(&mut empty, 2, 1, &[], &[], &[], &config).unwrap();
        assert_eq!(empty, [-1.0, 1.0]);
    }

    #[test]
    fn pinned_upstream_rescale_oracle() {
        // generate_standard_umap.py --init-rescale-only, umap-learn 0.6.0.
        // Source SHA256: 1441f803cf35a3412172c9d598f3ed6309e21dcfc4e416e4e677163faf05af88.
        let cases = [
            (
                InitRescale::Isotropic,
                [
                    -0.7508402,
                    -1.0726289,
                    -0.32178867,
                    -1.0726289,
                    -0.10726289,
                    -0.21452577,
                    1.1798917,
                    2.3597834,
                ],
            ),
            (
                InitRescale::PerAxis,
                [
                    -1.0130817,
                    -0.7949736,
                    -0.43417785,
                    -0.7949736,
                    -0.14472595,
                    -0.15899472,
                    1.5919855,
                    1.7489419,
                ],
            ),
            (
                InitRescale::GraphEdge,
                [-0.875, -1.25, -0.375, -1.25, -0.125, -0.25, 1.375, 2.75],
            ),
        ];
        for (mode, expected) in cases {
            let mut values = [0.0, 4.0, 2.0, 4.0, 3.0, 8.0, 9.0, 20.0];
            rescale(
                &mut values,
                4,
                2,
                &[0, 0, 1, 2, 3],
                &[1, 2, 0, 3, 2],
                &[2.0, 3.0, 1.0, 1.0, 4.0],
                &cfg(mode, 1.25),
            )
            .unwrap();
            for (actual, expected) in values.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
    }

    #[test]
    fn extents_overflow_and_nonfinite_output() {
        assert!(matches!(
            check_extents(usize::MAX, 2, 0, InitRescale::Isotropic),
            Err(UmapError::Overflow(_))
        ));
        assert!(matches!(
            check_extents(1, 1, usize::MAX, InitRescale::GraphEdge),
            Err(UmapError::Overflow(_))
        ));
        assert!(matches!(
            check_extents(usize::MAX / 8 + 1, 1, 0, InitRescale::PerAxis),
            Err(UmapError::Overflow(_))
        ));
        let mut values = [-1.0, 1.0];
        assert!(
            rescale(
                &mut values,
                2,
                1,
                &[0],
                &[1],
                &[1.0],
                &cfg(InitRescale::GraphEdge, f32::MAX)
            )
            .is_ok()
        );
        let mut values = [-2.0, 2.0, 100.0];
        assert!(matches!(
            rescale(
                &mut values,
                3,
                1,
                &[0],
                &[1],
                &[1.0],
                &cfg(InitRescale::GraphEdge, f32::MAX)
            ),
            Err(UmapError::Numerical(_))
        ));
    }
}
