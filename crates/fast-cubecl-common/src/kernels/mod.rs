use burn::cubecl::CubeDim;
#[cfg(any(feature = "cpu", feature = "cuda"))]
mod cube_backend;
#[cfg(any(feature = "cpu", feature = "cuda"))]
mod neighbors;

pub fn legal_cube_dim(client: &cubecl::client::Client, preferred: CubeDim) -> CubeDim {
    let h = &client.properties().hardware;
    legal_cube_dim_for_limits(preferred, h.max_cube_dim, h.max_units_per_cube)
}
fn legal_cube_dim_for_limits(
    preferred: CubeDim,
    (mx, my, mz): (u32, u32, u32),
    units: u32,
) -> CubeDim {
    let mut remaining = units.max(1);
    let x = preferred.x.min(mx).min(remaining).max(1);
    remaining /= x;
    let y = preferred.y.min(my).min(remaining).max(1);
    remaining /= y;
    let z = preferred.z.min(mz).min(remaining).max(1);
    CubeDim::new_3d(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_limits_bound_axes_and_cover_tails() {
        for (preferred, limits, units, expected) in [
            (CubeDim::new_1d(32), (32, 1, 1), 1, (1, 1, 1)),
            (CubeDim::new_1d(32), (32, 1, 1), 8, (8, 1, 1)),
            (CubeDim::new_1d(32), (32, 1, 1), 64, (32, 1, 1)),
            (CubeDim::new_1d(64), (64, 1, 1), 64, (64, 1, 1)),
            (CubeDim::new_1d(64), (16, 1, 1), 64, (16, 1, 1)),
            (CubeDim::new_2d(16, 16), (16, 16, 1), 256, (16, 16, 1)),
            (CubeDim::new_2d(32, 32), (32, 32, 1), 1024, (32, 32, 1)),
            (CubeDim::new_2d(32, 32), (32, 32, 1), 64, (32, 2, 1)),
        ] {
            let dim = legal_cube_dim_for_limits(preferred, limits, units);
            assert_eq!((dim.x, dim.y, dim.z), expected);
            assert!(67u32.div_ceil(dim.x) * dim.x >= 67);
        }
    }
}
