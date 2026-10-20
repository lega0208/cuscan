use super::legal_cube_dim;
use burn::{
    backend::{
        TensorMetadata,
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
    },
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

const DISTANCE_TILE: CubeDim = CubeDim::new_2d(16, 16);
const NEIGHBOR_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

#[cube(launch)]
fn exact_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    k: u32,
    exclude_self: u32,
    query_offset: u32,
    squared: u32,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let nr = reference.shape(0);
    let d = query.shape(1);
    let kw = k as usize;
    if row < query.shape(0) {
        for slot in 0..kw {
            indices[row * kw + slot] = I::cast_from(nr);
            distances[row * kw + slot] = if squared != 0 {
                F::new(f32::INFINITY)
            } else {
                F::new(f32::MAX)
            };
        }
        for r in 0..nr {
            if exclude_self == 0 || r != query_offset as usize + row {
                let mut distance = F::new(0.0f32);
                for f in 0..d {
                    let delta = query[row * d + f] - reference[r * d + f];
                    distance += delta * delta;
                }
                let worst = row * kw + kw - 1;
                let wi = u32::cast_from(indices[worst]) as usize;
                if distance < distances[worst] || (distance == distances[worst] && r < wi) {
                    let mut slot = kw - 1;
                    while slot > 0 {
                        let prev = row * kw + slot - 1;
                        let pi = u32::cast_from(indices[prev]) as usize;
                        if distance < distances[prev] || (distance == distances[prev] && r < pi) {
                            indices[row * kw + slot] = indices[prev];
                            distances[row * kw + slot] = distances[prev];
                            slot -= 1;
                        } else {
                            break;
                        }
                    }
                    indices[row * kw + slot] = I::cast_from(r);
                    distances[row * kw + slot] = distance;
                }
            }
        }
        if squared == 0 {
            for slot in 0..kw {
                let at = row * kw + slot;
                distances[at] = F::sqrt(distances[at]);
            }
        }
    }
}

#[cube(launch)]
fn distance_kernel<F: Float>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let q = ABSOLUTE_POS_X as usize;
    let r = ABSOLUTE_POS_Y as usize;
    let d = query.shape(1);
    if q < query.shape(0) && r < reference.shape(0) {
        let mut sum = F::new(0.0f32);
        for f in 0..d {
            let delta = query[q * d + f] - reference[r * d + f];
            sum += delta * delta;
        }
        output[q * reference.shape(0) + r] = sum;
    }
}

pub(super) fn exact_neighbors(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
    squared: bool,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [qr, qf] = query.shape().dims::<2>();
    let [rr, rf] = reference.shape().dims::<2>();
    assert!(qr > 0 && rr > 0, "neighbor inputs must be non-empty");
    assert!(qf > 0, "neighbor feature dimension must be non-empty");
    assert_eq!(qf, rf, "query and reference feature dimensions must match");
    assert_eq!(
        query.dtype, reference.dtype,
        "query and reference dtype must match"
    );
    assert!(k > 0, "neighbor count must be positive");
    let qr32 = u32::try_from(qr).expect("query row count exceeds u32");
    let rr32 = u32::try_from(rr).expect("reference row count exceeds u32");
    i32::try_from(rr).expect("reference indices exceed i32");
    let end = query_offset
        .checked_add(qr32)
        .expect("query offset plus block rows exceeds u32");
    if exclude_self {
        assert!(
            end <= rr32,
            "self-excluded query indices must be valid reference indices"
        );
    }
    if !squared {
        assert!(
            k <= rr32 - exclude_self as u32,
            "neighbor count exceeds available references"
        );
    }
    let n = qr
        .checked_mul(k as usize)
        .expect("neighbor output dimensions overflow usize");
    let ib = n
        .checked_mul(DType::I32.size())
        .expect("neighbor index allocation overflows usize");
    let db = n
        .checked_mul(query.dtype.size())
        .expect("neighbor distance allocation overflows usize");
    let ids = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([qr, k as usize]),
        query.client.empty(ib),
        DType::I32,
    );
    let ds = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([qr, k as usize]),
        query.client.empty(db),
        query.dtype,
    );
    let client = query.client.clone();
    let dim = legal_cube_dim(&client, NEIGHBOR_CUBE_DIM);
    exact_kernel::launch(
        &client,
        CubeCount::Static(qr32.div_ceil(dim.x), 1, 1),
        dim,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        k,
        exclude_self as u32,
        query_offset,
        squared as u32,
        ids.clone().into_tensor_arg(),
        ds.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(ds.dtype),
            dtype_to_storage_type(ids.dtype),
        ],
    );
    (ids, ds)
}

pub(super) fn squared_euclidean_distance_block(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
) -> FloatTensor<CubeBackend> {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [qr, qf] = query.shape().dims::<2>();
    let [rr, rf] = reference.shape().dims::<2>();
    assert!(qr > 0 && rr > 0, "distance blocks must be non-empty");
    assert!(qf > 0, "distance feature dimension must be non-empty");
    assert_eq!(qf, rf, "query and reference feature dimensions must match");
    let n = qr
        .checked_mul(rr)
        .expect("squared-distance output dimensions overflow usize");
    let bytes = n
        .checked_mul(query.dtype.size())
        .expect("squared-distance output allocation overflows usize");
    let qr32 = u32::try_from(qr).expect("query block row count exceeds u32");
    let rr32 = u32::try_from(rr).expect("reference row count exceeds u32");
    let out = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([qr, rr]),
        query.client.empty(bytes),
        query.dtype,
    );
    let client = query.client.clone();
    let dim = legal_cube_dim(&client, DISTANCE_TILE);
    distance_kernel::launch(
        &client,
        CubeCount::Static(qr32.div_ceil(dim.x), rr32.div_ceil(dim.y), 1),
        dim,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        dtype_to_storage_type(out.dtype),
    );
    out
}
