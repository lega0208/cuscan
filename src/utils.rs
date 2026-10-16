use core::f64;

use burn::backend::tensor::FloatTensor;
use burn::{
    backend::Dispatch,
    tensor::{Device, Tensor, TensorData},
};
use num::{Float, FromPrimitive, ToPrimitive};
// use prettytable::{row, Table};
use rand::{RngExt, distr::uniform::SampleUniform};

/// Generates random test data with the given number of samples and features.
///
/// # Arguments
/// * `num_samples` - The number of samples to generate.
/// * `num_features` - The number of features (columns) per sample.
///
/// # Returns
/// A `Vec<f64>` containing the randomly generated data.
///
/// This function uses the `rand` crate to generate a flat vector of random floating-point values.
pub fn generate_test_data<F: Float + FromPrimitive + SampleUniform>(
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
) -> Vec<F> {
    let mut rng = rand::rng();

    // Define the range for random numbers (e.g., [0.0, 1.0))
    let zero = F::from_f64(0.0).unwrap(); // 0.0 as a `F` type
    let one = F::from_f64(1.0).unwrap(); // 1.0 as a `F` type

    // Generate random data for the tensor (size = num_samples * num_features)
    let data: Vec<F> = (0..num_samples * num_features)
        .map(|_| rng.random_range(zero..one)) // Generate random number from the range [0.0, 1.0)
        .collect();

    data
}

/// Converts a vector of values into a 2D tensor on the specified device.
///
/// # Arguments
/// * `data` - A vector of `f64` values representing the data to convert.
/// * `num_samples` - The number of samples (rows).
/// * `num_features` - The number of features (columns).
/// * `device` - The device to place the tensor on (e.g., CPU, GPU).
///
/// # Returns
/// A `Tensor<2>` containing the data arranged as samples x features.
///
/// This function uses the `TensorData` struct to create a tensor from the given data, then places it
/// on the specified device (`CPU` or `GPU`).
pub fn convert_vector_to_tensor<F: Float>(
    data: Vec<F>,
    num_samples: usize,  // Number of samples
    num_features: usize, // Number of features (columns) per sample
    device: &Device,     // Device to place the tensor (CPU, GPU)
) -> Tensor<2>
where
    F: burn::tensor::Element,
{
    let tensor_data = TensorData::new(data, [num_samples, num_features]);
    Tensor::<2>::from_data(tensor_data, device)
}

/// Prints the content of a tensor in a table format with index and tensor values.
///
/// # Arguments
/// * `data` - The tensor to print, with dimensionality `D`.
///
/// This function prints the tensor's data in a table with each row corresponding to one sample.
/// The tensor data is printed in a format that makes it easy to inspect.
// pub fn print_tensor<const D: usize>(data: &Tensor<D>) {
//     let dims = data.dims();
//     let n_samples = match dims.len() > 0 {
//         true => dims[0],
//         false => 0,
//     };

//     let mut table = Table::new();
//     table.add_row(row!["Index", "Tensor"]);

//     for index in 0..n_samples {
//         let row = data.clone().slice([index..index + 1]);
//         let row = row.to_data().to_vec::<f32>().unwrap();
//         let row = format!("{row:?}");
//         table.add_row(row![index, format!("{:?}", row)]);
//     }

//     if dims.len() == 0 {
//         let row = data.to_data().to_vec::<f32>().unwrap();
//         let row = row.get(0).unwrap();
//         table.add_row(row![0, format!("{:?}", row)]);
//     }

//     table.printstd();
// }

/// Prints the content of a tensor with a title.
///
/// # Arguments
/// * `title` - A string title to print before displaying the tensor data.
/// * `data` - The tensor to print.
///
/// This function is similar to `print_tensor`, but with an added title to help distinguish different tensor prints.
// pub fn print_tensor_with_title<const D: usize>(title: &str, data: &Tensor<D>) {
//     println!("{title}");
//     print_tensor(data);
// }

/// Converts a 2D tensor into a nested vector for easier inspection or manipulation.
///
/// # Arguments
/// * `data` - A 2D tensor (samples x features) to convert into a vector of vectors.
///
/// # Returns
/// A `Vec<Vec<f64>>` where each inner `Vec<f64>` represents a row (sample) of the tensor.
///
/// This function extracts the data from a tensor and converts it into a `Vec<Vec<F>>` format. The conversion
/// assumes that the tensor is in a 2D shape and the precision is `f32` within the tensor.
pub fn convert_tensor_to_vector<F: Float>(data: Tensor<2>) -> Vec<Vec<F>>
where
    F: burn::tensor::Element,
{
    let n_components = data.dims()[1]; // usually 2 dimensional

    // Burn Tensor only has f32 precision inside the tensors, when you export to to_data
    let data = data.to_data().try_to_vec::<f32>().unwrap();

    let data: Vec<Vec<f32>> = data
        .chunks(n_components)
        .map(|chunk| chunk.to_vec())
        .collect();

    let data: Vec<Vec<F>> = data
        .into_iter()
        .map(|v| {
            v.into_iter()
                .map(|v| {
                    if v.is_nan() {
                        // TODO: fix this on a different level in the loss function!
                        // if NaN variables, replaces them with 0
                        F::from(0.0).unwrap()
                    } else {
                        F::from(v).unwrap()
                    }
                })
                .collect()
        })
        .collect();

    data
}

/// Formats a `Duration` into a human-readable string in hours, minutes, and seconds format.
///
/// # Arguments
/// * `duration` - The duration to format.
///
/// # Returns
/// A formatted string representing the duration in the format `HH:MM:SS`.
///
/// This function is useful for displaying elapsed times or durations in a more readable format.
pub fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

// a constant used to offset division by zero in the normalization function below
const SMALL_STD_DEV: f64 = 1e-6;

/// Normalizes the given dataset by centering each feature (column) to have mean 0
/// and standard deviation 1.
///
/// # Arguments
/// * `data` - A mutable slice representing the dataset, where each row is a sample,
///   and each column represents a feature. The data is assumed to be stored in
///   row-major order (i.e., `data[sample_idx * num_features + feature_idx]`).
/// * `num_samples` - The number of samples (rows) in the dataset.
/// * `num_features` - The number of features (columns) in the dataset.
///
/// # Example
/// ```
/// use fast_umap::utils::normalize_data;
///
/// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let num_samples = 2;
/// let num_features = 3;
/// normalize_data(&mut data, num_samples, num_features);
/// ```
/// The function will normalize each feature (column) across all samples (rows).
///
/// # Note
/// This function assumes that the dataset has at least one sample and one feature.
/// The data is normalized in-place, meaning the original data is modified directly.
pub fn normalize_data<F: Float>(
    data: &mut [F],
    num_samples: usize,
    num_features: usize,
) -> (Vec<f64>, Vec<f64>)
where
    F: num::FromPrimitive + ToPrimitive + Send + Sync,
{
    let mut means = Vec::with_capacity(num_features);
    let mut standard_deviations = Vec::with_capacity(num_features);
    // Sum sequentially to keep normalization statistics deterministic.
    (0..num_features).into_iter().for_each(|feature_idx| {
        // Calculate mean and standard deviation for the current feature
        let (sum, sum_sq) = (0..num_samples).fold(
            (F::zero(), F::zero()),
            |(acc_sum, acc_sum_sq), sample_idx| {
                let value = data[sample_idx * num_features + feature_idx];
                (acc_sum + value, acc_sum_sq + value * value)
            },
        );

        let mean = sum / F::from_usize(num_samples).unwrap();
        let variance = (sum_sq / F::from_usize(num_samples).unwrap()) - (mean * mean);
        let std_dev = variance.sqrt();

        // Avoid division by zero by adding SMALL_STD_DEV
        let safe_std_dev = std_dev + F::from_f64(SMALL_STD_DEV).unwrap();
        means.push(mean.to_f64().unwrap());
        standard_deviations.push(safe_std_dev.to_f64().unwrap());

        // Normalize the feature in place.
        (0..num_samples).into_iter().for_each(|sample_idx| {
            let idx = sample_idx * num_features + feature_idx;
            let value = data[idx];
            let normalized_value = (value - mean) / safe_std_dev;
            // Directly update the value in the `data` array
            data[idx] = normalized_value;
        });
    });
    (means, standard_deviations)
}

/// Normalizes a 1D tensor using min-max normalization.
///
/// This function performs min-max normalization on a 1D tensor, scaling its values
/// to a range between 0 and 1. If the minimum and maximum values of the tensor are
/// equal (i.e., the tensor has no variance), the original tensor is returned unmodified.
///
/// # Type Parameters
///
/// # Arguments
///
/// - `tensor`: A 1D tensor of type `Tensor<1>`, which represents the input data to be normalized.
///
/// # Returns
///
/// - A new tensor of the same type and shape as the input tensor, with normalized values.
///   If the minimum and maximum values of the tensor are equal, the original tensor is returned unchanged.
///
/// # Explanation
///
/// The normalization is performed using the following formula:
///
/// ```text
/// normalized = (tensor - min) / (max - min + epsilon)
/// ```
///
/// Where:
/// - `min`: The minimum value in the tensor.
/// - `max`: The maximum value in the tensor.
/// - `epsilon`: A small constant (`1e-6`) added to prevent division by zero, ensuring numerical stability.
///
/// The function first checks if the minimum and maximum values are equal. If they are, it avoids division by zero
/// and simply returns the original tensor. If they are not equal, it applies the min-max normalization formula.
///
/// # Example
///
/// ```rust
/// use burn::tensor::{Device, Tensor};
/// use fast_umap::utils::normalize_tensor;
///
/// let device = Device::cpu();
/// let tensor = Tensor::<1>::from_data([1.0f32, 2.0, 3.0], &device);
/// let normalized_tensor = normalize_tensor(tensor);
/// ```
pub fn normalize_tensor(tensor: Tensor<1>) -> Tensor<1> {
    let device = tensor.device();

    // Normalize the result using min-max normalization:
    let min_val = tensor.clone().min(); // Find the minimum value
    let max_val = tensor.clone().max(); // Find the maximum value

    // this is to prevent deleting by zero
    let offset_val = Tensor::<1>::from_data(TensorData::new(vec![1e-6], [1]), &device);

    // Read the scalar diff as f32.
    // We intentionally avoid .equal() + .to_vec::<bool>() because the WGPU
    // backend stores boolean tensors as u32 internally, and .to_vec::<bool>()
    // fails with TypeMismatch on that backend.
    let diff: f32 = (max_val.clone() - min_val.clone())
        .abs()
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .first()
        .copied()
        .unwrap_or(0.0);
    let are_equal = diff < 1e-6_f32;

    // Avoid division by zero by ensuring max_val != min_val
    let normalized = if !are_equal {
        (tensor - min_val.clone()) / (max_val - min_val + offset_val)
    } else {
        tensor.clone() // If all values are the same, return the original
    };

    // Return the normalized sum of the top K distances
    normalized
}

/// Print a raw [`FloatTensor`] primitive to stdout in a grid layout.
///
/// Wraps the primitive in a typed [`Tensor`] and delegates to [`print_tensor`].
///
/// # Arguments
///
/// * `tensor` — The raw float-tensor primitive to display.
/// * `rows`   — Maximum number of rows to print (clamped to at least `MIN_SIZE`).
/// * `cols`   — Maximum number of columns to print (clamped to at least `MIN_SIZE`).
pub fn print_primitive_tensor(tensor: &FloatTensor<Dispatch>, rows: usize, cols: usize) {
    let tensor: Tensor<2> = Tensor::from_dispatch(tensor.clone().into());
    print_tensor(&tensor, rows, cols)
}

/// Minimum number of rows / columns shown by [`print_tensor`] when `rows` or
/// `cols` is smaller than this value.
const MIN_SIZE: usize = 10;

/// Print the first `rows × cols` elements of a 2-D tensor to stdout.
///
/// Each element is formatted in scientific notation with three decimal places
/// (`{:10.3e}`).  Values beyond `rows` or `cols` are silently omitted.
///
/// # Arguments
///
/// * `tensor` — The 2-D tensor to display.
/// * `rows`   — Maximum number of rows to print (floored to `MIN_SIZE`).
/// * `cols`   — Maximum number of columns to print (floored to `MIN_SIZE`).
pub fn print_tensor(tensor: &Tensor<2>, rows: usize, cols: usize) {
    let shape = tensor.dims();
    let n = shape[0]; // Number of rows
    let d = shape[1]; // Number of columns

    let data = tensor.to_data().try_to_vec::<f32>().unwrap();

    let nn = n.min(rows.max(MIN_SIZE)); // Ensure nn is at least MIN_SIZE if rows > 0
    let dd = d.min(cols.max(MIN_SIZE)); // Ensure dd is at least MIN_SIZE if cols > 0

    // Print first few rows and columns with scientific notation
    for i in 0..nn {
        for j in 0..dd {
            // Print each element with scientific notation, limited to 3 decimal places
            print!("{:10.3e}", data[i * d + j]);
        }
        println!();
    }
}
