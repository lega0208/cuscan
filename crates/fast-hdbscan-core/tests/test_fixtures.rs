use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

const FIXTURES: &str = "../fast-hdbscan/tests/fixtures";

#[derive(Debug, PartialEq)]
struct NpyHeader {
    descr: String,
    fortran: bool,
    shape: Vec<usize>,
    payload: Vec<u8>,
}

fn parse_npy(bytes: &[u8]) -> Result<NpyHeader, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic/header".into());
    }
    let major = bytes[6];
    let (header_len, start): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
        _ => return Err(format!("unsupported NPY version {major}.{}", bytes[7])),
    };
    let end = start
        .checked_add(header_len)
        .ok_or("NPY header length overflow")?;
    if end > bytes.len() {
        return Err("truncated NPY header".into());
    }
    let header = std::str::from_utf8(&bytes[start..end]).map_err(|_| "NPY header is not UTF-8")?;
    let descr = header_value(header, "descr")?;
    let fortran_value = header_value(header, "fortran_order")?;
    let fortran = match fortran_value.as_str() {
        "True" => true,
        "False" => false,
        _ => return Err("invalid NPY fortran_order value".into()),
    };
    let shape_text = header_value(header, "shape")?;
    let shape = parse_shape(&shape_text)?;
    Ok(NpyHeader {
        descr,
        fortran,
        shape,
        payload: bytes[end..].to_vec(),
    })
}

fn parse_shape(text: &str) -> Result<Vec<usize>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let tokens: Vec<_> = text.split(',').collect();
    if tokens.last().is_some_and(|token| token.trim().is_empty()) {
        if tokens.len() == 1 {
            return Err("malformed NPY shape".into());
        }
    }
    let dimensions = if tokens.last().is_some_and(|token| token.trim().is_empty()) {
        &tokens[..tokens.len() - 1]
    } else {
        &tokens[..]
    };
    dimensions
        .iter()
        .map(|token| {
            let token = token.trim();
            if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(format!("invalid NPY shape dimension {token:?}"));
            }
            token
                .parse::<usize>()
                .map_err(|_| "NPY shape dimension overflow".to_owned())
        })
        .collect()
}

fn dtype_width(dtype: &str) -> Option<usize> {
    match dtype {
        "<i8" | "<f8" => Some(8),
        "<i4" | "<f4" => Some(4),
        _ => None,
    }
}

fn json_usize(value: &Value, label: &str) -> Result<usize, String> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label} must be a nonnegative integer"))
}

fn read_i64s(header: &NpyHeader) -> Result<Vec<i64>, String> {
    if header.descr != "<i8" || header.payload.len() % 8 != 0 {
        return Err("expected contiguous little-endian i64 payload".into());
    }
    Ok(header
        .payload
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn read_f32s(header: &NpyHeader) -> Result<Vec<f32>, String> {
    if header.descr != "<f4" || header.payload.len() % 4 != 0 {
        return Err("expected contiguous little-endian f32 payload".into());
    }
    Ok(header
        .payload
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn matrix_index_check(header: &NpyHeader, leaves: usize, label: &str) -> Result<(), String> {
    if header.shape.len() != 2 || header.shape[1] != 4 || header.descr != "<f8" {
        return Err(format!("{label} must be an f64 four-column linkage matrix"));
    }
    for (row, values) in header.payload.chunks_exact(32).enumerate() {
        let left = f64::from_le_bytes(values[0..8].try_into().unwrap());
        let right = f64::from_le_bytes(values[8..16].try_into().unwrap());
        let parent = leaves
            .checked_add(row)
            .ok_or("linkage cluster index overflow")?;
        for (name, value) in [("left", left), ("right", right)] {
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= parent as f64 {
                return Err(format!(
                    "{label} row {row} {name} is not a prior integral node index"
                ));
            }
        }
        if left == right {
            return Err(format!("{label} row {row} has identical children"));
        }
        let size = f64::from_le_bytes(values[24..32].try_into().unwrap());
        if !size.is_finite() || size <= 0.0 || size.fract() != 0.0 || size > leaves as f64 {
            return Err(format!("{label} row {row} has invalid cluster size"));
        }
    }
    Ok(())
}

fn structured_condensed_check(
    header: &NpyHeader,
    leaves: usize,
    label: &str,
) -> Result<(), String> {
    if header.shape.len() != 1 {
        return Err(format!("{label} must be rank-1"));
    }
    let dtype = header
        .descr
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let child_size_width = if dtype.contains("('child_size','<f8')") {
        8
    } else if dtype.contains("('child_size','<f4')") {
        4
    } else {
        return Err(format!("{label} has unsupported child_size dtype"));
    };
    let row_width = 8 + 8 + 8 + child_size_width;
    if header.payload.len()
        != header.shape[0]
            .checked_mul(row_width)
            .ok_or("condensed payload overflow")?
    {
        return Err(format!("{label} payload length mismatch"));
    }
    let max_node = leaves
        .checked_mul(2)
        .ok_or("condensed node bound overflow")?;
    for (row, bytes) in header.payload.chunks_exact(row_width).enumerate() {
        let parent = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let child = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let lambda = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let size = if child_size_width == 4 {
            f32::from_le_bytes(bytes[24..28].try_into().unwrap()) as f64
        } else {
            f64::from_le_bytes(bytes[24..32].try_into().unwrap())
        };
        if parent < leaves as i64
            || parent as usize >= max_node
            || child < 0
            || child as usize >= max_node
            || parent == child
        {
            return Err(format!(
                "{label} row {row} has out-of-range parent/child node"
            ));
        }
        if lambda.is_nan() || lambda < 0.0 {
            return Err(format!("{label} row {row} has invalid lambda"));
        }
        if !size.is_finite() || size <= 0.0 {
            return Err(format!("{label} row {row} has invalid child size"));
        }
    }
    Ok(())
}

fn header_value(header: &str, key: &str) -> Result<String, String> {
    let marker = format!("'{key}'");
    let start = header
        .find(&marker)
        .ok_or_else(|| format!("NPY header missing {key}"))?
        + marker.len();
    let tail = header[start..].trim_start();
    let tail = tail
        .strip_prefix(':')
        .ok_or_else(|| format!("malformed NPY {key}"))?
        .trim_start();
    if tail.starts_with('[') {
        let mut depth = 0usize;
        let mut end = None;
        for (index, byte) in tail.bytes().enumerate() {
            if byte == b'[' {
                depth += 1;
            }
            if byte == b']' {
                depth = depth.checked_sub(1).ok_or("malformed NPY list")?;
                if depth == 0 {
                    end = Some(index + 1);
                    break;
                }
            }
        }
        Ok(tail[..end.ok_or("unterminated NPY list")?].to_owned())
    } else if tail.starts_with('\'') || tail.starts_with('"') {
        let quote = tail.chars().next().unwrap();
        let rest = &tail[1..];
        let end = rest
            .find(quote)
            .ok_or_else(|| format!("unterminated NPY {key}"))?;
        Ok(rest[..end].to_owned())
    } else if key == "shape" {
        let end = tail.find(')').ok_or("malformed NPY shape")?;
        Ok(tail[..=end]
            .trim_matches(|c| c == '(' || c == ')')
            .to_owned())
    } else {
        let end = tail.find([',', '}']).unwrap_or(tail.len());
        Ok(tail[..end].trim().to_owned())
    }
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}
fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn assert_error_contains(result: Result<(), String>, expected: &str) {
    let error = result.expect_err("mutation unexpectedly validated");
    assert!(
        error.contains(expected),
        "expected error containing {expected:?}, got {error:?}"
    );
}

fn load_npz(path: &Path) -> Result<std::collections::BTreeMap<String, NpyHeader>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut zip = ZipArchive::new(file).map_err(|e| format!("invalid NPZ: {e}"))?;
    let mut arrays = std::collections::BTreeMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .ok_or("NPZ entry is not an NPY array")?
            .to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        let header = parse_npy(&bytes)?;
        if header.fortran {
            return Err(format!("{name} uses unsupported Fortran layout"));
        }
        let count = header
            .shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or("array size overflow")?;
        let width = if header.descr.ends_with("i8") || header.descr.ends_with("f8") {
            8
        } else if header.descr.ends_with("i4") || header.descr.ends_with("f4") {
            4
        } else {
            0
        };
        if !header.descr.starts_with('[')
            && width > 0
            && count.checked_mul(width) != Some(header.payload.len())
        {
            return Err(format!(
                "{name} payload length does not match shape and dtype"
            ));
        }
        if arrays.insert(name.clone(), header).is_some() {
            return Err(format!("duplicate array {name}"));
        }
    }
    Ok(arrays)
}

fn replace_npz_entry(path: &Path, entry_name: &str, mut mutate: impl FnMut(&mut Vec<u8>)) {
    let file = File::open(path).unwrap();
    let mut archive = ZipArchive::new(file).unwrap();
    let mut entries = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).unwrap();
        let name = entry.name().to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        if name == entry_name {
            mutate(&mut bytes);
        }
        entries.push((name, bytes));
    }
    let file = File::create(path).unwrap();
    let mut writer = zip::ZipWriter::new(file);
    for (name, bytes) in entries {
        writer
            .start_file(name, zip::write::FileOptions::default())
            .unwrap();
        writer.write_all(&bytes).unwrap();
    }
    writer.finish().unwrap();
}

fn validate_case(
    root: &Path,
    case_id: &str,
    manifest_case: &Value,
    manifest: &Value,
    schema: &Value,
) -> Result<(), String> {
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(root.join(format!("{case_id}.json"))).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for field in schema["metadata_required"]
        .as_array()
        .ok_or("schema metadata_required must be array")?
    {
        let field = field
            .as_str()
            .ok_or("schema metadata field must be string")?;
        if metadata.get(field).is_none() {
            return Err(format!("metadata missing schema-required {field}"));
        }
    }
    for field in schema["metadata_reference_required"]
        .as_array()
        .ok_or("schema metadata_reference_required must be array")?
    {
        let field = field
            .as_str()
            .ok_or("schema reference field must be string")?;
        if metadata["reference"].get(field).is_none() {
            return Err(format!("provenance missing {field}"));
        }
    }
    let manifest_reference = manifest["reference"]
        .as_object()
        .ok_or("manifest reference must be object")?;
    let required_reference = schema["manifest_reference_required"]
        .as_array()
        .ok_or("schema manifest_reference_required must be array")?;
    if required_reference.is_empty() {
        return Err("schema manifest_reference_required must be nonempty".into());
    }
    let mut required_reference_keys = HashSet::new();
    for field in required_reference {
        let field = field
            .as_str()
            .ok_or("schema manifest reference field must be string")?;
        if field.is_empty() || !required_reference_keys.insert(field) {
            return Err("schema manifest_reference_required has empty or duplicate key".into());
        }
        if !manifest_reference.contains_key(field) {
            return Err(format!("manifest reference missing required key {field}"));
        }
    }
    if manifest_reference.len() != required_reference_keys.len() {
        return Err("manifest reference has unknown keys".into());
    }
    if metadata["schema_version"] != 1 || metadata["case_id"] != case_id {
        return Err("metadata version or case id mismatch".into());
    }
    let status = metadata["status"].as_str().ok_or("status must be string")?;
    let status_contract = schema["case_statuses"]
        .get(status)
        .ok_or_else(|| format!("unknown status {status}"))?;
    for key in status_contract["required"]
        .as_array()
        .ok_or("schema status required must be array")?
    {
        let key = key.as_str().ok_or("schema array name must be string")?;
        if !metadata["arrays"].get(key).is_some() {
            return Err(format!("status {status} requires array {key}"));
        }
    }
    if status
        != manifest_case["status"]
            .as_str()
            .ok_or("manifest status must be string")?
    {
        return Err("manifest/metadata status mismatch".into());
    }
    for (key, expected) in manifest_case["params"]
        .as_object()
        .ok_or("manifest params must be object")?
    {
        if key == "persistence" && expected == "measured_boundary" {
            if !metadata["parameters"][key].is_number() {
                return Err("measured persistence boundary is not numeric".into());
            }
        } else if metadata["parameters"][key] != *expected {
            return Err(format!("metadata parameter {key} differs from manifest"));
        }
    }
    for (key, expected) in manifest_reference {
        if metadata["reference"][key] != *expected {
            return Err("fixture provenance does not match manifest release evidence".into());
        }
    }
    if json_usize(&metadata["reference"]["dimensions"], "reference dimensions")?
        != json_usize(&manifest_case["dimensions"], "manifest dimensions")?
        || metadata["reference"]["shape"] != metadata["input_shape"]
    {
        return Err("reference shape/dimensions metadata mismatch".into());
    }
    if status != "reference" && metadata["reason"].as_str().unwrap_or("").is_empty() {
        return Err("non-reference case needs an explicit reason".into());
    }
    let arrays = load_npz(&root.join(format!("{case_id}.npz")))?;
    for key in status_contract["forbidden"]
        .as_array()
        .ok_or("schema status forbidden must be array")?
    {
        let key = key.as_str().ok_or("schema array name must be string")?;
        if metadata["arrays"].get(key).is_some() {
            return Err(format!("status {status} forbids array {key}"));
        }
    }
    let catalogue = metadata["arrays"]
        .as_object()
        .ok_or("arrays catalogue must be object")?;
    if arrays.keys().collect::<HashSet<_>>() != catalogue.keys().collect::<HashSet<_>>() {
        return Err("NPZ arrays and JSON catalogue differ".into());
    }
    for (name, header) in &arrays {
        let item = &catalogue[name];
        let contract = schema["arrays"]
            .get(name)
            .ok_or_else(|| format!("schema has no array contract for {name}"))?;
        if header.shape.len()
            != contract["rank"]
                .as_u64()
                .ok_or("schema rank must be integer")? as usize
        {
            return Err(format!("{name} rank differs from schema"));
        }
        let schema_dtype = if let Some(options) = contract["dtype_options"].as_array() {
            options.iter().any(|option| item["dtype"] == *option)
        } else {
            item["dtype"] == contract["dtype"]
        };
        if !schema_dtype {
            return Err(format!("{name} catalogue dtype differs from schema"));
        }
        let shape: Vec<usize> = item["shape"]
            .as_array()
            .ok_or("catalogue shape must be array")?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize).ok_or("invalid shape value"))
            .collect::<Result<_, _>>()?;
        if header.shape != shape {
            return Err(format!("{name} shape differs from catalogue"));
        }
        if let Some(fields) = item["dtype"].as_array() {
            let expected_descr = format!(
                "[{}]",
                fields
                    .iter()
                    .map(|field| format!(
                        "('{}', '{}')",
                        field["name"].as_str().unwrap_or(""),
                        field["dtype"].as_str().unwrap_or("")
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if header
                .descr
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                != expected_descr
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
            {
                return Err(format!("{name} structured dtype fields/order mismatch"));
            }
            let mut field_width = 0usize;
            for field in fields {
                let field_dtype = field["dtype"]
                    .as_str()
                    .ok_or("structured field lacks dtype")?;
                field_width +=
                    dtype_width(field_dtype).ok_or("unsupported structured field dtype")?;
            }
            let elements = header
                .shape
                .iter()
                .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                .ok_or("structured array size overflow")?;
            if elements.checked_mul(field_width) != Some(header.payload.len()) {
                return Err(format!("{name} structured payload width mismatch"));
            }
        } else {
            let expected_dtype = item["dtype"]
                .as_str()
                .ok_or("catalogue dtype must be string")?;
            if header.descr != expected_dtype {
                return Err(format!(
                    "{name} dtype mismatch: {} vs {expected_dtype}",
                    header.descr
                ));
            }
        }
    }
    let input = &arrays["input"];
    if input.shape.len() != 2 || input.descr != "<f4" {
        return Err("input must be a rank-2 little-endian float32 array".into());
    }
    if input.shape
        != metadata["input_shape"]
            .as_array()
            .ok_or("input_shape must be array")?
            .iter()
            .map(|x| json_usize(x, "metadata input shape dimension"))
            .collect::<Result<Vec<_>, _>>()?
    {
        return Err("input shape metadata mismatch".into());
    }
    let raw_n = input.shape[0];
    if manifest_case["n"].as_u64() != Some(raw_n as u64)
        || manifest_case["dimensions"].as_u64() != Some(input.shape[1] as u64)
    {
        return Err("input dimensions differ from manifest".into());
    }
    if metadata["reference"]["shape"] != json!([raw_n, input.shape[1]]) {
        return Err("reference shape differs from NPY input shape".into());
    }
    let finite = &arrays["finite_index"];
    if finite.descr != "<i8" || finite.shape.len() != 1 {
        return Err("finite_index must be rank-1 int64".into());
    }
    let indices: Vec<usize> = finite
        .payload
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .map(|i| usize::try_from(i).map_err(|_| "finite_index contains negative value"))
        .collect::<Result<_, _>>()?;
    let leaf_count = indices.len();
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) || indices.iter().any(|&i| i >= raw_n) {
        return Err("finite_index is not strictly increasing or is out of range".into());
    }
    let metadata_indices: Vec<usize> = metadata["finite_index"]
        .as_array()
        .ok_or("metadata finite_index must be array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or("invalid metadata finite index")
        })
        .collect::<Result<_, _>>()?;
    if indices != metadata_indices {
        return Err("finite_index array differs from metadata".into());
    }
    if input.payload.len()
        != raw_n
            .checked_mul(input.shape[1])
            .and_then(|n| n.checked_mul(4))
            .ok_or("input dimensions overflow")?
    {
        return Err("input payload size mismatch".into());
    }
    let finite_set: HashSet<usize> = indices.iter().copied().collect();
    for row in 0..raw_n {
        let row_is_finite = input.payload[row * input.shape[1] * 4..(row + 1) * input.shape[1] * 4]
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite());
        if row_is_finite != finite_set.contains(&row) {
            return Err("finite_index does not match input finite rows".into());
        }
    }
    if let Some(neighbors) = arrays.get("neighbors") {
        if neighbors.descr != "<i8"
            || neighbors.shape.len() != 2
            || neighbors.shape[0] != indices.len()
        {
            return Err("neighbors shape/index contract violation".into());
        }
        let n = indices.len() as i64;
        if neighbors.payload.chunks_exact(8).any(|b| {
            let index = i64::from_le_bytes(b.try_into().unwrap());
            index < -1 || index >= n
        }) {
            return Err("neighbor index outside finite-local range or upstream -1 padding".into());
        }
    }
    for name in ["linkage", "remapped_linkage"] {
        if let Some(linkage) = arrays.get(name) {
            let leaves = if name == "linkage" { leaf_count } else { raw_n };
            matrix_index_check(linkage, leaves, name)?;
        }
    }
    for name in ["condensed_tree", "remapped_condensed_tree"] {
        if let Some(tree) = arrays.get(name) {
            let leaves = if name == "condensed_tree" {
                leaf_count
            } else {
                raw_n
            };
            structured_condensed_check(tree, leaves, name)?;
        }
    }
    for (name, dtype, length) in [
        ("labels", "<i8", raw_n),
        ("dbscan_labels", "<i8", raw_n),
        ("probabilities", "<f4", raw_n),
        ("core_distances", "<f4", indices.len()),
    ] {
        if let Some(array) = arrays.get(name) {
            if array.descr != dtype || array.shape != [length] {
                return Err(format!("{name} dtype/shape contract violation"));
            }
        }
    }
    if let Some(probabilities) = arrays.get("probabilities") {
        if probabilities.payload.chunks_exact(4).any(|b| {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            !v.is_finite() || !(0.0..=1.0).contains(&v)
        }) {
            return Err("probability outside finite [0,1] range".into());
        }
    }
    for name in ["labels", "dbscan_labels", "remapped_dbscan_labels"] {
        if let Some(labels) = arrays.get(name) {
            if read_i64s(labels)?
                .iter()
                .any(|&label| label < -1 || label >= raw_n as i64)
            {
                return Err(format!("{name} contains invalid label index"));
            }
        }
    }
    if let Some(core) = arrays.get("core_distances") {
        if read_f32s(core)?
            .iter()
            .any(|&distance| distance.is_nan() || distance < 0.0)
        {
            return Err("core distances must be nonnegative and not NaN".into());
        }
    }
    if status == "reference"
        && (arrays["mst_weights"].shape != [indices.len().saturating_sub(1)]
            || arrays["linkage"].shape != [indices.len().saturating_sub(1), 4]
            || arrays["remapped_linkage"].shape != [raw_n.saturating_sub(1), 4]
            || arrays["remapped_dbscan_labels"].shape != [raw_n])
    {
        return Err("MST/internal/remapped linkage cardinality contract violation".into());
    }
    if let Some(outliers) = arrays.get("outlier_order") {
        if outliers.descr != "<i8" || outliers.shape.len() != 1 {
            return Err("outlier_order must be rank-1 int64".into());
        }
        let order: Vec<usize> = outliers
            .payload
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .map(|i| usize::try_from(i).map_err(|_| "outlier index is negative"))
            .collect::<Result<_, _>>()?;
        let expected: HashSet<usize> = (0..raw_n).filter(|i| !finite_set.contains(i)).collect();
        if order.iter().copied().collect::<HashSet<_>>() != expected
            || order.len() != expected.len()
        {
            return Err("outlier_order is not a permutation of non-finite raw rows".into());
        }
    }
    if manifest_case["params"]["persistence"] == "measured_boundary" {
        for key in [
            "boundary_input",
            "boundary_below",
            "boundary_equal",
            "boundary_above",
        ] {
            if !arrays.contains_key(key) {
                return Err(format!("measured persistence case missing {key}"));
            }
        }
        let evidence = metadata
            .get("persistence_boundary_evidence")
            .ok_or("measured persistence evidence missing")?;
        let threshold = evidence["threshold"]
            .as_f64()
            .ok_or("persistence threshold must be numeric")? as f32;
        let birth_min = evidence["birth_min"]
            .as_f64()
            .ok_or("persistence birth_min must be numeric")? as f32;
        let death = evidence["death"]
            .as_f64()
            .ok_or("persistence death must be numeric")? as f32;
        if evidence["rule"] != ">="
            || !threshold.is_finite()
            || !birth_min.is_finite()
            || !death.is_finite()
            || (birth_min - death) != threshold
            || metadata["parameters"]["persistence"]
                .as_f64()
                .map(|v| v as f32)
                != Some(threshold)
        {
            return Err("persistence equality evidence does not reproduce float32 boundary".into());
        }
        for key in [
            "boundary_input",
            "boundary_below",
            "boundary_equal",
            "boundary_above",
        ] {
            let probe = &arrays[key];
            if probe.shape.len() != 1
                || probe.shape[0] == 0
                || probe.descr != arrays["condensed_tree"].descr
            {
                return Err(format!(
                    "{key} must be nonempty structured condensed-tree evidence"
                ));
            }
            structured_condensed_check(probe, leaf_count, key)?;
        }
        if arrays["boundary_below"].payload != arrays["boundary_equal"].payload
            || arrays["boundary_equal"].payload == arrays["boundary_above"].payload
        {
            return Err(
                "persistence boundary must be retained at equality but not above it".into(),
            );
        }
    }
    if metadata["divergence_classification"] == "overflow_neighbor_divergence" {
        for key in [
            "overflow_probe_input",
            "overflow_probe_squared_distances",
            "overflow_probe_neighbors",
        ] {
            if !arrays.contains_key(key) {
                return Err(format!("overflow divergence case missing {key}"));
            }
        }
        let input_probe = &arrays["overflow_probe_input"];
        let distances = &arrays["overflow_probe_squared_distances"];
        let neighbors = &arrays["overflow_probe_neighbors"];
        if input_probe.shape != [2, 2]
            || distances.shape != [2, 2]
            || read_f32s(distances)?.len() != 4
            || neighbors.descr != "<i4"
            || neighbors.shape != [2, 2]
        {
            return Err("overflow query probe shape/dtype mismatch".into());
        }
        let observed: Vec<i32> = neighbors
            .payload
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if observed != [0, -1, 1, -1] {
            return Err("overflow neighbor observations differ from release contract".into());
        }
        let squared = read_f32s(distances)?;
        if squared != [0.0, f32::INFINITY, 0.0, f32::INFINITY] {
            return Err(
                "overflow squared-distance observations differ from release contract".into(),
            );
        }
        let probe = metadata
            .get("overflow_neighbor_probe")
            .ok_or("overflow neighbor probe metadata missing")?;
        if probe["entrypoint"] != "fast_hdbscan.numba_kdtree.parallel_tree_query"
            || probe["output_rdist"] != true
            || probe["self_included"] != true
            || probe["intended_common_indices"] != json!([[0, 1], [1, 0]])
            || probe["intended_contract_source"] != "port contract, not upstream output"
        {
            return Err("overflow probe metadata differs from stored query observations".into());
        }
    }
    Ok(())
}

#[test]
fn all_manifest_cases_load_with_matching_metadata_and_npy_headers() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let schema = read_json(&root.join("schema.json"));
    assert_eq!(manifest["schema_version"], schema["schema_version"]);
    let cases = manifest["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let mut ids = HashSet::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        assert!(ids.insert(id), "duplicate manifest case {id}");
        validate_case(&root, id, case, &manifest, &schema).unwrap_or_else(|e| panic!("{id}: {e}"));
    }
}

#[test]
fn rejects_truncated_and_wrong_magic_npy_headers() {
    assert!(
        parse_npy(b"bad")
            .unwrap_err()
            .contains("invalid NPY magic/header")
    );
    assert!(
        parse_npy(b"\x93NUMPY\x01\x00\x10\x00x")
            .unwrap_err()
            .contains("truncated NPY header")
    );
    assert!(
        parse_shape("2,,3")
            .unwrap_err()
            .contains("invalid NPY shape dimension")
    );
    assert!(
        parse_shape("-1,2")
            .unwrap_err()
            .contains("invalid NPY shape dimension")
    );
    assert!(
        parse_shape("2,word")
            .unwrap_err()
            .contains("invalid NPY shape dimension")
    );
    assert_eq!(parse_shape("2,3,").unwrap(), vec![2, 3]);
    let header = b"{'descr': '<f4', 'fortran_order': Maybe, 'shape': (2,)}";
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header);
    assert!(
        parse_npy(&bytes)
            .unwrap_err()
            .contains("invalid NPY fortran_order value")
    );
}

#[test]
fn rejects_mutated_temp_copy_metadata_and_archive() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let id = manifest["cases"][0]["id"].as_str().unwrap();
    let temp = tempfile::tempdir().unwrap();
    let json_path = temp.path().join(format!("{id}.json"));
    std::fs::copy(root.join(format!("{id}.json")), &json_path).unwrap();
    let mut metadata = read_json(&json_path);
    metadata["case_id"] = json!("wrong-case");
    std::fs::write(&json_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    std::fs::copy(
        root.join(format!("{id}.npz")),
        temp.path().join(format!("{id}.npz")),
    )
    .unwrap();
    let schema = read_json(&root.join("schema.json"));
    assert_error_contains(
        validate_case(temp.path(), id, &manifest["cases"][0], &manifest, &schema),
        "metadata version or case id mismatch",
    );
    let broken = temp.path().join("broken.npz");
    std::fs::write(&broken, b"not a zip archive").unwrap();
    assert!(load_npz(&broken).unwrap_err().contains("invalid NPZ"));
}

#[test]
fn rejects_status_dtype_rank_provenance_finite_map_and_cardinality_mutations() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let schema = read_json(&root.join("schema.json"));
    let temp = tempfile::tempdir().unwrap();
    let reference = &manifest["cases"][0];
    let reference_id = reference["id"].as_str().unwrap();
    let degenerate = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["status"] == "validation_failure")
        .unwrap();
    let degenerate_id = degenerate["id"].as_str().unwrap();
    for id in [reference_id, degenerate_id] {
        std::fs::copy(
            root.join(format!("{id}.json")),
            temp.path().join(format!("{id}.json")),
        )
        .unwrap();
        std::fs::copy(
            root.join(format!("{id}.npz")),
            temp.path().join(format!("{id}.npz")),
        )
        .unwrap();
    }

    let ref_path = temp.path().join(format!("{reference_id}.json"));
    let original = read_json(&ref_path);
    for (name, mutate, expected_error) in [
        (
            "missing_required",
            0usize,
            "status reference requires array mst_weights",
        ),
        (
            "wrong_dtype",
            1,
            "labels catalogue dtype differs from schema",
        ),
        ("wrong_rank", 2, "labels shape differs from catalogue"),
        (
            "wrong_provenance",
            3,
            "fixture provenance does not match manifest release evidence",
        ),
        (
            "wrong_finite_mapping",
            4,
            "finite_index array differs from metadata",
        ),
        (
            "wrong_cardinality",
            5,
            "linkage shape differs from catalogue",
        ),
        (
            "wrong_structured_fields",
            6,
            "condensed_tree catalogue dtype differs from schema",
        ),
        ("unknown_status", 7, "unknown status invented_status"),
        (
            "nonintegral_shape",
            8,
            "metadata input shape dimension must be a nonnegative integer",
        ),
        (
            "wrong_reference_dimensions",
            9,
            "reference shape/dimensions metadata mismatch",
        ),
        (
            "wrong_comparison_blobs",
            10,
            "fixture provenance does not match manifest release evidence",
        ),
        (
            "wrong_signatures",
            11,
            "fixture provenance does not match manifest release evidence",
        ),
        (
            "wrong_release_evidence",
            12,
            "fixture provenance does not match manifest release evidence",
        ),
    ] {
        let mut value = original.clone();
        match mutate {
            0 => {
                value["arrays"]
                    .as_object_mut()
                    .unwrap()
                    .remove("mst_weights");
            }
            1 => value["arrays"]["labels"]["dtype"] = json!("<f4"),
            2 => value["arrays"]["labels"]["shape"] = json!([1, 48]),
            3 => value["reference"]["version"] = json!("0.0.0"),
            4 => value["finite_index"] = json!([47]),
            5 => value["arrays"]["linkage"]["shape"] = json!([1, 4]),
            6 => {
                value["arrays"]["condensed_tree"]["dtype"] = json!([{"name":"child","dtype":"<i8"}])
            }
            7 => value["status"] = json!("invented_status"),
            8 => {
                value["input_shape"] = json!(["invalid", 2]);
                value["reference"]["shape"] = value["input_shape"].clone();
            }
            9 => value["reference"]["dimensions"] = json!(50),
            10 => value["reference"]["comparison_commit_blob_ids"]["hdbscan.py"] = json!("wrong"),
            11 => value["reference"]["signatures"]["parallel_boruvka"] = json!("wrong"),
            12 => value["reference"]["release_source_evidence"] = json!("wrong"),
            _ => unreachable!(),
        }
        std::fs::write(&ref_path, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = validate_case(temp.path(), reference_id, reference, &manifest, &schema)
            .expect_err(&format!("accepted {name}"));
        assert!(
            error.contains(expected_error),
            "{name}: expected {expected_error:?}, got {error:?}"
        );
    }

    let degenerate_path = temp.path().join(format!("{degenerate_id}.json"));
    let mut value = read_json(&degenerate_path);
    value["arrays"]["mst_weights"] = json!({"dtype":"<f8","shape":[0]});
    std::fs::write(&degenerate_path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert_error_contains(
        validate_case(temp.path(), degenerate_id, degenerate, &manifest, &schema),
        "status validation_failure forbids array mst_weights",
    );

    let finite_npz = temp.path().join(format!("{reference_id}.npz"));
    replace_npz_entry(&finite_npz, "finite_index.npy", |bytes| {
        let header = parse_npy(bytes).unwrap();
        let payload_start = bytes.len() - header.payload.len();
        let last = payload_start + header.payload.len() - 8;
        bytes[last..last + 8].copy_from_slice(&999i64.to_le_bytes());
    });
    std::fs::write(&ref_path, serde_json::to_vec(&original).unwrap()).unwrap();
    assert_error_contains(
        validate_case(temp.path(), reference_id, reference, &manifest, &schema),
        "finite_index is not strictly increasing or is out of range",
    );
}

#[test]
fn rejects_invalid_linkage_condensed_label_and_core_payload_values() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let schema = read_json(&root.join("schema.json"));
    let case = &manifest["cases"][0];
    let id = case["id"].as_str().unwrap();
    for (array, bad_value, offset, expected_error) in [
        (
            "linkage.npy",
            999.0f64.to_le_bytes().to_vec(),
            0usize,
            "linkage row 0 left is not a prior integral node index",
        ),
        (
            "condensed_tree.npy",
            f64::NAN.to_le_bytes().to_vec(),
            16usize,
            "condensed_tree row 0 has invalid lambda",
        ),
        (
            "labels.npy",
            (-2i64).to_le_bytes().to_vec(),
            0usize,
            "labels contains invalid label index",
        ),
        (
            "core_distances.npy",
            (-1.0f32).to_le_bytes().to_vec(),
            0usize,
            "core distances must be nonnegative and not NaN",
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let json_path = temp.path().join(format!("{id}.json"));
        let npz_path = temp.path().join(format!("{id}.npz"));
        std::fs::copy(root.join(format!("{id}.json")), &json_path).unwrap();
        std::fs::copy(root.join(format!("{id}.npz")), &npz_path).unwrap();
        replace_npz_entry(&npz_path, array, |bytes| {
            let header = parse_npy(bytes).unwrap();
            let payload_start = bytes.len() - header.payload.len();
            let at = payload_start + offset;
            bytes[at..at + bad_value.len()].copy_from_slice(&bad_value);
        });
        assert_error_contains(
            validate_case(temp.path(), id, case, &manifest, &schema),
            expected_error,
        );
    }
}

#[test]
fn rejects_each_missing_manifest_provenance_key() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let schema = read_json(&root.join("schema.json"));
    let case = &manifest["cases"][0];
    let case_id = case["id"].as_str().unwrap();
    let temp = tempfile::tempdir().unwrap();
    std::fs::copy(
        root.join(format!("{case_id}.json")),
        temp.path().join(format!("{case_id}.json")),
    )
    .unwrap();
    std::fs::copy(
        root.join(format!("{case_id}.npz")),
        temp.path().join(format!("{case_id}.npz")),
    )
    .unwrap();

    let required = schema["manifest_reference_required"]
        .as_array()
        .expect("schema manifest_reference_required array");
    assert!(!required.is_empty());
    for key in required {
        let key = key.as_str().expect("manifest reference key string");
        let mut mutated = manifest.clone();
        assert!(
            mutated["reference"]
                .as_object_mut()
                .unwrap()
                .remove(key)
                .is_some()
        );
        assert_error_contains(
            validate_case(temp.path(), case_id, case, &mutated, &schema),
            &format!("manifest reference missing required key {key}"),
        );
    }
    let mut mutated = manifest.clone();
    mutated["reference"]["unknown_proof"] = Value::Null;
    assert_error_contains(
        validate_case(temp.path(), case_id, case, &mutated, &schema),
        "manifest reference has unknown keys",
    );
}
