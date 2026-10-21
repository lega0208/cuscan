use serde_json::Value;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use zip::ZipArchive;

const FIXTURES: &str = "../fast-hdbscan/tests/fixtures";

#[derive(Debug, PartialEq)]
pub(crate) struct NpyHeader {
    pub(crate) descr: String,
    pub(crate) fortran: bool,
    pub(crate) shape: Vec<usize>,
    pub(crate) payload: Vec<u8>,
}

pub(crate) fn parse_npy(bytes: &[u8]) -> Result<NpyHeader, String> {
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

pub(crate) fn parse_shape(text: &str) -> Result<Vec<usize>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let tokens: Vec<_> = text.split(',').collect();
    if tokens.last().is_some_and(|token| token.trim().is_empty()) && tokens.len() == 1 {
        return Err("malformed NPY shape".into());
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

pub(crate) fn read_i64s(header: &NpyHeader) -> Result<Vec<i64>, String> {
    if header.descr != "<i8" || !header.payload.len().is_multiple_of(8) {
        return Err("expected contiguous little-endian i64 payload".into());
    }
    Ok(header
        .payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&bytes| i64::from_le_bytes(bytes))
        .collect())
}

pub(crate) fn read_f32s(header: &NpyHeader) -> Result<Vec<f32>, String> {
    if header.descr != "<f4" || !header.payload.len().is_multiple_of(4) {
        return Err("expected contiguous little-endian f32 payload".into());
    }
    Ok(header
        .payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect())
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

pub(crate) fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}
pub(crate) fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub(crate) fn load_npz(
    path: &Path,
) -> Result<std::collections::BTreeMap<String, NpyHeader>, String> {
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
