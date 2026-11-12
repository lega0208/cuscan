//! Device memory availability for admission planning.

use crate::BackendError;
use burn::backend::DispatchDevice;
use burn::tensor::Device;
use std::path::{Component, Path, PathBuf};

pub const MEMORY_RESERVE_BYTES: u64 = 64 * 1024 * 1024;
pub const MEMORY_SAFETY_NUMERATOR: u64 = 9;
pub const MEMORY_SAFETY_DENOMINATOR: u64 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub limit_bytes: Option<u64>,
}

impl MemoryBudget {
    pub fn usable_bytes(&self, live_bytes: u64) -> u64 {
        let free = (self.free_bytes / MEMORY_SAFETY_DENOMINATOR)
            .saturating_mul(MEMORY_SAFETY_NUMERATOR)
            .saturating_add(
                (self.free_bytes % MEMORY_SAFETY_DENOMINATOR) * MEMORY_SAFETY_NUMERATOR
                    / MEMORY_SAFETY_DENOMINATOR,
            );
        free.saturating_add(live_bytes)
            .min(self.limit_bytes.unwrap_or(u64::MAX))
            .min(self.total_bytes)
            .saturating_sub(MEMORY_RESERVE_BYTES)
    }
}

pub fn resolve_memory_budget(
    device: &Device,
    limit_bytes: Option<u64>,
) -> Result<MemoryBudget, BackendError> {
    let DispatchDevice::Cube(cube) = device.as_dispatch();
    let client = cube.client();
    // Dropped dedicated buffers may remain cached until another allocation.
    // Reclaim only idle pages before observing free memory, otherwise a second
    // fit can reject a plan solely because the previous fit's cache is retained.
    client.memory_cleanup().map_err(|error| {
        BackendError::Unavailable(format!(
            "memory availability cleanup failed for {cube:?}: {error:?}"
        ))
    })?;
    cubecl::future::block_on(client.sync()).map_err(|error| {
        BackendError::Unavailable(format!(
            "memory availability synchronization failed for {cube:?}: {error:?}"
        ))
    })?;
    let free_bytes = match cube {
        cubecl::Device::Cuda(cuda) => {
            crate::cuda_backend::free_memory(cuda.index).map_err(BackendError::Unavailable)?
        }
        cubecl::Device::Cpu(_) => cpu_available(
            Path::new("/proc/meminfo"),
            Path::new("/proc/self/cgroup"),
            Path::new("/proc/self/mountinfo"),
        )?,
        other => {
            return Err(BackendError::Unavailable(format!(
                "unsupported memory device: {other:?}"
            )));
        }
    };
    let total_bytes = client.properties().memory.max_memory().ok_or_else(|| {
        BackendError::Unavailable(format!("device total memory unavailable: {cube:?}"))
    })?;
    Ok(MemoryBudget {
        total_bytes,
        free_bytes,
        limit_bytes,
    })
}

fn read(path: &Path) -> Result<String, BackendError> {
    std::fs::read_to_string(path)
        .map_err(|error| BackendError::Unavailable(format!("{}: {error}", path.display())))
}

fn invalid(path: &Path, detail: &str) -> BackendError {
    BackendError::Unavailable(format!("{}: {detail}", path.display()))
}

fn cpu_available(meminfo: &Path, cgroup: &Path, mountinfo: &Path) -> Result<u64, BackendError> {
    let text = read(meminfo)?;
    let mut available = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("MemAvailable:") {
            let mut fields = value.split_whitespace();
            let kib = fields
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| invalid(meminfo, "invalid MemAvailable"))?;
            if fields.next() != Some("kB") || fields.next().is_some() || available.is_some() {
                return Err(invalid(meminfo, "invalid or duplicate MemAvailable"));
            }
            available = Some(
                kib.checked_mul(1024)
                    .ok_or_else(|| invalid(meminfo, "MemAvailable overflow"))?,
            );
        }
    }
    let available = available.ok_or_else(|| invalid(meminfo, "missing MemAvailable"))?;
    Ok(cgroup_headroom(cgroup, mountinfo)?.map_or(available, |cap| available.min(cap)))
}

fn absolute_path(value: &str, source: &Path) -> Result<PathBuf, BackendError> {
    let path = Path::new(value);
    if !path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::CurDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid(source, "invalid cgroup path"));
    }
    Ok(path.to_path_buf())
}

fn unescape_mount(value: &str, source: &Path) -> Result<String, BackendError> {
    let bytes = value.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            if i + 3 >= bytes.len()
                || !bytes[i + 1..i + 4]
                    .iter()
                    .all(|b| (b'0'..=b'7').contains(b))
            {
                return Err(invalid(source, "invalid mountinfo escape"));
            }
            let code = (u16::from(bytes[i + 1] - b'0') * 64)
                + (u16::from(bytes[i + 2] - b'0') * 8)
                + u16::from(bytes[i + 3] - b'0');
            result
                .push(u8::try_from(code).map_err(|_| invalid(source, "invalid mountinfo escape"))?);
            i += 4;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(result).map_err(|_| invalid(source, "non-UTF-8 mountinfo path"))
}

fn cgroup_headroom(cgroup: &Path, mountinfo: &Path) -> Result<Option<u64>, BackendError> {
    let membership = match std::fs::read_to_string(cgroup) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(BackendError::Unavailable(format!(
                "{}: {error}",
                cgroup.display()
            )));
        }
    };
    let mut process = None;
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(invalid(cgroup, "malformed cgroup membership"));
        };
        if id == "0" && controllers.is_empty() {
            if process.replace(absolute_path(path, cgroup)?).is_some() {
                return Err(invalid(cgroup, "duplicate unified cgroup membership"));
            }
        }
    }
    let Some(process) = process else {
        return Ok(None);
    };
    let mounts = read(mountinfo)?;
    let mut best: Option<(PathBuf, PathBuf)> = None;
    let mut has_v2_mount = false;
    for line in mounts.lines() {
        let (left, right) = line
            .split_once(" - ")
            .ok_or_else(|| invalid(mountinfo, "malformed mountinfo"))?;
        let mut post = right.split_whitespace();
        if post.next() != Some("cgroup2") {
            continue;
        }
        has_v2_mount = true;
        let fields: Vec<_> = left.split_whitespace().collect();
        if fields.len() < 6 {
            return Err(invalid(mountinfo, "malformed cgroup2 mount"));
        }
        let root = absolute_path(&unescape_mount(fields[3], mountinfo)?, mountinfo)?;
        let mount = absolute_path(&unescape_mount(fields[4], mountinfo)?, mountinfo)?;
        if process.starts_with(&root)
            && best
                .as_ref()
                .is_none_or(|(old, _)| root.as_os_str().len() > old.as_os_str().len())
        {
            best = Some((root, mount));
        }
    }
    let Some((root, mount)) = best else {
        return if has_v2_mount {
            Err(invalid(
                mountinfo,
                "unified cgroup mount not found for process",
            ))
        } else {
            Ok(None)
        };
    };
    let relative = process
        .strip_prefix(&root)
        .map_err(|_| invalid(cgroup, "process outside cgroup mount"))?;
    let mut current = mount.join(relative);
    let mut cap: Option<u64> = None;
    loop {
        let max_path = current.join("memory.max");
        let usage_path = current.join("memory.current");
        let max = std::fs::read_to_string(&max_path);
        let usage = std::fs::read_to_string(&usage_path);
        match (max, usage) {
            (Ok(max), Ok(usage)) => {
                let usage = usage
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| invalid(&usage_path, "invalid memory.current"))?;
                if max.trim() != "max" {
                    let limit = max
                        .trim()
                        .parse::<u64>()
                        .map_err(|_| invalid(&max_path, "invalid memory.max"))?;
                    let remaining = limit.saturating_sub(usage);
                    cap = Some(cap.map_or(remaining, |previous| previous.min(remaining)));
                }
            }
            (Err(max), Err(usage))
                if max.kind() == std::io::ErrorKind::NotFound
                    && usage.kind() == std::io::ErrorKind::NotFound =>
            {
                let metadata = std::fs::metadata(&current).map_err(|error| {
                    BackendError::Unavailable(format!("{}: {error}", current.display()))
                })?;
                if !metadata.is_dir() {
                    return Err(invalid(&current, "cgroup is not a directory"));
                }
            }
            (Err(error), _) => {
                return Err(BackendError::Unavailable(format!(
                    "{}: {error}",
                    max_path.display()
                )));
            }
            (_, Err(error)) => {
                return Err(BackendError::Unavailable(format!(
                    "{}: {error}",
                    usage_path.display()
                )));
            }
        }
        if current == mount {
            break;
        }
        if !current.pop() {
            return Err(invalid(cgroup, "invalid cgroup hierarchy"));
        }
    }
    Ok(cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Fixture {
        dir: tempfile::TempDir,
        mem: PathBuf,
        group: PathBuf,
        mounts: PathBuf,
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let mem = dir.path().join("meminfo");
            let group = dir.path().join("cgroup");
            let mounts = dir.path().join("mountinfo");
            let root = dir.path().join("cg");
            fs::create_dir(&root).unwrap();
            fs::write(&mem, "MemAvailable: 1024 kB\n").unwrap();
            fs::write(&group, "0::/\n").unwrap();
            fs::write(
                &mounts,
                format!("1 0 0:1 / {} rw - cgroup2 cgroup rw\n", root.display()),
            )
            .unwrap();
            fs::write(root.join("memory.max"), "max\n").unwrap();
            fs::write(root.join("memory.current"), "0\n").unwrap();
            Self {
                dir,
                mem,
                group,
                mounts,
                root,
            }
        }
        fn available(&self) -> Result<u64, BackendError> {
            cpu_available(&self.mem, &self.group, &self.mounts)
        }
    }

    #[test]
    fn arithmetic_limits_and_overflow() {
        let b = MemoryBudget {
            total_bytes: u64::MAX,
            free_bytes: u64::MAX,
            limit_bytes: None,
        };
        assert_eq!(
            b.usable_bytes(0),
            (u64::MAX / 10 * 9 + (u64::MAX % 10) * 9 / 10) - MEMORY_RESERVE_BYTES
        );
        assert_eq!(b.usable_bytes(u64::MAX), u64::MAX - MEMORY_RESERVE_BYTES);
        assert_eq!(
            MemoryBudget {
                limit_bytes: Some(10),
                ..b
            }
            .usable_bytes(0),
            0
        );
        assert_eq!(
            MemoryBudget {
                total_bytes: MEMORY_RESERVE_BYTES + 7,
                ..b
            }
            .usable_bytes(0),
            7
        );
    }

    #[test]
    fn finite_max_and_absent_cgroup() {
        let f = Fixture::new();
        assert_eq!(f.available().unwrap(), 1_048_576);
        fs::write(f.root.join("memory.max"), "1000").unwrap();
        fs::write(f.root.join("memory.current"), "400").unwrap();
        assert_eq!(f.available().unwrap(), 600);
        fs::write(f.root.join("memory.current"), "1001").unwrap();
        assert_eq!(f.available().unwrap(), 0);
        fs::write(&f.group, "1:cpu:/\n").unwrap();
        assert_eq!(f.available().unwrap(), 1_048_576);
    }

    #[test]
    fn nested_hierarchy_checks_parent_and_mount_root() {
        let f = Fixture::new();
        let child = f.root.join("child");
        fs::create_dir(&child).unwrap();
        fs::write(&f.group, "0::/child\n").unwrap();
        fs::write(child.join("memory.max"), "5000").unwrap();
        fs::write(child.join("memory.current"), "2000").unwrap();
        fs::write(f.root.join("memory.max"), "1000").unwrap();
        fs::write(f.root.join("memory.current"), "800").unwrap();
        assert_eq!(f.available().unwrap(), 200);
    }

    #[test]
    fn root_without_controller_files_preserves_child_limit() {
        let f = Fixture::new();
        fs::remove_file(f.root.join("memory.max")).unwrap();
        fs::remove_file(f.root.join("memory.current")).unwrap();
        let child = f.root.join("init.scope");
        fs::create_dir(&child).unwrap();
        fs::write(&f.group, "0::/init.scope\n").unwrap();
        fs::write(child.join("memory.max"), "1000").unwrap();
        fs::write(child.join("memory.current"), "400").unwrap();
        assert_eq!(f.available().unwrap(), 600);
    }

    #[test]
    fn missing_controller_and_incomplete_pair() {
        let f = Fixture::new();
        fs::remove_file(f.root.join("memory.max")).unwrap();
        fs::remove_file(f.root.join("memory.current")).unwrap();
        let child = f.root.join("child");
        fs::create_dir(&child).unwrap();
        fs::write(&f.group, "0::/child\n").unwrap();
        assert_eq!(f.available().unwrap(), 1_048_576);
        fs::write(child.join("memory.max"), "1000").unwrap();
        assert!(
            f.available()
                .unwrap_err()
                .to_string()
                .contains("memory.current")
        );
        fs::remove_file(child.join("memory.max")).unwrap();
        fs::write(child.join("memory.current"), "400").unwrap();
        assert!(
            f.available()
                .unwrap_err()
                .to_string()
                .contains("memory.max")
        );
    }

    #[test]
    fn mount_root_offset_limits_visible_hierarchy() {
        let f = Fixture::new();
        let child = f.root.join("child");
        fs::create_dir(&child).unwrap();
        fs::write(&f.group, "0::/tenant/child\n").unwrap();
        fs::write(
            &f.mounts,
            format!(
                "1 0 0:1 /tenant {} rw - cgroup2 cgroup rw\n",
                f.root.display()
            ),
        )
        .unwrap();
        fs::write(f.root.join("memory.max"), "1000").unwrap();
        fs::write(f.root.join("memory.current"), "750").unwrap();
        fs::write(child.join("memory.max"), "2000").unwrap();
        fs::write(child.join("memory.current"), "500").unwrap();
        assert_eq!(f.available().unwrap(), 250);
    }

    #[test]
    fn malformed_or_unreadable_sources_fail() {
        let f = Fixture::new();
        fs::write(&f.mem, "MemAvailable: broken kB\n").unwrap();
        assert!(f.available().unwrap_err().to_string().contains("meminfo"));
        fs::write(&f.mem, "MemAvailable: 4 kB\n").unwrap();
        fs::remove_file(&f.mem).unwrap();
        assert!(f.available().unwrap_err().to_string().contains("meminfo"));
        fs::write(&f.mem, "MemAvailable: 4 kB\n").unwrap();
        fs::write(f.root.join("memory.max"), "bad").unwrap();
        assert!(
            f.available()
                .unwrap_err()
                .to_string()
                .contains("memory.max")
        );
        fs::write(f.root.join("memory.max"), "100").unwrap();
        fs::write(f.root.join("memory.current"), "bad").unwrap();
        assert!(
            f.available()
                .unwrap_err()
                .to_string()
                .contains("memory.current")
        );
        fs::write(&f.group, "malformed").unwrap();
        assert!(f.available().unwrap_err().to_string().contains("cgroup"));
        let _ = &f.dir;
    }
}
