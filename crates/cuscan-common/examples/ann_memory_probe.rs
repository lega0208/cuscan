//! P0-3 CUDA memory probe. The default run uses small allocations only.
//!
//! `--overallocate BYTES` is destructive: it can exhaust device memory, spill
//! to host memory, terminate this process, or disrupt other GPU workloads.

use cubecl::{MemoryScope, client::Client, future::block_on};
use cudarc::driver::CudaContext;
use std::{error::Error, io, sync::Arc};

fn snapshot(ctx: &CudaContext, label: &str) -> Result<usize, Box<dyn Error>> {
    let (free, total) = ctx.mem_get_info()?;
    println!("{label}: free_bytes={free} total_bytes={total}");
    Ok(free)
}

fn check_client(client: &Client, label: &str, dedicated: bool) -> Result<(), Box<dyn Error>> {
    let expected = [19u8, 37, 91, 203];
    let handle = if dedicated {
        client.memory_dedicated_allocation((), |_| client.create_from_slice(&expected))
    } else {
        client.create_from_slice(&expected)
    };
    let result = block_on(client.sync_buffers([&handle]));
    println!("{label}: sync_buffers={result:?}");
    result?;
    let actual = client.read_one(handle)?;
    let matches = &*actual == expected.as_slice();
    println!("{label}: readback_matches={matches}");
    if !matches {
        return Err(io::Error::other("client readback mismatch").into());
    }
    Ok(())
}

fn dedicated_probe(client: &Client, context: &CudaContext) -> Result<(), Box<dyn Error>> {
    println!(
        "dedicated_report_before={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    let mut handles = Vec::new();
    for size in [
        1usize,
        511,
        512,
        513,
        1024 * 1024,
        4 * 1024 * 1024,
        108_000_000,
        720_000_000,
    ] {
        let before = snapshot(context, "dedicated_allocation_before")?;
        // Large allocations are opt-in via --dedicated and leave both a fixed
        // reserve and room for an allocation-sized driver overhead.
        let required = size.saturating_mul(2).saturating_add(512 * 1024 * 1024);
        if size >= 108_000_000 && before < required {
            println!(
                "dedicated_allocation: requested_bytes={size} skipped=insufficient_headroom required_free_bytes={required}"
            );
            continue;
        }
        let handle = client.memory_dedicated_allocation((), |_| client.empty(size));
        let sync = block_on(client.sync());
        let buffers = block_on(client.sync_buffers([&handle]));
        let after = snapshot(context, "dedicated_allocation_after")?;
        println!(
            "dedicated_allocation: requested_bytes={size} aligned_bytes={} free_delta_bytes={} sync={sync:?} sync_buffers={buffers:?}",
            size.next_multiple_of(512),
            before as i128 - after as i128
        );
        println!(
            "dedicated_report_after_allocation_bytes_{size}={:#?}",
            client.memory_report(MemoryScope::Device)
        );
        sync?;
        buffers?;
        handles.push(handle);
    }
    println!(
        "dedicated_report_before_phase_drop={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    // Keep the 1 MiB buffer as a shared input-like lifetime across phases.
    let shared = handles.remove(4);
    drop(handles);
    block_on(client.sync())?;
    let after_drop = snapshot(context, "dedicated_phase_after_drop")?;
    println!(
        "dedicated_report_after_phase_drop={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    let next = client.memory_dedicated_allocation((), |_| client.empty(3 * 1024 * 1024));
    let sync = block_on(client.sync_buffers([&shared, &next]));
    let after_next = snapshot(context, "dedicated_phase_after_next_allocation")?;
    println!(
        "dedicated_phase_next: requested_bytes={} free_delta_since_drop_bytes={} sync_buffers={sync:?}",
        3 * 1024 * 1024,
        after_drop as i128 - after_next as i128
    );
    println!(
        "dedicated_report_after_next_allocation={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    sync?;
    drop(next);
    drop(shared);
    block_on(client.sync())?;
    snapshot(context, "dedicated_before_cleanup")?;
    println!(
        "dedicated_report_before_cleanup={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    client.memory_cleanup()?;
    block_on(client.sync())?;
    snapshot(context, "dedicated_after_cleanup")?;
    println!(
        "dedicated_report_after_cleanup={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    check_client(client, "dedicated_create_readback", true)?;
    println!(
        "dedicated_report_after_create_readback={:#?}",
        client.memory_report(MemoryScope::Device)
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut overallocate = None;
    let mut dedicated = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dedicated" if !dedicated => dedicated = true,
            "--overallocate" if overallocate.is_none() => {
                overallocate = Some(args.next().ok_or("missing BYTES")?.parse::<usize>()?)
            }
            "--help" => {
                println!(
                    "Usage: ann_memory_probe [--dedicated] [--overallocate BYTES]\n\
                Default: safe, small CUDA allocations on device 0.\n\
                --dedicated additionally probes individual dedicated allocations, including large sizes when headroom permits.\n\
                --overallocate BYTES deliberately risks device OOM, spill, process abort and other GPU workloads."
                );
                return Ok(());
            }
            _ => return Err("unexpected or repeated argument".into()),
        }
    }

    let client = cubecl::Device::cuda(0)?.client();
    block_on(client.sync())?;
    // CubeCL has already retained the device's primary context. The first
    // cudarc retain permits a baseline query; the second is measured below.
    let context = CudaContext::new(0)?;
    println!(
        "device=0 client={} cudarc_primary={}",
        client.name(),
        context.is_primary()
    );
    snapshot(&context, "before_estimator_thread")?;
    let thread_context = Arc::clone(&context);
    let thread = std::thread::spawn(move || -> Result<(), String> {
        let (free, total) = thread_context.mem_get_info().map_err(|e| e.to_string())?;
        println!("estimator_before_context: free_bytes={free} total_bytes={total}");
        let query_context = CudaContext::new(0).map_err(|e| e.to_string())?;
        let (free, total) = query_context.mem_get_info().map_err(|e| e.to_string())?;
        println!(
            "estimator_after_context_query: free_bytes={free} total_bytes={total} same_primary_context={}",
            query_context.cu_ctx() == thread_context.cu_ctx()
        );
        Ok(())
    });
    thread
        .join()
        .map_err(|_| "estimator thread panicked")?
        .map_err(io::Error::other)?;
    let free = snapshot(&context, "after_estimator_thread")?;
    println!(
        "note: free-memory deltas include concurrent GPU activity and allocator caching; context identity is between cudarc retains, not directly observed from CubeCL's server thread"
    );

    // Leave ample headroom for unrelated allocations and allocator page rounding.
    if free < 64 * 1024 * 1024 {
        return Err("less than 64 MiB free; refusing even the small-allocation probe".into());
    }
    let sizes = [1usize, 511, 512, 513, 1024 * 1024, 4 * 1024 * 1024];
    let mut handles = Vec::new();
    for size in sizes {
        let before = snapshot(&context, "allocation_before")?;
        let handle = client.empty(size);
        let sync = block_on(client.sync());
        let buffers = block_on(client.sync_buffers([&handle]));
        let after = snapshot(&context, "allocation_after")?;
        println!(
            "allocation: requested_bytes={size} free_delta_bytes={} sync={sync:?} sync_buffers={buffers:?}",
            before as i128 - after as i128
        );
        sync?;
        buffers?;
        handles.push(handle);
    }
    drop(handles);
    block_on(client.sync())?;
    snapshot(&context, "after_drop")?;
    let before = snapshot(&context, "reallocation_before")?;
    let size = 3 * 1024 * 1024;
    let handle = client.empty(size);
    let sync = block_on(client.sync());
    let buffers = block_on(client.sync_buffers([&handle]));
    let after = snapshot(&context, "reallocation_after")?;
    println!(
        "reallocation: requested_bytes={size} free_delta_bytes={} sync={sync:?} sync_buffers={buffers:?}",
        before as i128 - after as i128
    );
    sync?;
    buffers?;
    drop(handle);
    block_on(client.sync())?;
    snapshot(&context, "before_cleanup")?;
    let cleanup = client.memory_cleanup();
    println!("memory_cleanup={cleanup:?}");
    cleanup?;
    block_on(client.sync())?;
    snapshot(&context, "after_cleanup")?;
    check_client(&client, "after_cleanup_client", false)?;
    if dedicated {
        dedicated_probe(&client, &context)?;
    }

    if let Some(size) = overallocate {
        let free = snapshot(&context, "destructive_before")?;
        if size <= free {
            return Err("--overallocate BYTES must exceed current free bytes".into());
        }
        eprintln!(
            "DESTRUCTIVE: requesting {size} bytes with {free} bytes free; process or other workloads may fail"
        );
        let handle = if dedicated {
            client.memory_dedicated_allocation((), |_| client.empty(size))
        } else {
            client.empty(size)
        };
        let sync = block_on(client.sync());
        println!("destructive: requested_bytes={size} sync={sync:?}");
        let buffers = block_on(client.sync_buffers([&handle]));
        println!("destructive: sync_buffers={buffers:?}");
        match snapshot(&context, "destructive_after") {
            Ok(_) => (),
            Err(error) => eprintln!("destructive memory query failed: {error}"),
        }
        drop(handle);
        match check_client(&client, "after_destructive_client", dedicated) {
            Ok(()) => println!("after_destructive_client: usable=true"),
            Err(error) => eprintln!("after_destructive_client: usable=false error={error}"),
        }
        println!(
            "note: allocation success does not by itself prove physical residency or WSL2 spill; measure system memory and touch pages separately"
        );
    } else {
        println!(
            "destructive over-allocation and WSL2 spill behavior: not tested (requires --overallocate BYTES)"
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ann_memory_probe: {error}");
        std::process::exit(1);
    }
}
