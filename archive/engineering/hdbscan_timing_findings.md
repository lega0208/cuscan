### Timing results

Release builds, synthetic comparison defaults, one excluded warmup and three measured fits. All requested settings preserved. Auto resolves to complete-graph brute force, not KD-tree.

Median seconds:

| Rows | Backend | Fit | Core distances | MST |
| ----- | ----- | ----- | ----- | ----- |
| 20k | CPU | 1.535 | 0.305 | 1.230 |
| 20k | CUDA | 0.601 | 0.436 | 0.162 |
| 100k | CPU | 32.202 | 6.109 | 26.154 |
| 100k | CUDA | 15.181 | 10.015 | 5.144

The 20k processes—including warmup and all three fits—finished in 7.90s CPU / 3.02s CUDA, under enforced 10s timeouts.

### Where the time goes

CUDA core distances: underfilled kernels and excessive launches.

- At 100k: 76,636 distance/top-k launches, consuming 8.609s device time.
- Another 305,762 merge launches consume 0.476s.
- Main kernels launch at most 32 blocks of 32 threads. Representative counters show 2.08% occupancy and 0.21% DRAM throughput: this is not saturated memory bandwidth.

MST: overwhelmingly candidate scans, not component reduction or host merging.

- At 100k, candidate scans consume 26.132s CPU / 5.120s CUDA in synchronized timing scopes.
- Four rounds remain: 1,187 → 62 → 10 → 2 → 1 components.
- Each round still scans the complete candidate domain, filtering same-component points.
- Representative CUDA counters show 94% L1/TEX utilization, 72% load/store-throttle stalls, but only 19% DRAM throughput. The candidate mapping produces widely strided loads across neighboring lanes.

Optimization priorities: CUDA neighbor-search parallelism/launch reduction, then MST candidate tiling and data reuse—especially on CPU. Reducer optimization would barely affect these totals.

Added detailed tracing and retained results in [analyze/hdbscan_hotspots/findings.json](analyze/hdbscan_hotspots/findings.json), with raw captures alongside it.
