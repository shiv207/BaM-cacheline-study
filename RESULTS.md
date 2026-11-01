# Compressed and adaptive cache lines vs BaM's raw 4KB lines

## Read this first

- **The baseline is a reconstruction.** `bam_toy.py` was not in the repo or anywhere on this
  machine, so `bam_baseline.py` re-implements it from the brief and the BaM paper. No result here
  has been compared with your original. To swap it in, change `BamAdapter` and `LoadAllAdapter` in
  `stores.py`, then rerun. A test requires the new cache code to match the baseline's counters
  exactly in its raw, full-line mode, so that test will show any behavioural difference.
- **This measures bytes, device ops and hit rate, not GPU contention or GPU speed.** Nothing here
  supports a claim of GPU speedup.
- **Storage time is modelled, not measured.** Reads came from the OS page cache. "NAND" and "Optane"
  apply the paper's Table 2 read rates to the recorded op sizes. Decode time is measured.
- **The brief says macOS; this ran on Linux** (Python 3.14, numpy 2.5.4).

## Verdict

| Claim | Result |
|---|---|
| Compression reduces bytes | **Yes, but it depends on locality or cache size.** -1.3% on `random` and unordered `web-Google`, -15 to -20% with ID locality, -31 to -51% at a cache of 50% of the graph. |
| Compression reduces misses | **Yes**, by the same percentages. |
| Compression wins on total time | **No, not at the numpy decoder's speed.** It lost at every sensible line size. It needs to be about 2 to 48 times faster, depending on the case. |
| Adaptive reduces bytes | **Yes against 4KB lines** (-76 to -86%), **but a fixed 512B line does slightly better** on all four graphs. |
| Adaptive reduces misses | **No.** Raw adaptive has 1.6 to 11.5% more misses. Adaptive + compression has about 5% fewer only on graphs with locality. |
| Results stay correct | **Yes.** All 1,700 benchmark runs matched load-all, and all 423 tests pass. |

## Default point

Each row is the mean of 5 query seeds, 2,000 uniform-random 2-hop queries each, with a cold cache
at the start of each run. Cache is 2% of the raw graph, lines are 4KB and N=3. All stores share the
same cache budget in bytes.

- **I/O amp** is storage bytes divided by the raw uint32 bytes the queries asked for, using that
  same denominator for every format. **amp (unique)** counts each list once.
- **decode (measured)** is in-run, with one numpy call per frontier batch.
  **decode (bulk)** is calibrated ns/edge × edges, with no per-call overhead.
- `bam_raw` shows 0 decode because its copy happens inside the baseline, untimed. `adaptive_raw`'s
  roughly 6ms is the same kind of copy, not extra work.

| graph | store | storage MB | I/O amp | amp (unique) | misses | hit rate | coalesced % | decode ms (measured) | decode ms (bulk) | wall s | NAND I/O ms | Optane I/O ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | load_all | 64.00 | 29.48 | 30.03 | 0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.32 | 9.14 | 8.40 |
| random | bam_raw | 138.33 | 63.71 | 64.90 | 33772 | 0.020 | 0.0 | 0.0 | 0.0 | 0.45 | 45.03 | 22.51 |
| random | comp_straddle | 136.54 | 62.89 | 64.06 | 33334 | 0.028 | 0.1 | 135.0 | 28.0 | 0.59 | 44.45 | 22.22 |
| random | comp_pad | 135.10 | 62.23 | 63.39 | 32984 | 0.028 | 0.1 | 135.3 | 28.0 | 0.59 | 43.98 | 21.99 |
| random | adaptive_raw | 19.64 | 9.04 | 9.21 | 34360 | 0.003 | 0.0 | 6.1 | 0.0 | 0.48 | 45.92 | 7.02 |
| random | adaptive_comp | 19.07 | 8.78 | 8.95 | 34170 | 0.004 | 0.1 | 135.7 | 28.0 | 0.61 | 45.70 | 6.92 |
| powerlaw | load_all | 45.39 | 41.36 | 42.42 | 0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.33 | 6.48 | 5.96 |
| powerlaw | bam_raw | 52.65 | 47.64 | 48.88 | 12854 | 0.123 | 42.0 | 0.0 | 0.0 | 0.38 | 17.14 | 8.57 |
| powerlaw | comp_straddle | 42.30 | 38.24 | 39.24 | 10326 | 0.193 | 49.0 | 132.2 | 10.5 | 0.52 | 13.77 | 6.88 |
| powerlaw | comp_pad | 42.41 | 38.34 | 39.34 | 10354 | 0.191 | 48.9 | 131.7 | 10.5 | 0.52 | 13.81 | 6.90 |
| powerlaw | adaptive_raw | 12.89 | 11.68 | 11.99 | 14319 | 0.022 | 42.0 | 6.4 | 0.0 | 0.42 | 20.44 | 3.67 |
| powerlaw | adaptive_comp | 11.28 | 10.22 | 10.48 | 12300 | 0.038 | 49.0 | 135.8 | 10.5 | 0.55 | 18.01 | 3.22 |
| web-Google | load_all | 20.42 | 34.15 | 40.03 | 0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.27 | 2.92 | 2.68 |
| web-Google | bam_raw | 50.59 | 84.57 | 99.15 | 12352 | 0.022 | 0.1 | 0.0 | 0.0 | 0.35 | 16.47 | 8.23 |
| web-Google | comp_straddle | 49.93 | 83.45 | 97.85 | 12189 | 0.030 | 0.2 | 118.7 | 7.4 | 0.47 | 16.25 | 8.13 |
| web-Google | comp_pad | 49.52 | 82.77 | 97.04 | 12089 | 0.031 | 0.1 | 117.9 | 7.4 | 0.47 | 16.12 | 8.06 |
| web-Google | adaptive_raw | 6.98 | 11.67 | 13.68 | 12555 | 0.006 | 0.1 | 6.1 | 0.0 | 0.36 | 16.78 | 2.54 |
| web-Google | adaptive_comp | 6.80 | 11.37 | 13.33 | 12474 | 0.007 | 0.2 | 118.5 | 7.4 | 0.48 | 16.68 | 2.50 |
| web-Google-bfs | load_all | 20.42 | 33.96 | 40.07 | 0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.27 | 2.92 | 2.68 |
| web-Google-bfs | bam_raw | 32.50 | 54.03 | 63.74 | 7934 | 0.126 | 28.5 | 0.0 | 0.0 | 0.31 | 10.58 | 5.29 |
| web-Google-bfs | comp_straddle | 27.72 | 46.09 | 54.37 | 6767 | 0.185 | 34.1 | 110.8 | 6.3 | 0.41 | 9.02 | 4.51 |
| web-Google-bfs | comp_pad | 27.60 | 45.89 | 54.14 | 6738 | 0.185 | 34.1 | 109.6 | 6.3 | 0.40 | 8.98 | 4.49 |
| web-Google-bfs | adaptive_raw | 6.52 | 10.84 | 12.79 | 8625 | 0.050 | 28.5 | 5.5 | 0.0 | 0.30 | 12.23 | 2.04 |
| web-Google-bfs | adaptive_comp | 5.95 | 9.90 | 11.68 | 7547 | 0.092 | 34.1 | 111.1 | 6.3 | 0.42 | 10.96 | 1.84 |

Every sweep point, 340 rows in the same columns, is in `results/table.md`.

| Graph | Nodes | Edges | Raw bytes/edge | Compressed | Compressed size |
|---|---:|---:|---:|---:|---:|
| random | 1,000,000 | 15,999,870 | 4 | 2.77 | 69% |
| powerlaw (synthetic, ID locality) | 1,000,000 | 11,347,768 | 4 | 1.56 | 39% |
| web-Google (SNAP, original order) | 875,713 | 5,105,039 | 4 | 2.80 | 70% |
| web-Google-bfs (BFS relabel) | 875,713 | 5,105,039 | 4 | 1.86 | 47% |

## Findings

**1. Adaptive is roughly a fixed 512B line, with fewer device ops.** At the default point it cuts
bytes by 76 to 86%, but plain raw 512B lines cut slightly more. Amplification is 8.5 to 11.1 for
512B lines and 9.0 to 11.7 for adaptive. What adaptive gains is 3 to 20% fewer device ops than 512B
lines, and it keeps 4KB frames, so it needs one eighth of the cache tags (tag RAM is not counted
here). The upgrade rule did not pay for itself in bytes: amplification kept falling as N grew and
was lowest with upgrades off (`amp_vs_threshold.png`). It also makes large caches worse, because
lines stay resident long enough to trigger upgrades. On `random`, upgrades rise from 4 to 3,768
and amplification from 9.0 to 12.2 as the cache grows from 0.1% to 50%.

**2. A smaller file does not by itself mean fewer reads.** `random` and `web-Google` compress to
about 70% of their size, yet bytes read fall only 1.3%. A cold list costs one line either way, and
compression only helps when a line's extra lists get used. That needs locality (-15 to -20%, and
coalescing up from 28.5% to 34.1% on `web-Google-bfs`) or a cache big enough to benefit from the
higher density. At a 50% cache, misses fall 31% on `random` and `web-Google`, 50% on `powerlaw` and
51% on `web-Google-bfs`. BFS relabelling is free to the raw baseline too: it cuts `bam_raw` bytes
by 36%, more than compression does at this cache size.

**3. Compression loses on total time at this decoder speed.** The numpy decoder runs at 38 to 52
ns/edge, plus about 20µs per call. The I/O time compression saves is worth much less:

| Case (`comp_straddle`) | Break-even decode, NAND | Break-even decode, Optane |
|---|---:|---:|
| Default, no locality (`random`, `web-Google`) | 1.1 to 1.4 ns/edge | 0.5 to 0.7 ns/edge |
| Default, with locality | 10 to 12 ns/edge | 5 to 6 ns/edge |
| 50% cache | 15 to 20 ns/edge | 7.5 to 10 ns/edge |

So even an ideal bulk numpy decode loses in all of these cases. With measured decode, compressed
stores beat the baseline in 0% of runs. The only plain-compression wins on mean results were NAND at 64KB lines on
the graphs with locality, by under 2%, and that is a line size nobody would choose. A native
decoder might reach the 10 to 20 ns/edge range; that was not measured here, so it remains an open
question. Encoding is cheap: 0.3 to 1.1s per graph.

**4. Whether adaptive saves time depends on the device.** Under the Optane model raw adaptive cuts I/O time
by 57 to 69%, because small reads are cheaper there. Under the NAND model I/O time gets 2 to 19%
*worse*, because a 512B read uses the same IOPS slot as a 4KB read and adaptive issues more ops.
Lower hit rates come with it, for example 0.126 down to 0.050 on `web-Google-bfs`. Combining both
ideas adds little: `adaptive_comp` reads only 3 to 13% fewer bytes than `adaptive_raw` and pays the
full decode cost, so it rarely wins on time (24% of runs on Optane, 11% on NAND, mostly oversized
lines).

**5. On-demand loading can lose to load-all, and padding hardly matters.** Over this workload,
load-all reads less than `bam_raw` at depth 2 on `random` (64 vs 138 MB) and less than every store
at depth 3 on all four graphs (for example 20 vs 33 MB for the best store on `web-Google-bfs`). That reflects how much of the
graph 2,000 queries touch, and load-all needs the whole graph in RAM, which is the limit BaM exists
to avoid. Padding lists to line boundaries changed bytes by at most about 1 percentage point. It
costs 0.2 to 4.4% more file and 50% more index RAM, so straddling is the better default.

## Limitations

- **Baseline:** reconstructed, as stated above. It uses second-chance clock as the brief describes,
  whereas the paper's clock is a counter that only skips pinned slots.
- **Time model:** a throughput model with one device, no queueing and no latency, using datasheet
  rates from the paper. Finding 4's direction depends on it. Real SSD arrays and PCIe limits could
  shift the balance, especially since BaM's bottleneck can be link bandwidth, which favours adaptive.
- **Decode cost:** single-threaded numpy on a CPU, with up to 8 benchmark workers running at once on
  24 cores. It does not predict the cost of a native or GPU decoder.
- **Wall time:** dominated by the Python simulator, so use it for rough comparison only.
- **Hit rates:** they are not directly comparable across formats, because stores probe different
  lines. An adaptive partial hit counts as a miss. Compare bytes, misses and ops first.
- **Workload:** one traversal type, uniform random seeds, no skew, no concurrency. Warps are
  simulated as batches of 32 lists, not GPU threads. Skewed query popularity would give the cache
  more to do and could help compression.
- **Sample:** each synthetic graph is one instance (seed 0), and seeds vary only the queries. The
  powerlaw generator's locality is my own parameter choice. Only one real graph was run, not
  soc-LiveJournal1, and its 5 million edges are small for BaM.
- **Sweeps:** one factor at a time around the default, so interactions are mostly untested. The "% of
  runs" figures depend on which points were swept and are not population estimates.
- **Metadata:** tag and bitmap RAM are not counted. Index RAM is in the CSVs.
- **Fixed sectors:** the 512B fixed-line comparison uses raw lists. Real devices may be formatted
  for 4KB sectors, which would rule out sub-4KB reads altogether.

## Reproduce

```sh
uv venv .venv --system-site-packages && uv pip install --python .venv/bin/python pytest matplotlib
.venv/bin/python -m pytest -q tests
mkdir -p data/snap && curl -o data/snap/web-Google.txt.gz https://snap.stanford.edu/data/web-Google.txt.gz
.venv/bin/python bench.py && .venv/bin/python plots.py && .venv/bin/python analyze.py
```

To rebuild the paper after a run: `.venv/bin/python paper/make_tables.py`, then compile
`paper/main.tex` with any LaTeX toolchain that has `acmart` (`tectonic main.tex` from `paper/`).

The run takes about 3.5 minutes on 8 workers and writes about 1.3GB of images to `data/work/`,
which is safe to delete. Bytes, ops and misses are deterministic; timings are not.

## Files

| Path | Contents |
|---|---|
| `bam_baseline.py` | Reconstructed baseline, unchanged by the harness |
| `codec.py` | Delta + varint encode, numpy decode, line packing |
| `stores.py` | `Store` interface, adapters, `CompressedBamStore`, `AdaptiveStore`, `k_hop` |
| `graphs.py` | Random and power-law generators, SNAP loader, BFS relabel |
| `bench.py`, `analyze.py`, `plots.py` | Sweep, paired analysis, charts |
| `paper/`, `BaM_cache_paper.pdf` | ACM-format paper source, generated tables, compiled PDF |
| `results/` | `runs.csv`, `summary.csv`, `formats.csv`, `decode_calibration.csv`, `table.md`, `analysis.md`, PNG plots |
