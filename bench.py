"""Sweeps cache size, depth, block size and upgrade threshold; writes results/.

Measures bytes, ops and hit rate. It says nothing about GPU contention or GPU
throughput. Storage time is a device model, not a measurement: reads here come
from the OS page cache.
"""
import argparse
import csv
import hashlib
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import bam_baseline
import codec
import graphs
import stores

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WORK = ROOT / "data" / "work"

STORES = ["bam_raw", "comp_straddle", "comp_pad", "adaptive_raw", "adaptive_comp"]
DEFAULT = dict(cache_frac=0.02, depth=2, block=4096, threshold=3, order="sorted")
SWEEPS = {
    "cache_frac": [0.001, 0.005, 0.02, 0.1, 0.5],
    "depth": [1, 2, 3],
    "block": [512, 1024, 2048, 4096, 8192, 16384, 65536],
    "threshold": [0, 2, 3, 4, 6, 8, 99],
    "order": ["sorted", "shuffled"],
}

# Paper Table 2 read IOPS. One op costs its slower of command rate and transfer,
# with PCIe Gen4 x4 taken as 7 GB/s. Optane is a line through its 512B and 4KB rates.
DEVICES = {
    "nand": lambda size: max(1 / 750e3, size / 7e9),
    "optane": lambda size: 1 / 5.1e6 + max(size - 512, 0) * (1 / 1.5e6 - 1 / 5.1e6) / 3584,
}

LAYOUTS = {}  # filled before the workers fork


def build_graphs(args):
    made = []
    if "random" in args.graphs:
        made.append(("random", lambda: graphs.random_graph(args.nodes, 16, 0)))
    if "powerlaw" in args.graphs:
        made.append(("powerlaw", lambda: graphs.powerlaw_graph(args.nodes, 16, 0)))
    if "web-Google" in args.graphs:
        made.append(("web-Google", lambda: graphs.load_snap(args.snap, "web-Google")))
    if "web-Google-bfs" in args.graphs:
        made.append(("web-Google-bfs", lambda: graphs.load_snap(args.snap, "web-Google", "bfs")))
    out = {}
    for key, make in made:
        graph = make()
        graph.name = key
        out[key] = graph
    return out


def layout_for(graph, kind, block):
    """Builds each on-disk image once; workers inherit them through fork."""
    key = (graph.name, kind, block)
    if key not in LAYOUTS:
        folder = WORK / graph.name
        if kind == "raw":
            LAYOUTS[key] = stores.RawLayout(folder / "raw.bin", graph)
        else:
            straddle = kind == "straddle"
            started = time.perf_counter()
            LAYOUTS[key] = stores.CompressedLayout(folder / f"{kind}-{block}.bin", graph, block, straddle)
            LAYOUTS[key].encode_s = time.perf_counter() - started
    return LAYOUTS[key]


def make_store(name, graph, cfg):
    block = cfg["block"]
    raw_bytes = graph.m * stores.ID_BYTES
    lines = max(1, int(raw_bytes * cfg["cache_frac"]) // block)
    raw_path = WORK / graph.name / "raw.bin"
    if name == "load_all":
        return stores.LoadAllAdapter(raw_path, graph)
    if name == "bam_raw":
        return stores.BamAdapter(raw_path, graph, lines, block)
    if name == "comp_straddle":
        return stores.CompressedBamStore(layout_for(graph, "straddle", block), lines, block)
    if name == "comp_pad":
        return stores.CompressedBamStore(layout_for(graph, "pad", block), lines, block)
    kind = "raw" if name == "adaptive_raw" else "straddle"
    return stores.AdaptiveStore(layout_for(graph, kind, block), lines, block, cfg["threshold"])


def run_one(job):
    graph, name, cfg, seed, queries = job
    rng = np.random.default_rng([seed, cfg["depth"]])
    seeds = rng.integers(0, graph.n, queries).tolist()
    store = make_store(name, graph, cfg)
    digest = hashlib.sha256()
    started = time.perf_counter()
    for start in seeds:
        digest.update(stores.k_hop(store, start, cfg["depth"], graph.n, cfg["order"], rng).tobytes())
    wall = time.perf_counter() - started
    stats = store.stats()
    store.close()
    row = {"graph": graph.name, "store": name, "seed": seed, **cfg}
    row.update({k: stats[k] for k in ("storage_bytes", "needed_bytes", "unique_needed_bytes",
                                      "line_accesses", "probes", "hits", "misses", "file_bytes",
                                      "index_ram_bytes")})
    row["sector_misses"] = stats.get("sector_misses", 0)
    row["upgrades"] = stats.get("upgrades", 0)
    row["io_ops"] = sum(stats["op_sizes"].values())
    row["amp"] = stats["storage_bytes"] / max(stats["needed_bytes"], 1)
    row["amp_unique"] = stats["storage_bytes"] / max(stats["unique_needed_bytes"], 1)
    row["hit_rate"] = stats["hits"] / max(stats["probes"], 1)
    row["coalesced_pct"] = 100 * (stats["line_accesses"] - stats["probes"]) / max(stats["line_accesses"], 1)
    row["decode_s"] = stats["decode_s"]
    row["decode_ns_per_edge"] = 1e9 * stats["decode_s"] / max(stats["needed_bytes"] / stores.ID_BYTES, 1)
    row["wall_s"] = wall
    for device, cost in DEVICES.items():
        io = sum(cost(size) * count for size, count in stats["op_sizes"].items())
        row[f"{device}_io_s"] = io
        row[f"{device}_total_s"] = io + stats["decode_s"]
    row["digest"] = digest.hexdigest()
    return row


def jobs_for(all_graphs, seeds, queries):
    jobs, seen = [], set()
    for graph in all_graphs.values():
        for sweep, values in SWEEPS.items():
            for value in values:
                cfg = DEFAULT | {sweep: value}
                names = ["adaptive_raw", "adaptive_comp"] if sweep == "threshold" else STORES
                if sweep == "depth":
                    names = names + ["load_all"]
                for name in names:
                    for seed in seeds:
                        key = (graph.name, name, tuple(cfg.items()), seed)
                        if key not in seen:
                            seen.add(key)
                            jobs.append((graph, name, cfg, seed, queries))
    return jobs


def check_answers(rows):
    """Every store must give the load-all answer for the same queries."""
    truth = {(r["graph"], r["seed"], r["depth"]): r["digest"] for r in rows if r["store"] == "load_all"}
    for r in rows:
        if r["digest"] != truth[(r["graph"], r["seed"], r["depth"])]:
            raise SystemExit(f"WRONG ANSWER: {r['graph']} {r['store']} {r}")
    return len(rows)


def summarise(rows):
    keys = ("graph", "store", "cache_frac", "depth", "block", "threshold", "order")
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    skip = set(keys) | {"seed", "digest"}
    out = []
    for key, members in groups.items():
        row = dict(zip(keys, key))
        row["seeds"] = len(members)
        for field in (f for f in members[0] if f not in skip):
            row[field] = float(np.mean([m[field] for m in members]))
        row["amp_std"] = float(np.std([m["amp"] for m in members]))
        row["decode_s_std"] = float(np.std([m["decode_s"] for m in members]))
        out.append(row)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path, summary):
    head = ("graph", "store", "cache%", "depth", "block", "N", "storage MB", "I/O amp", "misses",
            "hit rate", "coalesced %", "decode ms", "wall s", "NAND total ms", "Optane total ms")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in summary:
        lines.append("| " + " | ".join([
            r["graph"], r["store"] + ("" if r["order"] == "sorted" else " (shuffled)"),
            f"{100 * r['cache_frac']:g}", str(r["depth"]), str(r["block"]), str(r["threshold"]),
            f"{r['storage_bytes'] / 1e6:.2f}", f"{r['amp']:.2f}", f"{r['misses']:.0f}",
            f"{r['hit_rate']:.3f}", f"{r['coalesced_pct']:.1f}", f"{1e3 * r['decode_s']:.1f}",
            f"{r['wall_s']:.2f}", f"{1e3 * r['nand_total_s']:.1f}", f"{1e3 * r['optane_total_s']:.1f}",
        ]) + " |")
    path.write_text("\n".join(lines) + "\n")


def calibrate_decode(all_graphs):
    """Sequential, before the pool starts. Batches inside a query are small, so the
    in-run decode_s is dominated by numpy call overhead; this separates the two."""
    rows = []
    for graph in all_graphs.values():
        layout = layout_for(graph, "straddle", 4096)
        stream = np.fromfile(layout.path, dtype=np.uint8)
        nodes = np.arange(graph.n)
        edges = np.diff(graph.offsets)
        order = np.argsort(-edges[:64], kind="stable")
        small = nodes[:64][order][:16]
        small_start = layout.start[small]
        small_buf = np.concatenate(
            [stream[begin : begin + size] for begin, size in zip(small_start, layout.list_bytes[small])]
        )
        whole, tiny = [], []
        for _ in range(7):
            tic = time.perf_counter()
            codec.decode_lists(stream, layout.list_bytes, nodes)
            whole.append(time.perf_counter() - tic)
        for _ in range(500):  # after the big decodes, so their cache misses are not charged here
            tic = time.perf_counter()
            codec.decode_lists(small_buf, layout.list_bytes[small], small)
            tiny.append(time.perf_counter() - tic)
        rows.append({
            "graph": graph.name,
            "ns_per_edge": 1e9 * float(np.median(whole)) / graph.m,
            "small_batch_us": 1e6 * float(np.median(tiny)),
        })
    return rows


def add_calibrated_totals(rows, calibration):
    """Totals with an ideal bulk numpy decoder: no per-call overhead."""
    ns = {c["graph"]: c["ns_per_edge"] for c in calibration}
    for row in rows:
        compressed = row["store"] in ("comp_straddle", "comp_pad", "adaptive_comp")
        decode = row["needed_bytes"] / stores.ID_BYTES * ns[row["graph"]] * 1e-9 if compressed else 0.0
        for device in DEVICES:
            row[f"{device}_total_bulk_s"] = row[f"{device}_io_s"] + decode


def format_rows(all_graphs):
    rows = []
    for graph in all_graphs.values():
        for kind in ("raw", "straddle", "pad"):
            layout = layout_for(graph, kind, 4096)
            rows.append({
                "graph": graph.name, "nodes": graph.n, "edges": graph.m, "format": kind,
                "file_bytes": layout.file_bytes, "bytes_per_edge": layout.file_bytes / graph.m,
                "index_ram_bytes": layout.ram_bytes, "encode_s": getattr(layout, "encode_s", 0.0),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["random", "powerlaw", "web-Google", "web-Google-bfs"])
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--snap", default=str(ROOT / "data" / "snap" / "web-Google.txt.gz"))
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    started = time.perf_counter()
    all_graphs = build_graphs(args)
    for graph in all_graphs.values():
        (WORK / graph.name).mkdir(parents=True, exist_ok=True)
        bam_baseline.write_csr(WORK / graph.name / "raw.bin", graph.neighbors)
    formats = format_rows(all_graphs)
    for graph in all_graphs.values():
        for block in SWEEPS["block"]:
            for kind in ("raw", "straddle", "pad"):
                layout_for(graph, kind, block)
    calibration = calibrate_decode(all_graphs)
    print(f"built graphs and images in {time.perf_counter() - started:.1f}s", flush=True)

    jobs = jobs_for(all_graphs, range(args.seeds), args.queries)
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} at {time.perf_counter() - started:.0f}s", flush=True)
    print(f"answers match load-all on {check_answers(rows)} runs", flush=True)

    add_calibrated_totals(rows, calibration)
    rows.sort(key=lambda r: (r["graph"], r["store"], str(r["cache_frac"]), r["depth"], r["block"],
                             r["threshold"], r["order"], r["seed"]))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    write_csv(out / "runs.csv", rows)
    summary = summarise(rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "formats.csv", formats)
    write_csv(out / "decode_calibration.csv", calibration)
    write_table(out / "table.md", summary)
    print(f"done in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
