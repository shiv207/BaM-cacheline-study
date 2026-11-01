"""Paired comparisons against the baseline; writes results/analysis.md."""
import csv
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
GRAPHS = ["random", "powerlaw", "web-Google", "web-Google-bfs"]
NEW = ["comp_straddle", "comp_pad", "adaptive_raw", "adaptive_comp"]
DEFAULT = dict(cache_frac=0.02, depth=2.0, block=4096.0, threshold=3.0, order="sorted")
CONFIG = ("cache_frac", "depth", "block", "threshold", "order")


def load():
    with open(RESULTS / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("graph", "store", "order", "digest"):
                r[k] = float(v)
    return rows


def is_default(r):
    return all(r[k] == v for k, v in DEFAULT.items())


def pairs(rows):
    """Yields (new, baseline) rows for the same graph, config and query seed."""
    baseline = {(r["graph"], r["seed"], *(r[k] for k in CONFIG)): r for r in rows if r["store"] == "bam_raw"}
    for r in rows:
        if r["store"] in NEW:
            match = baseline.get((r["graph"], r["seed"], *(r[k] for k in CONFIG)))
            if match is not None:
                yield r, match


def pct(new, old):
    return 100 * (new - old) / old


def main():
    rows = load()
    out = ["# Paired analysis", "",
           "Change vs `bam_raw` at the default point (cache 2%, depth 2, 4KB, N=3), mean of "
           "per-seed changes, with the seed range in brackets. Negative is better.", "",
           "| graph | store | storage bytes % | misses % | device ops % | NAND I/O % | Optane I/O % | "
           "bulk-decode break-even, NAND ns/edge | break-even, Optane ns/edge |",
           "|---|---|---|---|---|---|---|---|---|"]
    paired = list(pairs(rows))
    for graph in GRAPHS:
        for store in NEW:
            ps = [(new, old) for new, old in paired
                  if new["graph"] == graph and new["store"] == store and is_default(new)]

            def cell(key):
                v = [pct(new[key], old[key]) for new, old in ps]
                return f"{np.mean(v):+.1f} [{min(v):+.1f}, {max(v):+.1f}]"

            def breakeven(device):
                v = [1e9 * (old[f"{device}_io_s"] - new[f"{device}_io_s"]) / (new["needed_bytes"] / 4)
                     for new, old in ps]
                return f"{np.mean(v):.1f}"

            out.append(f"| {graph} | {store} | {cell('storage_bytes')} | {cell('misses')} | {cell('io_ops')} | "
                       f"{cell('nand_io_s')} | {cell('optane_io_s')} | {breakeven('nand')} | {breakeven('optane')} |")

    out += ["", "## Runs beating the baseline, across every sweep point and seed", "",
            "| store | bytes | misses | device ops | NAND total (bulk decode) | Optane total (bulk decode) | "
            "NAND total (measured decode) | Optane total (measured decode) |", "|---|---|---|---|---|---|---|---|"]
    for store in NEW:
        ps = [(new, old) for new, old in paired if new["store"] == store and new["threshold"] > 1]

        def share(key):
            return f"{100 * np.mean([new[key] < old[key] for new, old in ps]):.0f}%"

        out.append(f"| {store} | {share('storage_bytes')} | {share('misses')} | {share('io_ops')} | "
                   f"{share('nand_total_bulk_s')} | {share('optane_total_bulk_s')} | "
                   f"{share('nand_total_s')} | {share('optane_total_s')} |")

    means = {}
    for r in rows:
        key = (r["graph"], r["store"], *(r[k] for k in CONFIG))
        means.setdefault(key, []).append(r)

    def mean(graph, store, field, **change):
        cfg = DEFAULT | change
        return float(np.mean([r[field] for r in means[(graph, store, *(cfg[k] for k in CONFIG))]]))

    out += ["", "## Adaptive 4KB vs a plain fixed line of a smaller size (raw format)", "",
            "| graph | store | line | amp | hit rate | misses | device ops | Optane I/O ms | NAND I/O ms |",
            "|---|---|---|---|---|---|---|---|---|"]
    for graph in GRAPHS:
        for store, block in (("bam_raw", 4096.0), ("adaptive_raw", 4096.0), ("bam_raw", 1024.0), ("bam_raw", 512.0)):
            out.append(f"| {graph} | {store} | {block:.0f} | {mean(graph, store, 'amp', block=block):.2f} | "
                       f"{mean(graph, store, 'hit_rate', block=block):.3f} | "
                       f"{mean(graph, store, 'misses', block=block):.0f} | "
                       f"{mean(graph, store, 'io_ops', block=block):.0f} | "
                       f"{1e3 * mean(graph, store, 'optane_io_s', block=block):.2f} | "
                       f"{1e3 * mean(graph, store, 'nand_io_s', block=block):.2f} |")

    out += ["", "## Adaptive upgrades as the cache grows (adaptive_raw, N=3)", "",
            "| graph | cache % | amp | upgrades | sector misses |", "|---|---|---|---|---|"]
    for graph in GRAPHS:
        for frac in (0.001, 0.02, 0.1, 0.5):
            out.append(f"| {graph} | {100 * frac:g} | {mean(graph, 'adaptive_raw', 'amp', cache_frac=frac):.2f} | "
                       f"{mean(graph, 'adaptive_raw', 'upgrades', cache_frac=frac):.0f} | "
                       f"{mean(graph, 'adaptive_raw', 'sector_misses', cache_frac=frac):.0f} |")

    out += ["", "## Load-all vs on-demand, by depth (storage MB)", "",
            "| graph | depth | needed MB | load_all | bam_raw | comp_straddle | adaptive_raw | adaptive_comp |",
            "|---|---|---|---|---|---|---|---|"]
    for graph in GRAPHS:
        for depth in (1.0, 2.0, 3.0):
            mb = [mean(graph, s, "storage_bytes", depth=depth) / 1e6 for s in
                  ("load_all", "bam_raw", "comp_straddle", "adaptive_raw", "adaptive_comp")]
            need = mean(graph, "bam_raw", "needed_bytes", depth=depth) / 1e6
            out.append(f"| {graph} | {depth:.0f} | {need:.2f} | " + " | ".join(f"{m:.1f}" for m in mb) + " |")

    out += ["", "## Default point", "",
            "| graph | store | storage MB | I/O amp | amp (unique) | misses | hit rate | coalesced % | decode ms "
            "(measured) | decode ms (bulk) | wall s | NAND I/O ms | Optane I/O ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for graph in GRAPHS:
        for store in ["load_all", "bam_raw"] + NEW:
            f = lambda k, **c: mean(graph, store, k, **c)
            bulk = (f("nand_total_bulk_s") - f("nand_io_s")) * 1e3
            out.append(f"| {graph} | {store} | {f('storage_bytes') / 1e6:.2f} | {f('amp'):.2f} | {f('amp_unique'):.2f} | "
                       f"{f('misses'):.0f} | {f('hit_rate'):.3f} | {f('coalesced_pct'):.1f} | {1e3 * f('decode_s'):.1f} | "
                       f"{bulk:.1f} | {f('wall_s'):.2f} | {1e3 * f('nand_io_s'):.2f} | {1e3 * f('optane_io_s'):.2f} |")
    (RESULTS / "analysis.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
