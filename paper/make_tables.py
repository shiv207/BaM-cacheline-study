"""Generates the paper's tables and timing macros from results/, so no number is hand-copied."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import analyze

OUT = Path(__file__).resolve().parent / "generated"
T_95_DF4 = 2.776  # Student t, two-sided 95%, n = 5
GRAPHS = {"random": "random", "powerlaw": "powerlaw", "web-Google": "web-Google", "web-Google-bfs": "web-Google-BFS"}
NAMES = {"load_all": "Load-all", "bam_raw": "Raw-4K", "comp_straddle": "Comp-S", "comp_pad": "Comp-P",
         "adaptive_raw": "Adapt-R", "adaptive_comp": "Adapt-C"}
CONFIG = analyze.CONFIG
DEFAULT = analyze.DEFAULT


def write(name, text):
    (OUT / name).write_text(text)


def group(rows):
    out = {}
    for r in rows:
        out.setdefault((r["graph"], r["store"], *(r[k] for k in CONFIG)), []).append(r)
    return out


def mean(groups, graph, store, field, **change):
    cfg = DEFAULT | change
    return float(np.mean([r[field] for r in groups[(graph, store, *(cfg[k] for k in CONFIG))]]))


def ci(values):
    return T_95_DF4 * np.std(values, ddof=1) / np.sqrt(len(values))


def read_csv(name):
    with open(ROOT / "results" / name) as f:
        return list(csv.DictReader(f))


def datasets():
    rows = read_csv("formats.csv")
    body = []
    for graph, label in GRAPHS.items():
        by_format = {r["format"]: r for r in rows if r["graph"] == graph}
        raw, straddle, pad = by_format["raw"], by_format["straddle"], by_format["pad"]
        body.append(f"\\texttt{{{label}}} & {int(raw['nodes']):,} & {int(raw['edges']):,} & "
                    f"{float(straddle['bytes_per_edge']):.2f} & {float(pad['bytes_per_edge']):.2f} & "
                    f"{100 * int(straddle['file_bytes']) / int(raw['file_bytes']):.0f}\\% & "
                    f"{100 * (int(pad['file_bytes']) / int(straddle['file_bytes']) - 1):.1f}\\% \\\\")
    write("datasets.tex", "\n".join(body))


def default_table(groups):
    body = []
    for g, label in GRAPHS.items():
        body.append(f"\\midrule\n\\multicolumn{{11}}{{l}}{{\\texttt{{{label}}}}} \\\\")
        for s in ("load_all", "bam_raw", "comp_straddle", "comp_pad", "adaptive_raw", "adaptive_comp"):
            f = lambda k: mean(groups, g, s, k)
            bulk = 1e3 * (f("nand_total_bulk_s") - f("nand_io_s"))
            amp_sd = np.std([r["amp"] for r in groups[(g, s, *(DEFAULT[k] for k in CONFIG))]], ddof=1)
            probe = s != "load_all"
            body.append(" & ".join([
                NAMES[s], f"{f('storage_bytes') / 1e6:.1f}", f"{f('amp'):.1f}$\\pm${amp_sd:.1f}",
                f"{f('misses'):,.0f}" if probe else "--", f"{f('hit_rate'):.3f}" if probe else "--",
                f"{f('coalesced_pct'):.1f}" if probe else "--", f"{1e3 * f('decode_s'):.0f}", f"{bulk:.1f}",
                f"{f('wall_s'):.2f}", f"{1e3 * f('nand_io_s'):.1f}", f"{1e3 * f('optane_io_s'):.1f}"]) + " \\\\")
    write("default.tex", "\n".join(body))


def paired_table(rows):
    pairs = [p for p in analyze.pairs(rows) if analyze.is_default(p[0])]
    body = []
    for g, label in GRAPHS.items():
        body.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\texttt{{{label}}}}} \\\\")
        for s in analyze.NEW:
            ps = [(n, b) for n, b in pairs if n["graph"] == g and n["store"] == s]
            assert len(ps) == 5
            cells = []
            for key in ("storage_bytes", "misses", "io_ops", "nand_io_s", "optane_io_s"):
                v = [analyze.pct(n[key], b[key]) for n, b in ps]
                cells.append(f"${np.mean(v):+.1f}\\pm{ci(v):.1f}$")
            body.append(NAMES[s] + " & " + " & ".join(cells) + " \\\\")
    write("paired.tex", "\n".join(body))


def breakeven_table(groups):
    cal = {r["graph"]: float(r["ns_per_edge"]) for r in read_csv("decode_calibration.csv")}
    body = []
    for g, label in GRAPHS.items():
        cells = [f"\\texttt{{{label}}}", f"{cal[g]:.1f}"]
        for frac in (0.02, 0.5):
            edges = mean(groups, g, "comp_straddle", "needed_bytes", cache_frac=frac) / 4
            for device in ("nand", "optane"):
                saved = (mean(groups, g, "bam_raw", f"{device}_io_s", cache_frac=frac)
                         - mean(groups, g, "comp_straddle", f"{device}_io_s", cache_frac=frac))
                cells.append(f"{1e9 * saved / edges:.1f}")
            nand = 1e9 * (mean(groups, g, "bam_raw", "nand_io_s", cache_frac=frac)
                          - mean(groups, g, "comp_straddle", "nand_io_s", cache_frac=frac)) / edges
            cells.append(f"{cal[g] / nand:.1f}$\\times$")
        body.append(" & ".join(cells) + " \\\\")
    write("breakeven.tex", "\n".join(body))


def fixed_table(groups):
    body = []
    for g, label in GRAPHS.items():
        body.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\texttt{{{label}}}}} \\\\")
        for s, block, name in (("bam_raw", 4096.0, "Raw-4K"), ("bam_raw", 1024.0, "Raw-1K"),
                               ("bam_raw", 512.0, "Raw-512"), ("adaptive_raw", 4096.0, "Adapt-R")):
            f = lambda k: mean(groups, g, s, k, block=block)
            body.append(f"{name} & {f('amp'):.2f} & {f('hit_rate'):.3f} & {f('misses'):,.0f} & {f('io_ops'):,.0f} & "
                        f"{1e3 * f('nand_io_s'):.2f} & {1e3 * f('optane_io_s'):.2f} \\\\")
    write("fixed.tex", "\n".join(body))


def depth_table(groups):
    body = []
    for g, label in GRAPHS.items():
        for depth in (1.0, 2.0, 3.0):
            mb = [mean(groups, g, s, "storage_bytes", depth=depth) / 1e6 for s in
                  ("load_all", "bam_raw", "comp_straddle", "adaptive_raw", "adaptive_comp")]
            best = int(np.argmin(mb))
            cells = [f"\\textbf{{{m:.1f}}}" if i == best else f"{m:.1f}" for i, m in enumerate(mb)]
            need = mean(groups, g, "bam_raw", "needed_bytes", depth=depth) / 1e6
            first = f"\\texttt{{{label}}}" if depth == 1.0 else ""
            body.append(f"{first} & {depth:.0f} & {need:.2f} & " + " & ".join(cells) + " \\\\")
    write("depth.tex", "\n".join(body))


def macros():
    cal = read_csv("decode_calibration.csv")
    ns = [float(r["ns_per_edge"]) for r in cal]
    us = [float(r["small_batch_us"]) for r in cal]
    write("macros.tex", f"\\newcommand{{\\nsMin}}{{{min(ns):.0f}}}\n\\newcommand{{\\nsMax}}{{{max(ns):.0f}}}\n"
                        f"\\newcommand{{\\usMin}}{{{min(us):.0f}}}\n\\newcommand{{\\usMax}}{{{max(us):.0f}}}\n")


def main():
    OUT.mkdir(exist_ok=True)
    rows = analyze.load()
    groups = group(rows)
    datasets()
    default_table(groups)
    paired_table(rows)
    breakeven_table(groups)
    fixed_table(groups)
    depth_table(groups)
    macros()


if __name__ == "__main__":
    main()
