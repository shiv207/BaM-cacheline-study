"""Renders results/*.png from results/summary.csv. Needs matplotlib."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
GRAPHS = ["random", "powerlaw", "web-Google", "web-Google-bfs"]
DEFAULT = dict(cache_frac=0.02, depth=2.0, block=4096.0, threshold=3.0, order="sorted")
BLOCKS = [512, 2048, 8192, 65536]
THRESHOLDS = [1, 2, 3, 4, 8, 16]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
# Fixed slot per store, so a store keeps its colour in every figure. Markers
# give a second channel for colour-blind readers and greyscale print.
STYLE = {
    "bam_raw": ("#2a78d6", "o", "BaM raw (baseline)"),
    "comp_straddle": ("#eb6834", "s", "Compressed, straddle"),
    "comp_pad": ("#1baf7a", "^", "Compressed, padded"),
    "adaptive_raw": ("#eda100", "D", "Adaptive raw"),
    "adaptive_comp": ("#e87ba4", "v", "Adaptive compressed"),
}


SCREEN = dict(size=(13, 3.7), font=10, marker=6, line=2, ext="png", title=True, bottom=0.08)
PRINT = dict(size=(7.0, 2.15), font=7, marker=3.5, line=1.2, ext="pdf", title=False, bottom=0.13)
MODE = SCREEN


def load():
    rows = []
    with open(RESULTS / "summary.csv") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k not in ("graph", "store", "order"):
                    r[k] = float(v)
            rows.append(r)
    return rows


def select(rows, graph, store, sweep):
    fixed = {k: v for k, v in DEFAULT.items() if k != sweep}
    keep = [r for r in rows if r["graph"] == graph and r["store"] == store
            and all(r[k] == v for k, v in fixed.items())]
    return sorted(keep, key=lambda r: r[sweep])


def figure(rows, sweep, y, ylabel, title, name, xlabel, stores=STYLE, logx=True, logy=False,
           scale=1.0, xscale=1.0, xticks=None):
    plt.rcParams.update({"font.size": MODE["font"], "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=MODE["size"], sharex=True, facecolor=SURFACE)
    for ax, graph in zip(axes, GRAPHS):
        ax.set_facecolor(SURFACE)
        for store in stores:
            colour, marker, label = STYLE[store]
            pts = select(rows, graph, store, sweep)
            ax.plot([p[sweep] * xscale for p in pts], [p[y] * scale for p in pts], color=colour,
                    marker=marker, markersize=MODE["marker"], linewidth=MODE["line"], label=label,
                    markeredgecolor=SURFACE, markeredgewidth=MODE["line"] / 2)
        ax.set_title(graph, fontsize=MODE["font"], loc="left", color=INK)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(NullFormatter())
        if logy:
            ax.yaxis.set_major_formatter(plain)
            low, high = ax.get_ylim()
            if high / low < 30:  # beyond this the minor labels collide
                ax.yaxis.set_minor_formatter(plain)
                ax.tick_params(axis="y", which="minor", labelsize=MODE["font"] - 2, labelcolor=MUTED)
            else:
                ax.yaxis.set_minor_formatter(NullFormatter())
        if xticks:
            ax.set_xticks(xticks)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if MODE["title"]:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="lower center", ncol=len(stores), frameon=False)
    fig.tight_layout(rect=(0, MODE["bottom"], 1, 0.97 if MODE["title"] else 1))
    fig.savefig(RESULTS / f"{name}.{MODE['ext']}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    global MODE
    rows = load()
    for MODE in (SCREEN, PRINT):
        render(rows)


def render(rows):
    figure(rows, "cache_frac", "amp", "I/O amplification (log)",
           "Bytes read per raw byte needed, by cache size (depth 2, 4KB)",
           "amp_vs_cache", "cache, % of raw graph", logy=True, xscale=100)
    figure(rows, "cache_frac", "nand_total_bulk_s", "seconds (log)",
           "Modelled NAND I/O + bulk decode, by cache size (depth 2, 4KB)",
           "time_vs_cache", "cache, % of raw graph", logy=True, xscale=100)
    figure(rows, "depth", "amp", "I/O amplification (log)",
           "Bytes read per raw byte needed, by depth (cache 2%, 4KB)",
           "amp_vs_depth", "k-hop depth", logx=False, logy=True, xticks=[1, 2, 3])
    figure(rows, "block", "amp", "I/O amplification (log)",
           "Bytes read per raw byte needed, by line size (cache 2%, depth 2)",
           "amp_vs_block", "line size, bytes", logy=True, xticks=BLOCKS)
    figure(rows, "block", "io_ops", "device ops, thousands",
           "Device reads by line size (cache 2%, depth 2)",
           "ops_vs_block", "line size, bytes", scale=1e-3, xticks=BLOCKS)
    adaptive = ("adaptive_raw", "adaptive_comp")
    picked = [dict(r, threshold=max(r["threshold"], 1) if r["threshold"] < 99 else 16) for r in rows]
    figure(picked, "threshold", "amp", "I/O amplification",
           "Adaptive: upgrade threshold N (1 = full line, 16 = never; cache 2%, depth 2)",
           "amp_vs_threshold", "N, missed sectors", stores=adaptive,
           xticks=THRESHOLDS)
    figure(picked, "threshold", "hit_rate", "hit rate",
           "Adaptive: line hit rate by threshold N (cache 2%, depth 2)",
           "hit_vs_threshold", "N, missed sectors", stores=adaptive,
           xticks=THRESHOLDS)


if __name__ == "__main__":
    main()
