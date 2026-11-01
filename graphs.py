"""Deterministic graph sources. Every graph is canonical: sorted, deduplicated CSR."""
import gzip
from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    name: str
    offsets: np.ndarray  # int64, n + 1
    neighbors: np.ndarray  # uint32

    @property
    def n(self):
        return len(self.offsets) - 1

    @property
    def m(self):
        return len(self.neighbors)


def from_edges(name, src, dst, n):
    """Builds canonical CSR. Self loops are kept; duplicate edges are dropped."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    key = np.unique(src * n + dst)
    src, dst = key // n, key % n
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.add.at(offsets, src + 1, 1)
    return Graph(name, np.cumsum(offsets), dst.astype(np.uint32))


def random_graph(n, avg_degree, seed):
    """Uniform random directed graph: no locality, no skew."""
    rng = np.random.default_rng(seed)
    m = n * avg_degree
    return from_edges(f"random-{n}", rng.integers(0, n, m), rng.integers(0, n, m), n)


def powerlaw_graph(n, avg_degree, seed, exponent=2.1, local_fraction=0.8, spread=64.0):
    """Skewed out-degrees and clustered ids.

    Degrees are truncated Pareto. A local_fraction of edges land near the source
    with a heavy-tailed id distance; the rest hit skew-popular global targets.
    """
    rng = np.random.default_rng(seed)
    raw = rng.pareto(exponent - 1.0, n) + 1.0
    degree = np.minimum(raw * avg_degree / raw.mean(), n / 4).astype(np.int64)
    src = np.repeat(np.arange(n, dtype=np.int64), degree)
    local = rng.random(len(src)) < local_fraction
    sign = rng.choice(np.array([-1, 1]), len(src))
    distance = np.floor(spread * (rng.pareto(1.2, len(src)) + rng.random(len(src)))).astype(np.int64)
    dst = (src + sign * distance) % n
    popular = (n * rng.random(int((~local).sum())) ** 3).astype(np.int64)
    dst[~local] = rng.permutation(n)[popular]
    return from_edges(f"powerlaw-{n}", src, dst, n)


def load_snap(path, name=None, relabel=None):
    """Loads a SNAP edge list (.txt or .txt.gz; '#' comments).

    Ids are first compacted in numeric order, so 'identity' order keeps whatever
    locality the dataset has. relabel='bfs' then renumbers by breadth-first order.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        edges = np.loadtxt(f, dtype=np.int64, comments="#", ndmin=2)
    ids, compact = np.unique(edges, return_inverse=True)
    compact = compact.reshape(edges.shape)
    graph = from_edges(name or "snap", compact[:, 0], compact[:, 1], len(ids))
    return bfs_relabel(graph) if relabel == "bfs" else graph


def bfs_order(graph):
    """Rank of each node in a BFS over the undirected view, parents in visit order."""
    n = graph.n
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.offsets))
    dst = graph.neighbors.astype(np.int64)
    both = from_edges("view", np.concatenate((src, dst)), np.concatenate((dst, src)), n)
    offsets, neighbors = both.offsets, both.neighbors.astype(np.int64)
    roots = np.argsort(-np.diff(offsets), kind="stable").tolist()
    rank = np.full(n, -1, dtype=np.int64)
    next_rank = 0
    for root in roots:
        if rank[root] >= 0:
            continue
        frontier = np.array([root], dtype=np.int64)
        rank[root] = next_rank
        next_rank += 1
        while len(frontier):
            begin, count = offsets[frontier], offsets[frontier + 1] - offsets[frontier]
            if count.sum() == 0:
                break
            starts = np.concatenate(([0], np.cumsum(count[count > 0])))[:-1]
            at = np.arange(count.sum()) - np.repeat(starts, count[count > 0])
            found = neighbors[at + np.repeat(begin[count > 0], count[count > 0])]
            found = found[rank[found] < 0]
            unique, first = np.unique(found, return_index=True)
            frontier = unique[np.argsort(first)]
            rank[frontier] = next_rank + np.arange(len(frontier))
            next_rank += len(frontier)
    return rank


def bfs_relabel(graph):
    rank = bfs_order(graph)
    src = np.repeat(np.arange(graph.n, dtype=np.int64), np.diff(graph.offsets))
    return from_edges(f"{graph.name}-bfs", rank[src], rank[graph.neighbors.astype(np.int64)], graph.n)
