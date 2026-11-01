import numpy as np
import pytest

import bam_baseline
import graphs
import stores

KINDS = ["bam", "line_raw", "comp_straddle", "comp_pad", "adaptive_raw:0", "adaptive_raw:1",
         "adaptive_raw:3", "adaptive_raw:99", "adaptive_comp:3", "adaptive_comp:99"]


def edge_case_graph():
    """Degree-0 nodes at both ends and the middle, self loop, duplicate edges."""
    return graphs.from_edges("edges", [1, 1, 1, 5, 9], [9, 2, 2, 5, 0], 12)


def huge_degree_graph():
    """One list far longer than a line, and than a one-line cache."""
    hub = 3
    return graphs.from_edges("hub", [hub] * 5000 + [0, 4999], list(range(5000)) + [4999, hub], 5001)


GRAPHS = {
    "edges": edge_case_graph(),
    "hub": huge_degree_graph(),
    "random0": graphs.random_graph(800, 6, 0),
    "random1": graphs.random_graph(800, 6, 1),
    "powerlaw0": graphs.powerlaw_graph(800, 6, 0),
    "powerlaw1": graphs.powerlaw_graph(800, 6, 1),
    "powerlaw0-bfs": graphs.bfs_relabel(graphs.powerlaw_graph(800, 6, 0)),
}


def build(tmp_path, graph, kind, cache_lines=8, line_size=4096):
    raw = tmp_path / "raw.bin"
    bam_baseline.write_csr(raw, graph.neighbors)
    if kind == "bam":
        return stores.BamAdapter(raw, graph, cache_lines, line_size)
    if kind == "load_all":
        return stores.LoadAllAdapter(raw, graph)
    if kind == "line_raw":
        return stores.LineStore(stores.RawLayout(raw, graph), cache_lines, line_size)
    if kind.startswith("comp"):
        layout = stores.CompressedLayout(tmp_path / "c.bin", graph, line_size, kind == "comp_straddle")
        return stores.CompressedBamStore(layout, cache_lines, line_size)
    fmt, threshold = kind.split(":")
    layout = (stores.RawLayout(raw, graph) if fmt == "adaptive_raw"
              else stores.CompressedLayout(tmp_path / "c.bin", graph, line_size))
    return stores.AdaptiveStore(layout, cache_lines, line_size, int(threshold))


def python_k_hop(graph, seed, depth):
    """Reference BFS sharing no code with the stores."""
    seen, frontier = {seed}, [seed]
    for _ in range(depth):
        found = []
        for u in frontier:
            for v in graph.neighbors[graph.offsets[u] : graph.offsets[u + 1]].tolist():
                if v not in seen:
                    seen.add(v)
                    found.append(v)
        frontier = found
    return sorted(seen)


@pytest.mark.parametrize("name", GRAPHS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cache_lines,line_size", [(1, 512), (8, 4096), (64, 512), (2, 8192)])
def test_k_hop_matches_load_all(tmp_path, name, kind, cache_lines, line_size):
    graph = GRAPHS[name]
    store = build(tmp_path, graph, kind, cache_lines, line_size)
    truth = build(tmp_path, graph, "load_all")
    rng = np.random.default_rng(7)
    for seed in rng.integers(0, graph.n, 6).tolist():
        for depth in (0, 1, 2, 3):
            got = stores.k_hop(store, seed, depth, graph.n)
            assert got.tolist() == stores.k_hop(truth, seed, depth, graph.n).tolist()
            assert got.tolist() == python_k_hop(graph, seed, depth)
    store.close()


@pytest.mark.parametrize("name", GRAPHS)
@pytest.mark.parametrize("kind", KINDS)
def test_neighbors_batch_exact_lists_and_counts(tmp_path, name, kind):
    graph = GRAPHS[name]
    store = build(tmp_path, graph, kind, cache_lines=4)
    rng = np.random.default_rng(3)
    nodes = np.concatenate((rng.integers(0, graph.n, 500), [graph.n - 1, 0, graph.n - 1]))
    flat, counts = store.neighbors_batch(nodes)
    want = [graph.neighbors[graph.offsets[u] : graph.offsets[u + 1]] for u in nodes.tolist()]
    assert counts.tolist() == [len(x) for x in want]
    assert flat.dtype == np.uint32
    assert flat.tolist() == np.concatenate(want).tolist()
    store.close()


@pytest.mark.parametrize("kind", KINDS)
def test_all_degree_zero_batch(tmp_path, kind):
    graph = GRAPHS["edges"]
    store = build(tmp_path, graph, kind)
    flat, counts = store.neighbors_batch([0, 11, 7])
    assert len(flat) == 0 and counts.tolist() == [0, 0, 0]
    assert store.stats()["storage_bytes"] == 0
    store.close()


@pytest.mark.parametrize("kind", KINDS)
def test_last_block_past_eof(tmp_path, kind):
    """The last list ends in a short final line, so reads run past EOF."""
    graph = graphs.from_edges("tail", [0] * 1500 + [1499], list(range(1500)) + [7], 1500)
    store = build(tmp_path, graph, kind, cache_lines=2)
    flat, _ = store.neighbors_batch([1499])
    assert flat.tolist() == [7]
    store.close()


@pytest.mark.parametrize("twin", ["line_raw", "adaptive_raw:0", "adaptive_raw:1"])
@pytest.mark.parametrize("name", ["random0", "powerlaw0", "hub"])
@pytest.mark.parametrize("cache_lines,line_size", [(1, 4096), (6, 4096), (20, 512)])
def test_new_cache_code_reproduces_baseline_counters(tmp_path, twin, name, cache_lines, line_size):
    """Raw layout + full-line fetch must behave exactly as the baseline, so any
    measured difference comes from the format and not from the reimplemented cache."""
    graph = GRAPHS[name]
    a = build(tmp_path, graph, "bam", cache_lines, line_size)
    b = build(tmp_path, graph, twin, cache_lines, line_size)
    for seed in [3, 0] + np.random.default_rng(5).integers(0, graph.n, 10).tolist():
        stores.k_hop(a, seed, 2, graph.n)
        stores.k_hop(b, seed, 2, graph.n)
    keys = ("storage_bytes", "line_accesses", "probes", "hits", "misses", "needed_bytes")
    assert {k: a.stats()[k] for k in keys} == {k: b.stats()[k] for k in keys}
    assert a.stats()["op_sizes"] == b.stats()["op_sizes"]
    assert a.stats()["misses"] > 0
    a.close()
    b.close()


def adaptive_on(tmp_path, threshold, cache_lines=4):
    graph = graphs.from_edges("line", [0] * 1024, list(range(1024)), 1024)  # 4096 raw bytes
    raw = tmp_path / "raw.bin"
    bam_baseline.write_csr(raw, graph.neighbors)
    return stores.AdaptiveStore(stores.RawLayout(raw, graph), cache_lines, 4096, threshold)


def fake_list(store, node, start, size):
    store.layout.start = store.layout.start.copy()
    store.layout.list_bytes = store.layout.list_bytes.copy()
    store.layout.start[node], store.layout.list_bytes[node] = start, size
    return store.neighbors_batch([node])


def test_adaptive_reads_only_needed_sectors(tmp_path):
    store = adaptive_on(tmp_path, threshold=99)
    fake_list(store, 1, 0, 100)
    assert store.bytes_read == 512
    fake_list(store, 2, 10, 20)  # same sector: pure hit
    assert store.bytes_read == 512 and store.hits == 1
    fake_list(store, 3, 2048, 600)  # spans sectors 4 and 5: one contiguous op
    assert store.bytes_read == 1536
    assert store.op_sizes == {512: 1, 1024: 1}
    assert store.sector_misses == 3 and store.upgrades == 0


def test_adaptive_upgrade_fetches_exactly_the_rest(tmp_path):
    store = adaptive_on(tmp_path, threshold=2)
    fake_list(store, 1, 0, 100)
    assert store.upgrades == 0
    fake_list(store, 2, 2048, 100)  # second missed sector: upgrades
    assert store.upgrades == 1
    assert store.bytes_read == 4096  # nothing is read twice
    assert store.op_sizes == {512: 1, 3584: 1}
    before = store.bytes_read
    fake_list(store, 3, 3000, 100)
    assert store.bytes_read == before


def test_adaptive_bitmap_resets_on_eviction(tmp_path):
    store = adaptive_on(tmp_path, threshold=99, cache_lines=1)
    fake_list(store, 1, 0, 100)
    fake_list(store, 3, 4096, 100)  # beyond EOF, evicts line 0
    fake_list(store, 1, 0, 100)  # must refetch, not trust the stale bitmap
    assert store.bytes_read == 3 * 512


def test_warp_coalesces_same_line(tmp_path):
    graph = GRAPHS["powerlaw0"]
    store = build(tmp_path, graph, "bam", cache_lines=32)
    store.neighbors_batch(np.arange(32))
    s = store.stats()
    assert s["probes"] < s["line_accesses"]


def test_amplification_denominator_is_raw_bytes_for_every_format(tmp_path):
    graph = GRAPHS["powerlaw0"]
    nodes = np.arange(100)
    needed = {k: None for k in ("bam", "comp_straddle", "adaptive_comp:3")}
    for kind in needed:
        store = build(tmp_path, graph, kind)
        store.neighbors_batch(nodes)
        store.neighbors_batch(nodes)
        needed[kind] = (store.stats()["needed_bytes"], store.stats()["unique_needed_bytes"])
        store.close()
    want = int(np.diff(graph.offsets)[:100].sum()) * 4
    assert set(needed.values()) == {(2 * want, want)}


def test_padding_lengthens_the_file_but_not_the_answer(tmp_path):
    graph = graphs.powerlaw_graph(5000, 8, 2)
    plain = build(tmp_path, graph, "comp_straddle", line_size=512)
    padded = build(tmp_path, graph, "comp_pad", line_size=512)
    assert padded.layout.file_bytes > plain.layout.file_bytes
    assert padded.layout.ram_bytes > plain.layout.ram_bytes
    plain.close()
    padded.close()


def test_generators_are_deterministic():
    for make in (graphs.random_graph, graphs.powerlaw_graph):
        a, b, c = make(500, 5, 1), make(500, 5, 1), make(500, 5, 2)
        assert np.array_equal(a.neighbors, b.neighbors)
        assert not np.array_equal(a.neighbors, c.neighbors)


def test_snap_loader_and_bfs_relabel(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("# comment\n# FromNodeId ToNodeId\n10\t20\n20\t30\n10\t20\n99\t10\n")
    graph = graphs.load_snap(path)
    assert graph.n == 4 and graph.m == 3
    relabelled = graphs.load_snap(path, relabel="bfs")
    assert relabelled.n == 4 and relabelled.m == 3
    assert sorted(np.diff(relabelled.offsets).tolist()) == sorted(np.diff(graph.offsets).tolist())
    rank = graphs.bfs_order(graph)
    assert sorted(rank.tolist()) == [0, 1, 2, 3]


def test_bfs_relabel_preserves_structure():
    graph = GRAPHS["powerlaw0"]
    rank = graphs.bfs_order(graph)
    relabelled = graphs.bfs_relabel(graph)
    u = 17
    want = sorted(rank[graph.neighbors[graph.offsets[u] : graph.offsets[u + 1]]].tolist())
    v = rank[u]
    assert relabelled.neighbors[relabelled.offsets[v] : relabelled.offsets[v + 1]].tolist() == want
