import numpy as np
import pytest

import codec


def scalar_decode(buf, node):
    """Slow reference decoder, written independently of the numpy one."""
    values, v, shift = [], 0, 0
    for b in buf.tolist():
        v |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            values.append(v)
            v, shift = 0, 0
    if not values:
        return []
    z = values[0]
    out = [((z >> 1) ^ -(z & 1)) + node]
    for gap in values[1:]:
        out.append(out[-1] + gap)
    return out


def csr(lists):
    offsets = np.cumsum([0] + [len(adjacent) for adjacent in lists], dtype=np.int64)
    flat = [v for adjacent in lists for v in adjacent]
    return offsets, np.array(flat, dtype=np.uint32)


def roundtrip(lists):
    offsets, neighbors = csr(lists)
    stream, sizes = codec.encode_lists(offsets, neighbors)
    flat, counts = codec.decode_lists(stream, sizes, np.arange(len(lists)))
    assert flat.tolist() == neighbors.tolist()
    assert counts.tolist() == [len(x) for x in lists]
    at = 0
    for node, size in enumerate(sizes.tolist()):
        assert scalar_decode(stream[at : at + size], node) == lists[node]
        at += size


@pytest.mark.parametrize("lists", [
    [[]],
    [[], [], []],
    [[0]],
    [[2**32 - 1]],
    [[0, 2**32 - 1]],
    [[], [0, 1, 2], [], [2**32 - 2, 2**32 - 1], []],
    [[5, 5, 5]],
])
def test_roundtrip_edges(lists):
    roundtrip(lists)


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    top = 1 << int(rng.integers(1, 33))
    lists = [np.sort(rng.integers(0, top, int(rng.integers(0, 40)))).tolist() for _ in range(300)]
    roundtrip(lists)


def test_unsorted_rejected():
    with pytest.raises(ValueError):
        codec.encode_lists(*csr([[3, 1]]))


def test_decode_subset_in_any_order():
    lists = [[1, 2], [], [9, 100000], [0]]
    offsets, neighbors = csr(lists)
    stream, sizes = codec.encode_lists(offsets, neighbors)
    at = np.concatenate(([0], np.cumsum(sizes)))
    nodes = np.array([3, 1, 2])
    buf = np.concatenate([stream[at[i] : at[i + 1]] for i in nodes])
    flat, counts = codec.decode_lists(buf, sizes[nodes], nodes)
    assert flat.tolist() == [0, 9, 100000]
    assert counts.tolist() == [1, 0, 2]


@pytest.mark.parametrize("line", [64, 512, 4096])
def test_padding_never_straddles_a_list_that_fits(line):
    rng = np.random.default_rng(1)
    sizes = rng.integers(0, line * 2, 2000)
    start = codec.pack(sizes, line, straddle=False)
    fits = (sizes > 0) & (sizes <= line)
    assert np.all(start[fits] // line == (start[fits] + sizes[fits] - 1) // line)
    assert np.all(start[sizes > line] % line == 0)
    assert np.all(start[1:][sizes[:-1] > 0] >= (start[:-1] + sizes[:-1])[sizes[:-1] > 0])
