"""Delta + varint neighbour lists and their line layout.

Each list is sorted. Its first id is stored as zigzag(id - node), the rest as
gaps, all as little-endian base-128 varints. The value count is not stored: it
is the number of terminator bytes (high bit clear) in the list's byte range.
"""
import numpy as np


def _zigzag(x):
    return ((x << 1) ^ (x >> 63)).astype(np.uint64)


def encode_lists(offsets, neighbors):
    """Returns (encoded uint8 stream, per-node encoded byte lengths)."""
    offsets = np.asarray(offsets, dtype=np.int64)
    ids = np.asarray(neighbors, dtype=np.int64)
    degree = np.diff(offsets)
    n = len(degree)
    if len(ids) == 0:
        return np.empty(0, dtype=np.uint8), np.zeros(n, dtype=np.int64)
    source = np.repeat(np.arange(n, dtype=np.int64), degree)
    is_head = np.zeros(len(ids), dtype=bool)
    is_head[offsets[:-1][degree > 0]] = True
    gaps = np.diff(ids, prepend=0)
    if np.any(gaps[~is_head] < 0):
        raise ValueError("neighbour lists must be sorted")
    values = gaps.astype(np.uint64)
    values[is_head] = _zigzag(ids[is_head] - source[is_head])

    sizes = np.ones(len(values), dtype=np.int64)
    for bits in (7, 14, 21, 28, 35, 42, 49, 56):
        sizes += values >= (np.uint64(1) << np.uint64(bits))
    at = np.concatenate(([0], np.cumsum(sizes)))
    stream = np.empty(int(at[-1]), dtype=np.uint8)
    for k in range(int(sizes.max())):
        mask = sizes > k
        payload = (values[mask] >> np.uint64(7 * k)) & np.uint64(0x7F)
        more = (sizes[mask] > k + 1).astype(np.uint64) << np.uint64(7)
        stream[at[:-1][mask] + k] = (payload | more).astype(np.uint8)
    list_bytes = at[offsets[1:]] - at[offsets[:-1]]
    return stream, list_bytes


def decode_lists(buf, list_bytes, nodes):
    """Decodes concatenated lists. Returns (flat uint32 neighbours, counts)."""
    buf = np.asarray(buf, dtype=np.uint8)
    list_bytes = np.asarray(list_bytes, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    if len(buf) == 0:
        return np.empty(0, dtype=np.uint32), np.zeros(len(nodes), dtype=np.int64)
    terminator = buf < 0x80
    seen = np.concatenate(([0], np.cumsum(terminator)))
    counts = np.diff(seen[np.concatenate(([0], np.cumsum(list_bytes)))])

    ends = np.flatnonzero(terminator)
    starts = np.concatenate(([0], ends[:-1] + 1))
    position = np.arange(len(buf)) - np.repeat(starts, ends - starts + 1)
    shifted = (buf & 0x7F).astype(np.uint64) << (7 * position).astype(np.uint64)
    values = np.add.reduceat(shifted, starts).astype(np.int64)

    used = counts > 0
    heads = np.concatenate(([0], np.cumsum(counts[used])))[:-1]
    zigzagged = values[heads]
    first = ((zigzagged >> 1) ^ -(zigzagged & 1)) + nodes[used]
    values[heads] = first
    running = np.cumsum(values)
    out = running + np.repeat(first - running[heads], counts[used])
    return out.astype(np.uint32), counts


def pack(list_bytes, line_size, straddle):
    """Assigns a file offset to each list.

    straddle=True packs lists end to end. straddle=False moves a list to the next
    line when it would cross a boundary; lists longer than a line start on a
    boundary and must span lines anyway.
    """
    list_bytes = np.asarray(list_bytes, dtype=np.int64)
    if straddle:
        return np.concatenate(([0], np.cumsum(list_bytes)))[:-1]
    start = np.zeros(len(list_bytes), dtype=np.int64)
    cursor = 0
    for i, size in enumerate(list_bytes.tolist()):
        used = cursor % line_size
        if size and used and used + size > line_size:
            cursor += line_size - used
        start[i] = cursor
        cursor += size
    return start


def write_compressed(path, offsets, neighbors, line_size, straddle=True):
    """Writes the on-disk format. Returns (start, list_bytes, file_bytes)."""
    stream, list_bytes = encode_lists(offsets, neighbors)
    start = pack(list_bytes, line_size, straddle)
    if straddle:
        stream.tofile(path)
        return start, list_bytes, len(stream)
    used = list_bytes > 0
    size = int((start + list_bytes).max()) if len(start) else 0
    image = np.zeros(size, dtype=np.uint8)
    stream_at = np.concatenate(([0], np.cumsum(list_bytes)))[:-1]
    shift = np.repeat(start[used] - stream_at[used], list_bytes[used])
    image[np.arange(len(stream)) + shift] = stream
    image.tofile(path)
    return start, list_bytes, size
