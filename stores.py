"""Common Store interface, the new cache-line formats and the k-hop query."""
import os
import time
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

import bam_baseline
import codec

SECTOR_SIZE = 512
ID_BYTES = 4  # the denominator of I/O amplification is always raw uint32 ids


class Store(ABC):
    """neighbors_batch returns (flat uint32 neighbours in node order, counts)."""

    def __init__(self, n):
        self.needed_bytes = 0
        self.seen = np.zeros(n, dtype=bool)
        self.unique_needed_bytes = 0
        self.decode_s = 0.0

    def _account(self, nodes, counts):
        self.needed_bytes += int(counts.sum()) * ID_BYTES
        fresh = ~self.seen[nodes]
        # A batch may repeat a node, so dedupe before charging unique bytes.
        unique, first = np.unique(nodes[fresh], return_index=True)
        self.seen[unique] = True
        self.unique_needed_bytes += int(counts[fresh][first].sum()) * ID_BYTES

    def neighbors_batch(self, nodes):
        nodes = np.asarray(nodes, dtype=np.int64)
        flat, counts = self._neighbors(nodes)
        self._account(nodes, counts)
        return flat, counts

    @abstractmethod
    def _neighbors(self, nodes):
        ...

    @abstractmethod
    def stats(self):
        ...

    def _common_stats(self):
        return {
            "needed_bytes": self.needed_bytes,
            "unique_needed_bytes": self.unique_needed_bytes,
            "decode_s": self.decode_s,
        }

    def close(self):
        pass


def _stats(storage_bytes, line_accesses, probes, hits, misses, op_sizes, file_bytes, ram_bytes):
    return {
        "storage_bytes": storage_bytes,
        "line_accesses": line_accesses,
        "probes": probes,
        "hits": hits,
        "misses": misses,
        "op_sizes": op_sizes,
        "file_bytes": file_bytes,
        "index_ram_bytes": ram_bytes,
    }


class BamAdapter(Store):
    """The only coupling to the baseline. Its logic stays in bam_baseline.py."""

    def __init__(self, path, graph, cache_lines, line_size=4096):
        super().__init__(graph.n)
        self.inner = bam_baseline.BamStore(path, graph.offsets, cache_lines, line_size)
        self.file_bytes = os.path.getsize(path)
        self.ram_bytes = 8 * (graph.n + 1)

    def _neighbors(self, nodes):
        return self.inner.neighbors_batch(nodes)

    def stats(self):
        inner = self.inner
        return self._common_stats() | _stats(
            inner.bytes_read, inner.line_accesses, inner.probes, inner.hits, inner.misses,
            Counter({inner.line_size: inner.misses}), self.file_bytes, self.ram_bytes,
        )

    def close(self):
        self.inner.close()


class LoadAllAdapter(Store):
    def __init__(self, path, graph):
        super().__init__(graph.n)
        self.inner = bam_baseline.LoadAllStore(path, graph.offsets)
        self.file_bytes = os.path.getsize(path)
        self.ram_bytes = 8 * (graph.n + 1) + self.file_bytes

    def _neighbors(self, nodes):
        return self.inner.neighbors_batch(nodes)

    def stats(self):
        ops = Counter({self.inner.bytes_read: 1}) if self.inner.bytes_read else Counter()
        return self._common_stats() | _stats(
            self.inner.bytes_read, 0, 0, 0, 0, ops, self.file_bytes, self.ram_bytes
        )


class RawLayout:
    """The baseline's on-disk format seen through the new store code."""

    def __init__(self, path, graph):
        self.path = path
        self.n = graph.n
        self.start = graph.offsets[:-1] * ID_BYTES
        self.list_bytes = np.diff(graph.offsets) * ID_BYTES
        self.file_bytes = os.path.getsize(path)
        self.ram_bytes = 8 * (graph.n + 1)

    def decode(self, buf, list_bytes, nodes):
        return buf.view("<u4").astype(np.uint32), list_bytes // ID_BYTES


class CompressedLayout:
    """Byte offsets stay in RAM. Padding needs per-list lengths as well, since the
    gap to the next start no longer gives a list's size."""

    def __init__(self, path, graph, line_size, straddle=True):
        self.path = path
        self.n = graph.n
        self.start, self.list_bytes, self.file_bytes = codec.write_compressed(
            path, graph.offsets, graph.neighbors, line_size, straddle
        )
        self.ram_bytes = 8 * (graph.n + 1) + (0 if straddle else 4 * graph.n)

    def decode(self, buf, list_bytes, nodes):
        return codec.decode_lists(buf, list_bytes, nodes)


class LineStore(Store):
    """Fixed line cache over any layout: same clock and warp rules as the baseline."""

    def __init__(self, layout, cache_lines, line_size=4096, warp_size=bam_baseline.WARP_SIZE):
        if cache_lines < 1:
            raise ValueError("cache_lines must be >= 1")
        super().__init__(layout.n)
        self.layout = layout
        self.fd = os.open(layout.path, os.O_RDONLY)
        self.line_size = line_size
        self.warp_size = warp_size
        self.cache = np.zeros((cache_lines, line_size), dtype=np.uint8)
        self.tag = [-1] * cache_lines
        self.ref = bytearray(cache_lines)
        self.slot = {}
        self.hand = 0
        self.bytes_read = 0
        self.line_accesses = 0
        self.probes = 0
        self.hits = 0
        self.misses = 0
        self.op_sizes = Counter()

    def close(self):
        os.close(self.fd)

    def _read(self, row, offset, size, line):
        """One device read inside a line; bytes past EOF read as zero."""
        data = os.pread(self.fd, size, line * self.line_size + offset)
        row[offset : offset + size] = 0
        row[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
        self.bytes_read += size
        self.op_sizes[size] += 1

    def _victim(self, pinned):
        frames = len(self.tag)
        for _ in range(2 * frames):
            frame = self.hand
            self.hand = (self.hand + 1) % frames
            if frame in pinned:
                continue
            if self.ref[frame]:
                self.ref[frame] = 0
                continue
            return frame
        return None

    def _install(self, line, pinned):
        """Maps line to a frame, or returns None if the warp pinned every frame."""
        frame = self._victim(pinned)
        if frame is None:
            return None
        if self.tag[frame] >= 0:
            del self.slot[self.tag[frame]]
        self.tag[frame] = line
        self.slot[line] = frame
        self.ref[frame] = 1
        pinned.add(frame)
        return frame

    def _access(self, line, spans, pinned):
        """Returns a row holding every byte of spans, a list of (lo, hi) in the line."""
        self.probes += 1
        frame = self.slot.get(line)
        if frame is not None:
            self.hits += 1
            self.ref[frame] = 1
            pinned.add(frame)
            return self.cache[frame]
        self.misses += 1
        frame = self._install(line, pinned)
        row = self.cache[frame] if frame is not None else np.empty(self.line_size, dtype=np.uint8)
        self._read(row, 0, self.line_size, line)
        return row

    def _neighbors(self, nodes):
        begin = self.layout.start[nodes]
        sizes = self.layout.list_bytes[nodes]
        end = begin + sizes
        out = np.empty(int(sizes.sum()), dtype=np.uint8)
        dst = np.concatenate(([0], np.cumsum(sizes)))
        line_size = self.line_size
        for warp_start in range(0, len(nodes), self.warp_size):
            wanted = {}
            for i in range(warp_start, min(warp_start + self.warp_size, len(nodes))):
                first, last = int(begin[i]), int(end[i])
                if first == last:
                    continue
                for line in range(first // line_size, (last - 1) // line_size + 1):
                    self.line_accesses += 1
                    base = line * line_size
                    wanted.setdefault(line, []).append(
                        (i, max(first, base) - base, min(last, base + line_size) - base)
                    )
            pinned = set()
            for line, users in wanted.items():
                row = self._access(line, [(lo, hi) for _, lo, hi in users], pinned)
                base = line * line_size
                for i, lo, hi in users:
                    at = int(dst[i]) + base + lo - int(begin[i])
                    out[at : at + hi - lo] = row[lo:hi]
        start = time.perf_counter()
        flat, counts = self.layout.decode(out, sizes, nodes)
        self.decode_s += time.perf_counter() - start
        return flat, counts

    def stats(self):
        return self._common_stats() | _stats(
            self.bytes_read, self.line_accesses, self.probes, self.hits, self.misses,
            self.op_sizes, self.layout.file_bytes, self.layout.ram_bytes,
        )


def CompressedBamStore(layout, cache_lines, line_size=4096):
    """Compressed lines, cached compressed, decoded on read."""
    return LineStore(layout, cache_lines, line_size)


class AdaptiveStore(LineStore):
    """Line-sized frames filled sector by sector, with a per-frame valid bitmap.

    A miss reads only the missing sectors the warp needs, one device op per
    contiguous run. Once a resident line has missed `threshold` sectors the rest
    is fetched too. threshold <= 1 is therefore a plain full-line fetch.
    A partly filled frame still occupies a whole line of cache.
    """

    def __init__(self, layout, cache_lines, line_size=4096, threshold=3, sector_size=SECTOR_SIZE):
        super().__init__(layout, cache_lines, line_size)
        if line_size % sector_size:
            raise ValueError("line_size must be a multiple of sector_size")
        self.threshold = threshold
        self.sector_size = sector_size
        self.sectors = line_size // sector_size
        self.full = (1 << self.sectors) - 1
        self.valid = [0] * cache_lines
        self.missed = [0] * cache_lines
        self.sector_misses = 0
        self.upgrades = 0

    def _mask(self, spans):
        mask = 0
        sector_size = self.sector_size
        for lo, hi in spans:
            first, last = lo // sector_size, (hi - 1) // sector_size
            mask |= ((1 << (last - first + 1)) - 1) << first
        return mask

    def _fetch(self, row, mask, line):
        sector_size = self.sector_size
        i = 0
        while mask >> i:
            if not (mask >> i) & 1:
                i += 1
                continue
            run = i
            while (mask >> i) & 1:
                i += 1
            self._read(row, run * sector_size, (i - run) * sector_size, line)

    def _access(self, line, spans, pinned):
        self.probes += 1
        need = self._mask(spans)
        frame = self.slot.get(line)
        if frame is None:
            frame = self._install(line, pinned)
            if frame is None:
                self.misses += 1
                self.sector_misses += need.bit_count()
                row = np.empty(self.line_size, dtype=np.uint8)
                self._fetch(row, self.full if self.threshold <= 1 else need, line)
                return row
            self.valid[frame] = self.missed[frame] = 0
        else:
            self.ref[frame] = 1
            pinned.add(frame)
        missing = need & ~self.valid[frame]
        if not missing:
            self.hits += 1
            return self.cache[frame]
        self.misses += 1
        self.sector_misses += missing.bit_count()
        self.missed[frame] += missing.bit_count()
        if self.missed[frame] >= self.threshold:
            missing = self.full & ~self.valid[frame]
            self.upgrades += 1
        self._fetch(self.cache[frame], missing, line)
        self.valid[frame] |= missing
        return self.cache[frame]

    def stats(self):
        return super().stats() | {"sector_misses": self.sector_misses, "upgrades": self.upgrades}


def k_hop(store, seed, depth, n, order="sorted", rng=None):
    """Sorted ids reachable from seed within depth hops, one neighbors_batch per level."""
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    frontier = np.array([seed], dtype=np.int64)
    for _ in range(depth):
        if len(frontier) == 0:
            break
        if order == "shuffled":
            frontier = rng.permutation(frontier)
        flat, _ = store.neighbors_batch(frontier)
        reached = np.unique(flat).astype(np.int64)
        frontier = reached[~visited[reached]]
        visited[frontier] = True
    return np.flatnonzero(visited)
