"""RECONSTRUCTED BaM baseline: NOT the original bam_toy.py.

bam_toy.py was not in the repo when this harness was built, so this module
re-implements what the brief describes: on-disk CSR, a clock block cache and
warp-style coalescing. New code must not depend on anything here except through
stores.BamAdapter, so the real file can replace this one by editing that adapter.
"""
import os

import numpy as np

WARP_SIZE = 32


def write_csr(path, neighbors):
    """Neighbour ids as little-endian uint32; offsets stay in RAM."""
    np.asarray(neighbors, dtype="<u4").tofile(path)


class BamStore:
    """Fixed-size raw lines, clock replacement, per-warp line coalescing."""

    def __init__(self, path, offsets, cache_lines, line_size=4096, warp_size=WARP_SIZE):
        if cache_lines < 1:
            raise ValueError("cache_lines must be >= 1")
        self.fd = os.open(path, os.O_RDONLY)
        self.offsets = np.asarray(offsets, dtype=np.int64)
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

    def close(self):
        os.close(self.fd)

    def _read_line(self, line):
        """Reads one device block; bytes past EOF read as zero."""
        row = np.zeros(self.line_size, dtype=np.uint8)
        data = os.pread(self.fd, self.line_size, line * self.line_size)
        row[: len(data)] = np.frombuffer(data, dtype=np.uint8)
        self.bytes_read += self.line_size
        return row

    def _victim(self, pinned):
        """Second-chance sweep. None means every frame is pinned by this warp."""
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

    def _access(self, line, pinned):
        self.probes += 1
        frame = self.slot.get(line)
        if frame is not None:
            self.hits += 1
            self.ref[frame] = 1
            pinned.add(frame)
            return self.cache[frame]
        self.misses += 1
        row = self._read_line(line)
        frame = self._victim(pinned)
        if frame is None:
            return row  # warp touches more lines than the cache holds: bypass
        if self.tag[frame] >= 0:
            del self.slot[self.tag[frame]]
        self.tag[frame] = line
        self.slot[line] = frame
        self.ref[frame] = 1
        self.cache[frame] = row
        pinned.add(frame)
        return self.cache[frame]

    def neighbors_batch(self, nodes):
        """Returns (flat uint32 neighbours in node order, per-node counts)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        begin = self.offsets[nodes] * 4
        end = self.offsets[nodes + 1] * 4
        counts = (end - begin) // 4
        out = np.empty(int(end.sum() - begin.sum()), dtype=np.uint8)
        dst = np.concatenate(([0], np.cumsum(end - begin)))
        line_size = self.line_size
        for warp_start in range(0, len(nodes), self.warp_size):
            wanted = {}
            for i in range(warp_start, min(warp_start + self.warp_size, len(nodes))):
                first, last = int(begin[i]), int(end[i])
                if first == last:
                    continue
                for line in range(first // line_size, (last - 1) // line_size + 1):
                    self.line_accesses += 1
                    wanted.setdefault(line, []).append(i)
            pinned = set()
            for line, users in wanted.items():
                row = self._access(line, pinned)
                base = line * line_size
                for i in users:
                    lo = max(int(begin[i]), base)
                    hi = min(int(end[i]), base + line_size)
                    at = int(dst[i]) + lo - int(begin[i])
                    out[at : at + hi - lo] = row[lo - base : hi - base]
        return out.view("<u4").copy(), counts


class LoadAllStore:
    """Reads the whole file once, then serves from RAM."""

    def __init__(self, path, offsets):
        self.path = path
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.neighbors = None
        self.bytes_read = 0

    def neighbors_batch(self, nodes):
        if self.neighbors is None:
            self.neighbors = np.fromfile(self.path, dtype="<u4")
            self.bytes_read = self.neighbors.nbytes
        nodes = np.asarray(nodes, dtype=np.int64)
        begin = self.offsets[nodes]
        counts = self.offsets[nodes + 1] - begin
        if counts.sum() == 0:
            return np.empty(0, dtype=np.uint32), counts
        keep = counts > 0
        starts = np.concatenate(([0], np.cumsum(counts[keep])))[:-1]
        index = np.arange(counts.sum()) - np.repeat(starts, counts[keep])
        return self.neighbors[index + np.repeat(begin[keep], counts[keep])], counts
