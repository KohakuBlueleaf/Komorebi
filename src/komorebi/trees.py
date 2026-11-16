"""The collectives over the islands' trees an island run gives besides all_gather and
all_reduce: a reduce into one island's copy and a broadcast from one island's copy,
each down or up the routes' tree of that island (km::reduce_to, km::broadcast_from)."""

from komorebi import abi
from komorebi.errors import KomorebiError

SUMS = {"int32": 0, "uint32": 0, "float32": 1, "bfloat16": 2}  # km::Sum


class TreeCalls:
    """IslandRun's tree calls (it gives _collective, _served, runs and graph)."""

    def _root(self, root):
        """The root island, checked; its word (KM_COLL_ROOT: root + 1)."""
        if not 0 <= root < len(self.runs):
            raise KomorebiError(f"root island {root}: the run has {len(self.runs)}")
        return root + 1

    def reduce_to(self, tensor, scratch, root=0, first=0, pages=None, timeout=10.0):
        """Reduce of pages first .. first + pages - 1 of the copy `tensor` (each
        island's copy its partial) into island `root`'s copy, up its tree, in the
        tensor's format; the other copies are left partial. `scratch`: a copy of
        ports x pages (ports: the most neighbours of an island). One island: returns."""
        first, pages = self._collective(tensor, first, pages)
        if tensor.dtype not in SUMS:
            raise KomorebiError(f"reduce_to adds {sorted(SUMS)}: not {tensor.dtype}")
        word = self._root(root)
        if len(self.runs) < 2 or pages == 0:
            return None
        ports = max(len(self.graph.neighbours(i)) for i in range(len(self.runs)))
        self._collective(scratch, 0, ports * pages)
        if scratch.aligned_page_bytes != tensor.aligned_page_bytes:
            raise KomorebiError("reduce_to: the scratch's pages differ in size")
        transfers = [run.backend.transfer for run in self.runs]
        if any(t is None or t.chunk < tensor.aligned_page_bytes for t in transfers):
            raise KomorebiError("reduce_to: the run has no transfer buffer of a page")
        h = abi.headers()
        words = [0] * 8
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_SCRATCH"]] = scratch.address
        words[h["KM_COLL_SUM"]] = SUMS[tensor.dtype]
        words[h["KM_COLL_ROOT"]] = word
        by_island = []
        for transfer in transfers:  # the folds' L1: each island's run's own
            words[h["KM_COLL_BUFFER"]] = transfer.address
            words[h["KM_COLL_ROOM"]] = 2 * transfer.chunk
            by_island.append(list(words))
        return self._served(h["KM_HANDLER_REDUCE"], by_island, timeout)

    def broadcast_from(self, tensor, root=0, first=0, pages=None, timeout=10.0):
        """Broadcast of pages first .. first + pages - 1 of island `root`'s copy of the
        copy `tensor` into every other island's copy, down its tree. One island:
        returns at once."""
        first, pages = self._collective(tensor, first, pages)
        word = self._root(root)
        if len(self.runs) < 2 or pages == 0:
            return None
        h = abi.headers()
        words = [0] * 8
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_ROOT"]] = word
        return self._served(h["KM_HANDLER_GATHER"], [words] * len(self.runs), timeout)
