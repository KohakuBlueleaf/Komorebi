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

    def _tree(self, words, tree, root=None):
        """`tree` (island i's parent at i, the root its own; None: the routes' tree of
        each root), checked, into the command's words: KM_COLL_GIVEN in the root word,
        the parents 4 bits an island in words KM_COLL_TREE and the next. Its root."""
        if tree is None:
            return root
        h = abi.headers()
        n = len(self.runs)
        tree = [int(parent) for parent in tree]
        roots = [i for i, parent in enumerate(tree) if parent == i]
        if len(tree) != n or n > 16 or len(roots) != 1:
            raise KomorebiError(
                f"tree={tree}: a parent for each of the {n} islands (at most 16), one "
                "island its own parent (the root)"
            )
        for i, parent in enumerate(tree):
            if parent != i and parent not in self.graph.neighbours(i):
                raise KomorebiError(
                    f"tree={tree}: island {i}'s parent {parent} is not one link away "
                    f"(its neighbours: {list(self.graph.neighbours(i))})"
                )
            at, steps = i, 0
            while tree[at] != at and steps <= n:
                at, steps = tree[at], steps + 1
            if steps > n:
                raise KomorebiError(f"tree={tree}: island {i}'s parents make a loop")
        if root is not None and root != roots[0]:
            raise KomorebiError(f"tree={tree}: its root is {roots[0]}, not {root}")
        packed = sum(parent << 4 * i for i, parent in enumerate(tree))
        words[h["KM_COLL_ROOT"]] |= h["KM_COLL_GIVEN"]
        words[h["KM_COLL_TREE"]] = packed & 0xFFFFFFFF
        words[h["KM_COLL_TREE"] + 1] = packed >> 32
        return roots[0]

    def _ports(self):
        """The most neighbours of an island: a reduce's scratch holds this many partials
        of each page (one a child, at the port it arrives by)."""
        return max(len(self.graph.neighbours(i)) for i in range(len(self.runs)))

    def reduce_to(
        self, tensor, scratch, root=0, first=0, pages=None, tree=None, timeout=10.0
    ):
        """Reduce of pages first .. first + pages - 1 of the copy `tensor` (each
        island's copy its partial) into island `root`'s copy, up its tree (or `tree`,
        a parent an island, rooted at `root`), in the tensor's format; the others are
        left partial. `scratch`: a copy of ports x pages. One island: returns."""
        first, pages = self._collective(tensor, first, pages, "run.reduce_to()")
        if tensor.dtype not in SUMS:
            raise KomorebiError(f"reduce_to adds {sorted(SUMS)}: not {tensor.dtype}")
        word = self._root(root)
        if len(self.runs) < 2 or pages == 0:
            return None
        self._collective(scratch, 0, self._ports() * pages)
        if scratch.aligned_page_bytes != tensor.aligned_page_bytes:
            raise KomorebiError("reduce_to: the scratch's pages differ in size")
        transfers = [run.backend.transfer for run in self.runs]
        if any(t is None or t.chunk < tensor.aligned_page_bytes for t in transfers):
            raise KomorebiError("reduce_to: the run has no transfer buffer of a page")
        h = abi.headers()
        words = [0] * 10
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_SCRATCH"]] = scratch.address
        words[h["KM_COLL_SUM"]] = SUMS[tensor.dtype]
        words[h["KM_COLL_ROOT"]] = word
        self._tree(words, tree, root)
        by_island = []
        for transfer in transfers:  # the folds' L1: each island's run's own
            words[h["KM_COLL_BUFFER"]] = transfer.address
            words[h["KM_COLL_ROOM"]] = 2 * transfer.chunk
            by_island.append(list(words))
        return self._served(h["KM_HANDLER_REDUCE"], by_island, timeout)

    def broadcast_from(
        self, tensor, root=0, first=0, pages=None, tree=None, timeout=10.0
    ):
        """Broadcast of pages first .. first + pages - 1 of island `root`'s copy of the
        copy `tensor` into every other island's copy, down its tree (or `tree`, rooted
        at `root`). One island: returns at once."""
        first, pages = self._collective(tensor, first, pages, "run.broadcast_from()")
        word = self._root(root)
        if len(self.runs) < 2 or pages == 0:
            return None
        h = abi.headers()
        words = [0] * 10
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_ROOT"]] = word
        self._tree(words, tree, root)
        return self._served(h["KM_HANDLER_GATHER"], [words] * len(self.runs), timeout)
