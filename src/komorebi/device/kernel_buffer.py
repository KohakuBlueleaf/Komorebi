"""The kernel buffer that follows the built kernels: tt-metal fixes it when the card is
opened, so a load whose kernels need another size opens the card again at that size,
makes the session's tensors again at their addresses and writes into them what they
were given (docs/reference/host.md)."""

import gc
import sys
import time
import weakref

import numpy as np

from komorebi import formats
from komorebi.device import tensor as tensor_module
from komorebi.errors import KomorebiError

ALIGN = 64  # the allocator's base is aligned up to this
MARGIN = 1024  # bytes beyond the build's total


class Resize(Exception):
    """A load's kernels need a kernel buffer of `size` bytes: the load is undone and
    made again on the card reopened at that size."""

    def __init__(self, size, why):
        super().__init__(f"kernel buffer {size} B: {why}")
        self.size = size
        self.why = why


def need(total):
    """The kernel buffer that holds a build of `total` config bytes: the total and the
    margin, 64-aligned."""
    return -(-(total + MARGIN) // ALIGN) * ALIGN


def _freed(kept_ref, index):
    kept = kept_ref()
    if kept is not None and kept.history is not None:
        kept.history.append(("free", index, None))


class Kept:
    """The tensors a session made before its first load, in order: each make and free,
    so that a reopened card's allocator gives each the same address again, and what
    was written into each, as references to the host arrays given (never copies; a
    tensor made empty and not written holds none). `history` None: no longer kept."""

    def __init__(self):
        self.history = []
        self.made = []
        self.loading = False  # a load's own tensors are not kept: it frees them

    def keep(self, tensor, address=None, data=None):
        """Records the make of `tensor` (`address`: the one it was given; `data`: the
        array it was made from, as (array, transposed tiles))."""
        if self.history is None or self.loading:
            return
        index = len(self.made)
        params = (tensor.shape, tensor.dtype, tensor.layout, tensor.shard_grid, address)
        self.history.append(("make", index, params))
        self.made.append(weakref.ref(tensor))
        tensor.kept = [] if data is None else [("made", *data)]
        weakref.finalize(tensor, _freed, weakref.ref(self), index)

    def stop(self):
        """Nothing more is kept: a run was loaded, its kernels may have written."""
        for ref in self.made:
            tensor = ref()
            if tensor is not None:
                tensor.kept = None
        self.history, self.made = None, []

    def live(self):
        """The kept tensors still alive, in the order made."""
        gc.collect()  # a load undone leaves its own tensors to the collector
        return [ref() for ref in self.made if ref() is not None]

    def refused(self):
        """Why the card cannot be opened again under these tensors, or None."""
        if self.history is None:
            return "a run was loaded on it (its kernels may have written its tensors)"
        return None  # L1 tensors are made from the top of L1: the buffer moves none

    def remake(self, session):
        """Makes every kept tensor again on the reopened card, in the history's order
        (each freed one made and freed in turn), and gives each live one its new
        buffer; KomorebiError if one lands at another address."""
        alive = {index: ref() for index, ref in enumerate(self.made)}
        again = {}
        for kind, index, params in self.history:
            if kind == "free":
                again.pop(index, None)
                continue
            shape, dtype, layout, shard_grid, address = params
            again[index] = tensor_module.empty(
                session._module, session.device, shape, dtype, layout, shard_grid,
                address,
            )  # fmt: skip
        for index, tensor in alive.items():
            if tensor is None:
                continue
            made = again[index]
            if made.address != tensor.address:
                raise KomorebiError(
                    f"{tensor} at {tensor.address:#x} came back at {made.address:#x} "
                    "on the card reopened for the kernel buffer: its contents are "
                    "lost. Give the session the size (Session(kernel_bytes=...))"
                )
            tensor.buffer = made.buffer


def rewrite(session, tensor):
    """Writes into `tensor` again what it was given before the reopen, in order;
    the bytes written."""
    writes, tensor.kept = tensor.kept or [], None  # not kept twice
    done = 0
    try:
        for kind, data, extra in writes:
            if kind == "made":
                raw, _ = formats.to_bytes(data, tensor.dtype, tensor.layout, extra)
                session.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
            elif kind == "pages":
                session.write(tensor, np.ascontiguousarray(data, np.uint8), extra)
            else:  # Tensor.write
                tensor.write(data)
            done += np.asarray(data).nbytes
    finally:
        tensor.kept = writes
    return done


def reopen(session, size, worker_l1, why):
    """Reopens the session's card with a kernel buffer of `size` bytes (the runtime's
    `worker_l1` for it): closes the card, opens it again, makes every kept tensor
    again and writes into it what it was given, with a line before and after."""
    started = time.perf_counter()
    kept = session._kept
    before = session.l1()[0]
    session.log(
        f"kernel buffer {before:,} -> {size:,} B ({why}): the card is closed "
        "and opened again"
    )
    session.channel = None  # the mapping goes with the device
    session._close_device()
    session._worker_l1 = worker_l1
    session.open()
    kept.remake(session)
    tensors = kept.live()
    written = sum(rewrite(session, tensor) for tensor in tensors)
    took = time.perf_counter() - started
    session.log(
        f"kernel buffer {session.l1()[0]:,} B: card opened again, {len(tensors)} "
        f"tensor(s) made again and {written:,} B written ({took:.2f} s)"
    )
    if not getattr(session, "log_given", True):  # a reopen is said without a log too
        print(
            f"komorebi: kernel buffer {before:,} -> {size:,} B ({why}): the card was "
            f"opened again, {len(tensors)} tensor(s) made again ({took:.2f} s)",
            file=sys.stderr,
        )
