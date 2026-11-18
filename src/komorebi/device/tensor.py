"""Device buffers with the shape, data type and layout of what they hold, written from
and read into numpy arrays: interleaved in DRAM (a page a row, or a page a 32 x 32
tile), or one row in the L1 of every core of a grid.
"""

import numpy as np

from komorebi import formats
from komorebi.device import mock
from komorebi.errors import KomorebiError


class DramFull(KomorebiError):
    """A DRAM tensor that does not fit what is left of its card's DRAM, refused before
    tt-metal's allocator is asked: `said` the bytes a bank asked and left, `fits` the
    most pages of its page size that fit now."""

    def __init__(self, message, said, fits):
        super().__init__(message)
        self.said, self.fits = said, fits


def dram_room(device, shape, dtype, layout, size, page):
    """Refuses (DramFull) a DRAM tensor of `size` bytes in pages of `page` that the
    card's DRAM cannot take now: tt-metal's spread over its banks against each bank's
    largest free block (the allocator's own figures, device.dram_room)."""
    bank, banks, largest, free = device.dram_room(size, page)
    if bank <= largest:
        return
    pages = size // page
    aligned = bank // -(-pages // banks)  # a page's bytes in a bank, aligned
    fits = largest // aligned * banks
    said = f"{bank:,} bytes a bank asked, {largest:,} free in one piece"
    raise DramFull(
        f"a DRAM tensor of shape {tuple(shape)} ({dtype}, {layout}), {size:,} bytes, "
        f"does not fit what is left of the card's DRAM: {bank:,} bytes in each of its "
        f"{banks} banks, where a bank has {largest:,} free in one piece ({free:,} "
        f"free in all); at most {fits:,} pages of {page:,} bytes fit now: free "
        "tensors, or spread it over more cards",
        said,
        fits,
    )


class Tensor:
    """A device buffer (the native module's `Buffer`) and the shape on the card, the
    data type name and the layout ("row" or "tile") of its contents. `shard_grid`
    (x, y): one row in each core's L1, cores in row-major order; None: DRAM.
    `transposed_tiles`: each tile held transposed (a turned product's SrcB operand)."""

    def __init__(self, buffer, shape, dtype, layout, shard_grid=None):
        self.buffer = buffer
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.shard_grid = shard_grid
        self._transposed = False
        self.written = False  # the host wrote its data (transposed_tiles is fixed)
        self.page_tiles = 1  # a tile tensor's tiles a DRAM page
        self.freed = False
        self.kept = None  # what is written, kept by a session before its first load
        self.sent = None  # the chain's call that sent it direct: not in DRAM
        self.loaded = None  # its session's test: a run of it is loaded (read refused)
        # its shape before the padding to whole tiles (None: whole tiles), and its
        # padding zeros ("clean") or a handler's ("dirty") and by what (dsl.padding)
        self.logical = None
        self.padding, self.padding_by = "clean", None

    def __repr__(self):
        where = "DRAM" if self.shard_grid is None else f"L1 of {self.shard_grid}"
        paged = f", {self.page_tiles} tiles a page" if self.page_tiles != 1 else ""
        return f"Tensor({self.shape}, {self.dtype}, {self.layout}{paged}, {where})"

    @property
    def transposed_tiles(self):
        """Whether each tile is held transposed; set before the tensor's data is
        written (refused after: the stored tiles stay as written)."""
        return self._transposed

    @transposed_tiles.setter
    def transposed_tiles(self, value):
        refuse_turned_after(self, value)
        self._transposed = bool(value)

    @property
    def address(self):
        """The buffer's address: in every DRAM bank, or in every core's L1."""
        return self.buffer.address

    @property
    def page_bytes(self):
        return self.buffer.page_size

    @property
    def aligned_page_bytes(self):
        """A page's footprint in its bank."""
        return self.buffer.aligned_page_size

    @property
    def pages(self):
        return self.buffer.num_pages

    @property
    def nbytes(self):
        return self.buffer.size

    @property
    def in_dram(self):
        return self.shard_grid is None

    def accessor_args(self):
        """The compile-time arguments of a TensorAccessor over this buffer."""
        return list(self.buffer.accessor_args())

    def tensor_args(self):
        """What km::Tensor reads (compile arguments): the accessor's; with the
        tensor's reference word (`address`), what a kernel's page calls take."""
        return self.accessor_args()

    def write(self, data):
        """Writes `data` (this shape) over the whole buffer, once the cores of a
        launched program are done."""
        raw, shape = formats.to_bytes(
            data, self.dtype, self.layout, self.transposed_tiles
        )
        if shape != self.shape:
            raise ValueError(f"data of shape {shape}: the tensor is {self.shape}")
        refuse_live_write(self)
        self.buffer.write(raw)
        self.written = True
        self.padding, self.padding_by = "clean", None
        self.sent = None
        if self.kept is not None:
            self.kept.append(("data", data, None))

    def read(self):
        """The whole buffer as a numpy array (bfloat16 as float32), once the cores of
        a launched program are done."""
        return formats.from_bytes(
            self.read_bytes(),
            self.shape,
            self.dtype,
            self.layout,
            self.transposed_tiles,
        )

    def read_bytes(self):
        """The whole buffer's bytes (uint8), read as read() reads them."""
        in_dram(self)
        refuse_live(self)
        if mock.active():  # a mock run computed nothing: zeros
            refuse_closed(self)
            return np.zeros(self.buffer.size, np.uint8)
        return np.asarray(self.buffer.read())

    def free(self):
        """Gives the buffer's memory back; the tensor is not used after this (a kept
        one's free goes into its session's history: a reopen frees it in turn)."""
        self.buffer.free()
        self.freed = True
        noted = getattr(self, "kept_free", None)
        if noted is not None:
            noted()


def refuse_closed(tensor):
    """The card's refusal of a read once the session that made `tensor` (its `loaded`
    test's) closed its device: a mock's read refused as the card's buffer read is."""
    session = getattr(tensor.loaded, "__self__", None)
    if session is not None and getattr(session, "device", True) is None:
        raise RuntimeError("the device is closed")


def refuse_live(tensor):
    """A read of a tensor whose session has a run loaded, refused at once: the buffer
    read waits for the cores, which a loaded run keeps (tt-metal gives up after 30 s).
    """
    if tensor.loaded is None or not tensor.loaded():
        return
    live = (
        "session.read(tensor) reads its pages live, session.read_bf16(tensor) a "
        "bfloat16 tile tensor (by DRAM banks or the host channel: what no kernel "
        "writes any more)"
        if tensor.in_dram
        else "session.read(tensor) reads an L1 tensor live only through the host "
        "channel"
    )
    raise KomorebiError(
        f"{tensor!r} is read while a run is loaded: tensor.read() waits for the cores, "
        f"which the run keeps; read it after run.stop(), or {live}"
    )


def refuse_turned_after(tensor, value):
    """Refuses transposed_tiles changed on a tensor whose data the host wrote: its
    stored tiles stay as written, so its reads and a class's product would be wrong."""
    if tensor.written and bool(value) != tensor.transposed_tiles:
        raise KomorebiError(
            f"{tensor!r}: transposed_tiles={bool(value)} set after its data was "
            "written (its tiles stay as written): give session.tensor(..., "
            f"transposed_tiles={bool(value)}), or set it on an empty tensor before "
            "its first write"
        )


def refuse_live_write(tensor):
    """A write of a tensor whose session has a run loaded, refused at once: the buffer
    write waits for the cores, as the read does (refuse_live)."""
    if tensor.loaded is None or not tensor.loaded():
        return
    raise KomorebiError(
        f"{tensor!r} is written while a run is loaded: tensor.write() waits for the "
        "cores, which the run keeps; write it before the load or after run.stop(), or "
        "live with session.write(tensor, pages) (its pages as session.read gives "
        "them, through the run's transfer or the host channel)"
    )


def empty(
    module,
    device,
    shape,
    dtype,
    layout="row",
    shard_grid=None,
    address=None,
    page_tiles=1,
):
    """A tensor of `shape` (padded to whole tiles in a tile layout), not written; an
    L1 tensor may be given the `address` it takes in every core. `page_tiles`: a
    tile tensor's DRAM page holds that many tiles along the last dimension."""
    formats.check_dtype(dtype)
    logical = formats.logical_shape(shape, layout)
    shape = formats.padded_shape(shape, layout)
    page = formats.page_bytes(shape, dtype, layout, page_tiles)
    size = formats.buffer_bytes(shape, dtype, layout, page_tiles)
    if shard_grid is not None:
        if page_tiles != 1:
            raise ValueError(
                f"page_tiles={page_tiles} with shard_grid: an L1 tensor's page is a "
                "core's row"
            )
        if layout != "row" or len(shape) != 2:
            raise ValueError("an L1 tensor is [cores, words] rows, one a core")
        shard_grid = tuple(int(extent) for extent in shard_grid)
    if shard_grid is None and address is None:
        dram_room(device, shape, dtype, layout, size, page)
    buffer = module.Buffer(device, size, page, shard_grid, address)
    tensor = Tensor(buffer, shape, dtype, layout, shard_grid)
    tensor.page_tiles = page_tiles
    unwritten(tensor, logical)
    return tensor


def unwritten(tensor, logical):
    """Marks a tensor not written yet: its `logical` shape (None: whole tiles), its
    padding not known to hold zeros."""
    tensor.logical = logical
    if logical is not None:
        tensor.padding, tensor.padding_by = "dirty", "not written (made empty)"


def from_array(
    module, device, data, dtype, layout="row", shard_grid=None, page_tiles=1
):
    """A tensor holding `data`, written before it is returned."""
    raw, shape = formats.to_bytes(data, dtype, layout)
    tensor = empty(
        module, device, shape, dtype, layout, shard_grid, page_tiles=page_tiles
    )
    tensor.buffer.write(raw)
    tensor.written = True
    return tensor


def sent_direct(written, sent):
    """A chain submitted: the tensors its calls write are in DRAM again, those it
    sends direct ((tensor, the call)) are not until a later write."""
    for tensor in written:
        tensor.sent = None
    for tensor, call in sent:
        tensor.sent = call


def in_dram(*tensors):
    """Refuses a host read of a tensor a chain sent direct and nothing wrote since:
    its DRAM holds what it held before (a silent wrong)."""
    for tensor in tensors:
        if getattr(tensor, "sent", None) is not None:
            raise RuntimeError(
                f"a tensor of shape {tensor.shape} at {tensor.address:#x} was sent "
                f"direct by {tensor.sent}, not to DRAM: drop km.direct, or read the "
                "next tensor; a write or a chain that writes it makes it readable"
            )
