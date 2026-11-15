"""Device buffers with the shape, data type and layout of what they hold, written from
and read into numpy arrays: interleaved in DRAM (a page a row, or a page a 32 x 32
tile), or one row in the L1 of every core of a grid.
"""

import numpy as np

from komorebi import formats, prebuild


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
        self.transposed_tiles = False
        self.freed = False
        self.kept = None  # what is written, kept by a session before its first load
        self.sent = None  # the chain's call that sent it direct: not in DRAM

    def __repr__(self):
        where = "DRAM" if self.shard_grid is None else f"L1 of {self.shard_grid}"
        return f"Tensor({self.shape}, {self.dtype}, {self.layout}, {where})"

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
        self.buffer.write(raw)
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
        if prebuild.active():  # a mock run computed nothing: zeros
            return np.zeros(self.buffer.size, np.uint8)
        return np.asarray(self.buffer.read())

    def free(self):
        """Gives the buffer's memory back; the tensor is not used after this."""
        self.buffer.free()
        self.freed = True


def empty(module, device, shape, dtype, layout="row", shard_grid=None, address=None):
    """A tensor of `shape` (padded to whole tiles in a tile layout), not written; an
    L1 tensor may be given the `address` it takes in every core."""
    formats.check_dtype(dtype)
    shape = formats.padded_shape(shape, layout)
    page = formats.page_bytes(shape, dtype, layout)
    size = formats.buffer_bytes(shape, dtype, layout)
    if shard_grid is not None:
        if layout != "row" or len(shape) != 2:
            raise ValueError("an L1 tensor is [cores, words] rows, one a core")
        shard_grid = tuple(int(extent) for extent in shard_grid)
    buffer = module.Buffer(device, size, page, shard_grid, address)
    return Tensor(buffer, shape, dtype, layout, shard_grid)


def from_array(module, device, data, dtype, layout="row", shard_grid=None):
    """A tensor holding `data`, written before it is returned."""
    raw, shape = formats.to_bytes(data, dtype, layout)
    tensor = empty(module, device, shape, dtype, layout, shard_grid)
    tensor.buffer.write(raw)
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
