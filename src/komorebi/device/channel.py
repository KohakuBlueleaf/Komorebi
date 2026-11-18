"""The host channel: the 1 GiB of pinned host memory the driver maps when a card opens,
which the cores reach through the PCIe tile. A session reserves a region of it, and a
transfer moves pages between a DRAM tensor and the region. Pure functions and a view
of the mapping; the session issues the transfers."""

import ctypes
import re
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi.device.dram import DRAM_BANKS
from komorebi.errors import ChannelError

CHANNEL_BYTES = 1 << 30  # the driver's hugepage channel of one card
REGION_DEFAULT = 256 << 20  # the session's region: channel offsets [0, region)
REGION_ALIGN = 64 << 10
WORDS_BYTES = 64 << 10  # the region's last bytes: a sequence word a rank
CHUNK_DEFAULT = 16 << 10  # each of a core's two L1 chunks: KM_TRANSFER_CHUNK_BYTES
CORES_DEFAULT = 16  # cores of a transfer at most
# What the driver logs at the open when the channel is not where the cores write
WARNINGS = ("does not match the expected address", "already holding the sysmem")
# "va-end perms offset dev inode path" of /proc/self/maps
MAPS_LINE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s+(\S+)$")


class Region(NamedTuple):
    """A session's region: `bytes` from channel offset 0; data below `words`, the
    sequence words from `words` on."""

    bytes: int

    @property
    def words(self):
        return self.bytes - WORDS_BYTES

    @property
    def data(self):
        return self.bytes - WORDS_BYTES


def check_region(region):
    """The region's size, refused (ValueError) unless a multiple of REGION_ALIGN with
    room for data, inside the channel."""
    if (
        not isinstance(region, int)
        or region % REGION_ALIGN
        or not 2 * WORDS_BYTES <= region <= CHANNEL_BYTES
    ):
        raise ValueError(
            f"channel={region!r}: a multiple of {REGION_ALIGN} from {2 * WORDS_BYTES} "
            f"to {CHANNEL_BYTES} bytes (0 or None: no channel)"
        )
    return Region(region)


def warning_in(text):
    """The driver's warning about the channel's NoC address, if the open logged it."""
    for needle in WARNINGS:
        if needle in text:
            return needle
    return None


def find(maps, address):
    """(address, path) of the card's channel in a /proc/<pid>/maps text: the read-write
    1 GiB mapping of a file named ...tenstorrent that starts at `address`, the driver's
    record of its host memory; or (None, why)."""
    if not address:
        return None, "the driver mapped no host memory for the card"
    found = []
    for line in maps.splitlines():
        matched = MAPS_LINE.match(line.strip())
        if matched and int(matched[1], 16) == address:
            found.append(matched)
    if len(found) != 1 or not found[0][4].endswith("tenstorrent"):
        return None, f"no hugepage mapping of the card at {address:#x}"
    start, end, perms = int(found[0][1], 16), int(found[0][2], 16), found[0][3]
    if end - start != CHANNEL_BYTES or not perms.startswith("rw"):
        return None, f"{found[0][4]}: not a read-write mapping of {CHANNEL_BYTES} bytes"
    return start, found[0][4]


def view(address, region):
    """The region's bytes as a uint8 array over the mapping (not a copy)."""
    return np.ctypeslib.as_array((ctypes.c_uint8 * region.bytes).from_address(address))


def partition(items, index, count):
    """(start, length) of rank `index` of `count`, as km::partition: contiguous, the
    first items % count ranks one item more."""
    base, extra = divmod(items, count)
    return index * base + min(index, extra), base + (index < extra)


def rounds(pages, page, region):
    """[(first, pages)] of a transfer of `pages` pages of `page` bytes, each round as
    many pages as the region's data holds."""
    per = region.data // page
    if per < 1:
        raise ValueError(f"a page of {page} bytes does not fit the region's data")
    return [(first, min(per, pages - first)) for first in range(0, pages, per)]


def ranks(pages, page, chunk, most=CORES_DEFAULT):
    """How many cores share `pages` pages: one a chunk of pages, at most `most`."""
    chunk_pages = max(chunk // page, 1)
    return max(1, min(most, -(-pages // chunk_pages)))


def chunk_bytes(page, chunk):
    """The chunk a transfer uses: the most whole pages that fit `chunk` (at least one
    page; a larger page needs a buffer of its size)."""
    return max(chunk // page, 1) * page


def chunk_for(page, room):
    """The chunk of a transfer of pages of `page` bytes where a core can spare `room`
    bytes for its two chunks and a word slot: CHUNK_DEFAULT, or the page itself when
    that is larger; None when two of them do not fit."""
    chunk = chunk_bytes(page, CHUNK_DEFAULT)
    word = abi.headers()["KM_TRANSFER_WORD_BYTES"]
    return chunk if 2 * chunk + word <= room else None


DRAM_LINE = 64  # a DRAM read keeps its low 6 address bits (abi KM_COPY_ALIGN)


class Transfer(NamedTuple):
    """The ten argument words of a PUSH or PULL (abi KM_TRANSFER_* order)."""

    address: int
    page: int
    first: int
    pages: int
    offset: int
    words: int
    region: int
    nonce: int
    buffer: int
    chunk: int

    def args(self):
        h = abi.headers()
        out = [0] * 10
        for name in self._fields:
            out[h[f"KM_TRANSFER_{name.upper()}"]] = int(getattr(self, name))
        return out


def bound(transfer, count, l1_bytes, tensor_pages=None):
    """The first of the kernel's bounds (km::transfer_fits) the transfer breaks, named;
    None when it fits. An L1 tensor's floor (the firmware's L1) is the kernel's alone;
    `tensor_pages`: the pages first + pages may reach (None: not compared)."""
    h = abi.headers()
    t, word, line = transfer, h["KM_TRANSFER_WORD_BYTES"], DRAM_LINE
    end = t.offset + t.pages * t.page
    words_end = t.words + count * word
    rules = [
        (1 <= count <= h["KM_CORES_MAX"], f"1 to {h['KM_CORES_MAX']} cores"),
        (t.pages >= 1, "one page or more"),
        (t.page >= 16 and t.page % 16 == 0, "a page of whole 16 B"),
        (
            tensor_pages is None or t.first + t.pages <= tensor_pages,
            f"pages inside the tensor's {tensor_pages}",
        ),
    ]
    if t.chunk == h["KM_TRANSFER_L1"]:
        rules += [
            (t.pages == count, "an L1 tensor's page a core"),
            (t.address % 16 == 0, "an L1 page's address in whole 16 B"),
            (t.address + t.page <= l1_bytes, f"an L1 page inside L1 ({l1_bytes} B)"),
            (
                t.address + t.page <= t.buffer or t.buffer + word <= t.address,
                "an L1 page apart from the transfer slot",
            ),
        ]
    else:
        rows = -(-(t.first + t.pages) // DRAM_BANKS)
        rules += [
            (t.chunk >= t.page and t.chunk % t.page == 0, "a chunk of whole pages"),
            (
                t.address % line == 0 and t.page % line == 0 and t.buffer % line == 0,
                f"a DRAM tensor's address, page and the L1 buffer in whole {line} B",
            ),
            (
                t.address + rows * t.page <= 1 << 32,
                "a DRAM page inside its bank's 4 GiB",
            ),
        ]
    rules += [
        (
            t.offset % 16 == 0 and t.region % 16 == 0 and t.buffer % 16 == 0,
            "the offset, region and buffer in whole 16 B",
        ),
        (
            t.region <= h["KM_CHANNEL_BYTES"] and end <= t.region,
            f"the pages inside the region of {t.region} B",
        ),
        (
            t.buffer + 2 * t.chunk + word <= l1_bytes,
            f"two chunks of {t.chunk} B and the slot inside L1 ({l1_bytes} B)",
        ),
        (
            t.words % 16 == 0 and words_end <= t.region,
            "the sequence words in whole 16 B inside the region",
        ),
        (
            words_end <= t.offset or t.words >= end,
            "the sequence words apart from pages",
        ),
    ]
    return next((rule for held, rule in rules if not held), None)


def fits(transfer, count, l1_bytes):
    """Whether the transfer keeps every bound of the kernel's (bound)."""
    return bound(transfer, count, l1_bytes) is None


def check(transfer, count, l1_bytes, tensor_pages=None):
    """Raises ChannelError unless the transfer keeps every bound, naming the one it
    breaks."""
    broken = bound(transfer, count, l1_bytes, tensor_pages)
    if broken is not None:
        raise ChannelError(
            f"a transfer of pages {transfer.first} .. "
            f"{transfer.first + transfer.pages - 1} ({transfer.page} B each) by "
            f"{count} cores does not keep its bound: {broken}"
        )


def sequence_words(nonce, pages, page, count):
    """[count, 4] uint32: the word each rank writes after its pages (nonce, bytes,
    rank, the mark ^ nonce), a push's and a pull's alike."""
    mark = abi.headers()["KM_TRANSFER_MARK"]
    out = np.zeros((count, 4), np.uint32)
    for rank in range(count):
        out[rank] = [nonce, partition(pages, rank, count)[1] * page, rank, mark ^ nonce]
    return out


def words_view(region_bytes, region, count):
    """[count, 16] uint32 view of the ranks' word slots in the region: the sequence
    word (4 words), then the wall clock at the rank's start and end (2 x 2 words)."""
    stride = abi.headers()["KM_TRANSFER_WORD_BYTES"]
    raw = region_bytes[region.words : region.words + count * stride]
    return raw.view("<u4").reshape(count, stride // 4)


def cycles(words):
    """Cycles from the first rank's start to the last rank's end, from their words."""
    wall = words[:, 4:8].astype(np.int64)
    return int(
        (wall[:, 2] | wall[:, 3] << 32).max() - (wall[:, 0] | wall[:, 1] << 32).min()
    )


def l1_ranks(first, pages, width, grid_x, holds=None):
    """[(page, count, rank, field)] of pages first .. first + pages - 1 of an L1 tensor
    `width` cores wide (page p on core (p % width, p // width)) on a grid `grid_x` wide:
    runs of pages on consecutive ranks whose cores hold one program; None if one holds
    none (`holds` None: every core holds field 0)."""
    out = []
    for page in range(first, first + pages):
        rank = page // width * grid_x + page % width
        field = 0
        if holds is not None:
            fields = [i for i, held in enumerate(holds) if held[rank]]
            if not fields:
                return None
            field = fields[0]
        last = out[-1] if out else None
        if last and last[2] + last[1] == rank and last[3] == field:
            out[-1] = (last[0], last[1] + 1, last[2], field)
        else:
            out.append((page, 1, rank, field))
    return out


def l1_cores(first, pages, width):
    """The cores of pages first .. first + pages - 1 of an L1 tensor `width` cores
    wide, as (x, y) in page order and as rectangles (x0, y0, x1, y1), one a row."""
    places = [(page % width, page // width) for page in range(first, first + pages)]
    rects = []
    for x, y in places:
        if rects and rects[-1][3] == y and rects[-1][2] + 1 == x:
            rects[-1] = (rects[-1][0], y, x, y)
        else:
            rects.append((x, y, x, y))
    return places, rects


def transfer_group(holds, cores, count):
    """(first, count, program field) of `count` cores (fewer if no window that large
    holds one program) whose cores all hold one program; `holds` None: cores 0 on."""
    if holds is None:
        return 0, min(count, cores), 0
    for wanted in range(min(count, cores), 0, -1):
        for field, held in enumerate(holds):
            window = np.convolve(np.asarray(held, int), np.ones(wanted, int), "valid")
            at = np.flatnonzero(window == wanted)
            if at.size:
                return int(at[0]), wanted, field
    raise ChannelError("no core holds a program: a transfer needs one")
