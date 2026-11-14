"""The host channel: the 1 GiB of pinned host memory the driver maps when a card opens,
which the cores reach through the PCIe tile. A session reserves a region of it, and a
transfer moves pages between a DRAM tensor and the region. Pure functions and a view
of the mapping; the session issues the transfers."""

import ctypes
import re
from typing import NamedTuple

import numpy as np

from komorebi import abi
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


def fits(transfer, count, l1_bytes):
    """The kernel's bounds (km::transfer_fits) on the host, before anything is sent."""
    h = abi.headers()
    t, word = transfer, h["KM_TRANSFER_WORD_BYTES"]
    end = t.offset + t.pages * t.page
    words_end = t.words + count * word
    ok = 1 <= count <= h["KM_CORES_MAX"] and t.pages >= 1
    ok = ok and t.page >= 16 and t.page % 16 == 0
    ok = ok and t.chunk >= t.page and t.chunk % t.page == 0
    ok = ok and t.offset % 16 == 0 and t.region % 16 == 0 and t.buffer % 16 == 0
    ok = ok and t.region <= h["KM_CHANNEL_BYTES"] and end <= t.region
    ok = ok and t.buffer + 2 * t.chunk + word <= l1_bytes
    ok = ok and t.words % 16 == 0 and words_end <= t.region
    return bool(ok and (words_end <= t.offset or t.words >= end))


def check(transfer, count, l1_bytes):
    """Raises ChannelError unless the transfer fits its region and its buffer."""
    if not fits(transfer, count, l1_bytes):
        raise ChannelError(
            f"a transfer of pages {transfer.first} .. "
            f"{transfer.first + transfer.pages - 1} ({transfer.page} B each) by "
            f"{count} cores does not fit the region of {transfer.region} B or its L1"
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
