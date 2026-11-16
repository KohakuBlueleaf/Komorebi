"""The chain on the host: a call's description (`Call`), the records a list of calls
becomes (km_chain.h), and what is proven or checked before they are written: an
earlier input complete for every core that reads it, a direct input's senders."""

import functools
import importlib
import sys
import types
from typing import NamedTuple

import numpy as np

from komorebi import abi

RANGE_BASE, RANGE_STEP, RANGE_WIDTH = 0xFFF, 0xFF, 0x7FF  # a range word's fields


@functools.cache
def words():
    """km_chain.h's constants."""
    return abi.constants(abi.headers().chain)


class ChainRefused(ValueError):
    """run.chain refused the calls: the call, the tensor and the core in the text."""


class Range(NamedTuple):
    """What a core reads of a tensor's row of tiles: base + step x its group's index,
    `width` tiles; `root_only`: only rank 0 of each group (other ranks: nothing)."""

    base: int
    step: int = 0
    width: int = 1
    root_only: bool = False

    def word(self):
        if not (
            0 <= self.base <= RANGE_BASE
            and 0 <= self.step <= RANGE_STEP
            and 0 <= self.width <= RANGE_WIDTH
        ):
            raise ChainRefused(
                f"{self}: base 0 to 4095, step 0 to 255, width 0 to 2047"
            )
        return (
            self.base
            | self.step << 12
            | self.width << 20
            | (1 << 31 if self.root_only else 0)
        )


class Read(NamedTuple):
    """A tensor a call reads besides the output of the call before: written further
    back in the chain (`earlier`, proven) or before the chain (outside, never written
    by its calls). Its tiles a core: its share of K tiles (base, count) over the ranks
    of its group, or up to three ranges."""

    tensor: str
    k: tuple = None
    ranges: tuple = ()
    earlier: bool = True


class Direct(NamedTuple):
    """A direct input's landing area on every reading core: `tiles` tiles of
    `tile_bytes` bytes (the program's declared arrival area)."""

    tiles: int
    tile_bytes: int


class Call(NamedTuple):
    """A call of a chain: `handler` with its ten words on cores `group`, in groups of
    `size` whose root writes block g (`width` tiles on) of `output` and of each of
    `outputs` ((tensor, width)), then marks. With `input` (the call before's output)
    it waits for the marks of the blocks its share `k` or its `ranges` meet."""

    handler: object
    args: tuple  # a0 to a9
    group: tuple  # (first, count)
    size: int
    width: int
    output: str = None
    input: str = None
    k: tuple = None  # (base, count): K tiles shared out over the group's ranks
    ranges: tuple = ()  # up to three Range
    sends: bool = False  # its parts go up its group's tree (credits)
    reads: tuple = ()  # its other inputs (Read)
    direct: Direct = None  # its input sent into its cores' L1
    word: tuple = None  # (name, argument 0 to 9): the per-submit word, ORed into it
    row: int = 0  # its stamps' row
    name: str = None  # what a refusal calls it
    outputs: tuple = ()  # its other outputs: (tensor, width), block g each a group


def _label(call, index):
    return f"call {index} ({call.name})" if call.name else f"call {index}"


def _shape(call, index):
    """(groups, the cores) of a call, refused when the group does not divide."""
    first, count = call.group
    if not (1 <= call.size <= words()["kChainSizeMask"]) or count % call.size:
        raise ChainRefused(
            f"{_label(call, index)}: {count} cores in groups of {call.size} (1 to "
            f"{words()['kChainSizeMask']}, dividing its cores)"
        )
    for width in (call.width, *(width for _, width in call.outputs)):
        if not 0 <= width <= words()["kChainWidthMask"]:
            raise ChainRefused(f"{_label(call, index)}: width {width}, 0 to 255")
    return count // call.size


def width_of(call, tensor):
    """The width of `tensor`'s blocks as `call` writes them (None: it does not)."""
    if call.output == tensor:
        return call.width
    return dict(call.outputs).get(tensor)


def partition(items, index, count):
    """km::partition: (start, length) of part `index` of `count` over `items`."""
    base, extra = divmod(items, count)
    return index * base + min(index, extra), base + (index < extra)


def tiles_of(k, ranges, rank, index, size):
    """The tiles rank `rank` of group `index` reads, as (first, count) pairs: its share
    of K or its ranges (chain_need)."""
    if k is not None:
        start, length = partition(k[1], rank, size)
        return [(k[0] + start, length)] if length else []
    return [
        (item.base + item.step * index, item.width)
        for item in ranges
        if item.width and not (item.root_only and rank != 0)
    ]


def blocks_met(first, count, width, groups):
    """The blocks (of `width` tiles, `groups` of them) tiles first .. + count meet."""
    if count == 0 or width == 0 or first // width >= groups:
        return range(0)
    last = min((first + count - 1) // width, groups - 1)
    return range(first // width, last + 1)


def _cores(call):
    """(core, rank, group index) of each core of a call."""
    first, count = call.group
    return [
        (first + local, local % call.size, local // call.size) for local in range(count)
    ]


def _writers(calls, upto, tensor):
    """The calls before `upto` that write `tensor`, latest last."""
    return [
        index for index in range(upto) if width_of(calls[index], tensor) is not None
    ]


def check_calls(calls):
    """Every call's form, its input the output of the call before, and its other
    inputs stated: refused (ChainRefused) naming the call and the tensor."""
    if not calls:
        raise ChainRefused("0 calls: a chain has at least one")
    for index, call in enumerate(calls):
        label = _label(call, index)
        _shape(call, index)
        if len(call.args) > 10:
            raise ChainRefused(f"{label}: {len(call.args)} words; a call has 10")
        if len(call.ranges) > words()["kChainRangeCount"] or (
            call.k is not None and call.ranges
        ):
            raise ChainRefused(f"{label}: a K range or up to three ranges, not both")
        if call.input is not None:
            before = calls[index - 1].output if index else None
            if call.input != before:
                raise ChainRefused(
                    f"{label}: reads {call.input} as the call before's output, but the "
                    f"call before writes {before}; state it as an earlier input "
                    "(Read) or an outside tensor"
                )
        for read in call.reads:
            writers = _writers(calls, index, read.tensor)
            if not read.earlier and writers:
                raise ChainRefused(
                    f"{label}: reads {read.tensor} as an outside tensor, but "
                    f"{_label(calls[writers[-1]], writers[-1])} writes it"
                )
            if (
                read.earlier
                and writers
                and writers[-1] == index - 1
                and calls[index - 1].output == read.tensor
                and call.input != read.tensor
            ):
                raise ChainRefused(
                    f"{label}: reads {read.tensor}, which the call before writes, as "
                    "an earlier input; it is this call's input (marked)"
                )
        if call.word is not None and not 0 <= call.word[1] <= 9:
            raise ChainRefused(f"{label}: the per-submit word goes into a0 to a9")


def prove(calls):
    """Proves every block an earlier input's core reads complete when it reads it:
    what a core knows complete grows by the marks it waits for (with what each root
    knew, its group's too when its parts go up the tree) and by its own writes."""
    check_calls(calls)
    cores = 1 + max(call.group[0] + call.group[1] - 1 for call in calls)
    known = [set() for _ in range(cores)]  # (call, block) a core knows complete
    for index, call in enumerate(calls):
        if call.input is not None and index > 0:
            before = calls[index - 1]
            had = marked_knowledge(calls, index - 1, known)
            for group, held in enumerate(had):  # a root: its group's and its block
                known[before.group[0] + group * before.size] |= held
            learned = [
                (core, had[block])
                for core, rank, group in _cores(call)
                for first, count in tiles_of(
                    call.k, call.ranges, rank, group, call.size
                )
                for block in blocks_met(first, count, before.width, len(had))
            ]
            for core, held in learned:
                known[core] |= held
        for read in call.reads:
            _prove_read(calls, index, read, known)
    return True


def marked_knowledge(calls, index, known):
    """What each root of call `index` knows complete when it marks: its own (and its
    group's, when they send up the tree), and the block it wrote (in DRAM unless the
    next call takes it direct)."""
    call = calls[index]
    direct = index + 1 < len(calls) and calls[index + 1].direct is not None
    had = []
    for group in range(_shape(call, index)):
        first = call.group[0] + group * call.size
        members = range(first, first + call.size) if call.sends else [first]
        held = set().union(*(known[core] for core in members))
        if not direct:
            held.add((index, group))
        had.append(held)
    return had


def _prove_read(calls, index, read, known):
    call = calls[index]
    for core, rank, group in _cores(call):
        for first, count in tiles_of(read.k, read.ranges, rank, group, call.size):
            for tile in range(first, first + count):
                writer = next(
                    (
                        at
                        for at in reversed(_writers(calls, index, read.tensor))
                        if width_of(calls[at], read.tensor)
                        and tile // width_of(calls[at], read.tensor)
                        < _shape(calls[at], at)
                    ),
                    None,
                )
                if writer is None:
                    continue
                block = tile // width_of(calls[writer], read.tensor)
                if (writer, block) not in known[core]:
                    raise ChainRefused(
                        f"{_label(call, index)}: core {core} reads {read.tensor} tile "
                        f"{tile}, written by {_label(calls[writer], writer)} (block "
                        f"{block}), and nothing it waits for tells it that block is "
                        "complete"
                    )


def check_direct(calls):
    """A direct input's checks: the call before's output, read by a share of K, each
    core's share within the landing area (its tiles from one root or several, each
    root sending its block's part). The bytes each root of the call before sends and
    to how many cores, by call: {index: [(bytes, cores)]}."""
    report = {}
    for index, call in enumerate(calls):
        if call.direct is None:
            continue
        label = _label(call, index)
        if index == 0 or call.input is None or call.k is None:
            raise ChainRefused(
                f"{label}: a direct input is the call before's output, read by a share "
                "of K tiles"
            )
        before = calls[index - 1]
        if before.outputs:
            raise ChainRefused(
                f"{label}: a direct input from {_label(before, index - 1)}, which "
                "writes several outputs: a data mark does not wait for the others' "
                "writes (send it through DRAM)"
            )
        groups = _shape(before, index - 1)
        sends = [[0, 0] for _ in range(groups)]
        for core, rank, group in _cores(call):
            for first, count in tiles_of(call.k, (), rank, group, call.size):
                if count > call.direct.tiles:
                    raise ChainRefused(
                        f"{label}: core {core} reads {count} tiles of {call.input}; "
                        f"its arrival area holds {call.direct.tiles}"
                    )
                for block in blocks_met(first, count, before.width, groups):
                    low = max(first, block * before.width)
                    high = min(first + count, (block + 1) * before.width)
                    sends[block][0] += (high - low) * call.direct.tile_bytes
                    sends[block][1] += 1
        report[index] = [tuple(item) for item in sends]
    return report


def records(calls, address, handlers=None):
    """The calls' records (uint32, 16 words each) for a chain at DRAM `address`: each
    names the next one in the same bank (the last 0) and its handler's number from
    `handlers` (None: 0 each, the chain command's)."""
    w = words()
    arguments = abi.headers().positions[2]  # a0 to a9's words of a command
    out = np.zeros((len(calls), w["kChainWords"]), np.uint32)
    for index, call in enumerate(calls):
        flags = (
            (w["kChainMarked"] if call.input is not None and index > 0 else 0)
            | (w["kChainSends"] if call.sends else 0)
            | (w["kChainDataMark"] if call.direct is not None else 0)
        )
        word = arguments[call.word[1]] if call.word is not None else 0
        needs = w["kChainNeedsRanges"] if call.ranges else w["kChainNeedsK"]
        if not 0 <= call.row <= w["kChainRowMask"]:
            raise ChainRefused(f"{_label(call, index)}: stamp row 0 to 63")
        handler = 0 if handlers is None else handlers[index]
        if not 0 <= handler <= 0xFF:
            raise ChainRefused(f"{_label(call, index)}: handler {handler}, 1 to 255")
        out[index, : len(call.args)] = call.args
        out[index, w["kChainGroup"]] = call.group[0] | call.group[1] << 16
        out[index, w["kChainInfo"]] = (
            flags
            | word << w["kChainTokenShift"]
            | needs << w["kChainNeedsShift"]
            | call.size << w["kChainSizeShift"]
            | call.width << w["kChainWidthShift"]
            | handler << w["kChainHandlerShift"]
        )
        ranges = (
            [Range(call.k[0], 0, call.k[1]).word()]
            if call.k is not None
            else [item.word() for item in call.ranges]
        )
        out[index, w["kChainRanges"] : w["kChainRanges"] + len(ranges)] = ranges
        last = index + 1 == len(calls)
        next_address = 0 if last else address + (index + 1) * 64
        out[index, w["kChainNext"]] = next_address | call.row
    return out


class Chain:
    """Calls as records in DRAM, written once (run.chain); run.submit(chain, **word)
    runs them as one command. `report`: a direct input's bytes and cores a root."""

    def __init__(self, handler, group, mode, flags, word, tensor, report, calls):
        self.handler, self.group, self.mode, self.flags = handler, group, mode, flags
        self.word = word  # the per-submit word's name, or None
        self.tensor = tensor  # the records, a copy in every bank
        self.report = report
        self.calls = list(calls)
        self.direct, self.written = [], []  # a class's: tensors sent direct, written

    def args(self, words_given):
        """The chain command's a0 to a8 for a submit's words (the record in bank 0)."""
        names = set(words_given)
        if names != ({self.word} if self.word else set()):
            want = self.word or "none"
            raise ValueError(f"the chain's per-submit word is {want}, given {names}")
        value = int(words_given[self.word]) if self.word else 0
        bank = 1 << abi.headers()["KM_BLOCK_BANK_SHIFT"]  # bank 0, as a block's field
        return [
            value,
            self.flags,
            0,
            0,
            self.mode,
            0,
            0,
            self.tensor.address,
            64 | bank,
        ]

    @property
    def span(self):
        """The sequence numbers its command takes: one for each kChainRecordsMax
        records (KM_FLAG_SPAN), so its records' tags are no later command's."""
        return -(-len(self.calls) // words()["kChainRecordsMax"])

    def free(self):
        self.tensor.free()


class IslandChain:
    """A chain over islands: each island's records in its own DRAM (`chains`, a Chain
    an island); run.submit(chain, **word) runs it as one ticket, a part an island."""

    def __init__(self, chains, group):
        self.chains = list(chains)
        self.group = group
        self.calls = self.chains[0].calls
        self.report = self.chains[0].report
        self.held = None  # (region, address, bytes): records at one address

    @property
    def address(self):
        """The records' bank address when it is one on every island (a thread's chain
        command's a7), else None."""
        addresses = {chain.tensor.address for chain in self.chains}
        return addresses.pop() if len(addresses) == 1 else None

    def free(self):
        for chain in self.chains:
            chain.free()
        if self.held is not None:
            region, address, bytes_ = self.held
            region.give(address, bytes_)
            self.held = None


def build(calls):
    """The checks run.chain makes before writing records: the calls' form, every
    earlier input proven, the direct inputs checked; their report. Over islands the
    same: a mark, a credit or a direct input goes to a rank of any island."""
    prove(calls)
    report = check_direct(calls)
    names = {call.word[0] for call in calls if call.word is not None}
    if len(names) > 1:
        raise ChainRefused(f"one per-submit word a chain, not {sorted(names)}")
    return report, (names.pop() if names else None)


class _Spelled(types.ModuleType):
    """`km.chain` is this module and, called, the decorator of a class's chain
    method (komorebi.dsl.front.chain): `@km.chain` beside `km.chain.Call`."""

    def __call__(self, function):
        return importlib.import_module("komorebi.dsl.front").chain(function)


sys.modules[__name__].__class__ = _Spelled
