"""Komorebi-nn's tensors on the card and its op table, written by hand (a descriptor a
kernel: the op it serves, its shapes, a planner to a Call on decode's plans and words);
the device that appends calls to the executor's open list and syncs only on a host
read; the chain of a list's calls. No torch: front.py maps torch's ops onto it."""

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from komorebi import formats
from komorebi.errors import KomorebiError
from projects.decode import model as md
from projects.dense import plan as pl
from projects.nn import nn

D, S, CHAIN, RECORD = nn.D, nn.S, nn.card.CHAIN, nn.RECORD
TILE = md.TILE
EXCLUSIVE = nn.EXCLUSIVE
# Products beyond the decode layer's, as (K tiles, N tiles): 2048 x 6144 (x @ w1 and
# x @ w3 apart). Planned by decode's own planner with its constants (extra_plan()).
EXTRA_SHAPES = ((64, 192),)
# The executor's list: open lists from entry 0, closed lists after OPEN_ENTRIES.
OPEN_ENTRIES = 1024
ROWS_FIELD = D["DC_ROWS_MASK"] << D["DC_ROWS_SHIFT"]


def extra_plan(plans, shape):
    """The Split of a product of `shape` (K tiles, N tiles) as md.plans() makes the
    layer's: 1 x 8 blocks, groups as large as the cores allow up to GROUP_MOST, the
    layer plans' arrival buffer, chunk and W halves (every plan shares the buffers)."""
    first = next(iter(plans.values()))
    group = min(shape[0], md.CORES // pl.cdiv(shape[1], md.BLOCK_WIDTH), md.GROUP_MOST)
    if md.SETS * math.ceil(math.log2(group)) * md.BLOCK_WIDTH > first.arrive:
        raise ValueError(f"{shape}: its tree's rounds do not fit the arrival buffer")
    return pl.split_plan(
        1,
        *shape,
        group,
        1,
        md.BLOCK_WIDTH,
        first.kc,
        cores=md.CORES,
        arrive=first.arrive,
        pattern="AB",
        depth=first.depth,
        depth_b=first.depth_b,
    )


class Card(nn.Decode):
    """nn.Decode (decode's program with the executor) with the products of `shapes`
    added to its table of plans, after the layer's (their indices do not move)."""

    def __init__(self, *args, shapes=EXTRA_SHAPES, **kwargs):
        self.extra = tuple(shapes)
        super().__init__(*args, **kwargs)

    def program(self):
        for shape in self.extra:
            self.plans.setdefault(shape, extra_plan(self.plans, shape))
        self.plan_index = {shape: i for i, shape in enumerate(self.plans)}
        return super().program()


class Part:
    """A DRAM tensor's pages from a page that is a whole round of the banks on: a
    tensor of its own at `address` (as kernels address it) in `whole`."""

    def __init__(self, whole, address, shape):
        self.whole, self.address, self.shape = whole, address, shape

    def free(self):
        if self.address == self.whole.address:
            self.whole.free()


def tile_form(shape):
    """A tensor's [rows, columns] as the card holds it (bf16 tiles, rows padded to
    whole tiles), or None when it has none: columns not whole tiles, or rank 1."""
    if len(shape) < 2 or shape[-1] % TILE:
        return None
    rows = math.prod(shape[:-1])
    if len(shape) > 2 and rows % TILE:
        return None
    return rows, shape[-1]


class Tensor:
    """A tensor of Komorebi-nn: its shape (bf16), its DRAM tensor on the card (or None:
    it lives on the host), its host values when the host holds them, whether a call
    not yet run produces it, and its rows' sums of squares (a tensor) when a norm
    after it needs them."""

    def __init__(self, shape, card=None, host=None):
        self.shape = tuple(int(extent) for extent in shape)
        self.card, self.host = card, host
        self.pending = False
        self.squares = None

    @property
    def rows(self):
        """The rows that hold values: a call's DC_ROWS field."""
        return math.prod(self.shape[:-1])

    @property
    def kn(self):
        """A weight's (K tiles, N tiles)."""
        return self.shape[0] // TILE, self.shape[1] // TILE

    def __repr__(self):
        where = "card" if self.card is not None else "host"
        return f"Tensor({self.shape}, {where}{', pending' if self.pending else ''})"


@dataclass(frozen=True)
class Descriptor:
    """A kernel call of the table: the op it serves, its kernel and operation, its
    shapes and data types, and its planner (device, inputs, output, constants) -> Call.
    """

    name: str
    serves: str
    kernel: str
    shapes: str
    dtypes: str
    planner: Callable

    def plan(self, device, inputs, out, **constants):
        started = time.perf_counter()
        call = self.planner(device, inputs, out, **constants)
        device.clock["planner"] += time.perf_counter() - started
        return call


def reduce_word(device, rows):
    """A product's a3: the reduce form and the rows that hold a token."""
    if not 1 <= rows <= TILE:
        raise ValueError(f"{rows} rows: a call takes one tile row (1 to 32)")
    return device.words[D["DC_WORD_REDUCE"]] | rows << D["DC_ROWS_SHIFT"]


def product(device, a, weight, y, operation=0, other=(0, 0, 0), a3=0, info=0):
    """SPLIT FULL on `weight`'s plan (decode's plan word): a0 a, a1 W, a2 y, a6 the
    operation word (0: none), a7 to a9 `other`; the chain record's info `info`."""
    if a.card is None or weight.card is None or y.card is None:
        raise ValueError("a product's tensors are on the card")
    if a.shape[-1] != weight.shape[0] or y.shape[-1] not in (
        weight.shape[1],
        weight.shape[1] // 2,
    ):
        raise ValueError(f"{a.shape} @ {weight.shape} into {y.shape}")
    plan = device.plan_word(weight.kn)
    args = (
        a.card.address,
        weight.card.address,
        y.card.address,
        reduce_word(device, a.rows) | a3,
        0,
        plan & 0xFF,
        operation,
        *other,
    )
    return nn.Call(
        device.words[D["DC_WORD_SPLIT"]],
        (0, plan >> 8),
        args,
        EXCLUSIVE,
        0,
        operation & 0xFF,
        (RECORD["kChainSends"] | info, 0, 0, 0),
    )


def plan_matmul(device, inputs, out):
    a, w = inputs
    return product(device, a, w, out)


def plan_matmul_add(device, inputs, out):
    a, w, residual = inputs
    if device.plans[w.kn].groups() != D["DC_PARTS"]:
        raise ValueError(f"{w.shape}: the add takes plans of {D['DC_PARTS']} groups")
    if residual.card is None or out.squares is None:
        raise ValueError("the add: a residual on the card, the output's squares")
    other = (residual.card.address, device.constants, out.squares.card.address)
    return product(device, a, w, out, device.words[D["DC_WORD_ADD"]], other)


def norm_other(device, x):
    """The norm's a7 to a9 after a product: x's squares, the constants, epsilon."""
    squares = device.squares_of(x)
    return (squares.card.address, device.constants, device.words[D["DC_WORD_EPSILON"]])


def plan_norm_gate(device, inputs, out):
    x, w = inputs
    width = device.words[D["DC_WORD_GATE_TILES"]] << 16
    other = norm_other(device, x)
    return product(device, x, w, out, device.words[D["DC_WORD_GATE"]], other, 0, width)


def plan_norm_matmul(device, inputs, out):
    x, w = inputs
    other = norm_other(device, x)
    return product(device, x, w, out, device.words[D["DC_WORD_SCALE"]], other)


def plan_norm_qkv(device, inputs, out, layer, position):
    x, w = inputs
    squares = device.squares_of(x).card.address
    other = (squares, device.decode.tensors["qk"].address, position | layer << 16)
    scale = device.words[D["DC_WORD_SCALE"]]
    return product(device, x, w, out, scale, other, D["DC_QK"])


def plan_attend(device, inputs, out, position):
    qkv, keys, values = inputs
    words, decode = device.words, device.decode
    if (
        keys.card is None
        or keys.card.shape[0] != decode.model.kv_heads * decode.context
    ):
        raise ValueError(f"attend: a cache of {decode.context} rows a KV head")
    if qkv.rows != 1 or not 0 <= position < decode.context:
        raise ValueError("attend: one token's row, a position in the cache")
    plan = words[D["DC_WORD_PLAN_ATTEND"]]
    needs = words[D["DC_WORD_ATTEND_NEEDS"] : D["DC_WORD_ATTEND_NEEDS"] + 3]
    return nn.Call(
        words[D["DC_WORD_SPLIT"]],
        (0, plan >> 8),
        (
            qkv.card.address,
            keys.card.address,
            out.card.address,
            reduce_word(device, qkv.rows),
            S["kSplitOp"],
            plan & 0xFF,
            words[D["DC_WORD_ATTEND"]],
            values.card.address,
            decode.tensors["attend"].address,
            position | words[D["DC_WORD_CACHE_TILES"]] << 16,
        ),
        EXCLUSIVE,
        0,
        D["DC_OVERLAY_ATTEND"],
        (
            RECORD["kChainSends"]
            | RECORD["kChainNeedsRanges"] << RECORD["kChainNeedsShift"],
            *needs,
        ),
    )


SHAPES = "a [r, K] (r <= 32), W [K, N] of a plan"
TABLE = {
    "matmul": Descriptor(
        "matmul", "a @ W", "SPLIT full, no operation", SHAPES, "bf16", plan_matmul
    ),
    "matmul_add": Descriptor(
        "matmul_add",
        "r + a @ W (and the sum's squares)",
        "SPLIT full, add",
        SHAPES + ", N = 2048 (8 groups)",
        "bf16",
        plan_matmul_add,
    ),
    "norm_gate": Descriptor(
        "norm_gate",
        "silu(norm(x) Wg) * (norm(x) Wu)",
        "SPLIT full, gate",
        "x [r, 2048] with squares, W' [2048, 2M] (norm folded, 4 + 4 tiles a block)",
        "bf16",
        plan_norm_gate,
    ),
    "norm_matmul": Descriptor(
        "norm_matmul",
        "norm(x) @ W",
        "SPLIT full, scale",
        "x [r, 2048] with squares, W' [2048, N] (norm folded)",
        "bf16",
        plan_norm_matmul,
    ),
    "norm_qkv": Descriptor(
        "norm_qkv",
        "norm(x) @ [Wq Wk Wv], heads normed and rotated",
        "SPLIT full, scale with DC_QK",
        "x [1, 2048] with squares, W' [2048, 4096]",
        "bf16",
        plan_norm_qkv,
    ),
    "attend": Descriptor(
        "attend",
        "komorebi::attend (custom op, by name)",
        "SPLIT op, attend",
        "qkv [1, 4096], caches [8 x 4096, 128]",
        "bf16",
        plan_attend,
    ),
}


def fold(norm_weight, w):
    """W' = diag(weight) W in bf16 values, as md.fused() folds a norm into W."""
    return md.bf16(
        np.asarray(norm_weight, np.float32)[:, None] * np.asarray(w, np.float32)
    )


def interleave(gate, up):
    """The gate's and up's columns in turns of GATE_TILES tiles each (md.fused())."""
    k, n = gate.shape
    turn = md.GATE_TILES * TILE
    pair = np.stack(
        [gate.reshape(k, n // turn, turn), up.reshape(k, n // turn, turn)], axis=2
    )
    return pair.reshape(k, 2 * n)


def chained(calls):
    """A list's calls as items: runs where each call's input is the call before's
    output (the chain's marks, chain.h) as ("chain", calls), at most kChainRecordsMax a
    chain; the rest ("command", call)."""
    runs = []
    for call in calls:
        if (
            runs
            and runs[-1][-1].args[2] == call.args[0]
            and len(runs[-1]) < RECORD["kChainRecordsMax"]
        ):
            runs[-1].append(call)
        else:
            runs.append([call])
    items = []
    for run in runs:
        if len(run) == 1:
            items.append(("command", run[0]))
            continue
        marked = [run[0]]
        for call in run[1:]:
            info, *ranges = call.record
            marked.append(
                call._replace(record=(info | RECORD["kChainMarked"], *ranges))
            )
        items.append(("chain", marked))
    return items


def entries_of(items, first):
    """nn.Entries of chained() items from entry `first`."""
    entries = nn.Entries(first)
    for kind, item in items:
        if kind == "chain":
            entries.chain(item, CHAIN["kChainMode"], EXCLUSIVE)
        else:
            entries.command(item)
    return entries


def joint_pages(x, values):
    """The pages of an empty_joint tensor: x's values, then their squares."""
    rows = np.zeros(x.card.shape, np.float32)
    rows[: x.rows] = np.asarray(values, np.float32).reshape(x.rows, -1)
    both = np.concatenate([md.bf16(rows), md.squares(rows)], axis=1)
    return md.tile_pages(formats.bf16_bits(both))


def tensor_pages(tensor, values):
    """The pages of a tensor's DRAM tensor holding host values (float, to bf16)."""
    rows, columns = tensor.card.shape
    flat = np.zeros((rows, columns), np.float32)
    given = np.asarray(values, np.float32).reshape(-1, columns)
    flat[: given.shape[0]] = given
    return md.tile_pages(formats.bf16_bits(flat))


class TileWriter:
    """Writes bf16 bits [rows, columns] into tile pages (a OneSubmission.tiles view of
    a [32, columns] tensor), and with `squares` the rows' sums of squares in the 8
    tiles after them (md.squares: the first tile's diagonal): one strided copy."""

    def __init__(self, rows, columns, squares):
        self.rows, self.columns, self.squares = rows, columns, squares
        self.padded = np.zeros((TILE, columns), np.uint16)
        self.wide = np.zeros((rows, columns, 2), np.uint16)  # the bits as float32
        diagonal = np.arange(TILE)
        self.face, self.at = diagonal // 16, diagonal % 16

    def __call__(self, tiles, bits):
        padded, n = self.padded, self.columns // TILE
        padded[: self.rows] = bits
        tiles[:, :n] = padded.reshape(1, 2, 16, n, 2, 16).transpose(0, 3, 1, 4, 2, 5)
        if self.squares:
            self.wide[..., 1] = bits
            values = self.wide.view(np.float32).reshape(self.rows, self.columns)
            sums = np.zeros(TILE, np.float64)  # bf16 products are exact in float32
            sums[: self.rows] = np.add.reduce(values * values, axis=1, dtype=np.float64)
            tiles[:, n:] = 0
            tiles[0, n, self.face, self.face, self.at, self.at] = formats.bf16_bits(
                sums.astype(np.float32)
            )


def tile_bits(tiles, rows):
    """The first `rows` rows of tile pages (a tiles() view) as bf16 bits, a copy."""
    r, c = tiles.shape[:2]
    flat = tiles.transpose(0, 2, 4, 1, 3, 5).reshape(r * TILE, c * TILE)
    return np.ascontiguousarray(flat[:rows])


class Device:
    """Komorebi-nn on one card: the decode program with the executor (Card), the
    tensors, an open list the calls are appended to (synced on a host read only), the
    fallbacks counted, and the host's seconds by term (`clock`). `mirror`: a fallback's
    output keeps its host values and goes to the card only when a call takes it."""

    def __init__(self, session, model, deadline, log, mirror=False, fence=False):
        self.session, self.log, self.mirror, self.fence = session, log, mirror, fence
        self.decode = Card(session, model, 1, deadline, log, chain=True)
        self.plans = self.decode.plans
        self.executor = nn.Executor(session, self.decode.run)
        self.words = self.decode.words()
        self.constants = self.decode.tensors["constants"].address
        self.list = None
        self.closed_next = OPEN_ENTRIES
        self.fallbacks = Counter()
        self.clock = Counter()
        self.runs = []  # (name, stats slot, thread ticks) of every list run
        self.qk_layers = {}
        self.prepared = {}  # a compiled graph's prepared weights, by name and tensors
        self.pending_outputs = []
        self.ones = []  # the OneSubmission calls made

    def plan_word(self, kn):
        if kn not in self.decode.plan_index:
            raise ValueError(f"no plan for a product of {kn} tiles")
        return self.decode.plan_index[kn] | self.plans[kn].cores() << 8

    # Tensors

    def empty(self, shape, squares=False):
        """A tensor on the card, not written; with its squares' tensor."""
        form = tile_form(shape)
        if form is None:
            raise ValueError(f"{shape}: no tile form")
        started = time.perf_counter()
        out = Tensor(shape, self.session.empty(form, "bfloat16", "tile"))
        if squares:
            out.squares = Tensor(
                (TILE, D["DC_PARTS"] * TILE),
                self.session.empty((TILE, D["DC_PARTS"] * TILE), "bfloat16", "tile"),
            )
        self.clock["allocate"] += time.perf_counter() - started
        return out

    def write(self, tensor, values, term="write"):
        """Host values (float, rounded to bf16) into the tensor's DRAM tensor."""
        started = time.perf_counter()
        rows, columns = tensor.card.shape
        flat = np.asarray(values, np.float32).reshape(-1, columns)
        if flat.shape[0] != rows:
            padded = np.zeros((rows, columns), np.float32)
            padded[: flat.shape[0]] = flat
            flat = padded
        self.session.write(tensor.card, md.tile_pages(formats.bf16_bits(flat)))
        self.clock[term] += time.perf_counter() - started

    def upload(self, values, term="write"):
        """Host values as a tensor: on the card when it has a tile form (and the host
        keeps them), else on the host only."""
        values = md.bf16(np.asarray(values, np.float32))
        form = tile_form(values.shape)
        if form is None:
            return Tensor(values.shape, None, values)
        out = self.empty(values.shape)
        self.write(out, values, term)
        out.host = values
        return out

    def squares_of(self, x):
        """x's squares' tensor: an add's output has it; for a tensor the host holds
        the host computes it (md.squares) and writes it."""
        if x.squares is None:
            if x.host is None or x.shape[-1] != D["DC_ROW_TILES"] * TILE:
                raise ValueError("a norm after a product: x's squares are not known")
            self.write_squares(x, x.host)
        return x.squares

    def empty_joint(self, shape):
        """A [r, 2048] tensor and its squares as one DRAM tensor (its pages, then the
        squares' 8), so that one host write carries both (write_joint)."""
        rows, columns = tile_form(shape)
        banks = self.session.dram_banks  # page i in bank i % banks (the interleave)
        if rows > TILE or (columns // TILE) % banks:
            raise ValueError(f"{shape}: not one tile row of whole bank rounds")
        whole = self.session.empty(
            (TILE, columns + D["DC_PARTS"] * TILE), "bfloat16", "tile"
        )
        step = columns // TILE // banks * whole.aligned_page_bytes
        out = Tensor(shape, Part(whole, whole.address, (TILE, columns)))
        squares = (TILE, D["DC_PARTS"] * TILE)
        out.squares = Tensor(squares, Part(whole, whole.address + step, squares))
        return out

    def write_joint(self, x, values):
        """x's values and their squares in one host write (an empty_joint tensor)."""
        started = time.perf_counter()
        self.session.write(x.card.whole, joint_pages(x, values))
        self.clock["in_write"] += time.perf_counter() - started

    def write_squares(self, x, values):
        """x's rows' sums of squares (md.squares of host values) into its squares'
        tensor, given once."""
        if x.squares is None:
            x.squares = self.empty((TILE, D["DC_PARTS"] * TILE))
        rows = np.zeros((TILE, x.shape[-1]), np.float32)
        rows[: x.rows] = np.asarray(values, np.float32).reshape(x.rows, -1)
        self.write(x.squares, md.squares(rows), "squares")

    def to_card(self, tensor):
        """A tensor the host holds (a mirrored fallback output) onto the card."""
        if tensor.card is None:
            if tile_form(tensor.shape) is None:
                raise ValueError(f"{tensor.shape}: no tile form for the card")
            card = self.empty(tensor.shape).card
            tensor.card = card
            self.write(tensor, tensor.host, "fallback_write")
        return tensor

    def read(self, tensor, term="read"):
        """The tensor's host values: what the host holds, else synced and read."""
        if tensor.host is not None:
            return tensor.host
        if tensor.pending:
            self.sync()
        started = time.perf_counter()
        rows, columns = tensor.card.shape
        if rows == TILE:
            tiles = self.session.read_tiles_bf16(tensor.card, 0, columns // TILE)
            flat = tiles.transpose(1, 0, 2).reshape(TILE, columns)
        else:
            (flat,) = self.session.read_bf16(tensor.card)
        out = flat.reshape(-1)[: math.prod(tensor.shape)].reshape(tensor.shape)
        self.clock[term] += time.perf_counter() - started
        return out

    def free(self, tensor):
        if tensor.card is not None:
            tensor.card.free()
        if tensor.squares is not None:
            tensor.squares.card.free()

    # The open list

    def append(self, call, outputs):
        """`call` onto the open list (opened on the first); its outputs pending."""
        if self.list is None:
            started = time.perf_counter()
            if self.executor.slot == nn.NN["NN_SLOTS"]:
                self.executor.slot = 0
            self.list = self.executor.open(0, fence=self.fence)
            self.clock["open"] += time.perf_counter() - started
        elif self.list.next + 2 * RECORD["kChainRecordsMax"] > OPEN_ENTRIES:
            self.sync()
            return self.append(call, outputs)
        before = dict(self.list.times)
        self.list.append(call)
        for name in ("build", "store", "link"):
            self.clock[name] += self.list.times[name] - before[name]
        for out in outputs:
            out.pending = True
        self.pending_outputs.extend(outputs)

    def sync(self):
        """The open list's END appended and waited for: every pending output written."""
        if self.list is None:
            return
        started = time.perf_counter()
        ticks = self.list.sync()
        self.clock["sync"] += time.perf_counter() - started
        self.runs.append(("open", self.list.slot, ticks))
        self.list = None
        for out in self.pending_outputs:
            out.pending = False
        self.pending_outputs = []

    # Closed lists

    def reserve(self, entries):
        """Entries for a closed list of `entries`: its first index."""
        first = self.closed_next
        if first + entries > nn.NN["NN_ENTRIES_MAX"]:
            first = OPEN_ENTRIES
        self.closed_next = first + entries
        return first

    def write_list(self, entries):
        started = time.perf_counter()
        self.executor.write(entries)
        self.clock["list_write"] += time.perf_counter() - started

    def run_list(self, entries, name="closed"):
        """A written closed list run once and waited for (RUN): the thread's ticks."""
        self.sync()
        started = time.perf_counter()
        if self.executor.slot == nn.NN["NN_SLOTS"]:
            self.executor.slot = 0
        ticks, slot = self.executor.closed(entries)
        self.clock["run"] += time.perf_counter() - started
        self.runs.append((name, slot, ticks))
        return ticks

    def collect(self):
        """Every one-submission call's RUN event (off the calls' path)."""
        for one in self.ones:
            one.collect()

    def fallback(self, name):
        self.fallbacks[name] += 1

    def qk_layer(self, q_norm, k_norm):
        """The qk tensor's slot of a pair of head norm weights (written once)."""
        key = (q_norm.tobytes(), k_norm.tobytes())
        if key not in self.qk_layers:
            layer = len(self.qk_layers)
            if layer >= D["DC_QK_LAYERS"]:
                raise ValueError(f"the qk tensor holds {D['DC_QK_LAYERS']} layers")
            pages = md.qk_weight_pages(q_norm, k_norm)
            first = D["DC_QK_WEIGHTS"] + 2 * D["DC_HEAD_TILES"] * layer
            self.session.write(self.decode.tensors["qk"], pages, first=first)
            self.qk_layers[key] = layer
        return self.qk_layers[key]

    def stats(self):
        """{run index: its terms (nn.slot_terms)} of the last NN_SLOTS list runs (the
        slots before are overwritten), after the stop."""
        words = self.executor.stats()
        out = {}
        first = max(0, len(self.runs) - nn.NN["NN_SLOTS"])
        for index in range(first, len(self.runs)):
            name, slot, ticks, *entries = self.runs[index]
            out[index] = nn.slot_terms(words[slot]) | {"ticks": ticks, "name": name}
            if entries:  # a one-submission call: its first and last entries' commands
                at, last = entries[0].first, entries[0].first + len(entries[0]) - 1
                out[index]["pull"] = int(words[slot][nn.NN["NN_ST_ENTRY_TICKS"] + at])
                out[index]["push"] = int(words[slot][nn.NN["NN_ST_ENTRY_TICKS"] + last])
        return out


# A compiled call as one host submission (OneSubmission)
WORDS_TIMEOUT_S = 2.0  # a call's wait for y's sequence words
SPIN_S = 5e-3  # a call's spin before its looks pause: longer than a call


class Move(NamedTuple):
    """One transfer of a call: the whole DRAM tensor, its session.channel_transfer
    (its pages in the region, its wait) and the call's entry of it."""

    tensor: object
    transfer: object
    call: nn.Call


class OneSubmission:
    """A closed list as one host submission: its first entries pull the inputs' pages
    from the session's host channel region, its last push the outputs' there (README,
    "One submission a call"). A call: pages in, the nonce, one RUN, a spin on the last
    push's words, pages out; RUN's event is collected later (collect())."""

    def __init__(self, device, entries, inputs, outputs):
        session, run = device.session, device.decode.run
        self.device, self.session, self.moves = device, session, []
        for push, tensors in ((False, inputs), (True, outputs)):
            for tensor in tensors:
                transfer = session.channel_transfer(tensor, push)
                entry = transfer.entry
                call = nn.Call(entry.handler, entry.group, entry.args, 0, entry.field)
                self.moves.append(Move(tensor, transfer, call))
        self.pulls = [move for move in self.moves[: len(inputs)]]
        self.pushes = [move for move in self.moves[len(inputs) :]]
        out = nn.Entries(device.reserve(len(entries) + len(self.moves)))
        transfer = nn.NN["NN_KIND_TRANSFER"]
        for move in self.pulls:
            out.command(move.call, transfer)
        base = len(out.rows)
        for row, count in entries.items:
            out.items.append((base + row, count))
        out.rows += [row.copy() for row in entries.rows]
        for move in self.pushes:
            out.command(move.call, transfer)
        self.entries = out
        device.write_list(out)
        self.last = self.pushes[-1].transfer  # y's: its words end a call
        self.pending = []  # (handle, slot) of RUNs not collected
        self.local = (
            {  # each move's pages on the host, copied to or from the region whole
                id(move): np.zeros(move.tensor.pages * move.tensor.page_bytes, np.uint8)
                for move in self.moves
            }
        )
        thread = run.threads[nn.CLUSTER].thread  # the X280 thread the list runs on
        self.handler = thread.handler("run")  # RUN's words, the slot and nonce a call
        self.args = [device.executor.address(out.first), 1, 0, 0]
        self.spans = []  # cycles from the first pull's start to the last push's end
        device.ones.append(self)

    def rows(self, move):
        """[pages, page_bytes] uint8 view of a move's pages in the region."""
        return move.transfer.data

    def tiles(self, move):
        """A move's pages as bf16 bits [R, C, 2, 2, 16, 16] (tile rows, tile columns,
        faces, face rows and columns), a view of its host copy: the region itself is
        copied whole (a strided read or write of the channel costs three times)."""
        tensor = move.tensor
        rows, columns = tensor.shape
        if tensor.page_bytes != tensor.aligned_page_bytes:
            raise ValueError("tiles() takes pages without padding")
        raw = self.local[id(move)]
        return raw.view(np.uint16).reshape(rows // TILE, columns // TILE, 2, 2, 16, 16)

    def __call__(self, fill, take, spin=SPIN_S):
        """One call: fill() writes the inputs into the region (self.tiles), the nonce,
        one RUN, the spin (`spin` s without a pause), take() reads the outputs from
        it. KomorebiError with RUN's answer if y's words do not arrive."""
        clock = self.device.clock
        started = time.perf_counter()
        fill()
        for move in self.pulls:
            self.rows(move)[:] = self.local[id(move)].reshape(move.tensor.pages, -1)
        key = self.last.arm()  # RUN's a3: the executor XORs it into the nonce words
        copied = time.perf_counter()
        executor, run = self.device.executor, self.device.decode.run
        if executor.slot == nn.NN["NN_SLOTS"]:
            executor.slot = 0
        slot = executor.next_slot()
        self.args[2], self.args[3] = slot, key
        handle = run.submit(
            self.handler, cluster=nn.CLUSTER, args=self.args, result=True
        )
        queued = time.perf_counter()
        run.store()
        submitted = time.perf_counter()
        arrived = self.last.wait(WORDS_TIMEOUT_S, spin)
        seen = time.perf_counter()
        self.pending.append((handle, slot))
        if not arrived:
            event = run.wait(handle, timeout=WORDS_TIMEOUT_S)
            raise KomorebiError(f"y's sequence words did not arrive; RUN: {event}")
        for move in self.pushes:
            self.local[id(move)].reshape(move.tensor.pages, -1)[:] = self.rows(move)
        out = take()
        done = time.perf_counter()
        for name, span in (
            ("copy_in", copied - started),
            ("submit_queue", queued - copied),
            ("submit_store", submitted - queued),
            ("spin", seen - submitted),
            ("copy_out", done - seen),
        ):
            clock[name] += span
        return out

    def collect(self, timeout=10.0):
        """RUN's events of the calls so far (off a call's path): each run noted."""
        run = self.device.decode.run
        if len(self.pending) == 1:  # the last call's span on the cores, from its words
            first = self.pulls[0].transfer.words
            last = self.last.words.astype(np.int64)
            starts = first[:, 4].astype(np.int64) | first[:, 5].astype(np.int64) << 32
            self.spans.append(int((last[:, 6] | last[:, 7] << 32).max() - starts.min()))
        for handle, slot in self.pending:
            ticks = nn.ended(run.wait(handle, timeout=timeout))
            self.device.runs.append(("one", slot, ticks, self.entries))
        self.pending = []
