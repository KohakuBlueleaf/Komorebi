"""The decode step on a card: the K-split program with the operations as overlays and
the X280 thread that walks the layers, its tensors, the weights written through the
host channel, and the thread's job. Needs tt-metal's Python environment."""

import math
import time

import numpy as np

import komorebi as km
from komorebi import abi, formats
from projects.decode import bodies as emitted_bodies
from projects.decode import model as md
from projects.dense import card as dense_card
from projects.dense import plan as pl

D, OP, S = md.D, md.OP, pl.S
KERNELS = md.HERE / "kernels"
CLUSTER = 1
TICKS_PER_MICROSECOND = 50  # kmx_time() and the dispatcher count the 50 MHz timer
EXCLUSIVE = km.abi.headers().constants["KM_FLAG_EXCLUSIVE"]
ACTIVATIONS = ("x", "x1", "qkv", "k", "v", "gated", "sums", "squares1", "squares2", "a")
PARTS = {"reader": OP["kOpReader"], "compute": OP["kOpCompute"]}
ADD_PARTS = PARTS | {"writer": OP["kOpWriter"]}
GATE_PARTS = ADD_PARTS
ATTEND_PARTS = ADD_PARTS
LARGEST_PARTS = {"writer": OP["kOpWriter"]}
TEXT_START, TEXT_BYTES = 64, 1 << 16  # the thread's text region: count, then bytes
# attend's fp32 buffers that compute copies to DEST (with the K split's arrival)
TO_DEST = ("DC_CB_REFERENCE", "DC_CB_STATE", "DC_CB_RESCALE")
ATTEND_MODES = {None: 0, "read": 16, "local": 32, "exp_twice": 64}  # LAYERS' a4
# Operations whose vector bodies can be emitted (bodies.py), a switch each; none by
# default: each is right on the card and slower than the hand body (README, stage 5).
BODIES = ("add", "norm", "gate", "qk")
DEFAULT_BODIES = ()
# The layer as one command: each core goes on to the next product by itself (stage 6;
# the split's SPLIT_CHAIN), its records written by the thread; off: a command a step.
CHAIN = abi.constants(abi.headers().chain, dense_card.SPLIT_KERNELS / "chain.h")
CHAIN_PAGE = 2816  # the chain's words on each core (km_chain.hpp's kChainAreaBytes)
# The chain's records (LAYERS' a4 bits 12 on, in this order) whose input the roots
# before send into their cores' L1 (stage 8): one, or a pair whose areas never overlap.
DATA_STEPS = ("qkv", "attend", "o", "gate_up", "down")
DATA_PAIRS = ({"o", "down"}, {"qkv", "gate_up"})


def data_bits(plans, model, data):
    """LAYERS' a4 bits of `data` (DATA_STEPS' names), checked: attend's input is never
    sent; a core's K range fits its input area."""
    data = set(data)
    if not (data <= set(DATA_STEPS) - {"attend"}) or (
        len(data) > 1 and data not in DATA_PAIRS
    ):
        raise ValueError(
            f"data {sorted(data)}: one of {DATA_STEPS} but attend, or o "
            "and down, or qkv and gate_up"
        )
    bits = 0
    for name in data:
        plan = plans[md.plan_of(model, name)]
        room = (plan.arrive - CHAIN["kChainInputPage"]) * pl.S["kSumBytes"]
        if math.ceil(plan.kt / plan.s) * pl.TILE_BYTES > room:
            raise ValueError(
                f"data {name}: a K range of {plan.kt} / {plan.s} tiles "
                f"past the {room} bytes from arrival page "
                f"{CHAIN['kChainInputPage']}"
            )
        bits |= 1 << DATA_STEPS.index(name)
    return bits << D["DC_LAYERS_DATA_SHIFT"]


def scale_fp32(bodies):
    """Whether DC_CB_SCALE (1 / rms) is fp32: only when nothing hand-written reads it
    (those take it on the matrix unit's source B, in bf16 there)."""
    return {"norm", "gate", "qk"} <= set(bodies)


def operation_word(number, parts):
    """A SPLIT command's a6: the overlay's number and its parts."""
    return number | sum(parts.values())


def keys_most(context, group):
    """Key tiles a core of a KV head's group holds at most: its share of the full
    tiles of a cache of `context` rows, and the token's tile."""
    return math.ceil((context // md.TILE - 1) / group) + 1


def w_formats(weights):
    """(first, second, product) of md.WEIGHTS[weights]: every product's W in `first`
    but `product`'s, in `second` (None: one format), whose tiles fit `first`'s slots."""
    named = md.WEIGHTS[weights]
    first = named["qkv"]
    others = [name for name in md.PRODUCTS if named[name] != first]
    if len(others) > 1:
        raise ValueError(f"weights {weights}: W of {others} in a third format")
    if not others:
        return first, None, None
    second = named[others[0]]
    if formats.tile_bytes(second) > formats.tile_bytes(first):
        raise ValueError(f"weights {weights}: {second} tiles past {first}'s slots")
    return first, second, others[0]


def buffers(plan, most, bodies=DEFAULT_BODIES, chain=False, weights="bf16"):
    """The K split's buffers (every plan shares them; W's in w_formats(weights)) and the
    operations'; attend's for `most` key tiles a core; DC_CB_SCALE fp32 by
    scale_fp32(bodies); with `chain` the chain's words and record slots."""
    tile = pl.TILE_BYTES
    block = plan.oh * plan.ow
    head = D["DC_HEAD_TILES"]
    attend = [
        (D["DC_CB_QUERY"], D["DC_HEAD_GROUP"] * head),
        (D["DC_CB_ATTEND_CONSTANT"], 3),
        (D["DC_CB_MASK"], 2),
        (D["DC_CB_KEYS"], most * head),
        (D["DC_CB_VALUES"], most * head),
        (D["DC_CB_SCORES"], D["DC_HEAD_GROUP"] * most),
        (D["DC_CB_PROBABILITIES"], most),
        (D["DC_CB_ROW_MAX"], 1),
        (D["DC_CB_TOKEN"], 2 * head),
    ]
    state = [
        (D["DC_CB_REFERENCE"], 1),
        (D["DC_CB_STATE"], 2 * D["DC_STATE_ITEM"]),
        (D["DC_CB_RESCALE"], 2),
        (D["DC_CB_OUTBOX"], D["DC_STATE_ITEM"]),
    ]
    own = [
        (D["DC_CB_BLOCK"], block),
        (D["DC_CB_Y"], block),
        (D["DC_CB_PARTS"], D["DC_PARTS"]),
        (D["DC_CB_CONSTANT"], D["DC_CONSTANTS"]),
        (D["DC_CB_SQUARES"], 1),
        (D["DC_CB_UP"], block),
        (D["DC_CB_PART"], 1),
    ]
    (state if scale_fp32(bodies) else own).append((D["DC_CB_SCALE"], 1))
    words = [(CHAIN["kChainBuffer"], "uint32", CHAIN_PAGE, 1)] if chain else []
    first, second, _ = w_formats(weights)
    slot = formats.tile_bytes(first)
    split = [
        (
            (index, first, slot, pages)
            if index in (S["kCbSw"], S["kCbSwMoverB"])
            else (index, dtype, page, pages)
        )
        for index, dtype, page, pages in plan.buffers()
    ]
    if second is not None:  # srcA's format for the second W, a page never filled
        split.append((S["kCbSwSecondFormat"], second, slot, 1))
    return (
        [km.Buffer(*buffer) for buffer in split + words]
        + [km.Buffer(index, "bfloat16", tile, pages) for index, pages in own + attend]
        + [km.Buffer(index, "float32", S["kSumBytes"], pages) for index, pages in state]
    )


class Decode:
    """The model's layers on a session: `layers` layers of weights (`weights`), `vocab`
    the head's columns (0: none), `embedding` its own bf16 copy (else the head's table);
    `text` GENERATE's detokenizing tables; `trace` stamps; `chunk`, `patterns` plans';
    `context` the cache's rows; `bodies`; `chain` a layer as one; `data` sent inputs."""

    def __init__(
        self,
        session,
        model,
        layers,
        deadline,
        log,
        trace=0,
        chunk=md.CHUNK,
        patterns=None,
        context=4096,
        bodies=DEFAULT_BODIES,
        chain=True,
        stamps=False,
        data=(),
        weights="bf16",
        vocab=0,
        embedding=False,
        text=None,
    ):
        self.session, self.model, self.layers, self.log = session, model, layers, log
        self.weight_formats = weights
        _, _, self.second_w = w_formats(weights)
        self.chain = chain
        self.stamps = stamps and chain  # SPLIT_CHAIN_STAMPS: chain_stamps() reads them
        self.bodies = tuple(bodies)
        if not set(self.bodies) <= set(BODIES):
            raise ValueError(f"bodies {self.bodies}: of {BODIES}")
        if self.bodies:
            emitted_bodies.require()
        halves = md.CHAIN_HALVES if chain else (md.DEPTH, 1)
        self.plans = md.plans(
            model, chunk=chunk, patterns=patterns, halves=halves, head=vocab > 0
        )
        self.plan_index = {shape: i for i, shape in enumerate(self.plans)}
        self.data = tuple(data)
        data_bits(self.plans, model, self.data)
        for name in ("o", "down"):  # their roots write a row's squares, a tile a group
            if self.plans[md.plan_of(model, name)].groups() != D["DC_PARTS"]:
                raise ValueError(f"{name}: not {D['DC_PARTS']} groups")
        self.context = context
        self.attend_plan = self.plans[md.plan_of(model, "o")]
        self.check_attend(model, self.attend_plan, context)
        self.keys_most = keys_most(context, self.attend_plan.s)
        hidden = model.hidden
        products = model.products()
        widths = {"x": hidden, "x1": hidden, "gated": model.mlp}
        widths |= dict.fromkeys(("squares1", "squares2"), D["DC_PARTS"] * md.TILE)
        widths |= {"qkv": products["qkv"][1], "sums": products["gate_up"][1]}
        widths |= {name: model.shapes()[name][1] for name in ("k", "v")}
        widths |= {"a": model.heads * model.head_dim}
        self.tensors = {
            name: session.empty((md.TILE, widths[name]), "bfloat16", "tile")
            for name in ACTIVATIONS
        }
        self.tensors["constants"] = dense_card.bf16_tensor(session, md.constants())
        self.tensors["attend"] = dense_card.bf16_tensor(session, md.attend_constants())
        if self.stamps:  # a page of kChainStampPage bytes a core
            pages = np.zeros((self.cores(), CHAIN["kChainStampPage"] // 4), np.int32)
            self.stamp_tensor = session.tensor(pages, "int32")
        if layers > D["DC_QK_LAYERS"]:
            raise ValueError(
                f"{layers} layers: the qk tensor holds {D['DC_QK_LAYERS']}"
            )
        qk = md.qk_pages(model, context + md.TILE)  # a band past the cache's last row
        self.tensors["qk"] = session.empty(
            (md.TILE, md.TILE * len(qk)), "bfloat16", "tile"
        )
        session.write(self.tensors["qk"], qk)
        shapes = products | {name: model.shapes()[name] for name in md.APART}
        cache_shape = (model.kv_heads * context, model.head_dim)
        cache = dict.fromkeys(("keys", "values"), cache_shape)
        shapes |= cache
        self.weights = [
            {
                name: session.empty(
                    shape,
                    "bfloat16" if name in cache else md.format_of(weights, name),
                    "tile",
                )
                for name, shape in shapes.items()
            }
            for _ in range(layers)
        ]
        self.vocab = vocab
        self.text = text
        self.head_seconds = []  # head(): (its commands, its logits' read) a call
        head_width = md.TILE * md.head_plan(model)[1]
        self.head_format = md.format_of(weights, "qkv")  # the W pair's first format
        self.head_w = [
            session.empty((hidden, head_width), self.head_format, "tile")
            for _ in range(md.head_parts(vocab) if vocab else 0)
        ]
        self.embedding = (
            session.empty((vocab, hidden), "bfloat16", "row") if embedding else None
        )
        if vocab and not embedding and self.head_format != "bfloat16":
            raise ValueError("the embedding from a head not in bf16: its own copy")
        self.logits = [
            session.empty((md.TILE, head_width), "bfloat16", "tile")
            for _ in self.head_w
        ]
        # the final norm: its weight on a diagonal W, and x normed (the head's input)
        self.norm_w = session.empty((hidden, hidden), self.head_format, "tile")
        self.xn = session.empty((md.TILE, hidden), "bfloat16", "tile")
        # largest(): a page a block of each part, its largest value and index
        blocks = md.head_plan(model)[1] // md.BLOCK_WIDTH * max(len(self.head_w), 1)
        self.largest_pages = session.empty(
            (md.TILE, md.TILE * blocks), "bfloat16", "tile"
        )
        self.run = session.load(
            self.program(), deadline=deadline, wait=1.0, trace=trace
        )
        self.slot = 0

    @staticmethod
    def check_attend(model, plan, context):
        """attend takes o's plan: a group a KV head, whose block of a (1 x 8 tiles) is
        its query heads'; the merge's states fit the arrival buffer's slots."""
        head = D["DC_HEAD_TILES"]
        if model.head_dim != head * md.TILE or model.heads != D["DC_HEAD_GROUP"] * (
            model.kv_heads
        ):
            raise ValueError("attend: heads of 128 and two query heads a KV head")
        if plan.groups() != model.kv_heads or plan.ow != D["DC_HEAD_GROUP"] * head:
            raise ValueError("attend: o's plan is not a group a KV head")
        if plan.arrive // D["DC_STATE_ITEM"] < math.ceil(math.log2(plan.s)):
            raise ValueError("attend: the merge's rounds do not fit the arrival slots")
        if context % md.TILE or not md.TILE <= context < 1 << 16:
            raise ValueError(f"context {context}: whole tiles, below 65536 rows")
        if context // md.TILE > 0xFF:  # attend.hpp: a9's cache tiles, bits 16 to 23
            raise ValueError(
                f"context {context}: {context // md.TILE} cache tiles a head, past the "
                "8 bits of attend's a9 field (255 tiles, 8160 rows)"
            )

    def program(self):
        """The K split's kernels with a table of the layer's plans, four overlays and
        the thread."""
        plans = list(self.plans.values())
        table = pl.split_table(plans)
        tile_args = self.tensors["x"].tensor_args()  # A's and Y's: bf16 tiles
        w_args = self.weights[0]["qkv"].tensor_args()  # w_formats()' first W's
        if self.second_w is not None:  # its second W's, after the first's
            w_args += self.weights[0][self.second_w].tensor_args()
        program = km.Program(
            "decode",
            kernels=dense_card.SPLIT_KERNELS,
            buffers=buffers(
                plans[0], self.keys_most, self.bodies, self.chain, self.weight_formats
            ),
            args=lambda core: (table, table, table),
            defines=(
                ({"SPLIT_CHAIN": 1} if self.chain else {})
                | self.second_define()
                | ({"SPLIT_CHAIN_STAMPS": 1} if self.stamps else {})
            ),
            compile_args={
                "reader": tile_args + w_args,
                "writer": tile_args * 2 + w_args,
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    S["kCbArrive"]: "UnpackToDestFp32",
                    S["kCbSum"]: "UnpackToDestFp32",
                }
                | {D[name]: "UnpackToDestFp32" for name in self.to_dest()},
            },
            handlers=pl.SPLIT_H,
        )
        squares = KERNELS / "scale" / "reader.cpp"
        emit = {f"DC_EMIT_{name.upper()}": int(name in self.bodies) for name in BODIES}
        program.overlay(
            "add", KERNELS / "add", number=D["DC_OVERLAY_ADD"], defines=emit
        )
        program.overlay(
            "gate",
            {
                "reader": squares,
                "compute": KERNELS / "gate" / "compute.cpp",
                "writer": KERNELS / "gate" / "writer.cpp",
            },
            number=D["DC_OVERLAY_GATE"],
            defines=emit,
        )
        epsilon = np.array([self.model.eps], np.float32).view(np.uint32)[0]
        program.overlay(
            "scale",
            KERNELS / "scale",
            number=D["DC_OVERLAY_SCALE"],
            defines={"DC_EPSILON": int(epsilon)} | emit,
        )
        program.overlay(
            "largest",
            {"writer": KERNELS / "largest" / "writer.cpp"},
            number=D["DC_OVERLAY_LARGEST"],
        )
        program.overlay(
            "attend",
            KERNELS / "attend",
            number=D["DC_OVERLAY_ATTEND"],
            defines={"DC_KEYS_MOST": self.keys_most}
            | ({"SPLIT_CHAIN": 1} if self.chain else {}),
        )
        weights = np.zeros((self.layers, D["DC_LAYER_WORDS"]), np.uint32)
        for layer, tensors in enumerate(self.weights):
            for name, tensor in tensors.items():
                weights[layer, D[f"DC_W_{name.upper()}"]] = tensor.address
        addresses = np.zeros(D["DC_TENSORS"], np.uint32)
        for name, tensor in self.tensors.items():
            addresses[D[f"DC_TENSOR_{name.upper()}"]] = tensor.address
        if len(self.head_w) > D["DC_HEAD_PARTS_MOST"]:
            raise ValueError(f"the head in {len(self.head_w)} parts: too many")
        for part, w in enumerate(self.head_w):
            addresses[D["DC_TENSOR_HEAD"] + part] = w.address
        addresses[D["DC_TENSOR_LARGEST"]] = self.largest_pages.address
        addresses[D["DC_TENSOR_XN"]] = self.xn.address
        addresses[D["DC_TENSOR_NORM"]] = self.norm_w.address
        if self.embedding is not None:
            addresses[D["DC_TENSOR_EMBED"]] = self.embedding.address
        tile = md.TILE * md.TILE * 2
        largest = self.largest_pages.shape[1] // md.TILE * tile
        program.thread(
            cluster=CLUSTER,
            sources=[md.HERE / "thread" / "layers.c"],
            includes=[md.HERE, dense_card.SPLIT_KERNELS],
            regions={
                "weights": weights.reshape(-1),
                "tensors": addresses,
                "stats": 4 * D["DC_STATS_WORDS"] * D["DC_STATS_SLOTS"],
                "records": CHAIN["kChainRecordBytes"] * CHAIN["kChainRecordsMax"],
                # GENERATE's: the gathered column's tiles, x's and its squares' tiles,
                # the largest pages, the tokens made
                "gather": D["DC_ROW_TILES"] * tile,
                "row": (D["DC_ROW_TILES"] + D["DC_PARTS"]) * tile,
                "largest": largest,
                "made": 4 * D["DC_MADE_WORDS"] * D["DC_MADE_MOST"],
                "prompt": 4 * D["DC_PROMPT_MOST"],
                # detokenizing: the tables, the bytes out (a count, from byte 64 on)
                "offsets": np.zeros(1, np.uint32) if not self.text else self.text[0],
                "pieces": np.zeros(4, np.uint8) if not self.text else self.text[1],
                "text": TEXT_BYTES if self.text else 64,
            },
            words=self.words(),
            handlers={
                "layers": D["DC_HANDLER_LAYERS"],
                "generate": D["DC_HANDLER_GENERATE"],
            },
        )
        return program

    def second_define(self):
        """SPLIT_SECOND_W: the plan of the W in the second pair (one product's)."""
        if self.second_w is None:
            return {}
        shape = md.plan_of(self.model, self.second_w)
        sharing = [
            name
            for name in md.PRODUCTS + md.APART
            if name != self.second_w and md.plan_of(self.model, name) == shape
        ]
        if sharing:
            raise ValueError(f"{self.second_w}'s plan is {sharing}'s too: one W format")
        return {"SPLIT_SECOND_W": self.plan_index[shape]}

    def to_dest(self):
        """The operations' fp32 buffers that compute copies to DEST."""
        return TO_DEST + (("DC_CB_SCALE",) if scale_fp32(self.bodies) else ())

    def plan_word(self, weight):
        """A product's plan word: the plan's index | the cores it takes << 8."""
        shape = md.plan_of(self.model, weight)
        return self.plan_index[shape] | self.plans[shape].cores() << 8

    def words(self):
        """The thread's words (decode.h, DC_WORD_*)."""
        model = self.model
        epsilon = np.array([model.eps], np.float32).view(np.uint32)[0]
        head, parts = md.head_plan(model), len(self.head_w)
        return [
            S["kHandlerSplit"],
            pl.REDUCE_FORMS["tree"],
            EXCLUSIVE,
            *(self.plan_word(name) for name in ("qkv", "q", "k", "gate_up", "down")),
            operation_word(D["DC_OVERLAY_ADD"], ADD_PARTS),
            operation_word(D["DC_OVERLAY_GATE"], GATE_PARTS),
            operation_word(D["DC_OVERLAY_SCALE"], PARTS),
            int(epsilon),
            self.plan_word("o"),
            operation_word(D["DC_OVERLAY_ATTEND"], ATTEND_PARTS),
            self.context // md.TILE,
            md.GATE_TILES,
            *self.attend_needs(),
            (
                self.plan_index.get(head, 0) | self.plans[head].cores() << 8
                if parts
                else 0
            ),
            operation_word(D["DC_OVERLAY_LARGEST"], LARGEST_PARTS),
            parts,
            md.TILE * head[1],
            self.vocab,
        ]

    def attend_needs(self):
        """attend's three ranges of qkv for the chain (chain.h): a KV head's group reads
        its query heads' tiles, its root also the head's key and value tiles."""
        model, head = self.model, D["DC_HEAD_TILES"]
        query = D["DC_HEAD_GROUP"] * head
        keys = model.heads * model.head_dim // md.TILE
        values = keys + model.kv_heads * head
        root = 1 << 31
        return [
            0 | query << 12 | query << 20,
            keys | head << 12 | head << 20 | root,
            values | head << 12 | head << 20 | root,
        ]

    def attend(self, groups, position, rows=1, layer=0, timeout=5.0):
        """One ATTEND from the host on the first `groups` KV heads' groups, over layer
        `layer`'s cache, `rows` tokens from `position` (q, k, v from qkv): its event."""
        if not 0 <= position <= self.context - rows or not 1 <= rows <= md.TILE:
            raise ValueError(f"position {position}: {rows} rows past the cache")
        words = self.words()
        tensors, weights = self.tensors, self.weights[layer]
        args = [
            tensors["qkv"].address,
            weights["keys"].address,
            tensors["a"].address,
            words[D["DC_WORD_REDUCE"]] | rows << D["DC_ROWS_SHIFT"],
            S["kSplitOp"],
            words[D["DC_WORD_PLAN_ATTEND"]] & 0xFF,
            words[D["DC_WORD_ATTEND"]],
            weights["values"].address,
            tensors["attend"].address,
            position | words[D["DC_WORD_CACHE_TILES"]] << 16,
        ]
        group = (0, self.attend_plan.s * groups)
        handle = self.run.submit("split", group, args, exclusive=True)
        event = self.run.wait(handle, timeout=timeout)
        if event.kind != "done":
            raise km.KomorebiError(f"attend: {event}")
        return event

    def write_embedding(self, bits):
        """The embedding's own copy: bf16 bits [vocab, hidden], a row a page."""
        return self.write(self.embedding, np.asarray(bits, np.uint16).view(np.uint8))

    def write_head(self, parts, norm):
        """The head's table in parts (md.head_pages()) and the final norm's W
        (md.norm_pages()) into DRAM: seconds taken."""
        seconds = self.write(self.norm_w, norm)
        return seconds + sum(
            self.write(w, pages) for w, pages in zip(self.head_w, parts)
        )

    def head_commands(self, operation, outputs, timeout):
        """The final norm's SPLIT (x into xn, the scale) waited, then the head's, a part
        each (xn, its table, outputs[part]; a9 the part | its token columns << 16)."""
        words, tensors = self.words(), self.tensors
        reduce = words[D["DC_WORD_REDUCE"]] | 1 << D["DC_ROWS_SHIFT"]
        norm = md.plan_of(self.model, "q")
        args = [tensors["x"].address, self.norm_w.address, self.xn.address, reduce]
        args += [S["kSplitFull"], self.plan_index[norm], words[D["DC_WORD_SCALE"]]]
        args += [tensors["squares1"].address, tensors["constants"].address]
        args += [words[D["DC_WORD_EPSILON"]]]
        handles = [
            self.run.submit(
                "split", (0, self.plans[norm].cores()), args, exclusive=True
            )
        ]
        shape = md.head_plan(self.model)
        width = md.TILE * shape[1]
        for part, (w, output) in enumerate(zip(self.head_w, outputs)):
            if part == 0:  # xn written
                event = self.run.wait(handles.pop(), timeout=timeout)
                if event.kind != "done":
                    raise km.KomorebiError(f"the final norm: {event}")
            limit = min(width, self.vocab - part * width)
            args = [self.xn.address, w.address, output.address, reduce]
            args += [S["kSplitFull"], self.plan_index[shape], operation, 0, 0]
            args += [part | limit << 16]
            handles.append(
                self.run.submit(
                    "split", (0, self.plans[shape].cores()), args, exclusive=True
                )
            )
        for handle in handles:
            event = self.run.wait(handle, timeout=timeout)
            if event.kind != "done":
                raise km.KomorebiError(f"head: {event}")

    def largest(self, timeout=5.0):
        """The head on the card with the largest logit taken there: each block's
        largest and its index read back (a page a block), the token the largest of
        those, the first index on a tie (head_seconds: the two phases')."""
        started = time.perf_counter()
        operation = operation_word(D["DC_OVERLAY_LARGEST"], LARGEST_PARTS)
        self.head_commands(operation, [self.largest_pages] * len(self.head_w), timeout)
        done = time.perf_counter()
        best = self.largest_read()
        self.head_seconds.append((done - started, time.perf_counter() - done))
        return best

    def generate(
        self, tokens, layers, position, prompt, end, timeout=60.0, chain=True, rows=1
    ):
        """GENERATE: the `prompt` ids from `position` on (`rows` a command), then
        `tokens` tokens at most, embedded and chosen by the thread, to `end`; without
        `chain` a command a kernel, each waited for. The count made (made() after
        run.stop())."""
        if not self.head_w or not 0 < len(prompt) <= D["DC_PROMPT_MOST"]:
            raise ValueError("generate: a program with the head, a prompt that fits")
        self.run.threads[CLUSTER].write("prompt", np.asarray(prompt, np.uint32))
        slot = self.slot
        self.slot += 1
        mode = D["DC_LAYERS_HEAD"]
        if chain:
            mode |= D["DC_LAYERS_CHAIN"] | D["DC_LAYERS_TOKEN"]
            mode |= data_bits(self.plans, self.model, self.data)
        args = [tokens, layers, slot, 0, mode, position, rows, len(prompt), end]
        self.generate_slot = slot
        handle = self.run.submit("generate", cluster=CLUSTER, args=args, result=True)
        event = self.run.wait(handle, timeout=timeout)
        if event.kind != "done" or event.result >= 0xFFFFFFF0:
            raise km.KomorebiError(f"generate: {event}")
        return int(event.result)

    def made(self, count):
        """GENERATE's start and tokens, after run.stop(): the timer's ticks at the
        start, [(id, the ticks at its choice)]."""
        made = self.run.threads[CLUSTER].read("made").view(np.uint32)
        made = made.reshape(-1, D["DC_MADE_WORDS"])[: count + 1]
        ticks = [int(m[1]) | int(m[2]) << 32 for m in made]
        return ticks[0], [(int(m[0]), t) for m, t in zip(made[1:], ticks[1:])]

    def text_out(self):
        """GENERATE's detokenized bytes, after run.stop()."""
        raw = self.run.threads[CLUSTER].read("text")
        count = int(raw[:4].view(np.uint32)[0])
        return raw[TEXT_START : TEXT_START + count].tobytes()

    def largest_read(self):
        """The token from the largest pages the head's products wrote: the largest of
        the blocks', the first index on a tie."""
        (raw,) = self.session.read(self.largest_pages)
        words = raw.reshape(-1).view(np.uint32).reshape(-1, md.TILE * md.TILE // 2)
        bits, index = words[:, 0], words[:, 1].astype(np.int64)
        live = index != 0xFFFFFFFF
        per_part = md.head_plan(self.model)[1] // md.BLOCK_WIDTH
        width = md.TILE * md.head_plan(self.model)[1]
        index = index + np.arange(len(index)) // per_part * width  # the vocab's
        values = formats.bf16_values(bits[live].astype(np.uint16))
        index = index[live]
        return int(index[values == values.max()].min())

    def head(self, timeout=5.0):
        """The head on the card after a token: the final norm, then each part a SPLIT
        of xn with the table into its logits; the logits' row 0 read back: [vocab]
        float32 (head_seconds: the two phases')."""
        started = time.perf_counter()
        self.head_commands(0, self.logits, timeout)
        done = time.perf_counter()
        rows = []
        for logits in self.logits:  # row 0 of a tile: face 0's and face 1's first rows
            (raw,) = self.session.read(logits)
            half = raw.reshape(-1).view(np.uint16).reshape(-1, md.TILE * md.TILE)
            rows.append(np.concatenate([half[:, :16], half[:, 256:272]], axis=1))
        self.head_seconds.append((done - started, time.perf_counter() - done))
        return formats.bf16_values(np.concatenate(rows).reshape(-1))[: self.vocab]

    def cores(self):
        """The cores the layer's commands take at most (attend's, o's)."""
        return self.attend_plan.s * self.attend_plan.groups()

    def chain_stamps(self, timeout=5.0):
        """Every core's chain stamps since the last read (chain.h), then zeroed:
        [cores, kernels, records, words] uint32 sums of AICLK cycles."""
        tensor, cores = self.stamp_tensor, self.cores()
        args = [tensor.address, 0, 0, 0, CHAIN["kChainStampsMode"], 0, 0, 0, 0, 0]
        handle = self.run.submit("split", (0, cores), args, exclusive=True)
        event = self.run.wait(handle, timeout=timeout)
        if event.kind != "done":
            raise km.KomorebiError(f"chain stamps: {event}")
        (pages,) = self.session.read(tensor)
        shape = (CHAIN["kChainStampKernels"], CHAIN["kChainStampRecords"])
        words = pages.reshape(-1).view(np.uint32).reshape(cores, -1)
        return words[:, : CHAIN["kChainStampBytes"] // 4].reshape(cores, *shape, -1)

    def write_layer(self, layer, weights, pages=None):
        """Layer `layer`'s weights (layer_weights(): {name: (pages, values)}) into DRAM
        through the host channel, as the card's products take them (`pages`: md.fused()
        of them, made before): seconds taken."""
        seconds = 0.0
        if pages is None:
            pages = md.fused(weights, self.weight_formats)
        for name, page_rows in pages.items():
            seconds += self.write(self.weights[layer][name], page_rows)
        pages = md.qk_weight_pages(weights["q_norm"][1], weights["k_norm"][1])
        started = time.perf_counter()
        first = D["DC_QK_WEIGHTS"] + 2 * D["DC_HEAD_TILES"] * layer
        self.session.write(self.tensors["qk"], pages, first=first)
        return seconds + time.perf_counter() - started

    def write_cache(self, layer, keys, values):
        """A layer's cache ([KV heads, n, width] values each, rows past n zero) into its
        tensors through the host channel: seconds taken."""
        seconds = self.write(
            self.weights[layer]["keys"], md.cache_pages(keys, self.context)
        )
        return seconds + self.write(
            self.weights[layer]["values"], md.cache_pages(values, self.context)
        )

    def cache_rows(self, layer, name, position):
        """Row `position` of every KV head of a layer's cache ("keys" or "values"):
        [KV heads, width] float32, read back from the tiles holding it."""
        head_tiles, rows = D["DC_HEAD_TILES"], self.context // md.TILE
        out = []
        for head in range(self.model.kv_heads):
            first = (head * rows + position // md.TILE) * head_tiles
            tiles = self.session.read_tiles_bf16(
                self.weights[layer][name], first, head_tiles
            )  # [head_tiles, 32, 32]
            out.append(np.concatenate(list(tiles[:, position % md.TILE])))
        return np.array(out, np.float32)

    def write(self, tensor, pages):
        """Pages into a tensor through the host channel: seconds taken."""
        started = time.perf_counter()
        self.session.write(tensor, pages)
        return time.perf_counter() - started

    def write_activation(self, name, values):
        """A [32, width] activation (float values, rounded to bf16) into its tensor; x
        with its rows' sums of squares, as down's add leaves them."""
        bits = formats.bf16_bits(np.asarray(values, np.float32))
        self.session.write(self.tensors[name], md.tile_pages(bits))
        if name == "x":
            self.write_activation("squares1", md.squares(values))

    def read_activation(self, name):
        """A [32, width] activation as float32, after its command's event."""
        return dense_card.read_rows(self.session, self.tensors[name], range(md.TILE))

    def job(
        self,
        tokens,
        layers,
        first=0,
        bare=False,
        parts=False,
        apart=False,
        overlap=False,
        timeout=10.0,
        position=None,
        attend=None,
        rows=1,
        chain=None,
        pair=False,
        token=False,
        ahead=True,
        data=None,
        data_dram=False,
        head=False,
    ):
        """One LAYERS command: result word, stats slot. `bare` products, `parts` parts
        only, `apart` q k v as three, `overlap` two in flight; `rows` tokens a step from
        `position`; `attend` modes; `chain` a layer, `pair` 2 records, `token` a token,
        `head` with the head's; `ahead` W early; `data` sent inputs, DRAM acked."""
        slot = self.slot
        if slot >= D["DC_STATS_SLOTS"]:
            raise ValueError(f"job {slot}: the thread has {D['DC_STATS_SLOTS']} slots")
        self.slot += 1
        mode = 3 if parts else int(bare)
        mode |= (4 if apart or overlap else 0) | (8 if overlap else 0)
        mode |= ATTEND_MODES[attend]
        chain = self.chain and not (apart or overlap) if chain is None else chain
        if chain and (apart or overlap or not self.chain):
            raise ValueError("the chain: the fused layer, on a program built for it")
        mode |= D["DC_LAYERS_CHAIN"] if chain else 0
        mode |= D["DC_LAYERS_PAIR"] if chain and pair else 0
        if token and (pair or not chain):
            raise ValueError("a token as one command: the chain's whole layers")
        mode |= D["DC_LAYERS_TOKEN"] if token else 0
        if head and not (token and self.head_w):
            raise ValueError(
                "the head's records: a token's, on a program with the head"
            )
        mode |= D["DC_LAYERS_HEAD"] if head else 0
        mode |= 0 if ahead or not chain else D["DC_LAYERS_AHEAD_OFF"]
        data = (self.data if chain else ()) if data is None else data
        if data and not chain:
            raise ValueError("a sent input: the chain's records only")
        mode |= data_bits(self.plans, self.model, data)
        mode |= D["DC_LAYERS_DATA_DRAM"] if data and data_dram else 0
        steps = tokens * rows
        position = self.context - steps if position is None else position
        if not 0 <= position <= self.context - steps or not 1 <= rows <= md.TILE:
            raise ValueError(f"position {position}: {steps} rows past the cache")
        args = [tokens, layers, slot, first, mode, position, rows]
        handle = self.run.submit("layers", cluster=CLUSTER, args=args, result=True)
        event = self.run.wait(handle, timeout=timeout)
        if event.kind != "done" or event.result >= 0xFFFFFFF0:
            raise km.KomorebiError(f"thread: {event}")
        return {"thread_ticks": int(event.result), "stats_slot": slot}

    def stats(self):
        """Every stats slot of the thread, after run.stop(): [slots, words] uint32."""
        words = self.run.threads[CLUSTER].read("stats").view(np.uint32)
        count = D["DC_STATS_WORDS"] * D["DC_STATS_SLOTS"]
        return words[:count].reshape(D["DC_STATS_SLOTS"], D["DC_STATS_WORDS"])


THREAD_PARTS = ("records", "submit", "wait", "embed", "choose", "gather", "fetch")
STEPS = ("qkv", "q", "k", "v", "attend", "o", "gate_up", "down", "chain")  # DC_STEP_*


def decode_slot(words):
    """A stats slot: status, commands done, the job's microseconds, and per step the
    event microseconds summed, the most and the least."""
    out = {
        "status": int(words[D["DC_ST_STATUS"]]),
        "done": int(words[D["DC_ST_DONE"]]),
        "microseconds": (
            int(words[D["DC_ST_TICKS_LOW"]]) | int(words[D["DC_ST_TICKS_HIGH"]]) << 32
        )
        / TICKS_PER_MICROSECOND,
        "steps": {},
        "span_us": (
            int(words[D["DC_ST_SPAN_LOW"]]) | int(words[D["DC_ST_SPAN_HIGH"]]) << 32
        )
        / TICKS_PER_MICROSECOND,
    }
    out["thread"] = {}
    for part, name in enumerate(THREAD_PARTS):
        at = D["DC_ST_THREAD"] + 2 * part
        ticks = int(words[at]) | int(words[at + 1]) << 32
        out["thread"][name] = ticks / TICKS_PER_MICROSECOND
    for step, name in enumerate(STEPS):
        at = D["DC_ST_STEPS"] + D["DC_ST_STEP_WORDS"] * step
        ticks = int(words[at]) | int(words[at + 1]) << 32
        out["steps"][name] = {
            "sum_us": ticks / TICKS_PER_MICROSECOND,
            "most_us": int(words[at + 2]) / TICKS_PER_MICROSECOND,
            "least_us": int(words[at + 3]) / TICKS_PER_MICROSECOND if ticks else 0.0,
        }
    return out
