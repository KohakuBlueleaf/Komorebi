"""The decode step on a card: the K-split program with the operations as overlays and
the X280 thread that walks the layers, its tensors, the weights written through the
host channel, and the thread's job. Needs tt-metal's Python environment."""

import math
import time

import numpy as np

import komorebi as km
from komorebi import formats
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
# attend's fp32 buffers that compute copies to DEST (with the K split's arrival)
TO_DEST = ("DC_CB_REFERENCE", "DC_CB_STATE", "DC_CB_RESCALE")
ATTEND_MODES = {None: 0, "read": 16, "local": 32, "exp_twice": 64}  # LAYERS' a4


def operation_word(number, parts):
    """A SPLIT command's a6: the overlay's number and its parts."""
    return number | sum(parts.values())


def keys_most(context, group):
    """Key tiles a core of a KV head's group holds at most: its share of the full
    tiles of a cache of `context` rows, and the token's tile."""
    return math.ceil((context // md.TILE - 1) / group) + 1


def buffers(plan, most):
    """The K split's buffers (every plan shares them) and the operations'; attend's
    for `most` key tiles a core."""
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
        (D["DC_CB_SCALE"], 1),
        (D["DC_CB_UP"], block),
        (D["DC_CB_PART"], 1),
    ]
    return (
        [km.Buffer(*buffer) for buffer in plan.buffers()]
        + [km.Buffer(index, "bfloat16", tile, pages) for index, pages in own + attend]
        + [km.Buffer(index, "float32", S["kSumBytes"], pages) for index, pages in state]
    )


class Decode:
    """The model's layers loaded on a session: tensors, the program and its thread.
    `layers` distinct layers of weights in DRAM, written by write_layer(); `trace`:
    the commands each core stamps (the runtime's trace, 0: none); `chunk`, `patterns`:
    md.plans()'s; `context`: the cache's rows a KV head and layer."""

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
    ):
        self.session, self.model, self.layers, self.log = session, model, layers, log
        self.plans = md.plans(model, chunk=chunk, patterns=patterns)
        self.plan_index = {shape: i for i, shape in enumerate(self.plans)}
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
        shapes = products | {name: model.shapes()[name] for name in md.APART}
        cache = (model.kv_heads * context, model.head_dim)
        shapes |= dict.fromkeys(("keys", "values"), cache)
        self.weights = [
            {
                name: session.empty(shape, "bfloat16", "tile")
                for name, shape in shapes.items()
            }
            for _ in range(layers)
        ]
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

    def program(self):
        """The K split's kernels with a table of the layer's plans, four overlays and
        the thread."""
        plans = list(self.plans.values())
        table = pl.split_table(plans)
        tile_args = self.tensors["x"].tensor_args()  # every operand: bf16 tiles
        program = km.Program(
            "decode",
            kernels=dense_card.SPLIT_KERNELS,
            buffers=buffers(plans[0], self.keys_most),
            args=lambda core: (table, table, table),
            compile_args={"reader": tile_args * 2, "writer": tile_args * 3},
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    S["kCbArrive"]: "UnpackToDestFp32",
                    S["kCbSum"]: "UnpackToDestFp32",
                }
                | {D[name]: "UnpackToDestFp32" for name in TO_DEST},
            },
            handlers=pl.SPLIT_H,
        )
        squares = KERNELS / "scale" / "reader.cpp"
        program.overlay("add", KERNELS / "add", number=D["DC_OVERLAY_ADD"])
        program.overlay(
            "gate",
            {
                "reader": squares,
                "compute": KERNELS / "gate" / "compute.cpp",
                "writer": KERNELS / "gate" / "writer.cpp",
            },
            number=D["DC_OVERLAY_GATE"],
        )
        program.overlay("scale", KERNELS / "scale", number=D["DC_OVERLAY_SCALE"])
        program.overlay(
            "attend",
            KERNELS / "attend",
            number=D["DC_OVERLAY_ATTEND"],
            defines={"DC_KEYS_MOST": self.keys_most},
        )
        weights = np.zeros((self.layers, D["DC_LAYER_WORDS"]), np.uint32)
        for layer, tensors in enumerate(self.weights):
            for name, tensor in tensors.items():
                weights[layer, D[f"DC_W_{name.upper()}"]] = tensor.address
        addresses = np.zeros(D["DC_TENSORS"], np.uint32)
        for name, tensor in self.tensors.items():
            addresses[D[f"DC_TENSOR_{name.upper()}"]] = tensor.address
        program.thread(
            cluster=CLUSTER,
            sources=[md.HERE / "thread" / "layers.c"],
            includes=[md.HERE],
            regions={
                "weights": weights.reshape(-1),
                "tensors": addresses,
                "stats": 4 * D["DC_STATS_WORDS"] * D["DC_STATS_SLOTS"],
            },
            words=self.words(),
            handlers={"layers": D["DC_HANDLER_LAYERS"]},
        )
        return program

    def plan_word(self, weight):
        """A product's plan word: the plan's index | the cores it takes << 8."""
        shape = md.plan_of(self.model, weight)
        return self.plan_index[shape] | self.plans[shape].cores() << 8

    def words(self):
        """The thread's words (decode.h, DC_WORD_*)."""
        model = self.model
        epsilon = np.array([model.eps], np.float32).view(np.uint32)[0]
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
        ]

    def attend(self, groups, position, rows=1, layer=0, timeout=5.0):
        """One ATTEND from the host on the first `groups` KV heads' groups, over layer
        `layer`'s cache, `rows` tokens from `position` (q, k, v from qkv): its event."""
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

    def write_layer(self, layer, weights):
        """Layer `layer`'s weights (layer_weights(): {name: (pages, values)}) into DRAM
        through the host channel, as the card's products take them: seconds taken."""
        seconds = 0.0
        for name, pages in md.fused(weights).items():
            seconds += self.write(self.weights[layer][name], pages)
        return seconds

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
    ):
        """One LAYERS command: its result word and stats slot. `bare` products only,
        `parts` their parts only, `apart` q, k, v as three (no attention), `overlap` two
        in flight; `position` the first token's (default: the last ends the cache);
        `attend` a cut-down attention of ATTEND_MODES; `rows` tokens a step."""
        slot = self.slot
        self.slot += 1
        mode = 3 if parts else int(bare)
        mode |= (4 if apart or overlap else 0) | (8 if overlap else 0)
        mode |= ATTEND_MODES[attend]
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


STEPS = ("qkv", "q", "k", "v", "attend", "o", "gate_up", "down")  # DC_STEP_*


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
    for step, name in enumerate(STEPS):
        at = D["DC_ST_STEPS"] + D["DC_ST_STEP_WORDS"] * step
        ticks = int(words[at]) | int(words[at + 1]) << 32
        out["steps"][name] = {
            "sum_us": ticks / TICKS_PER_MICROSECOND,
            "most_us": int(words[at + 2]) / TICKS_PER_MICROSECOND,
            "least_us": int(words[at + 3]) / TICKS_PER_MICROSECOND if ticks else 0.0,
        }
    return out
