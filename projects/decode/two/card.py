"""The decode over two cards, form A: each card holds half of every product's output
columns (its 8 query heads and 4 KV heads, half of the MLP) and walks its half of the
layer as the chain's records on its own island; after each product its leader sends the
card's half across and releases the card when the other half is in (two.h)."""

import math
import time

import numpy as np

import komorebi as km
from komorebi import abi, formats
from projects.decode import card as one
from projects.decode import model as md
from projects.dense import card as dense_card
from projects.dense import plan as pl

HERE = md.HERE / "two"
D, S = md.D, pl.S
T = abi.constants(HERE / "two.h", known=abi.headers().constants)
CLUSTER = one.CLUSTER
BANKS = 8  # interleaved page i lies in bank i % 8 (session.DRAM_BANKS)
COPIES = ("x", "x1", "a", "gated", "squares1", "squares2")  # the crossings' tensors
CROSS_SERIAL = 0x80000000  # two.h's DC2_CROSS_SERIAL
CROSS_PARTS = {"reader": one.OP["kOpReader"], "writer": one.OP["kOpWriter"]}
# A crossing's single core: its K range takes every block the product before writes
# here (gate_up's 96 tiles at most); its width (the thread's) the next one's input.
CROSS_SHAPE = (96, md.BLOCK_WIDTH)


def half_shapes(model):
    """{name: (K tiles, N tiles)} of a card's products and of attend's plan."""
    hidden, mlp = model.hidden // md.TILE, model.mlp // md.TILE
    heads = model.heads * model.head_dim // md.TILE
    kv = model.kv_heads * model.head_dim // md.TILE
    return {
        "qkv": (hidden, (heads + 2 * kv) // 2),
        "o": (heads, hidden // 2),
        "gate_up": (hidden, mlp),
        "down": (mlp, hidden // 2),
        "attend": (heads // 2, D["DC_HEAD_GROUP"] * D["DC_HEAD_TILES"] * 4),
    }


def plans(model, cores, chunk=md.CHUNK):
    """{name: Split} of a card's half on `cores` ranks,
    as md.plans() makes them; and the crossing's plan of one core."""
    shapes = half_shapes(model)
    sizes = {
        name: min(k, cores // pl.cdiv(n, md.BLOCK_WIDTH), md.GROUP_MOST)
        for name, (k, n) in shapes.items()
    }
    slots = max(math.ceil(math.log2(s)) for s in sizes.values())
    arrive = md.SETS * slots * md.BLOCK_WIDTH
    depth, depth_b = md.CHAIN_HALVES
    out = {}
    for name, (k, n) in [*shapes.items(), ("cross", CROSS_SHAPE)]:
        out[name] = pl.split_plan(
            1,
            k,
            n,
            sizes.get(name, 1),
            1,
            md.BLOCK_WIDTH,
            chunk,
            cores=cores,
            arrive=arrive,
            pattern=md.PATTERNS.get(name, "AB"),
            depth=depth,
            depth_b=depth_b,
        )
    return out


def half_weights(model, weights, card):
    """Card `card`'s weights of a layer from md.layer_weights() or real.layer_weights():
    {name: pages} of qkv (its query heads, then its KV heads' k and v), o, gate_up (the
    gate's and up's turns) and down, each its half of the output columns; the norms'
    weights in the rows of the W after them, as md.fused() does."""
    values = {
        name: np.asarray(value, np.float32) for name, (_, value) in weights.items()
    }
    for name in ("q", "k", "v", "gate", "up"):
        norm = values["norm1" if name in md.APART else "norm2"]
        values[name] = norm[:, None] * values[name]

    def half(name):
        width = values[name].shape[1] // 2
        return values[name][:, card * width : (card + 1) * width]

    qkv = np.concatenate([half(name) for name in md.APART], axis=1)
    gate, up = half("gate"), half("up")
    k, n = gate.shape
    turn = md.GATE_TILES * md.TILE
    pair = np.stack(
        [gate.reshape(k, n // turn, turn), up.reshape(k, n // turn, turn)], axis=2
    ).reshape(k, 2 * n)
    out = {"qkv": qkv, "o": half("o"), "gate_up": pair, "down": half("down")}
    return {
        name: md.tile_pages(formats.bf16_bits(value)) for name, value in out.items()
    }


class TwoDecode:
    """The model's layers on a two-card session (km.Session(cards=[0, 1])): tensors,
    a program and thread an island. `context` the cache's rows a KV head and layer;
    `stamps` the chain's stamps (SPLIT_CHAIN_STAMPS); `slots` Program.slots' count."""

    def __init__(
        self,
        session,
        model,
        layers,
        deadline,
        log,
        context=256,
        stamps=False,
        slots=None,
        wait=1.0,
        serial=False,
    ):
        self.session, self.model, self.layers, self.log = session, model, layers, log
        self.context, self.stamps, self.serial = context, stamps, serial
        graph = session.graph
        self.firsts = [graph.first(i) for i in range(2)]
        self.counts = [graph.count(i) for i in range(2)]
        if self.counts[0] != self.counts[1]:
            raise ValueError(f"islands of {self.counts} ranks: two of one size")
        self.leader = 0  # the crossing's core: the island's first (all_gather's leader)
        self.plans = plans(model, self.counts[0])
        self.names = list(self.plans)
        self.attend_plan = self.plans["attend"]
        half = md.Model(
            model.name,
            model.hidden,
            model.mlp // 2,
            model.heads // 2,
            model.kv_heads // 2,
            model.head_dim,
            model.layers,
            model.eps,
            model.rope_theta,
        )
        self.half = half
        one.Decode.check_attend(half, self.attend_plan, context)
        for name in ("o", "down"):  # a card's squares: half of DC_PARTS
            if self.plans[name].groups() != T["DC2_SQUARES_HALF"]:
                raise ValueError(f"{name}: not {T['DC2_SQUARES_HALF']} groups a card")
        self.keys_most = one.keys_most(context, self.attend_plan.s)
        shapes = half_shapes(model)
        widths = {"x": model.hidden, "x1": model.hidden, "gated": model.mlp}
        widths |= dict.fromkeys(("squares1", "squares2"), D["DC_PARTS"] * md.TILE)
        widths |= {"a": model.heads * model.head_dim}
        self.copies = {
            name: session.empty((md.TILE, widths[name]), "bfloat16", "tile", copy=True)
            for name in COPIES
        }
        self.halves = {"x": 32, "x1": 32, "a": 32, "gated": shapes["gate_up"][1] // 2}
        for name, pages in self.halves.items():
            if pages % BANKS:
                raise ValueError(f"{name}: a card's {pages} pages, not whole banks")
        self.local = [
            {
                "qkv": s.empty(
                    (md.TILE, shapes["qkv"][1] * md.TILE), "bfloat16", "tile"
                ),
                "constants": dense_card.bf16_tensor(s, md.constants()),
                "attend": dense_card.bf16_tensor(s, md.attend_constants()),
                "qk": s.empty(
                    (md.TILE, md.TILE * len(md.qk_pages(model, context + md.TILE))),
                    "bfloat16",
                    "tile",
                ),
            }
            for s in session.sessions
        ]
        qk = md.qk_pages(model, context + md.TILE)  # a band past the cache's last row
        for island, s in enumerate(session.sessions):
            s.write(self.local[island]["qk"], qk)
        w_shapes = {name: (k * md.TILE, n * md.TILE) for name, (k, n) in shapes.items()}
        w_shapes.pop("attend")
        cache = (half.kv_heads * context, model.head_dim)
        w_shapes |= dict.fromkeys(("keys", "values"), cache)
        self.weights = [
            [
                {
                    name: s.empty(shape, "bfloat16", "tile")
                    for name, shape in w_shapes.items()
                }
                for _ in range(layers)
            ]
            for s in session.sessions
        ]
        if stamps:  # a page of kChainStampPage bytes a flat rank (a kernel's me)
            pages = np.zeros(
                (sum(self.counts), one.CHAIN["kChainStampPage"] // 4), np.int32
            )
            self.stamp_tensors = [s.tensor(pages, "int32") for s in session.sessions]
        programs = [self.program(island, slots) for island in range(2)]
        self.run = session.load(programs, deadline=deadline, wait=wait)
        self.slot = 0

    def addresses(self, island):
        """Region `tensors` of island `island` (two.h DC2_TENSOR_*)."""
        out = np.zeros(T["DC2_TENSORS"], np.uint32)
        for name, tensor in self.copies.items():
            base = tensor.parts[island].address
            key = name.upper()
            out[T[f"DC2_WORD_{key}"]] = tensor.address
            out[T[f"DC2_TENSOR_{key}"]] = base
            if name in self.halves:  # this card's pages: whole rows of the banks
                offset = island * self.halves[name] // BANKS * pl.TILE_BYTES
                out[T[f"DC2_TENSOR_{key}_HALF"]] = base + offset
        for name, tensor in self.local[island].items():
            out[T[f"DC2_TENSOR_{name.upper()}"]] = tensor.address
        return out

    def plan_word(self, name):
        """A plan's word: its index | the cores it takes << 8."""
        return self.names.index(name) | self.plans[name].cores() << 8

    def words(self, island):
        """The thread's words: decode.h's DC_WORD_* as one card's, then two.h's."""
        model = self.model
        epsilon = np.array([model.eps], np.float32).view(np.uint32)[0]
        words = [
            S["kHandlerSplit"],
            pl.REDUCE_FORMS["tree"],
            one.EXCLUSIVE,
            self.plan_word("qkv"),
            self.plan_word("o"),
            self.plan_word("o"),
            self.plan_word("gate_up"),
            self.plan_word("down"),
            one.operation_word(D["DC_OVERLAY_ADD"], one.ADD_PARTS),
            one.operation_word(D["DC_OVERLAY_GATE"], one.GATE_PARTS),
            one.operation_word(D["DC_OVERLAY_SCALE"], one.PARTS),
            int(epsilon),
            self.plan_word("attend"),
            one.operation_word(D["DC_OVERLAY_ATTEND"], one.ATTEND_PARTS),
            self.context // md.TILE,
            md.GATE_TILES,
            *self.attend_needs(),
        ]
        words += [0] * (T["DC2_WORD_PLAN_CROSS"] - len(words))
        words += [
            self.plan_word("cross"),
            one.operation_word(T["DC2_OVERLAY_CROSS"], CROSS_PARTS),
            self.firsts[island],
            # a keeper is a rank of its own island (km::keeper); serial: island 1 second
            self.leader | (CROSS_SERIAL if self.serial else 0),
            self.leader,
            island,
            self.plan_word("o"),
            self.halves["gated"],
        ]
        return words

    def attend_needs(self):
        """attend's three ranges of this card's qkv (one.Decode.attend_needs')."""
        half, head = self.half, D["DC_HEAD_TILES"]
        query = D["DC_HEAD_GROUP"] * head
        keys = half.heads * half.head_dim // md.TILE
        values = keys + half.kv_heads * head
        root = 1 << 31
        return [
            0 | query << 12 | query << 20,
            keys | head << 12 | head << 20 | root,
            values | head << 12 | head << 20 | root,
        ]

    def program(self, island, slots):
        """Island `island`'s program: the one-card decode's kernels and overlays on a
        card's plans, the crossing's overlay, its thread."""
        plans = list(self.plans.values())
        table = pl.split_table(plans)
        tile_args = self.copies["x"].tensor_args()
        program = km.Program(
            f"decode2-{island}",
            kernels=dense_card.SPLIT_KERNELS,
            buffers=one.buffers(plans[0], self.keys_most, (), True),
            args=lambda core: (table, table, table),
            defines={"SPLIT_CHAIN": 1}
            | ({"SPLIT_CHAIN_STAMPS": 1} if self.stamps else {}),
            compile_args={"reader": tile_args * 2, "writer": tile_args * 3},
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    S["kCbArrive"]: "UnpackToDestFp32",
                    S["kCbSum"]: "UnpackToDestFp32",
                }
                | {D[name]: "UnpackToDestFp32" for name in one.TO_DEST},
            },
            handlers=pl.SPLIT_H,
        )
        kernels = one.KERNELS
        emit = {f"DC_EMIT_{name.upper()}": 0 for name in one.BODIES}
        program.overlay(
            "add", kernels / "add", number=D["DC_OVERLAY_ADD"], defines=emit
        )
        program.overlay(
            "gate",
            {
                "reader": kernels / "scale" / "reader.cpp",
                "compute": kernels / "gate" / "compute.cpp",
                "writer": kernels / "gate" / "writer.cpp",
            },
            number=D["DC_OVERLAY_GATE"],
            defines=emit,
        )
        epsilon = np.array([self.model.eps], np.float32).view(np.uint32)[0]
        program.overlay(
            "scale",
            kernels / "scale",
            number=D["DC_OVERLAY_SCALE"],
            defines={"DC_EPSILON": int(epsilon)} | emit,
        )
        program.overlay(
            "attend",
            kernels / "attend",
            number=D["DC_OVERLAY_ATTEND"],
            defines={"DC_KEYS_MOST": self.keys_most, "SPLIT_CHAIN": 1},
        )
        program.overlay(
            "cross",
            HERE / "kernels" / "cross",
            number=T["DC2_OVERLAY_CROSS"],
        )
        if slots is not None:
            program.slots(count=slots)
        weights = np.zeros((self.layers, D["DC_LAYER_WORDS"]), np.uint32)
        for layer, tensors in enumerate(self.weights[island]):
            for name, tensor in tensors.items():
                weights[layer, D[f"DC_W_{name.upper()}"]] = tensor.address
        program.thread(
            cluster=CLUSTER,
            sources=[HERE / "thread" / "layers.c"],
            includes=[HERE, md.HERE, dense_card.SPLIT_KERNELS],
            regions={
                "weights": weights.reshape(-1),
                "tensors": self.addresses(island),
                "stats": 4 * D["DC_STATS_WORDS"] * D["DC_STATS_SLOTS"],
                "records": one.CHAIN["kChainRecordBytes"]
                * one.CHAIN["kChainRecordsMax"],
            },
            words=self.words(island),
            handlers={"layers": D["DC_HANDLER_LAYERS"]},
        )
        return program

    def write_layer(self, layer, weights):
        """Layer `layer`'s weights (md.layer_weights()' form) into both cards' DRAM,
        each its half: seconds taken."""
        started = time.perf_counter()
        for island, s in enumerate(self.session.sessions):
            for name, pages in half_weights(self.model, weights, island).items():
                s.write(self.weights[island][layer][name], pages)
            pages = md.qk_weight_pages(weights["q_norm"][1], weights["k_norm"][1])
            first = D["DC_QK_WEIGHTS"] + 2 * D["DC_HEAD_TILES"] * layer
            s.write(self.local[island]["qk"], pages, first=first)
        return time.perf_counter() - started

    def write_cache(self, layer, keys, values):
        """A layer's cache ([KV heads, n, width] each): each card its KV heads'."""
        started = time.perf_counter()
        heads = self.half.kv_heads
        for island, s in enumerate(self.session.sessions):
            mine = slice(island * heads, (island + 1) * heads)
            for name, cache in (("keys", keys), ("values", values)):
                pages = md.cache_pages(cache[mine], self.context)
                s.write(self.weights[island][layer][name], pages)
        return time.perf_counter() - started

    def write_activation(self, name, values):
        """A [32, width] activation into its copy on both cards; x with its squares."""
        bits = formats.bf16_bits(np.asarray(values, np.float32))
        self.session.write(self.copies[name], md.tile_pages(bits))
        if name == "x":
            self.write_activation("squares1", md.squares(values))

    def read_activation(self, name, island=0):
        """A [32, width] activation as float32, island `island`'s copy."""
        tensor = self.copies[name]
        (raw,) = self.session.sessions[island].read(tensor.parts[island])
        tiles = raw.reshape(-1).view(np.uint16).reshape(-1, md.TILE, md.TILE)
        rows = formats.bf16_values(tiles).transpose(1, 0, 2)
        return rows.reshape(md.TILE, -1)

    def job(
        self,
        tokens,
        layers,
        first=0,
        position=None,
        token=False,
        local=False,
        ahead=True,
        timeout=10.0,
    ):
        """One LAYERS2 on both threads, submitted back to back: each island's result
        word (ticks) and the stats slot."""
        slot = self.slot
        if slot >= D["DC_STATS_SLOTS"]:
            raise ValueError(f"job {slot}: the thread has {D['DC_STATS_SLOTS']} slots")
        self.slot += 1
        mode = (D["DC_LAYERS_TOKEN"] if token else 0) | (
            T["DC2_LAYERS_LOCAL"] if local else 0
        )
        mode |= 0 if ahead else D["DC_LAYERS_AHEAD_OFF"]
        position = self.context - tokens if position is None else position
        if not 0 <= position <= self.context - tokens:
            raise ValueError(f"position {position}: {tokens} rows past the cache")
        args = [tokens, layers, slot, first, mode, position, 1]
        runs = self.run.runs
        handles = [self.submit_thread(island, args) for island in range(2)]
        events = []
        for run, handle in zip(runs, handles):
            try:
                events.append(run.wait(handle, timeout=timeout))
            except km.KomorebiError as error:
                events.append(error)
        bad = [
            f"island {i}: {e}"
            for i, e in enumerate(events)
            if not hasattr(e, "kind") or e.kind != "done" or e.result >= 0xFFFFFFF0
        ]
        if bad:
            raise km.KomorebiError(f"threads: {'; '.join(bad)}")
        return {"thread_ticks": [int(e.result) for e in events], "stats_slot": slot}

    def submit_thread(self, island, args):
        """A LAYERS2 command to island `island`'s thread: its handle. On islands thread
        numbers are flat (island 1's cluster 1 is thread 4), which Run.submit's
        `cluster` (1 to 3: island 0's) does not name, so the group is given whole."""
        run = self.run.runs[island]
        number = self.session.graph.thread_numbers(island).start + CLUSTER - 1
        group = (abi.headers()["KM_GROUP_CLUSTER_BASE"] + number, 1)
        return run._queue_command(
            D["DC_HANDLER_LAYERS"], group, args, False, run._flag_result, 0
        )

    def stats(self):
        """Every stats slot of both threads, after run.stop(): [2, slots, words]."""
        count = D["DC_STATS_WORDS"] * D["DC_STATS_SLOTS"]
        return np.stack(
            [
                run.threads[CLUSTER]
                .read("stats")
                .view(np.uint32)[:count]
                .reshape(D["DC_STATS_SLOTS"], D["DC_STATS_WORDS"])
                for run in self.run.runs
            ]
        )
