"""The decode step on a card: the K-split program with the operations as overlays and
the X280 thread that walks the layers, its tensors, the weights written through the
host channel, and the thread's job. Needs tt-metal's Python environment."""

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
ACTIVATIONS = ("x", "h", "qkv", "k", "v", "gated", "sums")
PARTS = {"reader": OP["kOpReader"], "compute": OP["kOpCompute"]}
GATE_PARTS = {"compute": OP["kOpCompute"], "writer": OP["kOpWriter"]}
NORM_PARTS = PARTS | {"writer": OP["kOpWriter"]}


def operation_word(number, parts):
    """A SPLIT command's a6: the overlay's number and its parts."""
    return number | sum(parts.values())


def buffers(plan):
    """The K split's buffers (every plan shares them) and the operations'."""
    tile = pl.TILE_BYTES
    block = plan.oh * plan.ow
    own = [
        (D["DC_CB_BLOCK"], block),
        (D["DC_CB_ROW"], D["DC_ROW_TILES"]),
        (D["DC_CB_WEIGHT"], D["DC_NORM_SHARE"]),
        (D["DC_CB_CONSTANT"], D["DC_CONSTANTS"]),
        (D["DC_CB_SQUARES"], 1),
        (D["DC_CB_SCALE"], 1),
        (D["DC_CB_NORMED"], D["DC_NORM_SHARE"]),
        (D["DC_CB_UP"], block),
    ]
    return [km.Buffer(*buffer) for buffer in plan.buffers()] + [
        km.Buffer(index, "bfloat16", tile, pages) for index, pages in own
    ]


class Decode:
    """The model's layers loaded on a session: tensors, the program and its thread.
    `layers` distinct layers of weights in DRAM, written by write_layer()."""

    def __init__(self, session, model, layers, deadline, log):
        self.session, self.model, self.layers, self.log = session, model, layers, log
        self.plans = md.plans(model)
        self.plan_index = {shape: i for i, shape in enumerate(self.plans)}
        hidden = model.hidden
        products = model.products()
        widths = {"x": hidden, "h": hidden, "gated": model.mlp}
        widths |= {"qkv": products["qkv"][1], "sums": products["gate_up"][1]}
        widths |= {name: model.shapes()[name][1] for name in ("k", "v")}
        self.tensors = {
            name: session.empty((md.TILE, widths[name]), "bfloat16", "tile")
            for name in ACTIVATIONS
        }
        self.tensors["constants"] = dense_card.bf16_tensor(session, md.constants())
        shapes = products | {name: model.shapes()[name] for name in md.APART}
        self.weights = [
            {
                name: session.empty(shape, "bfloat16", "tile")
                for name, shape in shapes.items()
            }
            | {
                name: session.empty((md.TILE, hidden), "bfloat16", "tile")
                for name in ("norm1", "norm2")
            }
            for _ in range(layers)
        ]
        self.run = session.load(self.program(), deadline=deadline, wait=1.0)
        self.slot = 0

    def program(self):
        """The K split's kernels with a table of the layer's plans, three overlays and
        the thread."""
        plans = list(self.plans.values())
        table = pl.split_table(plans)
        tile_args = self.tensors["x"].tensor_args()  # every operand: bf16 tiles
        program = km.Program(
            "decode",
            kernels=dense_card.SPLIT_KERNELS,
            buffers=buffers(plans[0]),
            args=lambda core: (table, table, table),
            compile_args={"reader": tile_args * 2, "writer": tile_args * 3},
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    S["kCbArrive"]: "UnpackToDestFp32",
                    S["kCbSum"]: "UnpackToDestFp32",
                },
            },
            handlers=pl.SPLIT_H,
        )
        reader = KERNELS / "block_reader.cpp"
        program.overlay(
            "add",
            {"reader": reader, "compute": KERNELS / "add" / "compute.cpp"},
            number=D["DC_OVERLAY_ADD"],
        )
        program.overlay(
            "gate",
            {
                "compute": KERNELS / "gate" / "compute.cpp",
                "writer": KERNELS / "gate" / "writer.cpp",
            },
            number=D["DC_OVERLAY_GATE"],
        )
        program.overlay("norm", KERNELS / "norm", number=D["DC_OVERLAY_NORM"])
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
            operation_word(D["DC_OVERLAY_ADD"], PARTS),
            operation_word(D["DC_OVERLAY_GATE"], GATE_PARTS),
            operation_word(D["DC_OVERLAY_NORM"], NORM_PARTS),
            int(epsilon),
        ]

    def write_layer(self, layer, weights):
        """Layer `layer`'s weights (layer_weights(): {name: (pages, values)}) into DRAM
        through the host channel, as the card's products take them: seconds taken."""
        seconds = 0.0
        for name, pages in md.fused(weights).items():
            seconds += self.write(self.weights[layer][name], pages)
        return seconds

    def write(self, tensor, pages):
        """Pages into a tensor through the host channel: seconds taken."""
        started = time.perf_counter()
        self.session.write(tensor, pages)
        return time.perf_counter() - started

    def write_activation(self, name, values):
        """A [32, width] activation (float values, rounded to bf16) into its tensor."""
        bits = formats.bf16_bits(np.asarray(values, np.float32))
        self.session.write(self.tensors[name], md.tile_pages(bits))

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
    ):
        """One LAYERS command to the thread (`bare`: the products only; `parts`: their
        parts only, no sum and nothing written; `apart`: q, k and v as three products;
        `overlap`: and q, k, v, o two in flight): its result word and stats slot."""
        slot = self.slot
        self.slot += 1
        mode = 3 if parts else int(bare)
        mode |= (4 if apart or overlap else 0) | (8 if overlap else 0)
        args = [tokens, layers, slot, first, mode]
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


STEPS = ("norm1", "qkv", "q", "k", "v", "o", "norm2", "gate_up", "down")  # DC_STEP_*


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
