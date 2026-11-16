"""Running the matrix product on a card through `komorebi.Session`: the program of a
plan, the operand tensors, and the commands (LOAD, PRODUCT, PROF) submitted through the
dispatcher on X280 cluster 0. Needs tt-metal's Python environment.
"""

import time

import numpy as np

import komorebi as km
from komorebi import _env, abi, layout
from projects.dense import plan as pl
from reference import dense as ref

K, G, S = pl.K, pl.G, pl.S
KERNELS = pl.HERE / "kernels"
GRID_KERNELS = KERNELS / "grid"
SPLIT_KERNELS = KERNELS / "split"
BROADCAST_FIELDS = abi.struct_fields(
    _env.include_directory() / "komorebi/device/movement/km_broadcast.hpp",
    "BroadcastArgs",
)


def bf16_tensor(session, x):
    """A float array (bf16 values) as a bf16 tile tensor in DRAM."""
    return session.tensor(np.asarray(x, dtype=np.float32), "bfloat16", "tile")


def read_rows(session, y, rows, log=None):
    """Rows `rows` of the bf16 tile tensor Y as float32, read one row tile at a time
    (only the tiles holding them, at most 1 MB each), after the run's stop."""
    column_tiles = y.shape[1] // ref.TILE
    tiles = np.unique(np.asarray(rows) // ref.TILE)
    got = {}
    for i, tile in enumerate(tiles):
        part = session.read_tiles_bf16(
            y, int(tile) * column_tiles, column_tiles
        )  # [column_tiles, 32, 32]
        got[int(tile)] = part.transpose(1, 0, 2).reshape(
            ref.TILE, column_tiles * ref.TILE
        )
        if log:
            log(f"row tile {i + 1} of {len(tiles)} read")
    return np.stack([got[row // ref.TILE][row % ref.TILE] for row in rows])


def prof_tensor(session, cores):
    """Zeroed profile pages in DRAM, one page of kProfBytes per core (row-major)."""
    return session.tensor(np.zeros((cores, K["kProfBytes"] // 4), np.int32), "int32")


def program(plan, a_tensor, fidelity="HiFi2"):
    """The persistent program of `plan`: the three kernels and the plan's buffers."""
    tile_args = a_tensor.tensor_args()  # every operand's: bf16 tiles in DRAM
    buffers = [
        km.Buffer(index, "bfloat16", pl.TILE_BYTES, pages)
        for index, pages in plan.buffers()
    ]
    buffers += [
        km.Buffer(K["kCbProf"], "uint32", K["kProfBytes"], 1),
        km.Buffer(K["kCbLoaded"], "uint32", K["kLoadedBytes"], 1),
    ]
    return km.Program(
        "dense",
        kernels=KERNELS,
        buffers=buffers,
        compile_args={"reader": tile_args, "writer": tile_args},
        compute={"math_fidelity": fidelity, "fp32_dest_acc_en": True},
        handlers=pl.DENSE_H,
    )


class Product:
    """`plan` loaded on a session's grid with operand tensors a, w, y: LOAD at the
    start (resident form), then PRODUCT calls and PROF on the group (0, count)."""

    def __init__(self, session, plan, tensors, count, fidelity="HiFi2", deadline=2.0):
        self.plan, self.tensors, self.group = plan, tensors, (0, count)
        self.run = session.load(
            program(plan, tensors["a"], fidelity), deadline=deadline, wait=1.0
        )
        self.load_microseconds = None
        if plan.resident:
            w_address = tensors["w"].address
            handle = self.run.submit("load", self.group, pl.load_args(plan, w_address))
            self.load_microseconds = self.run.wait(handle, timeout=10.0).microseconds

    def words(self, mode):
        tensors = self.tensors
        return pl.product_args(
            self.plan,
            tensors["a"].address,
            tensors["y"].address,
            tensors["w"].address,
            mode,
        )

    def calls(self, commands, mode=K["kModeFull"], timeout=60.0):
        """`commands` PRODUCT commands back to back: (host seconds from the first
        submit to the last event, each command's submit-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (commands, 1))
        started = time.perf_counter()
        handles = self.run.submit_many("product", self.group[0], self.group[1], words)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [event for event in events if event.kind != "done"]
        if unfinished:
            raise km.KomorebiError(
                f"{len(unfinished)} commands not done: {unfinished[0]}"
            )
        return host_s, np.array([event.microseconds for event in events])

    def single(self, commands, mode=K["kModeFull"]):
        """`commands` PRODUCT commands one at a time (each waited for): microseconds
        each."""
        durations = []
        for _ in range(commands):
            handle = self.run.submit("product", self.group, self.words(mode))
            durations.append(self.run.wait(handle, timeout=10.0).microseconds)
        return np.array(durations)

    def profile(self, session, profile_tensor):
        """Each core's profile page (summed over the commands so far): [cores,
        words]."""
        handle = self.run.submit("prof", self.group, [profile_tensor.address])
        self.run.wait(handle, timeout=10.0)
        (pages,) = session.read(profile_tensor)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : K["kProfWords"]]

    def stop(self):
        """Stops the run; the kernel's result and the cores that gave up."""
        self.run.stop()
        return self.summary()

    def failed(self, error):
        """After a failed wait: the run stopped if it can be, and what its state shows
        (each read guarded, for the log before the driver leaves)."""
        report = {"error": f"{type(error).__name__}: {error}"}
        try:
            self.run.stop()
        except km.KomorebiError as stop_error:
            report["stop"] = f"{type(stop_error).__name__}: {stop_error}"
        try:
            report["broadcasts"] = broadcasts_at_stall(self.run.state())
            report |= self.summary()
        except km.KomorebiError as state_error:
            report["state"] = f"{type(state_error).__name__}: {state_error}"
        return report

    def summary(self):
        """The stopped run's kernel result, cores ended, give-ups and broadcast
        waits."""
        states = self.run.state()
        return {
            "cores": len(states),
            "kernel": (self.run.result() or {}).get("error_name"),
            "cores_ended": {
                key: sum(state[key] != 0 for state in states)
                for key in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {key: state[key] for key in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
            "broadcast_wait_cycles_max": [
                max(state["broadcasts"][broadcast]["wait_cycles"] for state in states)
                for broadcast in (0, 1)
            ],
            "broadcast_gave_up": [
                {"core": state["core"]} | area["gave_up"]
                for state in states
                for area in state["broadcasts"]
                if area["gave_up"]
            ][:8],
        }


def broadcasts_at_stall(states):
    """By broadcast: blocks delivered (min, max), the cores at the min (first 8),
    their VALID words, and the cores' commands completed (min, max)."""
    report = {
        "completed": [
            min(state["completed"] for state in states),
            max(state["completed"] for state in states),
        ]
    }
    for broadcast in (0, 1):
        blocks = [state["broadcasts"][broadcast]["blocks"] for state in states]
        lowest = [
            state
            for state in states
            if state["broadcasts"][broadcast]["blocks"] == min(blocks)
        ]
        report[f"broadcast{broadcast}"] = {
            "blocks": [min(blocks), max(blocks)],
            "at_min": [state["core"] for state in lowest][:8],
            "valid_at_min": [
                state["broadcasts"][broadcast]["valid"] for state in lowest
            ][:8],
        }
    return report


def grid_broadcasts(plan, nocs, group, along, noc, buffer):
    """Broadcast words by core for `plan`: broadcasts along the rect's rows or
    columns, or a broadcast of one core each where the layout gives every core its own
    operand."""
    common = (nocs.grid, nocs.panels, group)
    noc_places = (nocs.noc0, nocs.noc_xy)
    rect = plan.rect(nocs.grid)
    layout_args = layout.broadcast_args(*common, rect, along, noc, *noc_places, buffer)
    if {"row": "rows", "column": "cols"}[along] != plan.layout:
        return layout_args
    grid_columns = nocs.grid[0]
    for core in plan.places(nocs.grid):
        y, x = divmod(core, grid_columns)
        single = layout.broadcast_args(
            *common, (x, y, 1, 1), along, noc, *noc_places, buffer
        )
        layout_args[core] = single[core]
    return layout_args


def grid_program(plan, nocs, tensors, fidelity="HiFi2", first_row=0, command=None):
    """The grid form's program for `plan` on a grid whose NoCs are `nocs`: each core's
    broadcast words (row broadcast on NoC0, column broadcast on NoC1), place and plan
    words. The plan's rows start at row tile `first_row` of A and Y; `command`: (the
    cores (first, count) a command reaches here, its group word), default the rect's."""
    group = layout.Group(*plan.group(nocs.grid))
    group_word = group.first | group.count << 16
    if command is not None:
        group, group_word = layout.Group(*command[0]), command[1]
    row = grid_broadcasts(
        plan,
        nocs,
        group,
        "row",
        0,
        (G["kBroadcastRow"], G["kCbRow"], plan.wa, plan.bm * plan.kb),
    )
    column = grid_broadcasts(
        plan,
        nocs,
        group,
        "column",
        1,
        (G["kBroadcastCol"], G["kCbCol"], plan.ww, plan.kb * plan.bn),
    )
    places = plan.places(nocs.grid)
    idle = [0] * (len(BROADCAST_FIELDS) + G["kCoreWords"])
    # the row bound of the kernels is the plan's last row tile in A and Y, plus one
    plan_words = [first_row + plan.mt, *plan.words()[1:]]

    def args(core):
        if core not in group.indices:
            return idle, idle, idle[: G["kCoreWords"]] + [plan.wa, plan.ww]
        row_tile, column_tile = places.get(core, (0, 0))
        mine = [int(core in places), first_row + row_tile, column_tile, *plan_words]
        broadcasts = [
            [(words[core] | {"group": group_word})[field] for field in BROADCAST_FIELDS]
            for words in (row, column)
        ]
        return broadcasts[0] + mine, broadcasts[1] + mine, mine + [plan.wa, plan.ww]

    return km.Program(
        "dense_grid",
        kernels=GRID_KERNELS,
        buffers=[km.Buffer(*buffer) for buffer in plan.buffers()],
        args=args,
        compile_args={
            "reader": tensors["a"].tensor_args(),
            "writer": tensors["w"].tensor_args() + tensors["y"].tensor_args(),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {G["kCbPart"]: "UnpackToDestFp32"},
        },
        handlers=pl.GRID_H,
    )


class GridProduct:
    """`plan` (the grid form) loaded on a session with operand tensors a, w, y: one
    GRID command per call on the rectangle's group, exclusive (its broadcasts)."""

    handler = "grid"
    modes = {
        "full": G["kGridFull"],
        "broadcasts": G["kGridBroadcasts"],
        "product": G["kGridProduct"],
    }

    def __init__(
        self, session, plan, tensors, fidelity="HiFi2", deadline=2.0, wait=1.0
    ):
        self.plan, self.tensors = plan, tensors
        nocs = session.nocs()
        self.group = plan.group(nocs.grid)
        program = grid_program(plan, nocs, tensors, fidelity)
        self.run = session.load(program, deadline=deadline, wait=wait)
        self.load_microseconds = None

    def words(self, mode):
        tensors = self.tensors
        addresses = [tensors[name].address for name in ("a", "w", "y")]
        return pl.grid_args(self.plan, *addresses, mode)

    def calls(self, commands, mode=G["kGridFull"], timeout=60.0):
        """`commands` GRID commands back to back: (host seconds from the first submit
        to the last event, each command's submit-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (commands, 1))
        started = time.perf_counter()
        first, count = self.group
        handles = self.run.submit_many(
            self.handler, first, count, words, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [event for event in events if event.kind != "done"]
        if unfinished:
            raise km.KomorebiError(
                f"{len(unfinished)} commands not done: {unfinished[0]}"
            )
        return host_s, np.array([event.microseconds for event in events])

    def single(self, commands, mode=G["kGridFull"]):
        """`commands` GRID commands one at a time (each waited for): microseconds
        each."""
        durations = []
        for _ in range(commands):
            handle = self.run.submit(
                self.handler, self.group, self.words(mode), exclusive=True
            )
            durations.append(self.run.wait(handle, timeout=10.0).microseconds)
        return np.array(durations)

    def profile(self, session, profile_tensor):
        """Each core's profile page (summed over the commands so far): [cores,
        words]."""
        handle = self.run.submit("grid_prof", self.group, [profile_tensor.address])
        self.run.wait(handle, timeout=10.0)
        (pages,) = session.read(profile_tensor)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : G["kGpWords"]]

    stop, failed, summary = Product.stop, Product.failed, Product.summary


def split_program(plan, tensors, fidelity="HiFi2"):
    """The K-split form's program for `plan` (or a list of plans that share their
    buffers, a command's a5 picking one): every core gets the plan words; both movers
    read the K chunks (A and W to each), the writer also writes Y."""
    plans = list(plan) if isinstance(plan, list | tuple) else [plan]
    words = pl.split_table(plans)
    fp32 = "UnpackToDestFp32"
    return km.Program(
        "dense_split",
        kernels=SPLIT_KERNELS,
        buffers=[km.Buffer(*buffer) for buffer in plans[0].buffers()],
        args=lambda core: (words, words, words),
        compile_args={
            "reader": tensors["a"].tensor_args() + tensors["w"].tensor_args(),
            "writer": (
                tensors["y"].tensor_args()
                + tensors["a"].tensor_args()
                + tensors["w"].tensor_args()
            ),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {S["kCbArrive"]: fp32, S["kCbSum"]: fp32},
        },
        handlers=pl.SPLIT_H,
    )


class SplitProduct(GridProduct):
    """`plan` (K split) loaded on a session with operand tensors a, w, y: one SPLIT
    command per call on cores 0 to plan.cores() - 1, exclusive (its ports), by `form`.
    """

    handler = "split"
    modes = {"full": S["kSplitFull"], "parts": S["kSplitParts"]}

    def __init__(
        self,
        session,
        plan,
        tensors,
        form="owner",
        fidelity="HiFi2",
        deadline=2.0,
        wait=1.0,
    ):
        self.plan, self.tensors, self.form = plan, tensors, form
        self.group = (0, plan.cores())
        program = split_program(plan, tensors, fidelity)
        self.run = session.load(program, deadline=deadline, wait=wait)
        self.load_microseconds = None

    def words(self, mode):
        addresses = [self.tensors[name].address for name in ("a", "w", "y")]
        return pl.split_args(self.plan, *addresses, self.form, mode)
