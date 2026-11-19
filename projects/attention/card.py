"""Running attention on a card through `komorebi.Session`: the program of a plan, its
broadcast words from `komorebi.layout`, the tensors, and the ATTEND commands submitted
through the dispatcher on X280 cluster 0. Needs tt-metal's Python environment."""

import time

import numpy as np

import komorebi as km
from komorebi import abi, layout
from projects.attention import plan as pl

K = pl.K
KERNELS = pl.HERE / "kernels"
BROADCAST_FIELDS = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
BROADCAST_ID = 0


def bf16_tensor(session, values, **placed):
    """A float array of bf16 values [..., rows, cols] as a bf16 tile tensor in DRAM,
    flattened to [rows of all leading axes, cols] (the same page order); `placed`, on
    a session of islands, its placement (split, group)."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    return session.tensor(values, "bfloat16", "tile", **placed)


def table_tensor(session, words):
    """Job tables [cores, kTableBytes / 4] uint32 as a row-major DRAM tensor, one page
    per core."""
    return session.tensor(words, "uint32")


def prof_tensor(session, cores):
    """Profile pages [cores, kProfBytes / 4] uint32, zero, as a row-major DRAM tensor
    (one page of kProfBytes per core)."""
    return table_tensor(session, np.zeros((cores, K["kProfBytes"] // 4), np.uint32))


def broadcast_words(nocs, plan, group, noc=0, columns=None):
    """Each core's km::BroadcastArgs words: broadcasts of plan.length cores down every
    grid column (the first `columns`) on NoC `noc` (the sender first in its order), one
    band of rows after the other."""
    grid_columns, grid_rows = nocs.grid
    grid_columns = columns or grid_columns
    buffer = (BROADCAST_ID, K["kCbKv"], plan.window, plan.block_pages())
    by_core = {}
    for top in range(0, grid_rows, plan.length):
        rect = (0, top, grid_columns, plan.length)
        broadcast_arguments = layout.broadcast_args(
            nocs.grid,
            nocs.panels,
            group,
            rect,
            "column",
            noc,
            nocs.noc0,
            nocs.noc_xy,
            buffer,
        )
        for core, arguments in broadcast_arguments.items():  # a core in none: none
            if core not in by_core or arguments["rank"] != layout.BROADCAST_NONE:
                by_core[core] = arguments
    return {
        core: [arguments[field] for field in BROADCAST_FIELDS]
        for core, arguments in by_core.items()
    }


def card_plan(session, call, length=None, chunks=None):
    """A plan over one card's ranks and the grid it was made on: the session's grid,
    or, beside a Tensix dispatcher (the grid's last core), every column but the last
    (a column's broadcasts stay whole); its jobs by core, none on that column."""
    columns, rows = session.nocs().grid
    if session.ranks == columns * rows:
        return pl.make(call, (columns, rows), length=length, chunks=chunks), None
    grid = (columns - 1, rows)
    plan = pl.make(call, grid, length=length, chunks=chunks)
    jobs = [[] for _ in range(session.ranks)]
    for index, core_jobs in enumerate(plan.jobs):
        y, x = divmod(index, grid[0])
        jobs[y * columns + x] = core_jobs
    return pl.Plan(plan.dims, plan.r, plan.kb, plan.window, plan.length, jobs), grid


def island_plan(session, call, length=None, chunks=None):
    """A plan over the islands of `session` as one grid: their grids one below the
    other, every column but the last (an island's last core is a link core, not a
    rank), broadcasts inside an island (lengths dividing its rows). Its jobs by flat
    rank (none on the last column's ranks), and that grid's (columns, rows)."""
    graph = session.graph
    columns, rows = session.island(0).nocs().grid
    for island in range(len(graph.islands)):
        if graph.count(island) < rows * columns - 1:
            raise ValueError(f"island {island}: {graph.count(island)} ranks")
    grid = (columns - 1, rows * len(graph.islands))
    lengths = [n for n in range(1, rows + 1) if rows % n == 0]
    plan = pl.make(call, grid, length=length, chunks=chunks, lengths=lengths)
    if rows % plan.length:
        raise ValueError(f"broadcast length {plan.length}: not inside an island")
    jobs = [[] for _ in range(graph.ranks)]
    for index, core_jobs in enumerate(plan.jobs):
        y, x = divmod(index, grid[0])
        island = y // rows
        jobs[graph.first(island) + (y % rows) * columns + x] = core_jobs
    return pl.Plan(plan.dims, plan.r, plan.kb, plan.window, plan.length, jobs), grid


def island_broadcast_words(session, plan, grid, noc=0):
    """Each island's km::BroadcastArgs words by core (its ranks and link cores): the
    broadcasts of `plan` down the columns of `grid` (island_plan's), the sender
    first; a core in none of them is not in a broadcast."""
    graph = session.graph
    islands = [
        (session.island(i).nocs(), graph.first(i), graph.count(i))
        for i in range(len(graph.islands))
    ]
    buffer = (BROADCAST_ID, K["kCbKv"], plan.window, plan.block_pages())
    group = layout.Group(0, graph.ranks)
    by_island = [{} for _ in islands]
    for x in range(grid[0]):
        for top in range(0, grid[1], plan.length):
            per_island = layout.island_broadcast_args(
                islands, group, (x, top, 1, plan.length), "column", noc, buffer
            )
            for island, arguments in enumerate(per_island):
                for core, words in arguments.items():
                    if (
                        core not in by_island[island]
                        or words["rank"] != layout.BROADCAST_NONE
                    ):
                        by_island[island][core] = words
    none = next(
        words
        for mine in by_island
        for words in mine.values()
        if words["rank"] == layout.BROADCAST_NONE
    )
    out = []
    for island, mine in enumerate(by_island):
        cores = len(islands[island][0].words)
        out.append(
            {
                core: [mine.get(core, none)[field] for field in BROADCAST_FIELDS]
                for core in range(cores)
            }
        )
    return out


def program(
    plan,
    tensors,
    scale,
    approx=False,
    fidelity="HiFi2",
    broadcasts=None,
    noc=0,
):
    """The persistent program of `plan`: the three kernels, the plan's buffers, each
    core's broadcast words (`broadcasts`, by core) for the kernel on NoC `noc` (0: the
    reader, 1: the writer), the compute's window and the tensor arguments of q, k, v
    (reader) and o, k, v (writer) of `tensors`."""
    call = plan.dims
    scale_bits, threshold_bits = pl.exp_constants(scale, approx)
    tile_sizes = [call.dt, call.dvt, plan.r, plan.kb]
    defines = {"BROADCAST_ON_WRITER": "1"} if noc == 1 else {}
    if call.causal:
        defines["FLASH_CAUSAL"] = "1"
    if approx:
        defines["FLASH_APPROX"] = "1"

    def args(core):
        if noc == 1:
            return [], broadcasts[core], [plan.window]
        return broadcasts[core], [], [plan.window]

    return km.Program(
        "attention",
        kernels=KERNELS,
        buffers=[km.Buffer(*buffer) for buffer in plan.buffers()],
        args=args,
        compile_args={
            "reader": tile_sizes
            + [plan.window]
            + [word for name in "qkv" for word in tensors[name].tensor_args()]
            + tensors["tables"].tensor_args(),
            "writer": tile_sizes
            + [plan.window]
            + [word for name in "okv" for word in tensors[name].tensor_args()]
            + tensors["tables"].tensor_args()
            + tensors["prof"].tensor_args(),
            "compute": tile_sizes + [scale_bits, threshold_bits],
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": False,
            "math_approx_mode": approx,
            "unpack_to_dest_mode": {
                K["kCbO"]: "UnpackToDestFp32",
                K["kCbSt"]: "UnpackToDestFp32",
            },
        },
        defines=defines,
        handlers=pl.ATTENTION_H,
    )


class Attention:
    """A plan loaded on a session's grid with tensors q, k, v, o and the job tables:
    ATTEND commands on the group of every core. On a session of islands (`grid`, the
    plan's from island_plan), the group of every flat rank, the same kernels."""

    def __init__(
        self,
        session,
        plan,
        tensors,
        scale,
        approx=False,
        deadline=2.0,
        wait=2.0,
        noc=0,
        grid=None,
    ):
        islands = hasattr(session, "graph")
        cores = session.ranks
        if len(plan.jobs) != cores:
            raise ValueError(f"plan for {len(plan.jobs)} cores, grid has {cores}")
        if "prof" not in tensors:  # the writer's profile pages: its tensor arguments
            tensors = {**tensors, "prof": prof_tensor(session, cores)}
        self.plan, self.tensors, self.group = plan, tensors, (0, cores)
        if islands:
            loaded = [
                program(plan, tensors, scale, approx, broadcasts=words, noc=noc)
                for words in island_broadcast_words(session, plan, grid, noc)
            ]
        else:
            broadcasts = broadcast_words(
                session.nocs(), plan, layout.Group(0, cores), noc, grid and grid[0]
            )
            loaded = program(
                plan, tensors, scale, approx, broadcasts=broadcasts, noc=noc
            )
        self.run = session.load(loaded, deadline=deadline, wait=wait)

    def words(self, mode=K["kModeFull"], passes=1):
        tensors = self.tensors
        return pl.call_args(
            *(tensors[name].address for name in ("q", "k", "v", "o", "tables")),
            mode,
            passes,
        )

    def calls(self, count, mode=K["kModeFull"], passes=1, timeout=60.0):
        """`count` ATTEND commands back to back (each `passes` times over its jobs):
        (host seconds from the first submit to the last event, each command's
        microseconds)."""
        words = np.tile(np.array(self.words(mode, passes), np.int64), (count, 1))
        started = time.perf_counter()
        handles = self.run.submit_many(
            "attend", self.group[0], self.group[1], words, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        # over islands an event is its parts, one an island: the slowest part's time
        parts = [(event,) if hasattr(event, "kind") else event for event in events]
        unfinished = [part for event in parts for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(
                f"{len(unfinished)} commands not done: {unfinished[0]}"
            )
        return host_s, np.array(
            [max(part.microseconds for part in event) for event in parts]
        )

    def profile(self, session, zero=True, timeout=10.0):
        """Every core's profile page since the last zeroing ([cores, kProfWords]
        uint32, attention.h kPr*), stored by a PROF command into tensors["prof"] (made
        before the load); the pages are zeroed after (`zero`)."""
        profile_tensor = self.tensors["prof"]
        words = [profile_tensor.address, 1 if zero else 0]
        handle = self.run.submit("prof", self.group, words)
        self.run.wait(handle, timeout=timeout)
        (pages,) = session.read(profile_tensor)
        words = pages.reshape(-1).view(np.uint32).reshape(self.group[1], -1)
        return words[:, : K["kProfWords"]]

    def stop(self):
        """Stops the run; the kernel's result, the cores that ended, the give-ups, and
        `ok`: no kernel error, no give-up, every core's three kernels ended."""
        self.run.stop()
        states = self.run.state()
        results = self.run.result()
        kernels = [
            (result or {}).get("error_name")
            for result in (results if isinstance(results, list) else [results])
        ]
        report = {
            "kernel": kernels[0] if len(kernels) == 1 else kernels,
            "cores": len(states),
            "cores_ended": {
                name: sum(state[name] != 0 for state in states)
                for name in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {name: state[name] for name in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
            "broadcast_blocks": sum(
                state["broadcasts"][BROADCAST_ID]["blocks"] for state in states
            ),
        }
        ended = all(count == len(states) for count in report["cores_ended"].values())
        report["ok"] = (
            all(kernel == "none" for kernel in kernels)
            and not report["gave_up"]
            and ended
        )
        return report
