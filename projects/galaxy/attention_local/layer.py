"""A DeepSeek-V4.1-Flash layer's own attention on one Wormhole chip (pieces 7, 8, 9):
proj.py, window.py and attend.py as three runtime-form classes loaded as a dict, a
command at a time. `Layer.decode(x, pos)`, `Layer.prefill(x, start)` write y and extend
the window. README.md, "Why three programs"."""

import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.islands.records import IslandTensor
from projects.galaxy.attention_local import attend as at
from projects.galaxy.attention_local import proj as pj
from projects.galaxy.attention_local import window as wn

TILE = 32
WEIGHT_FORMAT = "bfp8_b"
STORE_EVERY = 4  # queued commands a send to the card (a send costs about 70 us)


def view(tensor, shape):
    """The same DRAM buffer seen as another shape of the same tiles (q's [32, 64 x 512]
    as [64 x 32, 512]: tile h x 16 + c either way); on islands, the same parts."""
    if isinstance(tensor, Tensor):
        seen = Tensor(tensor.buffer, shape, tensor.dtype, tensor.layout)
    else:
        seen = IslandTensor(
            tensor.parts,
            tensor.placement,
            shape,
            tensor.dtype,
            tensor.layout,
            tensor.address,
        )
        seen.session, seen.loaded = tensor.session, tensor.loaded
    seen.written = True
    return seen


def column(tensor):
    """A tile tensor seen as one tile column, its pages in order: an arena's form."""
    return view(tensor, (tensor.pages * TILE, TILE))


class Shared:
    """What every layer of the chip shares: the masks, the insert tables, the swap,
    the job tables of 5 .. 22 key tiles, the rope tables of both kinds of layer (a
    compress ratio of 0 or not) for `positions` positions."""

    def __init__(self, session, positions):
        tensor = lambda v: session.tensor(v, "bfloat16", "tile")  # noqa: E731
        self.positions = positions
        self.masks = tensor(at.mask_values())
        keep, move = wn.insert_tables()
        self.keep, self.move = tensor(keep), tensor(move)
        self.swap = tensor(wn.swap_tiles())
        self.tables = {
            k: at.tables_tensor(session, k) for k in range(5, at.KEY_TILES + 1)
        }
        self.rope = {}
        for compressed in (False, True):
            cos, sin, sin_inverse = wn.tables(wn.frequencies(compressed), positions)
            self.rope[compressed] = (tensor(cos), tensor(sin), tensor(sin_inverse))


class LayerTensors:
    """One layer's weights on the chip (`weights`: each linear an (arena, first page) of
    the loader's, or values (in, out) to store; q_norm, kv_norm, attn_sink values), its
    rope tables, its key list (README.md, "The key list") and the entries' mask."""

    def __init__(self, session, shared, weights, compressed):
        s = session
        tensor = lambda v: s.tensor(v, "bfloat16", "tile")  # noqa: E731
        self.w, self.first = {}, {}
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            if isinstance(weights[name], tuple):
                self.w[name], self.first[name] = weights[name]
            else:
                values = np.asarray(weights[name], np.float32)
                self.w[name] = column(s.tensor(values, WEIGHT_FORMAT, "tile"))
                self.first[name] = 0
        for name in ("q_norm", "kv_norm"):
            pages = pj.norm_pages(weights[name])
            self.w[name] = s.empty((pages.shape[0], 1024), "float32")
            s.write(self.w[name], pages.view(np.uint8).reshape(pages.shape[0], -1))
        self.sinks = tensor(at.sink_values(weights["attn_sink"]))
        self.cos, self.sin, self.sin_inverse = shared.rope[bool(compressed)]
        self.window = tensor(np.zeros((at.KEY_TILES * TILE, pj.HEAD_DIM), np.float32))
        self.entries = tensor(at.entry_values(np.zeros((TILE, 0), bool)))


class Stack:
    """The three programs and the activations of a chip, for any of its layers' tensors
    (the programs are made with `first`'s as examples: every layer's are alike)."""

    def __init__(self, session, shared, first, log=print):
        self.session, self.shared, self.log = session, shared, log
        s = session
        empty = lambda shape: s.empty(shape, "bfloat16", "tile")  # noqa: E731
        heads = (pj.HEADS * TILE, pj.HEAD_DIM)
        a = {
            "x": empty((TILE, pj.DIM)),
            "qa": empty((TILE, pj.Q_RANK)),
            "qr": empty((TILE, pj.Q_RANK)),
            "q": empty((TILE, pj.HEADS * pj.HEAD_DIM)),
            "kv0": empty((TILE, pj.HEAD_DIM)),
            "kv": empty((TILE, pj.HEAD_DIM)),
            "o": empty(heads),
            "oa": empty((TILE, pj.O_GROUPS * pj.O_RANK)),
            "y": empty((TILE, pj.DIM)),
            "turned": empty((pj.HEADS * TILE, wn.ROPE_DIM)),  # rope's x P
            "moved": empty((TILE, pj.HEAD_DIM)),  # the insert's e kv
        }
        a["q_heads"] = view(a["q"], heads)
        a["o_flat"] = view(a["o"], (TILE, pj.HEADS * pj.HEAD_DIM))
        self.a = a
        w = first.w
        self.proj = pj.Projections(
            x=a["x"],
            qr=a["qr"],
            o=a["o_flat"],
            oa=a["oa"],
            w_qa=w["wq_a"],
            w_kv=w["wkv"],
            w_qb=w["wq_b"],
            w_oa=w["wo_a"],
            w_ob=w["wo_b"],
            y=a["qa"],
            g=w["q_norm"],
            out=a["qr"],
            form="runtime",
            session=s,
            log=log,
        )
        self.window = wn.Window(
            t=a["q_heads"],
            swap=shared.swap,
            y=a["turned"],
            cos=first.cos,
            sin=first.sin,
            window=first.window,
            kv=a["kv"],
            keep=shared.keep,
            move=shared.move,
            moved=a["moved"],
            form="runtime",
            session=s,
            log=log,
        )
        self.attend = at.AttendLocal(
            q=a["q_heads"],
            kv=first.window,
            o=a["o"],
            tables=shared.tables[6],
            sinks=first.sinks,
            masks=shared.masks,
            entries=first.entries,
            form="runtime",
            session=s,
            rect=(0, 0, *at.grid(s)),  # the first island's rows (joined: not the grid)
            log=log,
        )
        self.groups = {
            self.proj: (0, at.island_ranks(s)),
            self.window: (0, at.island_ranks(s)),
            self.attend: at.group(s),
        }
        self.run = self.holding = None

    def load(self):
        """The dict's run: the attention on its rectangle, the others on the first
        island's cores left, the projections also on the other islands' ranks; every
        step switches the first island's cores (`step`)."""
        low, count = at.group(self.session)
        rest = at.island_ranks(self.session) - count
        self.run = self.session.load(
            {
                self.attend: (low, count),
                self.window: (count, rest // 2),
                self.proj: (count + rest // 2, self.session.ranks - count - rest // 2),
            }
        )
        self.holding = None
        return self

    def stop(self):
        if self.run is not None:
            self.run.stop()
            self.run = None

    def commands(self, layer, pos, prefill, entries=0, x=None, y=None):
        """The step's commands of `layer` (LayerTensors) in order: (name, program,
        handler, group, words); `entries` selected entries in the key list after the
        window (written there with their mask beforehand); x, y: tensors of the step's
        input and output ([32, 5120] bf16 tile) in place of the stack's own."""
        a, w, sh = dict(self.a), layer.w, self.shared
        a["x"] = self.a["x"] if x is None else x
        a["y"] = self.a["y"] if y is None else y
        ranks = lambda count: (0, count)  # noqa: E731
        columns = pj.HEAD_DIM // TILE
        if prefill:
            ring_row, slot_row = wn.RING_FIRST + pos % wn.WINDOW // TILE, wn.WHOLE
            keys, variant = 6, at.prefill_variant(pos)
        else:
            ring_row, slot_row = wn.ring_row(pos)
            keys, variant = 5, at.decode_variant(pos)
        if entries:
            keys = 1 + at.MASKS + (entries + TILE - 1) // TILE
        steps = [  # the projections, the window's, the attention, the window's, the
            # projections: four switches of program a step
            (
                "wq_a",
                self.proj,
                self.proj.wq_a,
                ranks(pj.cores("wq_a")),
                (a["x"], w["wq_a"], a["qa"], layer.first["wq_a"]),
            ),
            (
                "q_norm",
                self.proj,
                self.proj.norm_q,
                ranks(1),
                (a["qa"], w["q_norm"], a["qr"]),
            ),
            (
                "wq_b",
                self.proj,
                self.proj.wq_b,
                ranks(pj.cores("wq_b")),
                (a["qr"], w["wq_b"], a["q"], layer.first["wq_b"]),
            ),
            (
                "wkv",
                self.proj,
                self.proj.wkv,
                ranks(pj.cores("wkv")),
                (a["x"], w["wkv"], a["kv0"], layer.first["wkv"]),
            ),
            (
                "kv_norm",
                self.proj,
                self.proj.norm_kv,
                ranks(1),
                (a["kv0"], w["kv_norm"], a["kv"]),
            ),
            (
                "q_swap",
                self.window,
                self.window.rope_swap,
                ranks(pj.HEADS),
                (a["q_heads"], sh.swap, a["turned"]),
            ),
            (
                "q_rope",
                self.window,
                self.window.rope,
                ranks(pj.HEADS),
                (a["q_heads"], a["turned"], layer.cos, layer.sin, pos),
            ),
            (
                "kv_swap",
                self.window,
                self.window.rope_swap,
                ranks(1),
                (a["kv"], sh.swap, a["turned"]),
            ),
            (
                "kv_rope",
                self.window,
                self.window.rope,
                ranks(1),
                (a["kv"], a["turned"], layer.cos, layer.sin, pos),
            ),
            (
                "kv_move",
                self.window,
                self.window.insert_move,
                ranks(columns),
                (a["kv"], sh.move, a["moved"], slot_row),
            ),
        ]
        insert_ring = (
            "kv_insert",
            self.window,
            self.window.insert,
            ranks(columns),
            (layer.window, a["moved"], sh.keep, ring_row, slot_row),
        )
        if prefill:  # the block's own tile first, the ring after the attention
            steps.append(
                (
                    "kv_block",
                    self.window,
                    self.window.insert,
                    ranks(columns),
                    (layer.window, a["moved"], sh.keep, wn.BLOCK_ROW, wn.WHOLE),
                )
            )
        else:
            steps.append(insert_ring)
        steps.append(
            (
                "attend",
                self.attend,
                self.attend.attend,
                at.group(self.session),
                at.words(
                    a["q_heads"],
                    layer.window,
                    a["o"],
                    sh.tables[keys],
                    layer.sinks,
                    sh.masks,
                    variant,
                    layer.entries,
                ),
            )
        )
        if prefill:
            steps.append(insert_ring)
        steps += [
            (
                "o_swap",
                self.window,
                self.window.rope_swap,
                ranks(pj.HEADS),
                (a["o"], sh.swap, a["turned"]),
            ),
            (
                "o_rope",
                self.window,
                self.window.rope,
                ranks(pj.HEADS),
                (a["o"], a["turned"], layer.cos, layer.sin_inverse, pos),
            ),
            (
                "wo_a",
                self.proj,
                self.proj.wo_a,
                ranks(pj.cores("wo_a")),
                (a["o_flat"], w["wo_a"], a["oa"], layer.first["wo_a"]),
            ),
            (
                "wo_b",
                self.proj,
                self.proj.wo_b,
                ranks(pj.cores("wo_b")),
                (a["oa"], w["wo_b"], a["y"], layer.first["wo_b"]),
            ),
        ]
        return steps

    def step(self, layer, pos, prefill, entries=0, select=None, x=None, y=None):
        """`layer`'s step at pos, its commands queued at once and waited for at the
        end: {name: microseconds}. `select(qr)` gives the selected entries (rows [m,
        512], seen [32, m]) once q_norm is done, written before the attention."""
        if pos + (TILE if prefill else 1) > self.shared.positions:
            raise ValueError(
                f"position {pos}: the rope tables hold {self.shared.positions}"
            )
        where = f"at {pos}"
        if select is None:
            commands = self.commands(layer, pos, prefill, entries, x, y)
            return self.finish(self.queue(commands), where)
        commands = self.commands(layer, pos, prefill, 0, x, y)
        attend = [c[0] for c in commands].index("attend")
        queued = self.queue(commands[:attend])
        names = [name for name, _ in queued]
        q_norm = queued.pop(names.index("q_norm"))
        spent = self.finish([q_norm], where)
        rows, seen = select(self.read("qr"))  # the step's one look of the host
        self.write_entries(layer, rows, seen)
        later = self.commands(layer, pos, prefill, len(rows), x, y)[attend:]
        spent.update(self.finish(queued + self.queue(later), where))
        return spent

    def queue(self, commands):
        """Commands queued without a wait, each exclusive (after every earlier one on
        the cores it shares: all hold the group's first core), sent to the card every
        STORE_EVERY, a program loaded first where the cores hold another."""
        handles = []
        for name, program, handler, group, words in commands:
            if program is not self.holding:
                load = self.run.load(program, group=self.groups[program])
                handles.append((f"load_{name}", load))
                self.holding = program
            handle = self.run.submit(handler, group, args=words, exclusive=True)
            handles.append((name, handle))
            if (
                len(handles) % STORE_EVERY == 1
            ):  # the card runs while the host queues on
                self.run.store()
        return handles

    def finish(self, handles, where=""):
        """Waits for queued (name, handle)s: {name: microseconds}."""
        events = self.run.wait([h for _, h in handles], timeout=30.0)
        spent = {}
        for (name, _), event in zip(handles, events):
            if event.kind != "done":
                raise km.KomorebiError(f"{name} {where}: {event}")
            spent[name] = event.microseconds
        return spent

    def write_entries(self, layer, rows, seen):
        """Selected entries rows [m <= 512, 512] into `layer`'s key list from key tile
        6 on and their mask seen [32, m] (README.md, "The key list"), live."""
        if len(rows) > at.ENTRY_TILES * TILE:
            raise ValueError(f"{len(rows)} entries: the key list holds 512")
        if len(rows):
            tiles = -(-len(rows) // TILE)
            block = np.zeros((tiles * TILE, pj.HEAD_DIM), np.float32)
            block[: len(rows)] = rows
            raw, _ = formats.to_bytes(block, layer.window.dtype, "tile")
            pages = np.asarray(raw).reshape(tiles * pj.HEAD_DIM // TILE, -1)
            first = (1 + at.MASKS) * pj.HEAD_DIM // TILE
            self.session.write(layer.window, pages, first)
        self.write(layer.entries, at.entry_values(np.asarray(seen, bool)))

    def submit(self, commands, where=""):
        """Commands (name, program, handler, group, words) each waited for, each
        program loaded on its group first when the cores hold another: {name:
        microseconds}."""
        spent = {}
        for name, program, handler, group, words in commands:
            if program is not self.holding:
                event = self.run.wait(
                    self.run.load(program, group=self.groups[program]), timeout=30.0
                )
                if event.kind != "done":
                    raise km.KomorebiError(f"load before {name} {where}: {event}")
                spent[f"load_{name}"] = event.microseconds
                self.holding = program
            event = self.run.wait(
                self.run.submit(handler, group, args=words), timeout=30.0
            )
            if event.kind != "done":
                raise km.KomorebiError(f"{name} {where}: {event}")
            spent[name] = event.microseconds
        return spent

    def write(self, tensor, values):
        """values [rows, columns <= the tensor's] into the tensor's top left, zeros
        elsewhere, live (the run's transfer): the host microseconds."""
        if isinstance(tensor, str):
            tensor = self.a[tensor]
        full = np.zeros(tensor.shape, np.float32)
        values = np.asarray(values, np.float32)
        full[: values.shape[0], : values.shape[1]] = values
        started = time.monotonic()
        raw, _ = formats.to_bytes(full, tensor.dtype, "tile")
        self.session.write(tensor, np.asarray(raw).reshape(tensor.pages, -1))
        return (time.monotonic() - started) * 1e6

    def write_x(self, x):
        """x [tokens <= 32, 5120] into the command's row tile (zeros below)."""
        return self.write("x", x)

    def read(self, name):
        """An activation's (or a tensor's) values (float32), live."""
        tensor = self.a[name] if isinstance(name, str) else name
        (raw,) = self.session.read(tensor)
        return np.asarray(
            formats.from_bytes(
                np.asarray(raw).reshape(-1), tensor.shape, "bfloat16", "tile"
            ),
            np.float32,
        )


class Layer(Stack):
    """One layer on its own chip: its tensors and the stack made for them."""

    def __init__(self, session, shared, weights, compressed, log=print):
        self.tensors = LayerTensors(session, shared, weights, compressed)
        super().__init__(session, shared, self.tensors, log=log)

    def step(self, pos, prefill, entries=0):
        return super().step(self.tensors, pos, prefill, entries)

    def decode(self, pos):
        """One token (x's row 0) at pos: y's row 0, the window extended by its key."""
        return self.step(pos, False)

    def prefill(self, start):
        """A block of 32 tokens at start (a multiple of 32): y's rows, the window
        extended by its keys."""
        return self.step(start, True)
