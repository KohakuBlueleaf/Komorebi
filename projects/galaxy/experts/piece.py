"""Pieces 13, 14 and 15 in the driver's call interface: `make(session, W, layers)`,
then `piece(site, layer, *inputs)` for the sites gate, shared, routed and ffn. The
router and the shared expert run on the layer's home island, each routed expert on the
island that holds it; one load (the tree, moe.Moe added) kept across calls. README.md.
"""

import json
import os
import time

import numpy as np
import torch

import komorebi as km
from komorebi import formats
from projects.galaxy.experts import moe, tree

D, F, NE, TOP, SCALE = 5120, 2304, 384, 6, 1.5
SENDERS = 4  # the tree's cores an island
FIELDS = dict(SPLIT=5, SPLIT_U=2, WU=4, GQ=4, SPLIT_D=2, WD=5)  # PG: as fits
# the shared expert's copy: bfp8_b (compact) or bfloat16 (exact)
SHARED_FORMAT = os.environ.get("EXPERTS_SHARED_FORMAT", "bfp8_b")
PARTS = ("w1", "w3", "w2")
RELAY = [0, 11520, 23040]  # an island with no expert: its arena's first pages, weight 0


def _tiles(values, dtype):
    raw, _ = formats.to_bytes(np.ascontiguousarray(values), dtype, "tile")
    return np.asarray(raw)


def _rows(words):
    return np.asarray(formats.to_bytes(words, "uint32", "row")[0]).reshape(
        len(words), -1
    )


def _block(x):
    """A host [s, 5120] tensor as one tile row of 32 tokens (zero rows past s)."""
    out = np.zeros((32, D), np.float32)
    out[: x.shape[0]] = x.float().numpy()
    return _tiles(out, "bfloat16").reshape(160, -1)


class MoePiece:
    sites = ("gate", "shared", "routed", "ffn")

    def __init__(self, session, weights, layers, log=None):
        self.s, self.W, self.layers = session, weights, layers
        self.log = log or (lambda message: None)
        self.count = len(session.sessions)
        self.graph = session.graph
        self.calls = []
        ranks = min(self.graph.count(i) for i in range(self.count))
        held = {}  # the most routed experts an island holds a layer (13 on 31 islands)
        for layer in [one for one in layers if one is not None]:
            for e in range(NE):
                island = weights[f"layers.{layer}.ffn.experts.{e}.w1@bfp4_b"].island
                held[layer, island] = held.get((layer, island), 0) + 1
        self.e_most = e_most = max(held.values(), default=13)
        self.fields = {**FIELDS, "PG": min(e_most, ranks // FIELDS["GQ"])}

        def empty(shape, dtype, form="tile"):
            return session.empty(shape, dtype, form, copy=True)

        self.tx, self.ty = empty((32, D), "bfloat16"), empty((32, D), "bfloat16")
        self.tsum, self.tsy = empty((32, D), "bfloat16"), empty((32, D), "bfloat16")
        self.tl = empty((32, NE), "float32")
        self.hg = empty((32, e_most * 2 * F), "bfloat16")
        self.hid = empty((32, e_most * F), "bfloat16")
        self.table = empty((e_most, 32), "uint32", "row")
        self.stable = empty((e_most, 32), "uint32", "row")
        self.f = empty((32, e_most * F), "bfloat16")
        self.f1 = empty((32, e_most * F), "bfloat16")
        ones = _tiles(np.ones((32, e_most * F), np.float32), "bfloat16")
        for i in range(self.count):
            self._write(self.f1, i, ones.reshape(self.f1.pages, -1))
        # made with small copies of the arenas' forms (a near build needs copies); each
        # command names the arena of its own island
        dtypes = {"wg": "bfloat16", "w": "bfp4_b", "w8": SHARED_FORMAT}
        example = {k: empty((64 * 32, 32), d) for k, d in dtypes.items()}
        self.ex4, self.ex8 = (
            example["w"],
            example["w8"],
        )  # the unused weight of a command
        self.program = moe.Moe(
            session=session,
            form="runtime",
            **self.fields,
            x=self.tx,
            logits=self.tl,
            table=self.table,
            hg=self.hg,
            f=self.f,
            hidden=self.hid,
            y=self.ty,
            **example,
        )
        self.run, self.on, self.trees, self.bias = None, None, {}, {}
        for home in range(self.count):  # the tree from each island the home
            self.trees[home] = tree.bfs_tree(self.graph, self.count, home)
        kids = max(tree.children_most(t, h) for h, t in self.trees.items())
        self.tree = tree.program(SENDERS, kids, self.tx)
        self.tree.ahead(self.program)  # Moe's parts built at the load
        self.tree.slots(count=1, service="shared")

    def _write(self, tensor, island, pages):
        self.s.sessions[island].write(tensor.parts[island], np.ascontiguousarray(pages))

    def _read(self, tensor, island, shape, dtype):
        (raw,) = self.s.sessions[island].read(tensor.parts[island])
        raw = np.asarray(raw).reshape(-1)
        return np.asarray(formats.from_bytes(raw, shape, dtype, "tile"))

    def _use(self, program, times):
        """The one load (at the first call: the tree the dict's program, Moe added),
        switched to `program` when the cores hold the other."""
        if self.run is None:
            t0 = time.monotonic()
            self.run = self.s.load({self.tree: (0, self.s.ranks)})
            self.run.addon(self.program)
            self.on = self.tree
            times["loads_s"] = times.get("loads_s", 0.0) + time.monotonic() - t0
        if self.on is not program:
            t0 = time.monotonic()
            ticket = self.run.load(program, (0, self.s.ranks))
            event = self.run.wait(ticket, timeout=30.0)
            parts = event if isinstance(event, tuple) else (event,)
            if any(part.kind != "done" for part in parts):
                raise km.KomorebiError(f"a switch: {event}")
            self.on = program
            times["switches"] = times.get("switches", 0) + 1
            times["switch_s"] = times.get("switch_s", 0.0) + time.monotonic() - t0
        return self.run

    def _stop(self):
        if self.run is not None:
            run, self.run, self.on = self.run, None, None
            run.stop()

    def close(self):
        self._stop()

    def _commands(self, step, handler, islands, args_of, cores, times):
        """One command on each of `islands` (args_of(island)); the slowest card time."""
        run = self._use(self.program, times)
        tickets = []
        for island in islands:
            group = (self.graph.first(island), cores)
            tickets.append(run.submit(handler, group, args=args_of(island)))
        worst = 0.0
        for ticket in tickets:
            event = run.wait(ticket, timeout=60.0)
            parts = event if isinstance(event, tuple) else (event,)
            if any(part.kind != "done" for part in parts):
                raise km.KomorebiError(f"{step}: {event}")
            worst = max([worst] + [part.microseconds for part in parts])
        return round(worst, 2)

    def _home(self, layer):
        return self.W[f"layers.{layer}.ffn.gate@bfloat16"].island

    def _timed_write(self, times, writes):
        t0 = time.monotonic()
        for tensor, island, pages in writes:
            self._write(tensor, island, pages)
        times["writes_s"] = times.get("writes_s", 0.0) + time.monotonic() - t0

    def _bias(self, layer):
        if layer not in self.bias:
            h = self.W[f"layers.{layer}.ffn.gate.bias@float32"]
            part = h.arena.parts[h.island]
            (raw,) = self.s.sessions[h.island].read(part, first=h.first, pages=h.pages)
            raw = np.asarray(raw).reshape(-1)
            row = formats.from_bytes(raw, (32, NE), "float32", "tile")
            self.bias[layer] = np.asarray(row)[0].astype(np.float32)
        return self.bias[layer]

    def __call__(self, site, layer, *inputs):
        times = {"site": site, "layer": layer}
        t0 = time.monotonic()
        shape = inputs[0].shape
        x = inputs[0].reshape(-1, shape[-1])  # the ffn site's x is [1, s, 5120]
        n = int(x.shape[0])
        times["tokens"] = n
        block = _block(x)
        if site == "gate":
            out = self.gate(layer, block, n, times)
        elif site == "shared":
            out = self.shared(layer, block, n, times)
        elif site == "routed":
            out = self.routed(layer, block, n, *inputs[1:], times=times)
        elif site == "ffn":
            w, i = self.gate(layer, block, n, times)
            y = self.routed(layer, block, n, w, i, times=times)
            y = y + self.shared(layer, block, n, times).float()
            out = y.to(x.dtype).reshape(shape)
        else:
            raise KeyError(site)
        times["wall_s"] = round(time.monotonic() - t0, 4)
        for k in ("loads_s", "writes_s", "switch_s"):
            times[k] = round(times.get(k, 0.0), 4)
        self.calls.append(times)
        self.log(times)
        return out

    def gate(self, layer, block, n, times):
        home = self._home(layer)
        h = self.W[f"layers.{layer}.ffn.gate@bfloat16"]
        self._timed_write(times, [(self.tx, home, block)])
        p, c = self.program, moe.cores(self.fields, 1)
        args = (self.tx, h.arena, self.tl, h.first)
        route = self._commands(
            "route", p.route, [home], lambda i: args, c["route"], times
        )
        times["card_us"] = {"route": route}
        logits = self._read(self.tl, home, (32, NE), "float32")[:n].astype(np.float32)
        # score = sqrt(softplus(logit)), float32, on the host
        scores = np.sqrt(np.logaddexp(np.float32(0), logits)).astype(np.float32)
        idx = np.argsort(-(scores + self._bias(layer)), axis=1, kind="stable")[:, :TOP]
        chosen = np.take_along_axis(scores, idx, axis=1)
        w = chosen / (chosen.sum(axis=1, keepdims=True) + np.float32(1e-20))
        w = (w * np.float32(SCALE)).astype(np.float32)
        return torch.from_numpy(w), torch.from_numpy(idx.astype(np.int64))

    def shared(self, layer, block, n, times):
        home = self._home(layer)
        name = f"layers.{layer}.ffn.shared_experts"
        hs = [self.W[f"{name}.{m}@{SHARED_FORMAT}"] for m in PARTS]
        row = np.zeros((self.e_most, 32), np.uint32)
        row[0, :3] = [h.first for h in hs]
        writes = [(self.tx, home, block), (self.stable, home, _rows(row))]
        self._timed_write(times, writes)
        p, c, arena = self.program, moe.cores(self.fields, 1), hs[0].arena
        steps = (
            ("up", p.up, (self.tx, self.ex4, arena, self.stable, self.hg, 1, 1)),
            ("gate", p.gate, (self.hg, self.f1, self.hid, 1)),
            ("down", p.down, (self.hid, self.ex4, arena, self.stable, self.tsy, 1, 1)),
        )
        us = {}
        for step, handler, args in steps:
            us[step] = self._commands(
                step, handler, [home], lambda i, a=args: a, c[step], times
            )
        times["card_us_shared"] = us
        y = self._read(self.tsy, home, (32, D), "bfloat16")[:n]
        return torch.from_numpy(np.ascontiguousarray(y, np.float32)).to(torch.bfloat16)

    def _lists(self, layer, idx):
        """{island: [(expert, [first w1, w3, w2])]} of the experts the tokens chose."""
        lists = {}
        for e in sorted(set(int(v) for v in idx.reshape(-1))):
            name = f"layers.{layer}.ffn.experts.{e}"
            hs = [self.W[f"{name}.{m}@bfp4_b"] for m in PARTS]
            lists.setdefault(hs[0].island, []).append((e, [h.first for h in hs]))
        return lists

    def _tables(self, idx, wts, mine, e_list):
        """An island's table of list places and its weights a token (a short list padded
        with its last expert at weight 0; a relay, RELAY)."""
        table = np.zeros((self.e_most, 32), np.uint32)
        fw = np.zeros((32, self.e_most * F), np.float32)
        for pos in range(e_list):
            table[pos, :3] = mine[min(pos, len(mine) - 1)][1] if mine else RELAY
            if pos < len(mine):
                tok, k = np.nonzero(idx == mine[pos][0])
                fw[tok, pos * F : (pos + 1) * F] = wts[tok, k][:, None]
        return _rows(table), _tiles(fw, "bfloat16").reshape(self.f.pages, -1)

    def _tree(self, home, parent, phase, words, times):
        run = self._use(self.tree, times)
        us = tree.go(self.s, run, self.tree, parent, home, phase, SENDERS, *words)
        return round(us, 2)

    def routed(self, layer, block, n, weights, indices, times):
        home = self._home(layer)
        idx = indices.numpy().astype(np.int64)
        wts = weights.numpy().astype(np.float32)
        lists = self._lists(layer, idx)
        e_list = max(len(v) for v in lists.values())
        parent = tree.pruned(self.trees[home], home, list(lists))
        islands = list(parent)
        times["depth"] = tree.depth(parent, home)
        times.update(islands=len(islands), experts=sum(map(len, lists.values())))
        times["e_list"] = e_list
        writes = []
        for island in islands:
            table, fw = self._tables(idx, wts, lists.get(island, []), e_list)
            writes += [(self.table, island, table), (self.f, island, fw)]
        writes.append((self.tx, home, block))
        self._timed_write(times, writes)
        card = {}
        phase = tree.SYNC | tree.DOWN | tree.ACK
        card["down"] = self._tree(home, parent, phase, (self.tx.address,), times)
        p, c = self.program, moe.cores(self.fields, e_list)
        arena = {i: self.W.arenas[(i, "bfp4_b")] for i in islands}
        steps = (
            (
                "up",
                p.up,
                lambda i: (self.tx, arena[i], self.ex8, self.table, self.hg, e_list, 0),
            ),
            ("gate", p.gate, lambda i: (self.hg, self.f, self.hid, e_list)),
            (
                "down",
                p.down,
                lambda i: (
                    self.hid,
                    arena[i],
                    self.ex8,
                    self.table,
                    self.ty,
                    e_list,
                    0,
                ),
            ),
        )
        for step, handler, args_of in steps:
            us = self._commands(step, handler, islands, args_of, c[step], times)
            card[f"experts_{step}"] = us
        phase = tree.SYNC | tree.GO | tree.UP
        words = (self.ty.address, self.tsum.address)
        card["up"] = self._tree(home, parent, phase, words, times)
        y = self._read(self.tsum, home, (32, D), "bfloat16")[:n].astype(np.float32)
        times["card_us"] = card
        return torch.from_numpy(np.ascontiguousarray(y))


def make(session, weights, layers, log=None):
    if log is None and os.environ.get("EXPERTS_LOG"):  # each call's times as JSON

        def log(times):
            print("experts", json.dumps(times), flush=True)

    return MoePiece(session, weights, layers, log=log)
