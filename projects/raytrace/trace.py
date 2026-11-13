"""The frame with the card as the intersector: the host makes rays and shades with the
reference's own functions; every trace (primary, shadow, bounce, shadow at the bounce
hit) is one TRACE record on the card, checked against the reference's trace.
"""

import time

import numpy as np

from projects.raytrace import card, packets
from ref import bvh, raytrace

K = packets.K
F = np.float32
SORTED = ("bounce", "shadow2")  # incoherent passes; primary and shadow keep pixel order


def leaf_of(tree):
    """Leaf number of each scene triangle (the inverse of leaf_tri)."""
    out = np.empty(len(tree["leaf_tri"]), dtype=np.int32)
    out[tree["leaf_tri"]] = np.arange(len(out), dtype=np.int32)
    return out


class CardTracer:
    """Traces rays in an open session; keeps per-pass timings, profiles and checks."""

    def __init__(
        self,
        session,
        tree,
        cores=0,
        check=True,
        reps=1,
        cut=None,
        slots=4,
        entries=0,
        policy=K["kPolicyOnce"],
        balance=False,
        sort="none",
        per_record=0,
    ):
        self.session = session
        self.tree = tree
        self.cut = cut
        if cut is None:
            self.image = packets.tree_image(tree)
        else:
            self.treelets, table = packets.treelet_images(tree, cut)
            self.image = packets.top_image(cut, table)
            self.slot_bytes = int(table[:, 1].max()) * K["kTreePageBytes"]
            self.n_treelets = len(table)
        self.slots = slots
        self.entries = entries
        self.policy = policy
        self.least = 16  # packets per record the planner keeps before dropping slots
        self.per_record = per_record  # packets per core and record at most (0: plan)
        self.balance = balance
        self.sort = sort  # none, tile or pass: SORTED passes by coherence_key
        lo, hi = bvh.node_boxes(tree["nodes"][:1])
        self.box = lo[0].astype(np.float64), hi[0].astype(np.float64)
        self.tile_cost = {}  # pass name: cycles per tile, from the last trace
        self.frame = 0
        self.leaf = leaf_of(tree)
        self.workers = card.workers(session)
        self.cores = cores or self.workers
        self.check = check
        self.reps = reps
        self.power = None  # a komorebi.power.PowerSampler around the next program
        self.passes = []

    def trace(self, name, o, d, tmax, any_hit, pixels, layout=None):
        """The reference's trace() result for rays (o, d), computed on the card.
        `layout` (lanes, directory, tiles) replaces the split by tile."""
        info = {"pass": name, "rays": len(o), "any_hit": any_hit, "frame": self.frame}
        t0 = time.perf_counter()
        cost = self.tile_cost.get(name) if self.balance else None
        info["split"] = "cost" if cost is not None else "packets"
        info["det_bound"] = packets.check_det_range(self.tree["tris"], d)
        key = None
        if self.sort != "none" and name in SORTED:
            key = packets.coherence_key(o, d, *self.box)
        info["sort"] = self.sort if key is not None else "none"
        within = self.sort == "tile"
        if layout is None:
            layout = packets.split(pixels, self.cores, cost, key, within)
        lanes, directory, tiles = layout
        full = np.zeros((self.workers, directory.shape[1]), dtype=np.uint32)
        full[: self.cores] = directory
        pages = packets.packet_pages(lanes, o, d, tmax)
        info["host_pack_s"] = time.perf_counter() - t0
        mode = K["kModeAny"] if any_hit else K["kModeNearest"]
        seg, args, extra, sizes = K["kSegTrace"], [mode], (), None
        if self.cut is not None:
            most = int(directory[:, K["kDirCount"]].max())
            seg, extra = K["kSegTreelets"], (self.treelets,)
            per, slots, sizes = packets.plan_records(
                most,
                self.n_treelets,
                self.slot_bytes,
                len(self.image),
                self.slots,
                entries=self.entries,
                least=self.least,
            )
            if self.per_record and self.per_record < per:
                per = self.per_record
                sizes = packets.treelet_buffers(
                    per, sizes["entries"], self.n_treelets, slots, self.slot_bytes
                )
            args = [mode, sizes["entries"], self.slot_bytes, slots, self.policy]
            info["l1"] = {**sizes, "packets_per_record": per, "slot_count": slots}
            full = packets.chunks(full, per)
            info["records_per_trace"] = len(full)
        out, prof, res, times, power = card.run_segment(
            self.session,
            seg,
            full,
            pages,
            -(-len(pages) // K["kHitPagePackets"]),
            K["kHitPagePackets"] * K["kHitVectors"] * packets.VEC,
            tree=self.image,
            args=args,
            reps=self.reps,
            extra=extra,
            sizes=sizes,
            power=self.power,
        )
        if self.power is not None:
            info["power"] = self.power.result.as_dict()
        info["run"] = card.check_run(res, self.workers)
        info["times"] = times
        us = card.record_us(res)
        info["load_us"] = float(us[0])
        # A trace is its records in a row: their card times summed per round.
        per_trace = len(full) if isinstance(full, list) else 1
        info["trace_us"] = [float(x) for x in us[1:].reshape(-1, per_trace).sum(axis=1)]
        info["packets"] = len(pages)
        t0 = time.perf_counter()
        out = out.reshape(-1, K["kHitVectors"] * packets.VEC)[: len(pages)]
        hits = packets.decode_hits(out, lanes, len(o))
        info["host_decode_s"] = time.perf_counter() - t0
        cycles = hits.pop("packet_cycles")
        self.tile_cost[name] = np.bincount(tiles, cycles, minlength=packets.TILES)
        info["profile"] = card.profile_rows(prof, self.workers)
        info["walk_errors"] = int((info["profile"]["error"] != 0).sum())
        info["lanes"], info["tiles"] = lanes, tiles
        valid = hits["tri"] >= 0
        hit = {
            "t": hits["t"],
            "leaf": np.where(valid, self.leaf[np.maximum(hits["tri"], 0)], -1),
            "u": hits["u"],
            "v": hits["v"],
            "nodes": hits["nodes"],
            "leaves": hits["leaves"],
        }
        if self.check:
            t0 = time.perf_counter()
            card.log(f"{name}: reference trace of {len(o)} rays")
            ref = raytrace.trace(self.tree, o, d, tmax, any_hit)
            info["ref_s"] = time.perf_counter() - t0
            info["check"] = agreement(ref, hit, self.tree, any_hit)
        self.passes.append(info)
        card.log(f"{name}: {len(o)} rays, card {us[-1]:.1f} us, {info.get('check')}")
        return hit


def agreement(ref, got, tree, any_hit):
    """How the card's hits differ from the reference's for the same rays."""
    if any_hit:
        a, b = ref["leaf"] >= 0, got["leaf"] >= 0
        return {
            "rays": int(a.size),
            "occluded": int(a.sum()),
            "differ": int((a != b).sum()),
        }
    tri_a = np.where(ref["leaf"] >= 0, tree["leaf_tri"][np.maximum(ref["leaf"], 0)], -1)
    tri_b = np.where(got["leaf"] >= 0, tree["leaf_tri"][np.maximum(got["leaf"], 0)], -1)
    both = (tri_a >= 0) & (tri_b >= 0)
    tri = tri_a != tri_b
    tie = tri & both & (ref["t"] == got["t"])
    gap = np.abs(ref["t"].astype(np.float64) - got["t"])
    near = tri & both & ~tie & (gap <= 1e-5 * np.maximum(ref["t"], got["t"]))
    miss = tri & ((tri_a >= 0) != (tri_b >= 0))
    same = ~tri
    return {
        "rays": int(tri.size),
        "hits": int((tri_a >= 0).sum()),
        "tri_differs": int(tri.sum()),
        "exact_ties": int(tie.sum()),
        "near_ties": int(near.sum()),
        "hit_vs_miss": int(miss.sum()),
        "other": int((tri & ~tie & ~near & ~miss).sum()),
        "t_differs": int(
            (same & (ref["t"].view(np.uint32) != got["t"].view(np.uint32))).sum()
        ),
        "uv_differs": int(
            (same & ((ref["u"] != got["u"]) | (ref["v"] != got["v"]))).sum()
        ),
        "box_tests_ref": int(ref["nodes"].sum()),
        "box_tests_card": int(got["nodes"].sum()),
        "tri_tests_ref": int(ref["leaves"].sum()),
        "tri_tests_card": int(got["leaves"].sum()),
    }


def render_pixels(scene, tree, pixels, tracer):
    """The reference's render_pixels with every trace on the card (same arithmetic)."""
    o, d = raytrace.camera_rays(scene["camera"], pixels)
    hit = tracer.trace("primary", o, d, raytrace.TMAX, False, pixels)
    colour = raytrace.sky(d)
    h = np.nonzero(hit["leaf"] >= 0)[0]
    if len(h):
        sub = {k: v[h] for k, v in hit.items()}
        p = o[h] + d[h] * sub["t"][:, None]
        n, albedo = raytrace.surface(scene, tree, sub, d[h])
        o2, direct = direct_light(scene, p, n, tracer, "shadow", pixels[h])
        db = raytrace.bounce_directions(pixels[h], n)
        bounce = tracer.trace("bounce", o2, db, raytrace.TMAX, False, pixels[h])
        radiance = raytrace.sky(db)
        b = np.nonzero(bounce["leaf"] >= 0)[0]
        if len(b):
            sub1 = {k: v[b] for k, v in bounce.items()}
            p1 = o2[b] + db[b] * sub1["t"][:, None]
            n1, albedo1 = raytrace.surface(scene, tree, sub1, db[b])
            _, direct1 = direct_light(scene, p1, n1, tracer, "shadow2", pixels[h][b])
            radiance[b] = albedo1 * (raytrace.AMBIENT + direct1)[:, None]
        light = ((raytrace.AMBIENT + direct)[:, None]) + raytrace.GI * radiance
        colour[h] = albedo * light
    return colour, hit


def direct_light(scene, p, n, tracer, name, pixels):
    """The reference's direct_light with the shadow rays traced on the card."""
    o = p + n * scene["eps"][0]
    to_light = scene["light"] - o
    shadow = tracer.trace(name, o, to_light, F(1.0), True, pixels)["leaf"] >= 0
    cos = raytrace.dot(n, to_light) * (
        F(1.0) / np.sqrt(raytrace.dot(to_light, to_light))
    )
    return o, np.where(shadow, F(0), np.maximum(cos, F(0)))
