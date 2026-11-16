"""The frame with the card as the intersector: the host makes rays and shades with the
reference's own functions; every trace (primary, shadow, bounce, shadow at the bounce
hit) is one TRACE command on the card, checked against the reference's trace.
"""

import time

import numpy as np

from projects.raytrace import card, packets
from reference import bvh, raytrace

K = packets.K
F = np.float32
SORTED = ("bounce", "shadow2")  # incoherent passes; primary and shadow keep pixel order


def leaf_of(tree):
    """Leaf number of each scene triangle (the inverse of leaf_tri)."""
    leaves = np.empty(len(tree["leaf_tri"]), dtype=np.int32)
    leaves[tree["leaf_tri"]] = np.arange(len(leaves), dtype=np.int32)
    return leaves


class CardTracer:
    """Traces rays in an open session; keeps per-pass timings, profiles and checks."""

    def __init__(
        self,
        session,
        tree,
        cores=0,
        check=True,
        repetitions=1,
        cut=None,
        slots=4,
        entries=0,
        policy=K["kPolicyOnce"],
        balance=False,
        sort="none",
        per_command=0,
        scene=None,
    ):
        self.session = session
        self.tree = tree
        self.cut = cut
        self.scene = scene  # the triangles carry their shading attributes when given
        if cut is None:
            self.image = packets.tree_image(tree, scene)
        else:
            self.treelets, table = packets.treelet_images(tree, cut, scene)
            self.image = packets.top_image(cut, table)
            self.slot_bytes = int(table[:, 1].max()) * K["kTreePageBytes"]
            self.treelet_count = len(table)
        self.slots = slots
        self.entries = entries
        self.policy = policy
        self.least = 16  # packets per command the planner keeps before dropping slots
        self.per_command = per_command  # packets per core and command at most (0: plan)
        self.balance = balance
        self.sort = sort  # none, tile or pass: SORTED passes by coherence_key
        box_min, box_max = bvh.node_boxes(tree["nodes"][:1])
        self.box = box_min[0].astype(np.float64), box_max[0].astype(np.float64)
        self.tile_cost = {}  # pass name: cycles per tile, from the last trace
        self.frame = 0
        self.leaf = leaf_of(tree)
        self.workers = card.workers(session)
        self.cores = cores or self.workers
        self.check = check
        self.repetitions = repetitions
        self.power = None  # a komorebi.bench.power.PowerSampler around the next program
        self.passes = []

    def trace(self, name, origins, directions, tmax, any_hit, pixels, layout=None):
        """The reference's trace() result for rays (o, d), computed on the card.
        `layout` (lanes, directory, tiles) replaces the split by tile."""
        info = {
            "pass": name,
            "rays": len(origins),
            "any_hit": any_hit,
            "frame": self.frame,
        }
        started = time.perf_counter()
        cost = self.tile_cost.get(name) if self.balance else None
        info["split"] = "cost" if cost is not None else "packets"
        info["det_bound"] = packets.check_det_range(self.tree["tris"], directions)
        key = None
        if self.sort != "none" and name in SORTED:
            key = packets.coherence_key(origins, directions, *self.box)
        info["sort"] = self.sort if key is not None else "none"
        within_tiles = self.sort == "tile"
        if layout is None:
            layout = packets.split(pixels, self.cores, cost, key, within_tiles)
        lanes, directory, tiles = layout
        padded = np.zeros((self.workers, directory.shape[1]), dtype=np.uint32)
        padded[: self.cores] = directory
        pages = packets.packet_pages(lanes, origins, directions, tmax)
        info["host_pack_s"] = time.perf_counter() - started
        mode = K["kModeAny"] if any_hit else K["kModeNearest"]
        handler, args, extra, sizes = K["kHandlerTrace"], [mode], (), None
        if self.cut is not None:
            most = int(directory[:, K["kDirCount"]].max())
            handler, extra = K["kHandlerTreelets"], (self.treelets,)
            packets_per_command, slots, sizes = packets.plan_commands(
                most,
                self.treelet_count,
                self.slot_bytes,
                len(self.image),
                self.slots,
                entries=self.entries,
                least=self.least,
            )
            if self.per_command and self.per_command < packets_per_command:
                packets_per_command = self.per_command
                sizes = packets.treelet_buffers(
                    packets_per_command,
                    sizes["entries"],
                    self.treelet_count,
                    slots,
                    self.slot_bytes,
                )
            args = [mode, sizes["entries"], self.slot_bytes, slots, self.policy]
            info["l1"] = {
                **sizes,
                "packets_per_command": packets_per_command,
                "slot_count": slots,
            }
            padded = packets.chunks(padded, packets_per_command)
            info["commands_per_trace"] = len(padded)
        vectors = packets.hit_vectors(any_hit)
        output, profile, ran, times, power = card.run_handler(
            self.session,
            handler,
            padded,
            packets.group(pages),
            -(-len(pages) // K["kPagePackets"]),
            K["kPagePackets"] * vectors * packets.VEC,
            tree=self.image,
            args=args,
            repetitions=self.repetitions,
            extra=extra,
            sizes=sizes,
            power=self.power,
        )
        if self.power is not None:
            info["power"] = self.power.result.as_dict()
        info["run"] = card.check_run(ran, self.workers)
        info["times"] = times
        microseconds = card.command_microseconds(ran)
        info["load_microseconds"] = float(microseconds[0])
        # A trace is its commands in a row: their card times summed per round.
        per_trace = len(padded) if isinstance(padded, list) else 1
        info["trace_microseconds"] = [
            float(value)
            for value in microseconds[1:].reshape(-1, per_trace).sum(axis=1)
        ]
        info["packets"] = len(pages)
        started = time.perf_counter()
        output = packets.ungroup(output, vectors, len(pages))
        hits = packets.decode_hits(output, lanes, len(origins))
        attributes = hits.pop("attrs", None)
        info["host_decode_s"] = time.perf_counter() - started
        cycles = hits.pop("packet_cycles")
        self.tile_cost[name] = np.bincount(tiles, cycles, minlength=packets.TILES)
        info["profile"] = card.profile_rows(profile, self.workers)
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
            started = time.perf_counter()
            card.log(f"{name}: reference trace of {len(origins)} rays")
            reference = raytrace.trace(self.tree, origins, directions, tmax, any_hit)
            info["ref_s"] = time.perf_counter() - started
            info["check"] = agreement(reference, hit, self.tree, any_hit)
            if attributes is not None and self.scene is not None:
                info["check"]["attr_differs"] = attr_differs(
                    self.scene, self.tree, hit, attributes
                )
        self.passes.append(info)
        card.log(
            f"{name}: {len(origins)} rays, card {microseconds[-1]:.1f} us, "
            f"{info.get('check')}"
        )
        return hit


def attr_differs(scene, tree, hit, attributes):
    """Hits whose carried normal and colours are not their triangle's, bit for bit."""
    rows = hit["leaf"] >= 0
    want = packets.tri_attrs(scene, tree, hit["leaf"][rows])
    return int((packets.bits(attributes[rows]) != packets.bits(want)).any(axis=1).sum())


def agreement(reference, got, tree, any_hit):
    """How the card's hits differ from the reference's for the same rays."""
    if any_hit:
        reference_hit, card_hit = reference["leaf"] >= 0, got["leaf"] >= 0
        return {
            "rays": int(reference_hit.size),
            "occluded": int(reference_hit.sum()),
            "differ": int((reference_hit != card_hit).sum()),
        }
    reference_triangle = np.where(
        reference["leaf"] >= 0, tree["leaf_tri"][np.maximum(reference["leaf"], 0)], -1
    )
    card_triangle = np.where(
        got["leaf"] >= 0, tree["leaf_tri"][np.maximum(got["leaf"], 0)], -1
    )
    both = (reference_triangle >= 0) & (card_triangle >= 0)
    differs = reference_triangle != card_triangle
    tie = differs & both & (reference["t"] == got["t"])
    gap = np.abs(reference["t"].astype(np.float64) - got["t"])
    near = differs & both & ~tie & (gap <= 1e-5 * np.maximum(reference["t"], got["t"]))
    miss = differs & ((reference_triangle >= 0) != (card_triangle >= 0))
    same = ~differs
    return {
        "rays": int(differs.size),
        "hits": int((reference_triangle >= 0).sum()),
        "tri_differs": int(differs.sum()),
        "exact_ties": int(tie.sum()),
        "near_ties": int(near.sum()),
        "hit_vs_miss": int(miss.sum()),
        "other": int((differs & ~tie & ~near & ~miss).sum()),
        "t_differs": int(
            (same & (reference["t"].view(np.uint32) != got["t"].view(np.uint32))).sum()
        ),
        "uv_differs": int(
            (same & ((reference["u"] != got["u"]) | (reference["v"] != got["v"]))).sum()
        ),
        "box_tests_ref": int(reference["nodes"].sum()),
        "box_tests_card": int(got["nodes"].sum()),
        "tri_tests_ref": int(reference["leaves"].sum()),
        "tri_tests_card": int(got["leaves"].sum()),
    }


def render_pixels(scene, tree, pixels, tracer):
    """The reference's render_pixels with every trace on the card (same arithmetic)."""
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    hit = tracer.trace("primary", origins, directions, raytrace.TMAX, False, pixels)
    colour = raytrace.sky(directions)
    hit_rows = np.nonzero(hit["leaf"] >= 0)[0]
    if len(hit_rows):
        hits = {key: values[hit_rows] for key, values in hit.items()}
        points = origins[hit_rows] + directions[hit_rows] * hits["t"][:, None]
        normals, albedo = raytrace.surface(scene, tree, hits, directions[hit_rows])
        shadow_origins, direct = direct_light(
            scene, points, normals, tracer, "shadow", pixels[hit_rows]
        )
        bounce_directions = raytrace.bounce_directions(pixels[hit_rows], normals)
        bounce = tracer.trace(
            "bounce",
            shadow_origins,
            bounce_directions,
            raytrace.TMAX,
            False,
            pixels[hit_rows],
        )
        radiance = raytrace.sky(bounce_directions)
        bounce_rows = np.nonzero(bounce["leaf"] >= 0)[0]
        if len(bounce_rows):
            bounce_hits = {key: values[bounce_rows] for key, values in bounce.items()}
            bounce_points = (
                shadow_origins[bounce_rows]
                + bounce_directions[bounce_rows] * bounce_hits["t"][:, None]
            )
            bounce_normals, albedo1 = raytrace.surface(
                scene, tree, bounce_hits, bounce_directions[bounce_rows]
            )
            _, direct1 = direct_light(
                scene,
                bounce_points,
                bounce_normals,
                tracer,
                "shadow2",
                pixels[hit_rows][bounce_rows],
            )
            radiance[bounce_rows] = albedo1 * (raytrace.AMBIENT + direct1)[:, None]
        light = ((raytrace.AMBIENT + direct)[:, None]) + raytrace.GI * radiance
        colour[hit_rows] = albedo * light
    return colour, hit


def direct_light(scene, points, normals, tracer, name, pixels):
    """The reference's direct_light with the shadow rays traced on the card."""
    origins = points + normals * scene["eps"][0]
    to_light = scene["light"] - origins
    shadow = tracer.trace(name, origins, to_light, F(1.0), True, pixels)["leaf"] >= 0
    cosine = raytrace.dot(normals, to_light) * (
        F(1.0) / np.sqrt(raytrace.dot(to_light, to_light))
    )
    return origins, np.where(shadow, F(0), np.maximum(cosine, F(0)))
