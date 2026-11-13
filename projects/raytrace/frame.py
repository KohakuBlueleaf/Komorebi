"""A frame on the card: generation, the four traces, the shading and the colour as
records, each core keeping its tiles' packets from the first stage to the last; the
colour (with --stages, every stage) is checked against numpy. --reps reruns the frame
in the same load: times are medians, power the mean over the records.
"""

import argparse
import json
import os
import sys

import numpy as np

from projects.raytrace import card, packets
from projects.raytrace.reference import harness
from projects.raytrace.run import reference_frame
from ref import native, raytrace, scenes

K = packets.K
VEC = packets.VEC
F = np.float32
# Each step: name, segment, input streams, output streams (in the kernels' order).
STEPS = (
    ("gen", "kSegGen", ("desc",), ("primary_pk",)),
    ("primary", "kSegTrace", ("primary_pk",), ("primary_hit",)),
    (
        "shade1",
        "kSegShade1",
        ("primary_pk", "primary_hit"),
        ("shadow_pk", "bounce_pk", "state1"),
    ),
    ("shadow", "kSegTrace", ("shadow_pk",), ("shadow_hit",)),
    ("bounce", "kSegTrace", ("bounce_pk",), ("bounce_hit",)),
    (
        "shade2",
        "kSegShade2",
        ("state1", "shadow_hit", "bounce_pk", "bounce_hit"),
        ("shadow2_pk", "state2"),
    ),
    ("shadow2", "kSegTrace", ("shadow2_pk",), ("shadow2_hit",)),
    ("image", "kSegImage", ("state2", "shadow2_hit"), ("colour",)),
)
ANY_HIT = ("shadow", "shadow2")
STREAM_WORDS = (1, 2, 5, 6, 7, 8, 9)  # record word of stream k (stream_arg)


def vectors(stream):
    """Vectors of a stream's page per packet."""
    if stream.endswith("_pk"):
        return K["kPkVectors"]
    if stream.endswith("_hit"):
        return packets.hit_vectors(stream.split("_")[0] in ANY_HIT)
    named = {"desc": "kDescVectors", "state1": "kS1Vectors", "state2": "kS2Vectors"}
    return K[named.get(stream, "kColVectors")]


def records(tensors, directory_t, const_t, profs):
    """Every step's record (segment, ten words), in issue order."""
    out = []
    for (name, seg, ins, outs), prof in zip(STEPS, profs):
        words = [0] * 10
        words[0] = directory_t.buffer_address()
        words[4] = prof.buffer_address()
        if seg == "kSegTrace":
            words[1] = tensors[ins[0]].buffer_address()
            words[2] = tensors[outs[0]].buffer_address()
            words[3] = K["kModeAny"] if name in ANY_HIT else K["kModeNearest"]
        else:
            words[3] = const_t.buffer_address()
            for k, stream in enumerate(ins + outs):
                words[STREAM_WORDS[k]] = tensors[stream].buffer_address()
        out.append((K[seg], words))
    return out


def lanes_of(pages, stream):
    """A stream's pages (P, vectors * 32) as (vectors, P * 32): lane values in order."""
    v = vectors(stream)
    return pages.reshape(-1, v, VEC).transpose(1, 0, 2).reshape(v, -1)


def expected(scene, tree, pix):
    """The reference's values of every stage at the lanes' pixels (flat)."""
    o, d = raytrace.camera_rays(scene["camera"], pix)
    hit = raytrace.trace(tree, o, d, raytrace.TMAX)
    h = hit["leaf"] >= 0
    sub = {k: v[h] for k, v in hit.items()}
    p = o[h] + d[h] * sub["t"][:, None]
    n, albedo = raytrace.surface(scene, tree, sub, d[h])
    o2 = p + n * scene["eps"][0]
    tl = scene["light"] - o2
    cos = raytrace.dot(n, tl) * (F(1.0) / np.sqrt(raytrace.dot(tl, tl)))
    db = raytrace.bounce_directions(pix[h], n)
    colour = raytrace.render(scene, tree, pix)["image"]
    tri = np.where(h, tree["leaf_tri"][np.maximum(hit["leaf"], 0)], -1)
    return {"o": o, "d": d, "inv": raytrace.inverse_direction(d), "tri": tri, "h": h}, {
        "o2": o2,
        "tl": tl,
        "cos": cos,
        "db": db,
        "albedo": albedo,
        "sky": raytrace.sky(d),
        "colour": colour,
    }


def differ(a, b):
    """Lanes whose float32 bits differ (rows of a 2-D array count once)."""
    a, b = packets.bits(a), packets.bits(b)
    bad = a != b
    return int(bad.any(axis=1).sum() if bad.ndim == 2 else bad.sum())


def check_stages(got, scene, tree, pix):
    """Lanes that differ from the reference, stage by stage."""
    first, at = expected(scene, tree, pix)
    h = first["h"]
    pk = lanes_of(got["primary_pk"], "primary_pk")
    hit = lanes_of(got["primary_hit"], "primary_hit")
    sh = lanes_of(got["shadow_pk"], "shadow_pk")
    bo = lanes_of(got["bounce_pk"], "bounce_pk")
    s1 = lanes_of(got["state1"], "state1")
    col = s1[K["kS1Col"] : K["kS1Col"] + 3].T
    want_col = at["sky"].copy()
    want_col[h] = at["albedo"]
    out = {
        "gen_o": differ(pk[K["kPkO"] : K["kPkO"] + 3].T.view(F), first["o"]),
        "gen_d": differ(pk[K["kPkD"] : K["kPkD"] + 3].T.view(F), first["d"]),
        "gen_inv": differ(pk[K["kPkInv"] : K["kPkInv"] + 3].T.view(F), first["inv"]),
        "primary_tri": int((hit[K["kHitTri"]].view(np.int32) != first["tri"]).sum()),
        "shade1_hit_flag": int(((s1[K["kS1Hit"]] != 0) != h).sum()),
        "shade1_colour": differ(col.view(F), want_col),
        "shade1_cos": differ(s1[K["kS1Cos"]].view(F)[h], at["cos"]),
        "shadow_o": differ(sh[K["kPkO"] : K["kPkO"] + 3].T.view(F)[h], at["o2"]),
        "shadow_d": differ(sh[K["kPkD"] : K["kPkD"] + 3].T.view(F)[h], at["tl"]),
        "bounce_d": differ(bo[K["kPkD"] : K["kPkD"] + 3].T.view(F)[h], at["db"]),
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles")
    p.add_argument("--stages", action="store_true", help="read and check every stage")
    p.add_argument("--reps", type=int, default=1, help="frames in the one load")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    image = packets.tree_image(tree, scene)
    report = {"scene": args.scene, "tiles": args.tiles}
    with card.open_session() as session:
        workers = card.workers(session)
        pix, directory, desc = packets.frame_layout(args.tiles, workers)
        n = len(pix)
        report["packets"] = n
        dev = session.device
        names = {s for _, _, ins, outs in STEPS for s in ins + outs}
        tensors = {
            s: card.tensor(dev, packets.group(np.zeros((n, vectors(s) * VEC))))
            for s in sorted(names)
        }
        tensors["desc"] = card.tensor(dev, packets.group(desc))
        dir_t = card.tensor(dev, directory)
        const_t = card.tensor(dev, packets.frame_constants(scene))
        tree_t = card.tensor(dev, image)
        prof_rows = np.zeros((workers, K["kOutPageBytes"] // 4), dtype=np.uint32)
        profs = [card.tensor(dev, prof_rows) for _ in STEPS]
        recs = [(K["kSegLoad"], [tree_t.buffer_address(), image.nbytes])]
        recs += records(tensors, dir_t, const_t, profs) * args.reps
        card.log(f"frame: {n} packets, {len(recs)} records")
        meter = card.Power()
        report["power_idle_w"] = meter.idle()
        report["aiclk_mhz"] = card.aiclk_mhz()
        report["load_avg"] = os.getloadavg()[0]
        res, samples = card.run_records(session, recs, len(image), power=meter)
        run = card.check_run(res, workers)
        us = card.record_us(res)[1:].reshape(args.reps, len(STEPS))
        report["run"] = run
        report["frame_us_median"] = float(np.median(us.sum(axis=1)))
        watts = [w for t, w in samples if t >= samples[-1][0] - res.host_s]
        report["power_mean_w"] = float(np.mean(watts)) if watts else None
        report["steps"] = []
        rows = card.read_each(session, profs, [f"profile {s[0]}" for s in STEPS])
        for (name, *_), ev_us, prof in zip(STEPS, us.T, rows):
            pr = card.profile_rows(prof.astype(np.uint64), workers)
            step = {"step": name, "us": float(np.median(ev_us))}
            step["errors"] = int((pr["error"] != 0).sum())
            step["busy_max"] = int(pr["total"].max())
            used = pr["packets"] > 0
            for key in ("load", "box", "out", "wait_in", "wait_out", "total"):
                per = pr[key][used].sum() / max(pr["packets"][used].sum(), 1)
                step[f"{key}_per_packet"] = round(float(per), 1)
            report["steps"].append(step)
        wanted = sorted(names) if args.stages else ["colour"]
        got = card.read_each(session, [tensors[s] for s in wanted], wanted)
        got = {s: packets.ungroup(g, vectors(s), n) for s, g in zip(wanted, got)}
    flat = pix.reshape(-1)
    colour = lanes_of(got["colour"], "colour").T.view(F)
    if args.tiles is None:
        name, ref = reference_frame(args.scene)
        want = ref["image"].reshape(-1, 3)[flat]
    else:
        name, want = "numpy", raytrace.render(scene, tree, flat)["image"]
    report["image_check"] = {"against": name, "pixels": len(flat)}
    report["image_check"]["differs"] = differ(colour, want)
    if args.stages:
        report["stages"] = check_stages(got, scene, tree, flat)
    ok = card.ran_cleanly(run, workers) and report["image_check"]["differs"] == 0
    ok &= all(s["errors"] == 0 for s in report["steps"])
    ok &= all(v == 0 for v in report.get("stages", {}).values())
    report["ok"] = bool(ok)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
