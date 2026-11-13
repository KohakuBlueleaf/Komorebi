"""A frame on the card: generation, the four traces, the shading and the colour as
commands, each core keeping its tiles' packets; the colour (--stages: every stage) is
checked against numpy, --reps reruns the frame, --pictures DIR writes PNGs and JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from komorebi import session as km_session  # noqa: E402
from projects.raytrace import card, packets, planner  # noqa: E402
from projects.raytrace import ties as ties_mod  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from projects.raytrace.run import reference_frame  # noqa: E402
from ref import bvh, native, png, raytrace, scenes  # noqa: E402

K = packets.K
VEC = packets.VEC
F = np.float32
# Each step: name, handler, input streams, output streams (in the kernels' order).
STEPS = (
    ("gen", "kHandlerGen", ("desc",), ("primary_pk",)),
    ("primary", "kHandlerTrace", ("primary_pk",), ("primary_hit",)),
    (
        "shade1",
        "kHandlerShade1",
        ("primary_pk", "primary_hit"),
        ("shadow_pk", "bounce_pk", "state1"),
    ),
    ("shadow", "kHandlerTrace", ("shadow_pk",), ("shadow_hit",)),
    ("bounce", "kHandlerTrace", ("bounce_pk",), ("bounce_hit",)),
    (
        "shade2",
        "kHandlerShade2",
        ("state1", "shadow_hit", "bounce_pk", "bounce_hit"),
        ("shadow2_pk", "state2"),
    ),
    ("shadow2", "kHandlerTrace", ("shadow2_pk",), ("shadow2_hit",)),
    ("image", "kHandlerImage", ("state2", "shadow2_hit"), ("colour",)),
)
ANY_HIT = ("shadow", "shadow2")
SORTED = ("bounce", "shadow2")  # traces whose rays --sort orders on each core first


def sorted_steps():
    """STEPS with each SORTED trace on its rays in the core's order (SORT before it,
    UNSORT of its hits after it)."""
    out = []
    for name, handler, ins, outs in STEPS:
        if name not in SORTED:
            out.append((name, handler, ins, outs))
            continue
        pk, hit, order = f"{name}_s_pk", f"{name}_s_hit", f"order_{name}"
        out.append((f"sort_{name}", "kHandlerSort", ins, (pk, order)))
        out.append((name, handler, (pk,), (hit,)))
        out.append((f"unsort_{name}", "kHandlerSort", (hit, order), outs))
    return tuple(out)


def sort_key(tree):
    """SORT's words a6 to a9: the root box's low corner in fixed point (frac fraction
    bits, the largest coordinate under 2^30) and the shift to 2^kSortCellBits cells."""
    lo, hi = (b[0].astype(np.float64) for b in bvh.node_boxes(tree["nodes"][:1]))
    big = max(np.abs(lo).max(), np.abs(hi).max(), 1e-30)
    frac = int(np.clip(29 - np.ceil(np.log2(big)), 0, 60))
    lo_fix = np.floor(lo * 2.0**frac).astype(np.int64)
    span = int(np.ceil((hi - lo).max() * 2.0**frac)) + 1
    shift = max(0, span.bit_length() - K["kSortCellBits"])
    return [int(v) & 0xFFFFFFFF for v in lo_fix] + [frac | shift << 8]


STREAM_WORDS = (1, 2, 5, 6, 7, 8, 9)  # command word of stream k (stream_arg)


def vectors(stream):
    """Vectors of a stream's page per packet."""
    if stream.endswith("_pk"):
        return K["kPkVectors"]
    if stream.endswith("_hit"):
        return packets.hit_vectors(stream.split("_")[0] in ANY_HIT)
    if stream.startswith("order_"):
        return 1
    named = {"desc": "kDescVectors", "state1": "kS1Vectors", "state2": "kS2Vectors"}
    return K[named.get(stream, "kColVectors")]


def step_words(step, addr, directory, const, prof):
    """A step's ten command words; `addr` maps a stream to its address (or a graph's
    register)."""
    name, handler, ins, outs = step
    words = [0] * 10
    words[0], words[4] = directory, prof
    if handler == "kHandlerTrace":
        words[1], words[2] = addr[ins[0]], addr[outs[0]]
        words[3] = K["kModeAny"] if name in ANY_HIT else K["kModeNearest"]
    elif handler == "kHandlerSort":
        streams = ins + outs
        order = next(s for s in streams if s.startswith("order_"))
        src, dst = (s for s in streams if s != order)
        words[1], words[2], words[3] = addr[src], addr[dst], addr[order]
        words[5] = vectors(src) if name.startswith("unsort") else 0
    else:
        words[3] = const
        for k, stream in enumerate(ins + outs):
            words[STREAM_WORDS[k]] = addr[stream]
    return words


def commands(tensors, dir_ts, const_t, profs, treelets=None, steps=STEPS, key=()):
    """Every step's commands (handler, ten words), in submit order. With treelets (the
    tensor and four words) a trace has a command per chunk dir_ts[1:], others dir_ts[0].
    """
    addr = {s: t.address for s, t in tensors.items()}
    c = const_t.address
    out = []
    for step, step_profs in zip(steps, profs):
        chunks = dir_ts[1:] if treelets and step[1] == "kHandlerTrace" else dir_ts[:1]
        for d, prof in zip(chunks, step_profs):
            words = step_words(step, addr, d.address, c, prof.address)
            handler = K[step[1]]
            if treelets and step[1] == "kHandlerTrace":
                handler = K["kHandlerTreelets"]
                words[5:10] = [treelets[0].address, *treelets[1]]
            if step[1] == "kHandlerSort":
                words[6:10] = key
            out.append((handler, words))
    return out


def load_cut(scene, tree, kib):
    """The scene's treelets of at most `kib` KiB: the stored cut, else a new one."""
    path = native.data_dir() / f"{scene}_treelets_{kib}k.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    return bvh.cut_treelets(tree, kib * 1024)


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


def pictures(out, scene, flat, colour, want, report):
    """The card's image, the reference's and the difference as PNGs, and frame.json."""
    out.mkdir(parents=True, exist_ok=True)
    side = scenes.WIDTH
    card_img = np.zeros((side * side, 3), dtype=F)
    ref_img = np.zeros((side * side, 3), dtype=F)
    card_img[flat], ref_img[flat] = colour, want
    card_img, ref_img = card_img.reshape(side, side, 3), ref_img.reshape(side, side, 3)
    mask = (packets.bits(card_img) != packets.bits(ref_img)).any(axis=2)
    png.write_png(out / f"{scene}_card.png", png.to_srgb8(card_img))
    png.write_png(out / f"{scene}_reference.png", png.to_srgb8(ref_img))
    png.write_png(out / f"{scene}_diff.png", png.diff_picture(card_img, mask))
    with open(out / f"{scene}_frame.json", "w") as f:
        json.dump(report, f, indent=1)


SAMPLE_CORES = (0, 40, 80, 119)  # cores whose trace hit pages --profile keeps


def save_profile(path, session, raw, command_us, directory, tensors, steps=STEPS):
    """Every command's counters per core, the command times, the directory and the hit
    pages of SAMPLE_CORES' packets for each trace (first kHitAnyVectors vectors)."""
    out = {"command_us": command_us, "directory": directory}
    for (name, *_), rows in zip(steps, raw):
        out[f"prof_{name}"] = np.stack(rows)
    per = K["kPagePackets"]
    for name, handler, _, outs in steps:
        if handler != "kHandlerTrace":
            continue
        v = vectors(outs[0])
        parts = []
        for c in SAMPLE_CORES:
            first, count = (
                int(x) for x in directory[c, [K["kDirFirst"], K["kDirCount"]]]
            )
            lo, hi = first // per, (first + count - 1) // per + 1
            (pages,) = session.read(tensors[outs[0]], first=lo, pages=hi - lo)
            pk = pages.view(np.uint32).reshape(-1, v, VEC)
            pk = pk[first - lo * per : first - lo * per + count, : K["kHitAnyVectors"]]
            parts.append(pk)
        out[f"hits_{name}"] = np.concatenate(parts)
    np.savez_compressed(path, **out)


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


def planned(
    args, session, recs, steps, per_step, treelets, directory, profs, shape, report
):
    """The frames driven by the X280 thread (planner.py): command us per frame and
    step, the profile tensors of the last frame, the host's seconds, power samples."""
    tree_pages, sizes, workers, n, per = shape
    cap = min(per, planner.P["RP_COST_MAX"])
    entries = planner.entries_of(recs, steps, per_step, treelets)
    tab = planner.table(
        entries,
        directory,
        [ps for s, ps in zip(steps, profs) if s[1] == "kHandlerTrace"],
        workers,
    )
    prog = planner.program(
        card.program(tree_pages, sizes),
        tab,
        args.plan,
        args.reps,
        workers,
        n,
        cap,
        len(entries),
    )
    card.log(f"frame: {n} packets, {len(entries)} table commands, plan {args.plan}")
    meter = card.Power()
    frames, status, state, samples = planner.run(
        session, prog, args.reps, meter, card.log
    )
    report["plan"] = {"mode": args.plan, "cap": cap, "thread": status}
    kinds = [e[0] & 0xFF for e in entries]
    us, last = [], None
    for i, f in enumerate(frames):
        chunks = [f["chunks"][k - 1] if k else 1 for k in kinds]
        chunks[0] = 1 if i == 0 else 0  # LOAD: the cold frame only
        rec = np.asarray(f["command_us"], dtype=np.float64)
        if len(rec) != sum(chunks):
            break
        step_us = np.bincount(np.repeat(np.arange(len(kinds)), chunks), rec)[1:]
        us.append(step_us)
        last = f["chunks"]
    report["frames"] = [
        {
            "frame_ms": float(u.sum()) / 1000,
            "step_ms": (u / 1000).tolist(),
            "wall_ms": (f["wall_us"] - (f["command_us"][0] if i == 0 else 0)) / 1000,
            "plan_ms": [x / 1000 for x in f["plan_us"]],
            "chunks": f["chunks"],
            "plan_parts_ms": [x / 1000 for x in f["plan_parts_us"]],
        }
        for i, (u, f) in enumerate(zip(us, frames))
    ]
    if len(us) > 1:
        report["repeated_frame_ms"] = float(np.median([u.sum() for u in us[1:]])) / 1000
    done = len(frames) == args.reps and len(us) == args.reps and status["code"] == "ok"
    ran = card.Ran([], status["kernel"], state, status["host_s"])
    report["run"] = card.check_run(ran, workers)
    report["run"].update(done=int(done), commands=1)
    report["host"] = {"commands": 0, "command_bytes": 0, "wall_s": status["host_s"]}
    kind_of = iter(k for k in kinds[1:] if k)
    read = []
    for s, ps in zip(steps, profs):
        read.append(ps[: last[next(kind_of) - 1]] if s[1] == "kHandlerTrace" else ps)
    us = np.stack(us) if us else np.zeros((1, len(steps)))
    last_us = np.asarray(frames[-1]["command_us"]) if frames else np.zeros(0)
    return us, read, status["host_s"], samples, last_us


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles")
    p.add_argument("--stages", action="store_true", help="read and check every stage")
    p.add_argument("--reps", type=int, default=1, help="frames in the one load")
    p.add_argument("--json", default=None)
    p.add_argument("--pictures", type=Path, default=None, help="folder for the PNGs")
    p.add_argument("--treelets", type=int, default=0, help="KiB: treelets from DRAM")
    p.add_argument("--slots", type=int, default=4, help="treelet slots per core")
    p.add_argument("--profile", default=None, help="NPZ: every core's counters")
    p.add_argument("--sort", action="store_true", help="sort rays on each core")
    p.add_argument(
        "--plan", choices=planner.MODES, default=None, help="X280 thread cuts traces"
    )
    args = p.parse_args()
    if (args.sort or args.plan) and not args.treelets:
        p.error("--sort and --plan need --treelets")
    if args.plan and planner.same_numbers():
        p.error(f"plan.h differs from raytrace.h: {planner.same_numbers()}")
    steps = sorted_steps() if args.sort else STEPS
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    cut = load_cut(args.scene, tree, args.treelets) if args.treelets else None
    if cut is None:
        image = packets.tree_image(tree, scene)
    else:
        treelet_pages, table = packets.treelet_images(tree, cut, scene)
        image = packets.top_image(cut, table)
        slot_bytes = int(table[:, 1].max()) * K["kTreePageBytes"]
    report = {"scene": args.scene, "tiles": args.tiles, "treelets_kib": args.treelets}
    report["sort"] = args.sort
    with card.open_session() as session:
        workers = card.workers(session)
        pix, directory, desc = packets.frame_layout(args.tiles, workers)
        n = len(pix)
        report["packets"] = n
        names = {s for _, _, ins, outs in steps for s in ins + outs}
        tensors = {
            s: card.tensor(session, packets.group(np.zeros((n, vectors(s) * VEC))))
            for s in sorted(names)
        }
        tensors["desc"] = card.tensor(session, packets.group(desc))
        dirs, sizes, treelets = [directory], None, None
        if cut is not None:
            most = int(directory[:, K["kDirCount"]].max())
            # SORT works in the slot buffer (free between treelet commands).
            sort_bytes, short = packets.sort_bytes(most) if args.sort else 0, 0
            for _ in range(4):
                per, slots, sizes = packets.plan_commands(
                    most,
                    len(table),
                    slot_bytes,
                    len(image),
                    args.slots,
                    budget=packets.L1_BUDGET - short,
                )
                if sort_bytes - sizes["slots"] <= short:
                    break
                short = sort_bytes - sizes["slots"]
            sizes["slots"] = max(sizes["slots"], sort_bytes)
            dirs = packets.chunks(directory, per)
            tl_t = card.tensor(session, treelet_pages)
            tl_words = [sizes["entries"], slot_bytes, slots, K["kPolicyOnce"]]
            treelets = (tl_t, tl_words)
            report["treelets"] = {"count": len(table), "packets_per_command": per}
            report["treelets"].update(sizes, slot_count=slots, chunks=len(dirs))
        dir_ts = [
            card.tensor(session, d)
            for d in [directory] + dirs[1 if cut is None else 0 :]
        ]
        const_t = card.tensor(session, packets.frame_constants(scene))
        tree_t = card.tensor(session, image)
        prof_rows = np.zeros((workers, K["kOutPageBytes"] // 4), dtype=np.uint32)
        per_step = [
            len(dirs) if cut is not None and s[1] == "kHandlerTrace" else 1
            for s in steps
        ]
        most = [planner.P["RP_CHUNKS"] if args.plan else k for k in per_step]
        most = [m if s[1] == "kHandlerTrace" else 1 for m, s in zip(most, steps)]
        profs = [[card.tensor(session, prof_rows) for _ in range(k)] for k in most]
        recs = [(K["kHandlerLoad"], [tree_t.address, image.nbytes])]
        key = sort_key(tree) if args.sort else ()
        one = commands(tensors, dir_ts, const_t, profs, treelets, steps, key)
        meter = card.Power()
        report["power_idle_w"] = meter.idle()
        report["aiclk_mhz"] = card.aiclk_mhz()
        report["x280_mhz"] = km_session.X280_MHZ
        report["load_avg"] = os.getloadavg()[0]
        if args.plan:
            us, read_profs, host_s, samples, command_us = planned(
                args,
                session,
                recs + one,
                steps,
                per_step,
                treelets,
                directory,
                profs,
                (len(image), sizes, workers, n, per),
                report,
            )
        else:
            recs += one * args.reps
            card.log(f"frame: {n} packets, {len(recs)} commands")
            res, samples = card.run_commands(
                session, recs, len(image), power=meter, sizes=sizes
            )
            report["run"] = card.check_run(res, workers)
            # A step's time is its commands' (one per directory chunk) summed.
            step_of = np.repeat(np.arange(len(steps)), per_step)
            per_rep = card.command_us(res)[1:].reshape(args.reps, len(step_of))
            us = np.stack([np.bincount(step_of, r, len(steps)) for r in per_rep])
            report["host"] = {**res.host, "wall_s": res.host_s}
            host_s, read_profs = res.host_s, profs
            command_us = card.command_us(res)
        run = report["run"]
        report["frame_us_median"] = float(np.median(us.sum(axis=1)))
        report["frame_ms"] = report["frame_us_median"] / 1000
        if args.plan:
            report["frame_ms"] = float(us[0].sum()) / 1000
        watts = [w for t, w in samples if t >= samples[-1][0] - host_s]
        report["power_mean_w"] = float(np.mean(watts)) if watts else None
        report["steps"] = []
        raw = [
            card.read_each(session, ps, [s[0]] * len(ps))
            for s, ps in zip(steps, read_profs)
        ]
        rows = [sum(r.astype(np.uint64) for r in rs) for rs in raw]
        if args.profile:
            save_profile(
                args.profile,
                session,
                raw,
                command_us,
                directory,
                tensors,
                steps,
            )
        for (name, *_), ev_us, prof in zip(steps, us.T, rows):
            pr = card.profile_rows(prof.astype(np.uint64), workers)
            step = {"step": name, "us": float(np.median(ev_us))}
            step["errors"] = int((pr["error"] != 0).sum())
            step["busy_max"] = int(pr["total"].max())
            used = pr["packets"] > 0
            for key in ("load", "box", "out", "wait_in", "wait_out", "total"):
                per = pr[key][used].sum() / max(pr["packets"][used].sum(), 1)
                step[f"{key}_per_packet"] = round(float(per), 1)
            report["steps"].append(step)
        report["command_ms"] = {s["step"]: s["us"] / 1000 for s in report["steps"]}
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
    bad = (packets.bits(colour) != packets.bits(want)).any(axis=1)
    ties, wrong = ties_mod.classify(scene, tree, flat[bad], colour[bad])
    report["image_check"]["differs"], report["image_check"]["ties"] = wrong, ties
    report["image_check"]["max_abs_error"] = float(np.abs(colour - want).max())
    report["image_check"]["first_pixels"] = flat[bad][:64].tolist()
    if args.stages:
        report["stages"] = check_stages(got, scene, tree, flat)
    ok = card.ran_cleanly(run, workers) and report["image_check"]["differs"] == 0
    ok &= all(s["errors"] == 0 for s in report["steps"])
    ok &= all(v == 0 for v in report.get("stages", {}).values())
    report["ok"] = bool(ok)
    print(json.dumps(report), flush=True)
    if args.pictures is not None:
        pictures(args.pictures, args.scene, flat, colour, want, report)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
