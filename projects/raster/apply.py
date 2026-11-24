"""APPLY: one apply y = M x of a scene's light-transport set (projects/transport) as a
hand-written handler of the rasterizer's kernels (kernels/apply_*.hpp): its own
kernel-form program here, checked as projects/transport/run.py checks the class.
`python -m projects.raster.apply <set folder> [--calls 32]` prints one JSON line."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.program import several as programs
from projects.raster import card, card_thread, lists
from projects.transport import plan as pl
from projects.transport import run as transport_run
from reference import indirect

KERNELS = lists.HERE / "kernels"
APPLY_H = KERNELS / "apply.h"
A = card_thread.A  # apply.h's numbers
BLOCK = 8  # weight tiles a block (transport.py's BLOCK)
BLOCKS_A_HALF = 4  # a weight half's blocks (the class compiler's 32 pages of 8)
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class SetRefused(ValueError):
    """A set whose APPLY buffers do not fit a core's L1: the numbers."""


ENCODING = (
    1  # w's encoding (plan.encode_weights): a cached file of another is made again
)
SET_KEYED = ("blocks.npz", "weights.f32")  # the files a cached w depends on


def cache_folder(choice, data):
    """The folder of the cached w files: <data>/cache/apply/ for `choice` None or
    "default", none for "none", else `choice`."""
    if choice in (None, "default"):
        return Path(data) / "cache" / "apply"
    return None if choice == "none" else Path(choice)


def weights_file(folder, cores, cache):
    """A set's cached w in the folder `cache` (None: no cache): <scene>_<key>.npy, the
    key of SET_KEYED's sizes and modification times, `cores`, ENCODING."""
    if cache is None:
        return None
    folder = Path(folder)
    stats = [(folder / name).stat() for name in SET_KEYED]
    text = json.dumps(
        [[st.st_size, st.st_mtime_ns] for st in stats] + [int(cores), ENCODING]
    )
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    scene = folder.name.removesuffix("_transport")
    return Path(cache) / f"{scene}_{key}.npy"


def encoded(s, path):
    """w's bytes (plan.encode_weights): read from `path` when it is there, else made
    and written there whole (a temporary name, then renamed); the scene's files of
    other keys removed."""
    if path is not None and path.exists():
        return np.load(path, mmap_mode="r")
    raw = pl.encode_weights(s, None)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with open(part, "wb") as out:
            np.save(out, raw)
        part.replace(path)
        scene = path.name.rsplit("_", 1)[0]
        for stale in path.parent.glob(f"{scene}_*.npy"):
            if stale != path and stale.name.rsplit("_", 1)[0] == scene:
                stale.unlink()
    return raw


def prepare(folder, cores=120, cache=None):
    """The set in `folder` dealt to `cores` cores: (set, tables, sizes, w's bytes),
    w cached in the folder `cache` (weights_file; None: none)."""
    s = pl.load(folder)
    first = pl.deal(s, cores)
    _, pages = pl.weight_pages(s)
    t = pl.tables(s, first, pages - 1)
    raw = encoded(s, weights_file(folder, cores, cache))
    return s, t, pl.sizes(s, t), raw


def defines(sizes, block=BLOCK):
    """The kernels' build defines of a set's sizes (plan.sizes)."""
    return {
        "APPLY_OWN": str(sizes["own_most"]),
        "APPLY_KMAX": str(sizes["k_most"]),
        "APPLY_RANK": str(sizes["rank_most"]),
        "APPLY_PAGES": str(sizes["table_pages"]),
        "APPLY_BLOCK": str(block),
    }


def buffers(sizes, block=BLOCK):
    """APPLY's buffers, each whole: {name: (format, page bytes, pages)} in apply.h's
    order (x's pair first): the read streams' halves, y's and the profile's, the
    table pages, the sums, t, the token."""
    kmax, own, rank = sizes["k_most"], sizes["own_most"], sizes["rank_most"]
    weights = BLOCKS_A_HALF * block
    return {
        "kCbApplyX": ("bfloat16", A["kApplyXBytes"], kmax),
        "kCbApplyXB": ("bfloat16", A["kApplyXBytes"], kmax),
        "kCbApplyW": ("bfp8_b", A["kApplyWBytes"], weights),
        "kCbApplyWB": ("bfp8_b", A["kApplyWBytes"], weights),
        "kCbApplyY": ("float32", A["kApplyYBytes"], A["kApplyYPages"]),
        "kCbApplyProf": ("uint32", A["kApplyProfBytes"], 1),
        "kCbApplyEntries": ("uint32", A["kApplyTableBytes"], 1),
        "kCbApplyEntriesB": ("uint32", A["kApplyTableBytes"], 1),
        "kCbApplyHeadPage": ("uint32", A["kApplyTableBytes"], 1),
        "kCbApplyHeadPageB": ("uint32", A["kApplyTableBytes"], 1),
        "kCbApplySums": ("float32", A["kApplySumsBytes"], own),
        "kCbApplyHidden": ("bfloat16", A["kApplyHiddenBytes"], rank),
        "kCbApplyReady": ("bfloat16", A["kApplyHiddenBytes"], A["kApplyReadyPages"]),
    }


def l1_bytes(sizes, block=BLOCK):
    """APPLY's buffers' bytes, each whole."""
    return sum(page * pages for _, page, pages in buffers(sizes, block).values())


WINDOW_ALIGN = 64  # a window's start (the movers' DRAM reads land on 64 bytes)


def layout(sizes, hosts, block=BLOCK):
    """APPLY's windows over `hosts` ([(name, index, bytes)]): x's pair in the first,
    then the largest buffer first at the first place it fits (first fit decreasing),
    on WINDOW_ALIGN bytes: {name: (host index, byte, pages)}; SetRefused with the
    numbers when a buffer fits nowhere."""
    used = [0] * len(hosts)
    windows = {}
    rows = buffers(sizes, block)
    x_pair = ["kCbApplyX", "kCbApplyXB"]
    rest = sorted(
        (name for name in rows if name not in x_pair),
        key=lambda name: -rows[name][1] * rows[name][2],
    )
    for name in x_pair + rest:
        _, page, pages = rows[name]
        need = page * pages
        for h, (_, index, size) in enumerate(hosts[:1] if name in x_pair else hosts):
            at = -(-used[h] // WINDOW_ALIGN) * WINDOW_ALIGN
            if at + need <= size:
                windows[name] = (index, at, pages)
                used[h] = at + need
                break
        else:
            left = ", ".join(
                f"{host} {size - u:,} of {size:,} B"
                for (host, _, size), u in zip(hosts, used)
            )
            raise SetRefused(
                f"APPLY of a set with OWN {sizes['own_most']}, KMAX {sizes['k_most']}, "
                f"RANK {sizes['rank_most']} needs {l1_bytes(sizes, block):,} B: "
                f"{name} ({need:,} B) fits in no host buffer"
                + (" (x's pair goes in the first)" if name in x_pair else "")
                + f"; left: {left}"
            )
    return {name: windows[name] for name in rows}


def window_defines(windows):
    """The kernels' defines of a layout: APPLY_WINDOWS and each buffer's host, byte
    and pages (apply_window.hpp)."""
    out = {"APPLY_WINDOWS": "1"}
    for name, (host, at, pages) in windows.items():
        out[f"APPLY_HOST_{name}"] = str(host)
        out[f"APPLY_AT_{name}"] = str(at)
        out[f"APPLY_PAGES_{name}"] = str(pages)
    return out


def declared(sizes, block=BLOCK, windows=False):
    """APPLY's km.Buffers: whole, or (windows) one page each."""
    return [
        km.Buffer(A[name], fmt, page, 1 if windows else pages, name)
        for name, (fmt, page, pages) in buffers(sizes, block).items()
    ]


def host_bytes(sizes, block=BLOCK):
    """The own program's host buffer (--windows): every window end to end."""
    hosts = [("kCbApplyHost", A["kCbApplyHost"], 1 << 30)]
    windows = layout(sizes, hosts, block)
    rows = buffers(sizes, block)
    end = max(at + rows[name][1] * pages for name, (_, at, pages) in windows.items())
    return -(-end // 4096) * 4096


def dead_buffers():
    """The frame buffers dead between a frame's commands, APPLY's windows' room, in
    order ({index: name}): the page slots (and the white texel's page), SHADE's work,
    LIGHT's work, the planes, the table, SHOW's ring; each written by its owner before
    it reads it within each of its commands (README, "APPLY")."""
    sh, show = card_thread.SH, card_thread.SHOW
    names = ("kCbShadeSlots", "kCbShadeWork", "kCbLightWork", "kCbShadePlanes")
    rows = {sh[name]: name for name in names + ("kCbShadeTable",)}
    rows[show["kCbShow"]] = "kCbShow"
    return rows


def frame_runs(cbs, dead, align=programs.CB_ALIGN):
    """The host ranges of the buffers `cbs` ((index, format, page bytes, pages)): each
    a maximal run of `dead` buffers ({index: name}) that follow one another at
    programs.buffer_layout's addresses (a gap inside a run unused), as (names joined
    by "+", the first's index, the run's bytes), the run of `dead`'s first first."""
    placed, _ = programs.buffer_layout(sorted(cbs), 0, align)
    runs, run = [], None
    for index in sorted(placed):
        at, size = placed[index][:2]
        if index in dead and run is not None and at == -(-run[2] // align) * align:
            run[1].append(index)
            run[2] = at + size
        elif index in dead:
            run = [at, [index], at + size]
            runs.append(run)
        else:
            run = None
    rank = {index: place for place, index in enumerate(dead)}
    runs.sort(key=lambda run: min(rank[index] for index in run[1]))
    return [
        ("+".join(dead[index] for index in members), members[0], end - start)
        for start, members, end in runs
    ]


def frame_hosts(buffers):
    """The host ranges APPLY's windows lie over in a frame program of `buffers` (its
    buffers but APPLY's, card_thread.frame_buffers): frame_runs of dead_buffers()."""
    halves = [
        half
        for item in buffers
        for half in (item.halves() if isinstance(item, km.BufferPair) else [item])
    ]
    return frame_runs(programs.program_cbs(halves), dead_buffers())


def frame_parts(sizes, buffers, block=BLOCK):
    """APPLY in the frame program of `buffers` (card_thread.program's `apply_parts`;
    `buffers`: card_thread.frame_buffers): its buffers one page each, the build defines
    (sizes, windows over frame_hosts), its float32 sums unpacked to Dst, the movers'
    tensor arguments, the windows. SetRefused when the set does not fit."""
    windows = layout(sizes, frame_hosts(buffers), block)
    return {
        "buffers": declared(sizes, block, windows=True),
        "defines": {**defines(sizes, block), **window_defines(windows)},
        "unpack_to_dest_mode": {A["kCbApplySums"]: "UnpackToDestFp32"},
        "args": tensor_args(),
        "windows": windows,
    }


def header(made, cores, words=1024):
    """The update region's words (thread/frame.h RF_U_*): APPLY's command words a0 to
    a9, its group (cores 0 to cores - 1); the rest 0."""
    out = np.zeros(words, dtype=np.uint32)
    command = apply_words(made)
    at = card_thread.F["RF_U_A0"]
    out[at : at + len(command)] = command
    out[card_thread.F["RF_U_GROUP"]] = cores << 16
    return out


def tensors(session, s, t, raw, x, cores=120):
    """x (bf16 tiles, plan.x_tiles), w (bfp8_b, tiles stored transposed), y (float32
    tiles), the tables, the profile pages: {name: tensor}."""
    tw = session.empty((len(raw), pl.TILE, pl.TILE), "bfp8_b", "tile")
    tw.transposed_tiles = True
    session.write(tw, np.asarray(raw))
    return {
        "x": session.tensor(pl.x_tiles(x), "bfloat16", "tile"),
        "w": tw,
        "y": session.empty((pl.TILE, pl.TILE * s.tiles), "float32", "tile"),
        "tables": session.tensor(t.words, "uint32"),
        "profile": session.empty((cores, 1024), "uint32"),
    }


ORDER = ("x", "w", "y", "tables", "profile")


def apply_words(made):
    """APPLY's command words a0 to a5 (apply.h)."""
    return [made[name].address for name in ORDER] + [0]


def tensor_args(made=None):
    """The movers' compile arguments of the five tensors, in ORDER: `made`'s, or
    (None) those of DRAM tensors of their page sizes."""
    pages = (
        A["kApplyXBytes"],
        A["kApplyWBytes"],
        A["kApplyYBytes"],
        A["kApplyTableBytes"],
        A["kApplyProfBytes"],
    )
    if made is None:
        return [word for page in pages for word in (card.DRAM_ARGS, page)]
    return [word for name in ORDER for word in made[name].tensor_args()]


def compute(frame_compute):
    """The frame program's compute configuration with APPLY's float32 sums unpacked
    straight to Dst (as the class's own float32 field is: through SrcA a copy keeps
    10 bits of the mantissa)."""
    modes = dict(frame_compute.get("unpack_to_dest_mode", {}))
    modes[A["kCbApplySums"]] = "UnpackToDestFp32"
    return {**frame_compute, "unpack_to_dest_mode": modes}


def program(made, sizes, block=BLOCK, windows=False):
    """APPLY's own kernel-form program (kernels/apply_*.cpp), the frame program's
    compute configuration (HiFi4, float32 sums); its buffers whole, or (`windows`)
    one page each with windows over one uint32 host buffer (kCbApplyHost)."""
    args = tensor_args(made)
    more, extra = [], {}
    if windows:
        size = host_bytes(sizes, block)
        more = [km.Buffer(A["kCbApplyHost"], "uint32", 4096, size // 4096, "host")]
        hosts = [("kCbApplyHost", A["kCbApplyHost"], size)]
        extra = window_defines(layout(sizes, hosts, block))
    return km.Program(
        "raster_apply",
        kernels={
            role: KERNELS / f"apply_{role}.cpp"
            for role in ("reader", "compute", "writer")
        },
        buffers=[*declared(sizes, block, windows), *more],
        compile_args={"reader": args, "writer": args},
        compute=compute(card.program().compute),
        defines={**defines(sizes, block), **extra},
        depth=1,
        handlers={"apply": A["kHandlerApply"]},
    )


def refuse_l1(session, made_program, sizes):
    """SetRefused when the program's L1 budget does not fit a core."""
    rows = session.l1_rows(made_program)
    used = sum(size for _, size in rows)
    free = session.l1()[1]
    if used > free:
        raise SetRefused(
            f"APPLY of a set with OWN {sizes['own_most']}, KMAX {sizes['k_most']}, "
            f"RANK {sizes['rank_most']}: its buffers take {l1_bytes(sizes):,} B, a "
            f"core's L1 budget {used:,} B of {free:,} B"
        )


def check(got, emulated, s, lit):
    """The card's y against the emulation and y_float.npy (transport run's errors)."""
    return {
        "emulated": transport_run.errors(got, emulated, lit),
        "y_float": transport_run.errors(got, s.vector("y_float.npy"), lit),
        "emulated_vs_y_float": transport_run.errors(
            emulated, s.vector("y_float.npy"), lit
        ),
        "lit": int(lit.sum()),
    }


def lit_of(s, folder):
    """The elements counted in the errors (transport run's rule)."""
    lit = s.vector("y_exact.npy") @ transport_run.LUMA > transport_run.LIT
    lit[np.load(Path(folder) / "perm.npy") < 0] = False
    return lit


def scene_set(scene, data):
    """A scene's (or a view's) transport set: <data>/public/<scene>_transport/."""
    return indirect.model_path(scene, data).with_name(
        indirect.model_path(scene, data).name.replace("_indirect.npz", "_transport")
    )


SET_FILES = ("blocks.npz", "weights.f32", "perm.npy")  # what field_set reads


def field_set(scene, data, cores, cache="default"):
    """A scene's transport set made ready for the field's update in the thread
    (live.py --card-field; w cached as cache_folder(`cache`) says): {folder, set,
    tables, sizes, raw, x (0)}; FileNotFoundError naming the files missing."""
    folder = scene_set(scene, data)
    missing = [str(folder / name) for name in SET_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"the field's update on the card needs {scene}'s transport set: "
            f"missing {', '.join(missing)}"
        )
    s, t, sizes, raw = prepare(folder, cores, cache_folder(cache, data))
    x = np.zeros((pl.TILE * s.tiles, 3), np.float32)
    return {
        "folder": str(folder),
        "set": s,
        "tables": t,
        "sizes": sizes,
        "raw": raw,
        "x": x,
    }


def checked_set(folder, cores, cache=None):
    """A set made ready to apply and check on the host: {folder, set, tables, sizes,
    raw (w's bytes), x, lit, emulated}."""
    s, t, sizes, raw = prepare(folder, cores, cache)
    x = s.vector("x.npy")
    return {
        "folder": str(folder),
        "set": s,
        "tables": t,
        "sizes": sizes,
        "raw": raw,
        "x": x,
        "lit": lit_of(s, folder),
        "emulated": pl.emulate(s, pl.decode_weights(raw), x),
    }


def y_check(session, made, ready):
    """y read back and checked (check()) against the emulation and y_float; its
    hash."""
    s = ready["set"]
    got = transport_run.y_of(session, made["y"], s.tiles)
    out = check(got, ready["emulated"], s, ready["lit"])
    out["y_sha256"] = hashlib.sha256(got.tobytes()).hexdigest()[:16]
    return out


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="the set's files (blocks.npz, weights.f32, ...)")
    parser.add_argument("--cache", default=None, help="a folder for w's bytes")
    parser.add_argument("--cores", type=int, default=120)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument(
        "--windows", action="store_true", help="buffers as windows over a host buffer"
    )
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main():
    args = parse()
    name = Path(args.folder).name
    cache = Path(args.cache) if args.cache else None
    s, t, sizes, raw = prepare(args.folder, args.cores, cache)
    report = {"set": name, "sizes": sizes, "l1_buffers": l1_bytes(sizes)}
    report["windows"] = args.windows
    log(f"sizes {json.dumps(sizes)}; buffers {report['l1_buffers']:,} B")
    x = s.vector("x.npy")
    lit = lit_of(s, args.folder)
    emulated = pl.emulate(s, pl.decode_weights(raw), x)
    log("emulated")
    index = power.visible_card()
    with km.Session(log=log) as session:
        made = tensors(session, s, t, raw, x, args.cores)
        built = program(made, sizes, windows=args.windows)
        refuse_l1(session, built, sizes)
        run = session.load(built)
        group, command = (0, args.cores), apply_words(made)
        handler = A["kHandlerApply"]
        event = run.wait(run.submit(handler, group, args=command), timeout=60.0)
        if event.kind != "done":
            raise km.KomorebiError(f"apply: {event}")
        got = transport_run.y_of(session, made["y"], s.tiles)
        log(f"first apply {event.microseconds:.1f} us")
        card_us, wall_us = [], []
        before = power.aiclk_mhz(index)
        for _ in range(args.calls):
            started = time.perf_counter()
            event = run.wait(run.submit(handler, group, args=command))
            wall_us.append((time.perf_counter() - started) * 1e6)
            if event.kind != "done":
                raise km.KomorebiError(f"apply: {event}")
            card_us.append(event.microseconds)
        last = transport_run.y_of(session, made["y"], s.tiles)
        report["time"] = {
            "calls": args.calls,
            "card_us_median": float(np.median(card_us)),
            "card_us_least": float(np.min(card_us)),
            "wall_us_median": float(np.median(wall_us)),
            "wall_us_least": float(np.min(wall_us)),
            "mhz": [before, power.aiclk_mhz(index)],
        }
        log(f"time {json.dumps(report['time'])}")
        run.stop()
        report["kernel"] = run.result()["error_name"]
    report["check"] = check(got, emulated, s, lit)
    report["last_equals_first"] = bool(np.array_equal(got, last))
    report["y_sha256"] = hashlib.sha256(got.tobytes()).hexdigest()[:16]
    report["ok"] = report["kernel"] == "none" and report["last_equals_first"]
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
