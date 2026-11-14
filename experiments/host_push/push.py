"""Push bytes from Tensix cores into the host's memory channel and check every byte.
Mode pattern: a pattern, then a sequence word; mode dram: pages of a DRAM tensor.
Needs FR_LOG (the watched job's log) and TT_VISIBLE_DEVICES=1; one JSON line a push.
Example: python push.py --mode pattern --bytes 4096 --cores 1"""

import argparse
import ctypes
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
PAGE = 4096  # bytes of a source page
CHANNEL = 1 << 30  # the host channel the driver maps
LIMIT = 64 << 20  # a push stays below this offset (the kernel checks it too)
SEQ_AT = 32 << 20  # the sequence words: 64 bytes a core from here
GUARD = 1 << 20  # bytes after the data that must stay as they were
BUFFER_PAGES, BUFFER_PAGE = 33, 2048  # L1 buffer of a core: two chunks and a slot
MAGIC = 0x50555348
MULT = 2654435761
OK_STATUS = 1
T0 = time.monotonic()
# Report words, as push.h names them.
STATUS, PUSHED, START, DATA, MIDCLEAR, CHUNKS, END = 0, 1, 2, 4, 6, 7, 8


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def sysfs(card, name):
    try:
        return Path(f"/sys/class/tenstorrent/tenstorrent!{card}/{name}").read_text()
    except OSError:
        return ""


def link(card):
    """(speed, width) of the card's PCIe link, from its device directory."""
    base = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/device").resolve()

    def read(name):
        path = base / name
        return path.read_text().strip() if path.exists() else "?"

    return read("current_link_speed"), read("current_link_width")


class Clock(threading.Thread):
    """Samples the card's AICLK (MHz) until stopped."""

    def __init__(self, card):
        super().__init__(daemon=True)
        self.card, self.samples, self.stop = card, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            text = sysfs(self.card, "tt_aiclk").strip()
            if text.isdigit():
                self.samples.append(int(text))
            self.stop.wait(0.02)

    def stats(self):
        s = self.samples or [0]
        return min(s), round(sum(s) / len(s), 1)


def refuse_on_warning(path):
    """Stop before any kernel if the open log shows the NoC address warning."""
    text = Path(path).read_text(errors="replace")
    for needle in ("does not match the expected address", "already holding the sysmem"):
        if needle in text:
            sys.exit(f"stop: the open log shows '{needle}'; no kernel was run")


def map_channel():
    """The channel's mapping in this process, found in /proc/self/maps, as a numpy view
    of its first LIMIT bytes (bytes of a pinned host page; not copied)."""
    with open("/proc/self/maps") as maps:
        lines = [ln for ln in maps if "hugepages" in ln]
    if len(lines) != 1:
        sys.exit(f"stop: {len(lines)} hugepage mappings, expected one")
    m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s+(\S+)", lines[0])
    va, end, perms, path = int(m[1], 16), int(m[2], 16), m[3], m[4]
    if end - va != CHANNEL or not perms.startswith("rw"):
        sys.exit(f"stop: mapping {lines[0].strip()} is not a read-write 1 GiB channel")
    log(f"channel {path} at {va:#x}")
    return np.ctypeslib.as_array((ctypes.c_uint8 * LIMIT).from_address(va))


def slices(units, cores):
    """[(first unit, units)] of the cores that get work; the first cores one more."""
    base, rem = divmod(units, cores)
    out, first = [], 0
    for j in range(cores):
        n = base + (j < rem)
        if n:
            out.append((first, n))
        first += n
    return out


def pattern(first_word, words, nonce):
    i = np.arange(first_word, first_word + words, dtype=np.uint64)
    return ((i * MULT) & 0xFFFFFFFF ^ nonce).astype("<u4")


def run(args, session, channel, card):
    pages = args.bytes // PAGE
    parts = slices(pages, args.cores)
    ends = [(first + n) * PAGE for first, n in parts]
    # The host's own check of every offset, as the kernel does it.
    assert ends[-1] == args.bytes <= SEQ_AT, "the data must end below the words"
    assert SEQ_AT + 64 * args.cores <= LIMIT and args.bytes + GUARD <= SEQ_AT
    rng = np.random.default_rng(secrets.randbits(32))
    nonce = secrets.randbits(32)
    if args.mode == "dram":
        src = rng.integers(0, 256, (pages, PAGE), dtype=np.uint8)
        want = src.reshape(-1)
        tensor = session.tensor(src, "uint8", "row")
        assert tensor.aligned_page_bytes == PAGE, tensor.aligned_page_bytes
    else:
        want = pattern(0, args.bytes // 4, nonce).view(np.uint8)
    # Sentinels: every byte of the data region differs from what is expected there.
    channel[: args.bytes] = ~want
    channel[args.bytes : args.bytes + GUARD] = 0x5A
    channel[SEQ_AT : SEQ_AT + 4096] = 0
    flags = session.tensor(
        np.zeros((len(parts), 16), np.uint32), "uint32", shard_grid=(len(parts), 1)
    )
    launch = session.launch()
    ct = tensor.accessor_args() if args.mode == "dram" else []
    defines = [("KM_PUSH_DRAM", "1")] if args.mode == "dram" else []
    cores = [(0, 0, len(parts) - 1, 0)]
    runtime = []
    for j, (first, n) in enumerate(parts):
        words = [
            flags.address,
            1 if args.mode == "dram" else 0,
            tensor.address if args.mode == "dram" else 0,
            first,
            n * PAGE,
            PAGE,
            first * PAGE,
            args.chunk,
            nonce,
            SEQ_AT + 64 * j,
            j,
            BUFFER_PAGES * BUFFER_PAGE,
        ]
        runtime.append((j, 0, words))
    launch.kernel(
        str(KERNELS / "push.cpp"), cores, 1, ct, defines, [], runtime, noc=args.noc
    )
    launch.cb(cores, 0, "bfloat16", BUFFER_PAGE, BUFFER_PAGE * BUFFER_PAGES)
    clock = Clock(card)
    clock.start()
    load = os.getloadavg()[0]
    log(f"launch: {len(parts)} cores, {args.bytes} bytes, mode {args.mode}")
    t_start = time.monotonic()
    launch.run()
    t_run = time.monotonic()
    seen, deadline = 0, t_run + args.wait
    seq = channel[SEQ_AT : SEQ_AT + 64 * len(parts)].view("<u4").reshape(-1, 16)
    while time.monotonic() < deadline:
        seen = sum(int(seq[j, 3]) == MAGIC ^ nonce for j in range(len(parts)))
        if seen == len(parts):
            break
        time.sleep(0.0005)
    t_seen = time.monotonic()
    if seen != len(parts):
        clock.stop.set()
        return {"hang": True, "seen": seen, "waited_s": round(t_seen - t_run, 2)}
    session.device.finish()
    t_fin = time.monotonic()
    clock.stop.set()
    clock.join()
    rep = flags.read().astype(np.int64)
    log("kernels done")
    # Every byte, and the guards.
    got = channel[: args.bytes]
    bad = np.flatnonzero(got != want)
    guard_ok = bool((channel[args.bytes : args.bytes + GUARD] == 0x5A).all())
    words_ok = all(
        list(seq[j, :4]) == [nonce, n * PAGE, j, MAGIC ^ nonce]
        for j, (_, n) in enumerate(parts)
    )

    def wall(r, k):
        return int(r[k]) | int(r[k + 1]) << 32

    t0 = min(wall(r, START) for r in rep)
    t1 = max(wall(r, DATA) for r in rep)
    t2 = max(wall(r, END) for r in rep)
    amin, amean = clock.stats()
    hz = 1.35e9  # the wall clock runs at the nominal AICLK; see the README
    return {
        "mode": args.mode,
        "cores": len(parts),
        "bytes": args.bytes,
        "chunk": args.chunk,
        "noc": args.noc,
        "exact": bool(bad.size == 0),
        "differing_bytes": int(bad.size),
        "first_bad": int(bad[0]) if bad.size else None,
        "guard_untouched": guard_ok,
        "sequence_words_ok": bool(words_ok),
        "status": [int(r[STATUS]) for r in rep],
        "pushed": [int(r[PUSHED]) for r in rep],
        "routing_clear": [int(r[MIDCLEAR]) for r in rep],
        "card_cycles_data": t1 - t0,
        "card_cycles_all": t2 - t0,
        "card_mb_s": round(args.bytes / ((t1 - t0) / hz) / 1e6, 1),
        "host_s_run": round(t_run - t_start, 5),
        "host_s_to_seen": round(t_seen - t_start, 5),
        "host_mb_s": round(args.bytes / (t_seen - t_start) / 1e6, 1),
        "host_s_to_finish": round(t_fin - t_start, 5),
        "aiclk_min_mean": [amin, amean],
        "load": round(load, 1),
    }


def readback(session, tensor_bytes, chunk):
    """Session.read_chunks over a tensor of the same bytes: seconds and MB/s."""
    pages = tensor_bytes // PAGE
    src = np.zeros((pages, PAGE), np.uint8)
    tensor = session.tensor(src, "uint8", "row")
    t0, got = time.monotonic(), 0
    for _, rows in session.read_chunks(tensor, chunk):
        got += rows.size
        log(f"read_chunks: {got >> 20} MiB at {time.monotonic() - t0:6.2f} s")
    dt = time.monotonic() - t0
    return {"bytes": got, "s": round(dt, 3), "mb_s": round(got / dt / 1e6, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["pattern", "dram"], default="pattern")
    p.add_argument("--bytes", type=int, default=4096, help="a multiple of 4096")
    p.add_argument("--cores", type=int, default=1, help="cores 0..n-1 of row 0")
    p.add_argument("--chunk", type=int, default=32768, help="bytes a write batch")
    p.add_argument("--noc", type=int, default=0, choices=[0, 1])
    p.add_argument("--wait", type=float, default=10.0, help="s for the host to wait")
    p.add_argument("--reps", type=int, default=1, help="pushes in one session")
    p.add_argument("--readback", action="store_true", help="also time read_chunks")
    args = p.parse_args()
    if os.environ.get("TT_VISIBLE_DEVICES") != "1":
        sys.exit("stop: TT_VISIBLE_DEVICES must be 1 (card 0 is not for this run)")
    if not os.environ.get("FR_LOG"):
        sys.exit("stop: FR_LOG names the log of the watched job")
    card = 1
    if args.bytes % PAGE or not 0 < args.bytes <= 20 << 20:
        sys.exit("stop: --bytes is a multiple of 4096 up to 20 MiB")
    if args.chunk % PAGE or not PAGE <= args.chunk <= 32768:
        sys.exit("stop: --chunk is a multiple of 4096 up to 32768")
    if not 1 <= args.cores <= 12:
        sys.exit("stop: --cores is 1 to 12")
    speed, width = link(card)
    log(f"link {speed}, x{width}; aiclk {sysfs(card, 'tt_aiclk').strip()} MHz")
    with km.Session() as session:
        log("device open")
        refuse_on_warning(os.environ["FR_LOG"])
        channel = map_channel()
        for rep in range(args.reps):
            out = run(args, session, channel, card)
            out.update(rep=rep, link=[speed, width])
            print(json.dumps(out), flush=True)
            if out.get("hang") or not (out["exact"] and out["guard_untouched"]):
                sys.exit(3)
        if args.readback:
            out = readback(session, args.bytes, 2 << 20)
            print(json.dumps({"read_chunks": out}), flush=True)


if __name__ == "__main__":
    main()
