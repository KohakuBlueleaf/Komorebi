"""The timing both TT-NN reference benchmarks share: a loop of calls and one device
synchronize, sized to about a second, five loops, the median; whole-card power, AICLK
and temperature sampled over each loop; a cool-down as long as the loops after them.
The device calls are the callers' (`call`, `sync`), so the helpers run without a card.
"""

import contextlib
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from komorebi import power  # noqa: E402

LOOPS = 5
TARGET_S = 1.0
MIN_CALLS = 3
PROBE_S = 0.2
HEARTBEAT_S = 10.0
AICLK = "/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"


def log(msg):
    """One progress line, flushed (the watcher reads the log's growth)."""
    print(msg, flush=True)


@contextlib.contextmanager
def heartbeat(what, every=HEARTBEAT_S):
    """Prints a line every `every` seconds while a host-side step runs, so a long
    build, upload or read-back is not mistaken for a hang."""
    stop, t0 = threading.Event(), time.monotonic()

    def beat():
        while not stop.wait(every):
            log(f"{what}: {time.monotonic() - t0:.0f} s")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def calls_for(per_call_s, target_s=TARGET_S, minimum=MIN_CALLS):
    """The calls a loop of about `target_s` takes at `per_call_s` each (at least
    `minimum`)."""
    return max(minimum, round(target_s / per_call_s))


def size_loop(call, sync, target_s=TARGET_S, probe_s=PROBE_S):
    """The calls per loop: probes of 3, 12, 48, ... calls back to back until one takes
    `probe_s`, then that probe's rate scaled to `target_s`."""
    n = MIN_CALLS
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        sync()
        elapsed = time.perf_counter() - t0
        if elapsed >= probe_s or n >= 1 << 20:
            return calls_for(elapsed / n, target_s)
        n *= 4


def sample_clock(card, samples, stop):
    """AICLK (MHz) every 20 ms into `samples` until `stop` is set."""
    path = Path(AICLK.format(card=card))
    while not stop.wait(0.02):
        samples.append(int(path.read_text().split()[0]))


def visible_card():
    """The card index the process sees (the first of TT_VISIBLE_DEVICES, else 0)."""
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])


def spread(values):
    """[min, mean, max] of a list, the mean to one decimal; None when it is empty."""
    return (
        [min(values), round(statistics.mean(values), 1), max(values)]
        if values
        else None
    )


def summarize(loops):
    """One config's figures from its loops, each {"microseconds", "w", "peak_w",
    "temp_c", "aiclk"}: the median time and the means of power and temperature."""
    watts = [x["w"] for x in loops if x["w"] is not None]
    temps = [x["temp_c"] for x in loops if x["temp_c"] is not None]
    clocks = [c for x in loops for c in x["aiclk"]]
    return {
        "microseconds_per_call": statistics.median(x["microseconds"] for x in loops),
        "microseconds_per_loop": [x["microseconds"] for x in loops],
        "mean_w": statistics.mean(watts) if watts else None,
        "w_per_loop": [x["w"] for x in loops],
        "peak_w": max((x["peak_w"] for x in loops if x["peak_w"]), default=None),
        "temp_c": statistics.mean(temps) if temps else None,
        "aiclk_mhz": spread(clocks),
    }


def measure(call, sync, card, loops=LOOPS, target_s=TARGET_S):
    """The loops of `call`: the first two calls (compile, program cache) and the
    sizing probes are outside; each loop is n calls and one `sync`. Returns the
    summary with n and the first call's time; sleeps as long as the loops took."""
    t0 = time.perf_counter()
    call()
    sync()
    first_ms = (time.perf_counter() - t0) * 1e3
    call()
    sync()
    n = size_loop(call, sync, target_s)
    log(f"first call {first_ms:.1f} ms; {loops} loops of {n} calls")
    meter, rows = power.PowerSampler(card), []
    with meter:
        for i in range(loops):
            clock, stop = [], threading.Event()
            sampler = threading.Thread(target=sample_clock, args=(card, clock, stop))
            meter.mark(f"loop{i}")
            sampler.start()
            t0 = time.perf_counter()
            for _ in range(n):
                call()
            sync()
            seconds = time.perf_counter() - t0
            stop.set()
            sampler.join()
            rows.append(
                {"seconds": seconds, "microseconds": seconds / n * 1e6, "aiclk": clock}
            )
            log(f"loop {i}: {rows[-1]['microseconds']:.2f} us per call")
        meter.mark("after")
    for i, row in enumerate(rows):
        phase = meter.result.phases[f"loop{i}"]
        row.update(w=phase.mean_w, peak_w=phase.peak_w, temp_c=phase.mean_temp_c)
    out = summarize(rows)
    out.update(calls_per_loop=n, first_call_ms=first_ms, loops=loops)
    out["open_w"] = meter.result.idle_mean_w
    log(f"cool-down {sum(r['seconds'] for r in rows):.0f} s")
    time.sleep(sum(r["seconds"] for r in rows))
    return out


def command(argv):
    """The command line as the result records it: the script's name and its options,
    without the output path."""
    args = list(argv[1:])
    if "--json" in args:
        i = args.index("--json")
        del args[i : i + 2]
    return [Path(argv[0]).name, *args]


def tt_metal_version(home):
    """`git describe` of the tt-metal tree at `home`, else None."""
    done = subprocess.run(
        ["git", "-C", str(home), "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return done.stdout.strip() or None
