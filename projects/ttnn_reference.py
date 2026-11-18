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

from komorebi.bench import power  # noqa: E402

LOOPS = 5
TARGET_S = 1.0
MIN_CALLS = 3
PROBE_S = 0.2
HEARTBEAT_S = 10.0
AICLK = "/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"


def log(message):
    """One progress line, flushed (the watcher reads the log's growth)."""
    print(message, flush=True)


@contextlib.contextmanager
def heartbeat(what, every=HEARTBEAT_S):
    """Prints a line every `every` seconds while a host-side step runs, so a long
    build, upload or read-back is not mistaken for a hang."""
    stop, started = threading.Event(), time.monotonic()

    def beat():
        while not stop.wait(every):
            log(f"{what}: {time.monotonic() - started:.0f} s")

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
    calls = MIN_CALLS
    while True:
        started = time.perf_counter()
        for _ in range(calls):
            call()
        sync()
        elapsed = time.perf_counter() - started
        if elapsed >= probe_s or calls >= 1 << 20:
            return calls_for(elapsed / calls, target_s)
        calls *= 4


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
    watts = [loop["w"] for loop in loops if loop["w"] is not None]
    temps = [loop["temp_c"] for loop in loops if loop["temp_c"] is not None]
    clocks = [clock for loop in loops for clock in loop["aiclk"]]
    return {
        "microseconds_per_call": statistics.median(
            loop["microseconds"] for loop in loops
        ),
        "microseconds_per_loop": [loop["microseconds"] for loop in loops],
        "mean_w": statistics.mean(watts) if watts else None,
        "w_per_loop": [loop["w"] for loop in loops],
        "peak_w": max(
            (loop["peak_w"] for loop in loops if loop["peak_w"]), default=None
        ),
        "temp_c": statistics.mean(temps) if temps else None,
        "aiclk_mhz": spread(clocks),
    }


def measure(call, sync, card_index, loops=LOOPS, target_s=TARGET_S):
    """The loops of `call`: the first two calls (compile, program cache) and the
    sizing probes are outside; each loop is `calls_per_loop` calls and one `sync`.
    Returns the summary with the calls per loop and the first call's time; sleeps as
    long as the loops took."""
    started = time.perf_counter()
    call()
    sync()
    first_ms = (time.perf_counter() - started) * 1e3
    call()
    sync()
    calls_per_loop = size_loop(call, sync, target_s)
    log(f"first call {first_ms:.1f} ms; {loops} loops of {calls_per_loop} calls")
    meter, measurements = power.PowerSampler(card_index), []
    with meter:
        for loop_index in range(loops):
            clock_samples, stop = [], threading.Event()
            sampler = threading.Thread(
                target=sample_clock, args=(card_index, clock_samples, stop)
            )
            meter.mark(f"loop{loop_index}")
            sampler.start()
            started = time.perf_counter()
            for _ in range(calls_per_loop):
                call()
            sync()
            seconds = time.perf_counter() - started
            stop.set()
            sampler.join()
            measurements.append(
                {
                    "seconds": seconds,
                    "microseconds": seconds / calls_per_loop * 1e6,
                    "aiclk": clock_samples,
                }
            )
            log(
                f"loop {loop_index}: {measurements[-1]['microseconds']:.2f} us per call"
            )
        meter.mark("after")
    for loop_index, measurement in enumerate(measurements):
        phase = meter.result.phases[f"loop{loop_index}"]
        measurement.update(
            w=phase.mean_w, peak_w=phase.peak_w, temp_c=phase.mean_temp_c
        )
    summary = summarize(measurements)
    summary.update(calls_per_loop=calls_per_loop, first_call_ms=first_ms, loops=loops)
    summary["open_w"] = meter.result.idle_mean_w
    log(
        f"cool-down {sum(measurement['seconds'] for measurement in measurements):.0f} s"
    )
    time.sleep(sum(measurement["seconds"] for measurement in measurements))
    return summary


def read_clock(card):
    """AICLK (MHz) of `card` now."""
    return int(Path(AICLK.format(card=card)).read_text().split()[0])


def pairs_and_32(call, sync, card_index, pairs=3, loop_calls=32, pause_s=1.0):
    """The short timing: `pairs` x (a warm-up call, one timed call waited for, a
    pause), their median; then `loop_calls` calls back to back and one sync. AICLK is
    read around each timed part."""
    singles = []
    for _ in range(pairs):
        call()
        sync()
        before = read_clock(card_index)
        started = time.perf_counter()
        call()
        sync()
        singles.append(
            {
                "microseconds": (time.perf_counter() - started) * 1e6,
                "aiclk": [before, read_clock(card_index)],
            }
        )
        time.sleep(pause_s)
    before = read_clock(card_index)
    started = time.perf_counter()
    for _ in range(loop_calls):
        call()
    sync()
    loop_us = (time.perf_counter() - started) * 1e6 / loop_calls
    summary = {
        "single_microseconds": statistics.median(
            one["microseconds"] for one in singles
        ),
        "singles": singles,
        "loop_microseconds_per_call": loop_us,
        "loop_calls": loop_calls,
        "loop_aiclk": [before, read_clock(card_index)],
        "microseconds_per_call": loop_us,
        "aiclk_mhz": [before, read_clock(card_index)],
    }
    log(
        f"single {summary['single_microseconds']:.2f} us, "
        f"{loop_calls} back to back {loop_us:.2f} us a call"
    )
    return summary


def command(argv):
    """The command line as the result records it: the script's name and its options,
    without the output path."""
    args = list(argv[1:])
    if "--json" in args:
        index = args.index("--json")
        del args[index : index + 2]
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
