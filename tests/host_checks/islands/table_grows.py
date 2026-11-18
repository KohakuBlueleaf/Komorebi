"""The island tensor table grows past 32 like one card's tensors: 40 then 130 tensors
on two cards, each summed by example 26's kernel through its table word, exact; a live
run's full table refuses the next tensor by name. Under kmr prebuild (the mock) it
checks the growth and the refusal, and builds the loads. Prints one JSON line."""

import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.device import mock

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT / "examples" / "26_islands_sum"))
import sum as example  # noqa: E402

TILE, TILES = 32, 64
STARTED = time.monotonic()


def made(session, count, seed):
    """`count` tensors of TILES tiles, a copy on every island, and their data."""
    generator = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        data = generator.integers(-2, 3, (TILES * TILE, TILE)).astype(np.float32)
        out.append((session.tensor(data, "float32", "tile", copy=True), data))
    return out


def summed(session, tensors, output):
    """Each tensor summed by the kernel through its word: the wrong ones' words."""
    run = session.load(example.program(tensors[0][0], output), deadline=1.0)
    island0, wrong = run.island(0), []
    for tensor, data in tensors:
        words = [tensor.address, TILES, output.address, 0, 1, 0, 0, 1]
        event = island0.wait(island0.submit("sum", (0, session.ranks), words), 5.0)
        (raw,) = session.sessions[0].read(output.parts[0])
        got = formats.from_bytes(raw, output.shape, output.dtype, output.layout)
        want = data.reshape(-1, TILE, TILE).sum(0)
        if event.kind != "done" or not np.array_equal(got[:TILE], want):
            wrong.append(tensor.address)
    return run, wrong


def main():
    report = {"mock": mock.active()}
    with km.Session(cards=[0, 1], links=1) as session:
        output = session.tensor(
            np.zeros((TILE, TILE), np.float32), "float32", "tile", copy=True
        )
        first = made(session, 39, 1)  # with the output: 40 tensors
        report["entries_40"] = len(session._entries)
        run, wrong = summed(session, first, output)
        report["table_40"] = [session._table_size, bool(session._drams)]
        report["aligned_40"] = [t.address % 64 == 0 for t in session._tables]
        report["wrong_40"] = wrong
        run.stop()
        second = first + made(session, 90, 2)  # 130 tensors
        report["entries_130"] = len(session._entries)
        run, wrong = summed(session, second, output)
        report["table_130"] = [session._table_size, bool(session._drams)]
        report["aligned_130"] = [t.address % 64 == 0 for t in session._tables]
        report["wrong_130"] = wrong
        live = made(session, session._table_size - 130, 3)  # every word, run live
        try:
            made(session, 1, 4)
            report["refused"] = None
        except km.KomorebiError as error:
            report["refused"] = str(error)
        run.stop()
        for tensor, _ in live:
            tensor.free()
    exact = report["mock"] or not (report["wrong_40"] or report["wrong_130"])
    report["ok"] = bool(
        exact
        and report["entries_40"] == 64
        and report["entries_130"] == 256
        and report["table_130"] == [256, True]
        and all(report["aligned_40"] + report["aligned_130"])
        and "live run's tensor table is full" in (report["refused"] or "")
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
