"""Y = X W at 128 x 896 x 128 as a class: 16 cores, each Y's tile (row r, column n) over
the whole K, W's column held in a field (filled once) or read from DRAM each call.
Each form one call at a time, 32 as a kept graph the X280 submits, and 32 as one chain
command (call c its own Y); every Y against float64. Prints one JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.engines.graph import Graph  # noqa: E402
from reference import dense as ref  # noqa: E402

RMS_BOUND = 1e-2  # the dense project's product bound
CALLS = 32
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Small(km.Program):
    """Y [32 M, 32 N] = X [32 M, 32 K] W [32 K, 32 N]; core r N + n makes Y's tile
    (r, n), written to Y's page c M N + r N + n by call c."""

    M_TILES = km.constant(4)
    K_TILES = km.constant(28)
    N_TILES = km.constant(4)
    CHUNK = km.constant(4)  # K tiles a block
    w: km.L1[km.bf16, K_TILES, 1, 32, 32]  # this core's column of W

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def load(self, w: km.Tensor):
        n = self.rank % self.N_TILES
        self.w.fill(km.Stream(w.tiles[0 : self.K_TILES, n : n + 1]))

    @km.handler
    def held(self, x: km.Tensor, y: km.Tensor, call: int):
        r = self.rank // self.N_TILES
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[r : r + 1, 0 : self.K_TILES], block=self.K_TILES):
            acc += self.project(xb, self.w)
        y.pages(call * self.M_TILES * self.N_TILES + self.rank, 1).write(acc)

    @km.handler
    def read(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, call: int):
        r = self.rank // self.N_TILES
        n = self.rank % self.N_TILES
        acc = km.tile.zeros(km.fp32, 1)
        for xb, wb in zip(
            km.Stream(x.tiles[r : r + 1, 0 : self.K_TILES], block=self.CHUNK),
            km.Stream(w.tiles[0 : self.K_TILES, n : n + 1], block=self.CHUNK),
        ):
            acc += self.project(xb, wb)
        y.pages(call * self.M_TILES * self.N_TILES + self.rank, 1).write(acc)


def kept_graph(run, handler, group, words, rounds, restarts=3):
    """`rounds` calls as a kept graph the X280 submits (each after the one before is
    done), restarted `restarts` times: the median wall microseconds a call."""
    kept = Graph(deadline_microseconds=5_000_000, max_runs=rounds, req=12, keep=True)
    iteration = kept.register()
    command = kept.command(handler, group=group, args=words, exclusive=True)
    step = kept.cpu(after=[command])
    step.addi(iteration, iteration, 1)
    step.lt(kept.again, iteration, kept.const(rounds))
    upload = run.upload(kept)
    run.wait_engine(run.start(upload, kept=True), timeout=60.0)
    each = []
    for _ in range(restarts):
        started = time.perf_counter()
        run.wait_engine(run.restart(upload.req), timeout=60.0)
        each.append((time.perf_counter() - started) * 1e6 / rounds)
    run.release(upload)
    return float(np.median(each))


def form(session, run, index, name, given, group, y):
    """One form: single calls, the kept graph, the chain of CALLS calls (y zeroed
    before its timed command): its times and the clock around them."""
    result, handler = {}, getattr(Small, name)
    before = power.aiclk_mhz(index)
    singles = []
    for _ in range(6):
        event = run.wait(run.submit(handler, group, args=(*given, 0)))
        if event.kind != "done":
            raise km.KomorebiError(f"{handler}: {event}")
        singles.append(event.microseconds)
    result["single_card_us"] = float(np.median(singles[1:]))
    words = [int(word) for word in handler.args(*given, 0)]
    result["graph32_us_per_call"] = kept_graph(run, int(handler), group, words, CALLS)
    chain = run.chain([handler(*given, c, cores=group) for c in range(CALLS)])
    run.wait(run.submit(chain), timeout=30.0)  # its first
    session.write(y, np.zeros((y.pages, 32 * 32 * 4), np.uint8))
    started = time.perf_counter()
    event = run.wait(run.submit(chain), timeout=30.0)
    host_s = time.perf_counter() - started
    if event.kind != "done":
        raise km.KomorebiError(f"chain: {event}")
    result["chain32_us_per_call"] = host_s * 1e6 / CALLS
    result["chain32_card_us_per_call"] = event.microseconds / CALLS
    result["mhz"] = [before, power.aiclk_mhz(index)]
    log(f"{name}: {json.dumps(result)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    m, k, n = (c.default * 32 for c in (Small.M_TILES, Small.K_TILES, Small.N_TILES))
    rng = np.random.default_rng(args.seed)
    x = ref.bf16(rng.uniform(-1, 1, (m, k)).astype(np.float32))
    w = ref.bf16((rng.uniform(-1, 1, (k, n)) / np.sqrt(k)).astype(np.float32))
    want = ref.matmul(x, w)
    report = {"m_k_n": [m, k, n]}
    index = power.visible_card()
    group = (0, Small.M_TILES.default * Small.N_TILES.default)
    with km.Session(log=log) as session:
        tx = session.tensor(x, "bfloat16", "tile")
        tw = session.tensor(w, "bfloat16", "tile")
        ys = {
            name: session.empty((CALLS * m, n), "float32", "tile")
            for name in ("read", "held")
        }
        program = Small(log=log, session=session, chain=True, x=tx, w=tw, y=ys["read"])
        run = session.load(program)
        for name, given in (("read", (tx, tw, ys["read"])), ("held", (tx, ys["held"]))):
            if name == "held":
                event = run.wait(run.submit(Small.load, group, args=(tw,)))
                report["load_us"] = event.microseconds
            y = ys[name]
            report[name] = form(session, run, index, name, given, group, y)
        run.stop()
        report["kernel"] = run.result()["error_name"]
        outputs = {name: y.read() for name, y in ys.items()}
    ok = report["kernel"] == "none"
    for name, values in outputs.items():
        got = values.reshape(CALLS, m, n)
        each = [ref.errors(got[c], want) for c in range(CALLS)]
        worst = {key: max(e[key] for e in each) for key in ("rms_rel", "max_rel")}
        worst["ok"] = worst["rms_rel"] <= RMS_BOUND
        report[name]["check"] = worst
        ok = ok and worst["ok"]
    report["ok"] = ok
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
