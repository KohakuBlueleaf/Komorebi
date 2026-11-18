"""Y = X W at 1024 cubed as a class in the grid form (experiments/language/dense_class's
`grid`): core (x, y) of 8 x 8 makes Y's block of 4 x 4 tiles, X's K chunks cast along
its row and W's down its column. Calls one at a time, 32 as a kept graph the X280
submits, then 32 as one chain command (call c its own X and Y); every Y against
float64. Prints one JSON line; exits 1 when the check fails."""

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
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Grid(km.Program):
    """Y [32 M, 32 N] = X [32 M, 32 K] W [32 K, 32 N] on COLUMNS x ROWS cores, OH x OW
    tiles a core; call c writes Y's tile rows from c M."""

    M_TILES = km.constant(32)
    K_TILES = km.constant(32)
    N_TILES = km.constant(32)
    COLUMNS = km.constant(8)
    ROWS = km.constant(8)
    KC = km.constant(2)  # K tiles a cast
    OH = M_TILES // ROWS
    OW = N_TILES // COLUMNS
    part: km.L1[km.fp32, OH, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, call: int):
        along = km.Broadcast(self, along="row", window=2)
        down = km.Broadcast(self, along="column", window=2)
        top = self.y * self.OH
        left = self.x * self.OW
        row = call * self.M_TILES + top  # call c's X and Y: rows from c M
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if along.sender():
                xs = along.cast(x.tiles[row : row + self.OH, k0 : k0 + self.KC].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[row : row + self.OH, left : left + self.OW].write(self.part)


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
    return float(np.median(each)), each


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kc", type=int, default=2, help="K tiles a cast")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main():
    args = parse()
    size = Grid.M_TILES.default * 32
    rng = np.random.default_rng(args.seed)
    calls = 32  # the chain's: call c reads X's rows from c M and writes Y's
    x = ref.bf16(rng.uniform(-1, 1, (calls * size, size)).astype(np.float32))
    w = ref.bf16((rng.uniform(-1, 1, (size, size)) / 32).astype(np.float32))
    report = {"m_k_n": [size] * 3, "kc": args.kc}
    index = power.visible_card()
    with km.Session(log=log) as session:
        tx = session.tensor(x, "bfloat16", "tile")
        tw = session.tensor(w, "bfloat16", "tile")
        ty = session.empty((calls * size, size), "bfloat16", "tile")
        rect = (0, 0, Grid.COLUMNS.default, Grid.ROWS.default)
        program = Grid(
            log=log,
            session=session,
            rect=rect,
            chain=True,
            KC=args.kc,
            x=tx,
            w=tw,
            y=ty,
        )
        run = session.load(program)
        group = program.rectangle["group"]
        singles = []
        for _ in range(6):
            event = run.wait(run.submit(Grid.grid, group, args=(tx, tw, ty, 0)))
            if event.kind != "done":
                raise km.KomorebiError(f"grid: {event}")
            singles.append(event.microseconds)
        before = power.aiclk_mhz(index)
        report["single_card_us"] = float(np.median(singles[1:]))
        log(f"SINGLE {report['single_card_us']} us on the card")
        words = [int(word) for word in Grid.grid.args(tx, tw, ty, 0)]
        report["graph32_us_per_call"], report["graph_restarts"] = kept_graph(
            run, int(Grid.grid), group, words, 32
        )
        log(f"GRAPH {report['graph32_us_per_call']} us a call")
        chain = run.chain([Grid.grid(tx, tw, ty, c, cores=group) for c in range(calls)])
        run.wait(run.submit(chain), timeout=30.0)  # its first
        session.write(ty, np.zeros((ty.pages, 32 * 32 * 2), np.uint8))
        started = time.perf_counter()
        event = run.wait(run.submit(chain), timeout=30.0)
        host_s = time.perf_counter() - started
        if event.kind != "done":
            raise km.KomorebiError(f"chain: {event}")
        report["chain32_us_per_call"] = host_s * 1e6 / calls
        report["chain32_card_us_per_call"] = event.microseconds / calls
        log(f"CHAIN {report['chain32_us_per_call']} us a call")
        report["mhz"] = [before, power.aiclk_mhz(index)]
        run.stop()
        report["kernel"] = run.result()["error_name"]
        got = ty.read().reshape(calls, size, size)
    xs = x.reshape(calls, size, size)
    checks = [ref.errors(got[c], ref.matmul(xs[c], w)) for c in range(calls)]
    report["check"] = {key: max(c[key] for c in checks) for key in checks[0]}
    report["check"]["calls"] = calls
    report["ok"] = (
        report["check"]["rms_rel"] <= RMS_BOUND and report["kernel"] == "none"
    )
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
