"""The class's attend as 32 calls of one chain command (shape causal64), call c its
own Q, K, V (seed c + 1) and O; every O against float64 at the attention project's
bounds, the chain run three times. One JSON line; exits 1 when a check fails."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from attention import DT, DV, KB, LENGTH, WINDOW, Attention, R

import komorebi as km

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from komorebi.bench import power  # noqa: E402
from projects.attention import card  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

RMS_BOUND, MAX_BOUND = 1.5e-2, 6e-2  # projects/attention/run.py's


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--save", default=None, help="every call's O to this .npy")
    args = parser.parse_args()
    calls = args.calls
    batches, heads, kv_heads, length, head_dim, causal = pl.SHAPES["causal64"]
    sizes = (batches, heads, kv_heads, length, length, head_dim)
    call = pl.dims(*sizes, head_dim, causal)
    scale = 1.0 / np.sqrt(head_dim)
    data = [sdpa.inputs(*sizes, head_dim, seed + 1) for seed in range(calls)]
    report = {"shape": "causal64", "calls": calls}
    index = power.visible_card()
    with km.Session() as session:
        # Over the run's ranks, at the class's broadcast length (its own grid's may
        # differ: 5 on a Wormhole chip's 7 x 10 beside its dispatcher).
        plan, grid = card.card_plan(session, call, length=LENGTH)
        made = (plan.r, plan.kb, plan.window, plan.length, call.dt, call.dvt)
        if made != (R, KB, WINDOW, LENGTH, DT, DV):
            raise SystemExit(f"plan {made}: the class is of {(R, KB, WINDOW, LENGTH)}")
        tables = card.table_tensor(session, pl.tables(plan))
        zeros = np.zeros((*data[0][0].shape[:-1], head_dim), np.float32)
        ts = [[card.bf16_tensor(session, x) for x in qkv] for qkv in data]
        os_ = [card.bf16_tensor(session, zeros) for _ in range(calls)]
        scale_bits, threshold_bits = pl.exp_constants(scale)
        defines = {
            "FLASH_SCALE_BITS": scale_bits,
            "FLASH_THRESHOLD_BITS": threshold_bits,
            "FLASH_CAUSAL": 1,
        }
        q, k, v = ts[0]
        program = Attention(
            q=q, k=k, v=v, o=os_[0], tables=tables, defines=defines,
            session=session, chain=True, **({"rect": (0, 0, *grid)} if grid else {}),
        )  # fmt: skip
        run = session.load(program)
        group = (0, len(plan.jobs))
        words = (tables, pl.K["kModeFull"], 1, 0, 0)
        singles = []
        for _ in range(3):
            event = run.wait(
                run.submit(Attention.attend, group, args=(q, k, v, os_[0], *words))
            )
            if event.kind != "done":
                raise km.KomorebiError(f"single: {event}")
            singles.append(event.microseconds)
        report["single_card_us"] = float(np.median(singles))
        chain = run.chain(
            [
                Attention.attend(*ts[c], os_[c], *words, cores=group)
                for c in range(calls)
            ]
        )  # fmt: skip
        before = power.aiclk_mhz(index)
        run.wait(run.submit(chain), timeout=30.0)  # its first
        for o in os_:
            session.write(o, np.zeros((o.pages, 32 * 32 * 2), np.uint8))
        walls, cards = [], []
        for _ in range(3):
            started = time.perf_counter()
            event = run.wait(run.submit(chain), timeout=30.0)
            walls.append((time.perf_counter() - started) * 1e6 / calls)
            if event.kind != "done":
                raise km.KomorebiError(f"chain: {event}")
            cards.append(event.microseconds / calls)
        report["mhz"] = [before, power.aiclk_mhz(index)]
        report["chain32_us_per_call"] = float(np.median(walls))
        report["chain32_card_us_per_call"] = float(np.median(cards))
        report["chain32_walls"] = walls
        run.stop()
        report["kernel"] = run.result()["error_name"]
        got = [o.read() for o in os_]
    if args.save:  # for a comparison between chips, word for word
        np.save(args.save, np.stack([np.asarray(o, np.float32) for o in got]))
    worst = {"rms_rel": 0.0, "max_rel": 0.0}
    group_size = heads // kv_heads
    rows = np.arange(length)
    for c in range(calls):
        qs, ks, vs = data[c]
        o = np.asarray(got[c], np.float32).reshape(batches, heads, length, head_dim)
        for b in range(batches):
            for h in range(heads):
                want = sdpa.attention(
                    qs[b, h], ks[b, h // group_size], vs[b, h // group_size],
                    rows, causal, scale,
                )  # fmt: skip
                errors = sdpa.errors(o[b, h], want)
                for key in worst:
                    worst[key] = max(worst[key], float(errors[key]))
    worst["ok"] = worst["rms_rel"] <= RMS_BOUND and worst["max_rel"] <= MAX_BOUND
    report["check"] = worst
    report["ok"] = worst["ok"] and report["kernel"] == "none"
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
