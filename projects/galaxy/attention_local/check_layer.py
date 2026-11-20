"""The layer's attention (layer.py) on one chip with random weights over a sequence of
steps (prefill blocks, then decode tokens past the window's wrap), each step's y
against float64 of model.py's attention from x with the weights as stored, the window
kept across the steps. Prints one JSON line a step. README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from projects.galaxy.attention_local import attend as at  # noqa: E402
from projects.galaxy.attention_local import layer as ly  # noqa: E402
from projects.galaxy.attention_local import proj as pj  # noqa: E402
from projects.galaxy.attention_local import window as wn  # noqa: E402

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def bf16(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def errors(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
    }


class Reference:
    """model.py's Attention.forward in float64 (no fp8 activation quantization): the
    weights (in, out) as stored, every position's rotated key kept."""

    def __init__(self, weights, compressed):
        self.w = {k: np.asarray(v, np.float64) for k, v in weights.items()}
        self.freqs = wn.frequencies(compressed)
        self.keys = []

    @staticmethod
    def norm(x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + pj.EPS) * g

    def forward(self, x, start):
        w, tokens = self.w, len(x)
        x = np.asarray(x, np.float64)
        qr = self.norm(x @ w["wq_a"], w["q_norm"])
        q = (qr @ w["wq_b"]).reshape(tokens, pj.HEADS, pj.HEAD_DIM).transpose(1, 0, 2)
        q = wn.rotate(q, self.freqs, start)
        kv = wn.rotate(self.norm(x @ w["wkv"], w["kv_norm"]), self.freqs, start)
        self.keys[start:] = list(kv)
        keys = np.asarray(self.keys)
        visible = np.zeros((tokens, len(keys)), bool)
        for i in range(tokens):
            p = start + i
            visible[i, max(0, p - wn.WINDOW + 1) : p + 1] = True
        o = at.reference(q, keys, w["attn_sink"], visible)  # [64, tokens, 512]
        o = wn.rotate(o, self.freqs, start, inverse=True)
        o = o.transpose(1, 0, 2).reshape(tokens, -1)
        oa = np.concatenate(
            [
                o[:, g * 4096 : (g + 1) * 4096] @ w["wo_a"][g * 4096 : (g + 1) * 4096]
                for g in range(pj.O_GROUPS)
            ],
            axis=1,
        )
        return oa @ w["wo_b"]


def random_weights(rng):
    shapes = {
        "wq_a": (pj.DIM, pj.Q_RANK),
        "wq_b": (pj.Q_RANK, pj.HEADS * pj.HEAD_DIM),
        "wkv": (pj.DIM, pj.HEAD_DIM),
        "wo_a": (pj.HEADS * pj.HEAD_DIM, pj.O_RANK),
        "wo_b": (pj.O_GROUPS * pj.O_RANK, pj.DIM),
    }
    weights = {}
    for name, (k, n) in shapes.items():
        fan_in = k // pj.O_GROUPS if name == "wo_a" else k
        w = rng.standard_normal((k, n), dtype=np.float32) / np.float32(np.sqrt(fan_in))
        weights[name] = pj.stored(w, ly.WEIGHT_FORMAT)
    weights["q_norm"] = bf16(
        1.0 + 0.1 * rng.standard_normal(pj.Q_RANK, dtype=np.float32)
    )
    weights["kv_norm"] = bf16(
        1.0 + 0.1 * rng.standard_normal(pj.HEAD_DIM, dtype=np.float32)
    )
    weights["attn_sink"] = (0.3 * rng.standard_normal(pj.HEADS)).astype(np.float32)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compressed", type=int, default=0, help="1: YaRN's layer")
    parser.add_argument("--blocks", type=int, default=5, help="prefill blocks from 0")
    parser.add_argument("--decode", type=int, default=3, help="decode tokens after")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dump", help="a folder: each step's oa, y and wo_b, .npz")
    a = parser.parse_args()
    rng = np.random.default_rng(a.seed)
    weights = random_weights(rng)
    reference = Reference(weights, bool(a.compressed))
    steps = [(32 * b, True) for b in range(a.blocks)]
    steps += [(32 * a.blocks + i, False) for i in range(a.decode)]
    ok = True
    with km.Session(log=log) as s:
        shared = ly.Shared(s, 32 * a.blocks + a.decode + 32)
        layer = ly.Layer(s, shared, weights, a.compressed, log=log).load()
        try:
            for pos, prefill in steps:
                tokens = 32 if prefill else 1
                x = bf16(rng.standard_normal((tokens, pj.DIM), dtype=np.float32))
                host_us = layer.write_x(x)
                spent = layer.step(pos, prefill)
                got = layer.read("y")[:tokens]
                want = reference.forward(x, pos)
                report = {
                    "step": "prefill" if prefill else "decode",
                    "pos": pos,
                    "compressed": a.compressed,
                }
                report.update(errors(got, want))
                # each output step from the chip's own input: where an error starts
                w = reference.w
                o_chip = layer.read("o_flat").astype(np.float64)[:tokens]
                oa_chip = layer.read("oa").astype(np.float64)[:tokens]
                oa_own = np.concatenate(
                    [
                        o_chip[:, g * 4096 : (g + 1) * 4096]
                        @ w["wo_a"][g * 4096 : (g + 1) * 4096]
                        for g in range(pj.O_GROUPS)
                    ],
                    axis=1,
                )
                report["wo_a_own"] = errors(oa_chip, oa_own)
                report["wo_b_own"] = errors(got, oa_chip @ w["wo_b"])
                if a.dump:
                    np.savez(
                        f"{a.dump}/step_{pos}.npz", oa=oa_chip, y=got, wo_b=w["wo_b"]
                    )
                rows = [errors(got[i], want[i])["rms_rel"] for i in range(tokens)]
                worst = int(np.argmax(rows))
                column = int(np.argmax(np.abs(got[worst] - want[worst])))
                report["worst_row"] = {
                    "row": worst,
                    "rms_rel": rows[worst],
                    "column": column,
                    "got": float(got[worst, column]),
                    "want": float(want[worst, column]),
                }
                report["us_total"] = round(sum(spent.values()), 1)
                report["us"] = {k: round(v, 1) for k, v in spent.items()}
                report["x_write_us"] = round(host_us, 1)
                ok &= report["rms_rel"] < 2e-2
                log(json.dumps(report))
        finally:
            layer.stop()
    log(json.dumps({"ok": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
