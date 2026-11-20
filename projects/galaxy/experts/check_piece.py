"""The experts piece against golden/ (as galaxy.driver.verify compares) and, with
--stored, routed and shared also against float64 of the pages the chips hold (read from
the loader's cache). One JSON line a call; on fewer islands than the placement's,
placement island p goes to session island p mod the islands. README.md."""

import argparse
import json
import os

import numpy as np

import komorebi as km
from komorebi import formats
from projects.galaxy.driver.verify import compare, load_site
from projects.galaxy.experts import piece as experts
from projects.galaxy.weights import loader

SITES = {"gate": 13, "ffn": 13, "shared": 14, "routed": 15}
TILE = {"bfp4_b": 576, "bfp8_b": 1088, "bfloat16": 2048}


def stored(entry):
    """A weight's values as the chips hold them, from the loader's cache file."""
    name = f"i{entry['island']:02d}-{entry['dtype']}.bin"
    pages = np.memmap(os.path.join(loader.CACHE, name), np.uint8, "r")
    size = TILE[entry["dtype"]]
    raw = np.asarray(
        pages[entry["first"] * size : (entry["first"] + entry["pages"]) * size]
    )
    shape = tuple(entry["shape"])
    return np.asarray(
        formats.from_bytes(raw, shape, entry["dtype"], "tile"), np.float64
    )


def bf16(a):
    bits = np.asarray(a, np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32).astype(np.float64)


def expert(x, w1, w3, w2, r):
    h, g = np.minimum(bf16(x @ w1), 10.0), np.clip(bf16(x @ w3), -10.0, 10.0)
    return bf16(h / (1 + np.exp(-h)) * g * r[:, None]) @ w2


def emulate(entries, site, layer, ins):
    """float64 of routed or shared on the stored values, bf16 where the piece rounds."""
    x = ins[0].reshape(-1, 5120).float().numpy().astype(np.float64)
    parts = ("w1", "w3", "w2")
    if site == "shared":
        name = f"layers.{layer}.ffn.shared_experts"
        w = [stored(entries[f"{name}.{m}@{experts.SHARED_FORMAT}"]) for m in parts]
        return expert(x, *w, np.ones(len(x)))
    wts, idx = ins[1].numpy(), ins[2].numpy()
    y = np.zeros((len(x), 5120))
    for e in sorted(set(idx.reshape(-1).tolist())):
        name = f"layers.{layer}.ffn.experts.{e}"
        w = [stored(entries[f"{name}.{m}@bfp4_b"]) for m in parts]
        tok, k = np.nonzero(idx == e)
        r = np.zeros(len(x))
        r[tok] = wts[tok, k]
        y += expert(x, *w, r)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True, help="e.g. 20,21,22,23")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--layers", default="3")
    ap.add_argument("--sites", default="gate,shared,routed,ffn")
    ap.add_argument("--prompts", default="en")
    ap.add_argument("--steps", default="prefill,decode0,decode1")
    ap.add_argument("--stored", action="store_true")
    a = ap.parse_args()
    cards = [int(c) for c in a.cards.split(",")]
    layers = [int(v) for v in a.layers.split(",")]
    with open(loader.PLACEMENT) as f:
        entries = {e["name"]: e for e in json.load(f)["entries"]}
    with km.Session(cards=cards, links=1) as session:
        n = len(session.sessions)
        islands = {e["island"]: e["island"] % n for e in entries.values()}
        wanted = [
            k for k in entries if k.split(".")[1:3] in ([str(L), "ffn"] for L in layers)
        ]
        weights = loader.load(
            session, names=wanted, islands=islands, log=lambda m: None
        )
        piece = experts.make(session, weights, layers)
        for prompt in a.prompts.split(","):
            with open(os.path.join(a.golden, prompt, "manifest.json")) as f:
                manifest = json.load(f)
            for step in a.steps.split(","):
                for layer in layers:
                    for site in a.sites.split(","):
                        base = f"p{SITES[site]}_{site}_L{layer:02d}.npz"
                        path = os.path.join(a.golden, prompt, step, base)
                        ins, ref = load_site(path, f"{step}/{base}", manifest)
                        out = piece(site, layer, *ins)
                        row = {"prompt": prompt, "step": step, "site": site}
                        row.update(layer=layer, outputs=compare(out, ref))
                        row["times"] = piece.calls[-1]
                        if a.stored and site in ("routed", "shared"):
                            want = emulate(entries, site, layer, ins)
                            err = out.reshape(want.shape).double().numpy() - want
                            row["stored_rel_rms"] = float(
                                np.linalg.norm(err) / np.linalg.norm(want)
                            )
                        print(json.dumps(row), flush=True)
        piece.close()


if __name__ == "__main__":
    main()
