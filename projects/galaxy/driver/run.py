"""The driver: a prompt in, tokens out, every site on the host or on the box by a switch
(README.md, `driver/run.py`)."""

import argparse
import json
import os
import time

import numpy as np
import torch

from ..reference.model_loader import load_reference
from ..reference.prompts import encode, prompt_text
from . import state
from .interface import install, load_factory, load_weights, set_hooks, weights_for
from .path import SITE_PIECE, TokenPath

BLOCK = 32  # the pieces' one prefill block (tokens, from position 0)


def parse_layers(text, n_layers):
    if text in ("-", "none"):
        return [None]
    if text == "all":
        return list(range(n_layers))
    out = []
    for part in text.split(","):
        if part in ("-", "none"):  # the global sites
            out.append(None)
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_cards(text):
    return parse_layers(text, 0) if text else None


def open_session(km, cards, kernel_bytes=None):
    """A one-card session, or a session of islands over several cards."""
    given = {"kernel_bytes": kernel_bytes} if kernel_bytes else {}
    if len(cards) == 1:
        return km.Session(card=cards[0], **given).__enter__()
    return km.Session(cards=cards, links=1, **given).__enter__()


def open_checkpoint(path):
    """The Checkpoint, of the module name the pieces check its type against."""
    try:
        from galaxy.reference.model_loader import Checkpoint  # noqa: PLC0415
    except ImportError:
        from ..reference.model_loader import Checkpoint  # noqa: PLC0415
    return Checkpoint(path)


def dram_left(session):
    """Prints island 0's free DRAM after the pieces are made (tt-metal's allocator)."""
    first = session.sessions[0] if hasattr(session, "sessions") else session
    bank, banks, largest, free = first.device.dram_room(2048, 2048)
    print(
        f"island 0 DRAM free: {free * banks:,} B ({banks} banks, "
        f"{largest:,} B the largest piece a bank)",
        flush=True,
    )


def compare(logits, ref_logits):
    a, b = logits.float(), torch.from_numpy(ref_logits).float()
    rel = ((a - b).norm() / b.norm()).item()
    return rel, (a - b).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--prompt", default="en")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--piece", action="append", default=[])
    ap.add_argument(
        "--cards", default=None, help="the session's cards, e.g. 0-15,17-31"
    )
    ap.add_argument(
        "--load",
        default=None,
        help="the loader's W: 'all' or layers@kinds[@formats] (by weights_for)",
    )
    ap.add_argument(
        "--kernel-bytes",
        type=int,
        default=None,
        help="a joined session's kernel buffer",
    )
    ap.add_argument(
        "--one-load",
        action="store_true",
        help="the session's pieces made with run= one shared load (load/shared.py)",
    )
    ap.add_argument(
        "--prompt-by-decode",
        action="store_true",
        help="a prompt longer than 32 tokens: its first 32 as the prefill block, every "
        "further prompt token through a decode step with that token forced",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--report", default=None, help="write the report as JSON here")
    a = ap.parse_args()

    ref = load_reference(a.ckpt, max_seq_len=4096, threads=a.threads)
    path = TokenPath(ref)
    gdir = os.path.join(a.golden, a.prompt)
    with open(os.path.join(gdir, "tokens.json")) as f:
        gold = json.load(f)
    gold_logits = np.load(os.path.join(gdir, "logits.npy"), mmap_mode="r")
    ids = encode(ref, a.ckpt, prompt_text(ref, a.ckpt, a.prompt))
    assert ids == gold["prompt_ids"]

    session, W, pieces, remote = None, None, [], []
    ckpt = open_checkpoint(a.ckpt)
    hooks = {
        "compress_hook": lambda i, *x: path.compress_kv(
            i, path.model.layers[i].attn, *x
        )
    }
    if any("#" not in spec for spec in a.piece):
        import komorebi as km  # noqa: PLC0415 (only a run with box pieces needs it)

        session = open_session(km, parse_cards(a.cards), a.kernel_bytes)
        if a.load:
            t0 = time.time()
            W = load_weights(session, a.load, parse_layers)
            print(f"loaded {len(W)} weights in {time.time() - t0:.1f} s", flush=True)
    shared = None
    if a.one_load:  # the pieces of the driver's session hand their programs to one load
        from projects.galaxy.load.shared import Shared  # noqa: PLC0415 (box runs only)

        shared = Shared(session)
    if a.piece:
        n_layers = len(ref.model.layers)
        for n, spec in enumerate(a.piece):
            spec_, _, where = spec.partition("#")
            cards, _, load = where.partition("%")
            fac, _, rest = spec_.partition("@")
            groups = []  # (layers, sites) a group: `layers@sites+layers@sites...`
            for group in (rest or "all").split("+"):
                lay, _, sites = group.partition("@")
                groups.append((parse_layers(lay or "all", n_layers), sites))
            every = sorted(
                {x for g, _ in groups for x in g}, key=lambda x: (x is None, x)
            )
            if cards:  # a process of its own, only these cards visible to it
                from .remote import RemotePiece  # noqa: PLC0415 (remote imports run)

                log = f"{os.path.splitext(a.report or 'run')[0]}.piece{n}.log"
                every_text = ",".join("-" if x is None else str(x) for x in every)
                piece = RemotePiece(
                    fac, cards, every_text, a.ckpt, log, hooks, a.kernel_bytes, load
                )
                remote.append(piece)
            else:
                extra = {} if shared is None else {"run": shared}
                factory = load_factory(fac)
                weights = weights_for(factory, W, ckpt)
                piece = factory(session, weights, every, **extra)
                set_hooks(piece, hooks)
            for layers, sites in groups:
                install(
                    path,
                    piece,
                    layers=None if layers == [None] else layers,
                    sites=sites.split(",") if sites else None,
                )
            pieces.append((spec, piece.sites))
            print(f"piece {spec}: sites {piece.sites}", flush=True)
    if shared is not None:
        t0 = time.time()
        shared.start()
        print(f"the one load: {time.time() - t0:.1f} s", flush=True)
    if session is not None:
        dram_left(session)

    state.reset(ref)
    steps, tokens, first_diff = [], [], None
    pos, inp = 0, torch.tensor([ids])
    by_decode = None
    if a.prompt_by_decode and len(ids) > BLOCK:  # the block, then forced decode steps
        inp, forced, t0 = torch.tensor([ids[:BLOCK]]), ids[BLOCK:], time.time()
        for nxt in forced:  # the last prompt token is step 0's input
            path.forward(inp, pos)
            pos += inp.size(1)
            inp = torch.tensor([[nxt]])
        by_decode = {"block": BLOCK, "forced_steps": len(forced)}
        by_decode["seconds"] = round(time.time() - t0, 3)
        print(json.dumps({"prompt_by_decode": by_decode}), flush=True)
    n_steps = min(a.new, len(gold["generated_ids"])) if a.force else a.new
    for n in range(n_steps):
        path.seconds.clear()
        t0 = time.time()
        tok, logits = path.forward(inp, pos)
        dt = time.time() - t0
        t = int(tok[0])
        tokens.append(t)
        row = {"step": n, "pos": pos, "token": t, "seconds": round(dt, 3)}
        for where in ("host", "box"):
            row[f"{where}_s"] = round(
                sum(v for (_, w), v in path.seconds.items() if w == where), 3
            )
        by_piece = {}
        for (site, where), v in path.seconds.items():
            k = f"{SITE_PIECE[site]}/{where}"
            by_piece[k] = by_piece.get(k, 0.0) + v
        if n < len(gold["generated_ids"]):
            row["ref_token"] = gold["generated_ids"][n]
            rel, mx = compare(logits[0], gold_logits[n])
            row["logits_rel_rms"], row["logits_max_abs"] = rel, mx
            if t != row["ref_token"] and first_diff is None:
                first_diff = row
        print(json.dumps(row), flush=True)
        row["seconds_by_piece"] = {k: round(v, 4) for k, v in sorted(by_piece.items())}
        steps.append(row)
        pos += inp.size(1)
        nxt = gold["generated_ids"][n] if a.force else t
        inp = torch.tensor([[nxt]])
        if not a.force and t == ref.tokenizer.eos_token_id:
            break
    dec = [s["seconds"] for s in steps[1:]]
    report = {
        "prompt": a.prompt,
        "pieces": [{"spec": s, "sites": list(x)} for s, x in pieces],
        "box_calls": {
            f"{k[0]}@{k[1]}": v for k, v in sorted(path.calls.items(), key=str)
        },
        "mode": "teacher-forced" if a.force else "free",
        "tokens": tokens,
        "text": ref.tokenizer.decode(tokens),
        "ref_tokens": gold["generated_ids"][: len(tokens)],
        "same_tokens": tokens == gold["generated_ids"][: len(tokens)],
        "first_difference": first_diff,
        "prompt_by_decode": by_decode,
        "prefill_s": steps[0]["seconds"],
        "decode_steps": len(dec),
        "decode_s": round(sum(dec), 3),
        "tokens_per_s": round(len(dec) / sum(dec), 4) if dec else None,
        "last_step_seconds_by_piece": steps[-1]["seconds_by_piece"],
        "steps": steps,
    }
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "steps"}, ensure_ascii=False
        ),
        flush=True,
    )
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    for piece in remote:
        piece.close()
    if session is not None:
        session.__exit__(None, None, None)


if __name__ == "__main__":
    main()
