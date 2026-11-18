"""A real prompt on the card: the model's own weights (bf16), a token a step, the
embedding, final norm and head on the host, the largest logit taken greedily; against
torch_reference.py's tokens and hidden states (its tokenizer writes the prompt's
tokens). The checkpoint's safetensors are read with numpy alone."""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import komorebi as km
from komorebi import formats
from komorebi.bench import power
from projects.decode import card, text
from projects.decode import model as md

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshot", required=True, help="the model's snapshot folder")
    parser.add_argument("--reference", required=True, help="torch_reference.py's .npz")
    parser.add_argument("--context", type=int, default=256, help="the cache's rows")
    parser.add_argument(
        "--new", type=int, help="tokens generated (default: the reference's count)"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="--head loop: the thread detokenizes each token while the next runs",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=1,
        help="--head loop: the prompt's tokens a command (1 to 32), the head on the "
        "last row of the last",
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="--head loop: the thread submits a command a kernel and waits for each "
        "(the same build), not a token as one chain command",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="--head loop: generate from this position on a cache of zeros, from the "
        "prompt's last id (the rate at a cache's length; tokens not compared)",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--tokens", default=None, help=".npz of the card's tokens")
    parser.add_argument(
        "--bodies",
        nargs="*",
        default=list(card.DEFAULT_BODIES),
        choices=list(card.BODIES),
        help="the operations whose vector bodies are emitted (bodies.py)",
    )
    parser.add_argument(
        "--no-chain",
        action="store_true",
        help="a command a step (before stage 6), not the layer as one command",
    )
    parser.add_argument(
        "--token",
        action="store_true",
        help="a token as one command (the chain's records of every layer, once)",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=[],
        choices=list(card.DATA_STEPS),
        help="the records whose input the roots before send to their cores (stage 8)",
    )
    parser.add_argument(
        "--weights", default="bf16", choices=list(md.WEIGHTS), help="the W's formats"
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=None,
        help="a folder of the layers' W pages in --weights' formats, made if missing",
    )
    parser.add_argument(
        "--head",
        default="host",
        choices=("host", "logits", "card", "chain", "loop"),
        help="the final norm and the head: numpy on the host, products on the card "
        "with the logits read back, or with the largest taken on the card too (chain: "
        "as records of the token's command; loop: and the whole loop on the thread, "
        "the embedding read from the head's table)",
    )
    return parser.parse_args()


def model_end(args):
    """The end token's id (the config's eos_token_id, the first of a list)."""
    end = json.loads((Path(args.snapshot) / "config.json").read_text())["eos_token_id"]
    return end[0] if isinstance(end, list) else end


def head_parts(model, table, dtype, folder):
    """md.head_pages() of the head's table (values) in `dtype`, from `folder` (made
    there when missing; None: made, not kept)."""
    name = model.name.replace("/", "--")
    count = md.head_parts(table.shape[0])
    paths = [
        None if folder is None else folder / f"{name}-table-{dtype}-{p}.npy"
        for p in range(count)
    ]
    if folder is None or not all(path.exists() for path in paths):
        parts = md.head_pages(model, table, dtype)
        if folder is None:
            return parts
        folder.mkdir(parents=True, exist_ok=True)
        for path, part in zip(paths, parts):
            np.save(path, part)
    return [np.load(path) for path in paths]


def layer_pages(checkpoint, model, layer, weights, folder):
    """md.fused() of a layer's W in WEIGHTS[weights]' formats, from `folder` (made
    there when missing; None: made, not kept)."""
    if folder is None:
        return md.fused(layer_weights(checkpoint, layer), weights)
    path = folder / f"{model.name.replace('/', '--')}-{weights}-real-{layer}.npz"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(path, **md.fused(layer_weights(checkpoint, layer), weights))
    with np.load(path) as kept:
        return dict(kept)


class Checkpoint:
    """A safetensors checkpoint read by numpy: get(name) is a tensor's bf16 bits."""

    def __init__(self, folder):
        self.folder = Path(folder)
        index = json.loads((self.folder / "model.safetensors.index.json").read_text())
        self.where, self.headers = index["weight_map"], {}

    def get(self, name):
        shard = self.where[name]
        if shard not in self.headers:
            with open(self.folder / shard, "rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(size))
            self.headers[shard] = (8 + size, header)
        start, header = self.headers[shard]
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name}: {entry['dtype']}, not BF16")
        low, high = entry["data_offsets"]
        bits = np.memmap(
            self.folder / shard, np.uint16, "r", start + low, ((high - low) // 2,)
        )
        return np.asarray(bits).reshape(entry["shape"])


def layer_weights(checkpoint, layer):
    """Layer `layer`'s weights as md.layer_weights() gives them: {name: (pages,
    values)}, each W as [K, N] (the checkpoint's [N, K] transposed)."""
    prefix = f"model.layers.{layer}."
    names = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }
    out = {}
    for name, key in names.items():
        bits = np.ascontiguousarray(checkpoint.get(prefix + key + ".weight").T)
        pages = md.tile_pages(bits) if name in ("o", "down") else None
        out[name] = (pages, formats.bf16_values(bits))
    norms = {
        "norm1": "input_layernorm",
        "norm2": "post_attention_layernorm",
        "q_norm": "self_attn.q_norm",
        "k_norm": "self_attn.k_norm",
    }
    for name, key in norms.items():
        out[name] = (
            None,
            formats.bf16_values(checkpoint.get(prefix + key + ".weight")),
        )
    return out


def final_norm(x, weight, eps):
    """The model's last norm as its library computes it: float32, cast to bf16, times
    the weight in bf16."""
    x = np.asarray(x, np.float32)
    rms = np.float32(1.0) / np.sqrt(
        (x * x).mean(axis=-1, keepdims=True) + np.float32(eps)
    )
    return md.bf16(np.asarray(weight, np.float32) * md.bf16(x * rms))


def relative_rms(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean()))


def main():
    args = parse()
    reference = np.load(args.reference)
    prompt, want = reference["prompt"].tolist(), reference["tokens"].tolist()
    want = want[: args.new] if args.new else want
    hidden = reference["hidden"]  # [positions, layers, hidden]; the last one normed
    model = md.from_config(Path(args.snapshot) / "config.json")
    checkpoint = Checkpoint(args.snapshot)
    embedding = checkpoint.get("model.embed_tokens.weight")
    head = formats.bf16_values(checkpoint.get("lm_head.weight"))
    last_norm = formats.bf16_values(checkpoint.get("model.norm.weight"))
    steps = len(prompt) + len(want) - 1
    if steps > args.context:
        raise ValueError(f"{steps} positions past a cache of {args.context}")
    report = {"model": model.name, "prompt_tokens": len(prompt), "steps": steps}
    report["bodies"] = args.bodies
    report["chain"] = not args.no_chain
    report["token"] = args.token
    report["data"] = args.data
    report["weights"] = md.WEIGHTS[args.weights]
    report["head"] = args.head
    vocab = head.shape[0]
    parts = []
    if args.head != "host":  # before the card is opened
        parts = head_parts(model, head, md.format_of(args.weights, "qkv"), args.pages)
        log(f"the head's {len(parts)} parts made")
    tied = args.head != "loop" or np.array_equal(
        checkpoint.get("lm_head.weight"), embedding
    )
    if not tied:  # the loop reads the embedding from the head's table: one copy
        raise ValueError("the head and the embedding differ: not one table")
    own_embedding = args.head == "loop" and md.format_of(args.weights, "qkv") != (
        "bfloat16"
    )
    loop_from = 0 if args.head == "loop" else steps  # the loop: the prompt in it too
    given = prompt if args.start is None else prompt[-1:]  # the loop's prompt
    if args.start is not None:
        steps = args.start + len(want)
    if args.head == "loop" and steps > args.context:
        raise ValueError(f"{steps} positions past a cache of {args.context}")
    for layer in range(model.layers) if args.pages else ():  # before the card opens
        layer_pages(checkpoint, model, layer, args.weights, args.pages)
        log(f"layer {layer}: its pages in {args.pages}")
    tokens, per_layer, step_us, made_at = list(prompt), [], [], []
    with km.Session(log=log) as session:  # its kernel buffer follows the kernels
        decode = card.Decode(
            session,
            model,
            model.layers,
            1.0 if args.head != "loop" else 1.0 + 0.03 * (len(given) + len(want)),
            log,
            context=args.context,
            bodies=args.bodies,
            chain=not args.no_chain,
            data=args.data,
            weights=args.weights,
            vocab=vocab if args.head != "host" else 0,
            embedding=own_embedding,
            text=text.pieces(args.snapshot, vocab) if args.text else None,
        )
        head_format = md.format_of(args.weights, "qkv")
        report["head_format"] = head_format
        report["embedding_format"] = "bfloat16, " + (
            "its own copy" if own_embedding else "the head's table"
        )
        norm = md.norm_pages(model, last_norm, head_format)
        written = decode.write_head(parts, norm)
        if own_embedding:  # as the official stack: a bf16 copy beside the head
            written += decode.write_embedding(embedding)
        empty = np.zeros((model.kv_heads, 0, model.head_dim), np.float32)
        for layer in range(model.layers):
            pages = layer_pages(checkpoint, model, layer, args.weights, args.pages)
            written += decode.write_layer(
                layer, layer_weights(checkpoint, layer), pages
            )
            written += decode.write_cache(layer, empty, empty)
        log(f"weights written ({written:.2f} s of writes)")
        try:
            for position in range(loop_from):
                decode.slot = 0  # one stats slot for every job (the thread's has 32)
                x = np.zeros((md.TILE, model.hidden), np.float32)
                x[0] = formats.bf16_values(embedding[tokens[position]])
                decode.write_activation("x", x)
                if position < hidden.shape[0]:  # layer by layer, each output kept
                    errors = []
                    for layer in range(model.layers):
                        decode.job(
                            1, 1, first=layer, position=position, token=args.token
                        )
                        decode.slot = 0
                        row = decode.read_activation("x")[0]
                        if layer == model.layers - 1:
                            row = final_norm(row, last_norm, model.eps)
                        errors.append(relative_rms(row, hidden[position, layer]))
                    per_layer.append(errors)
                    log(
                        f"position {position}: per layer {np.round(errors, 4).tolist()}"
                    )
                    on_chain = False
                else:  # the head's records in the token's command (--head chain)
                    on_chain = args.head == "chain" and position >= len(prompt) - 1
                    ran = decode.job(
                        1,
                        model.layers,
                        position=position,
                        token=args.token,
                        head=on_chain,
                    )
                    step_us.append(ran["thread_ticks"] / card.TICKS_PER_MICROSECOND)
                if on_chain:
                    started = time.perf_counter()
                    chosen = decode.largest_read()
                    decode.head_seconds.append((0.0, time.perf_counter() - started))
                elif args.head in ("card", "chain"):
                    chosen = decode.largest()
                elif args.head == "logits":
                    logits = decode.head()
                    chosen = int(np.argmax(logits))
                else:
                    row = decode.read_activation("x")[0]
                    logits = head @ final_norm(row, last_norm, model.eps)
                    chosen = int(np.argmax(logits))
                if args.head != "host" and position == len(prompt) - 1:
                    row = decode.read_activation("x")[0]  # the head's check, once
                    host = head @ final_norm(row, last_norm, model.eps)
                    if args.head == "logits":
                        report["head_check"] = relative_rms(logits, host)
                    else:  # the card's choice against the host's top two
                        top = np.argsort(host)[::-1][:2]
                        report["head_check"] = {
                            "card": chosen,
                            "host": top.tolist(),
                            "margin": float(host[top[0]] - host[top[1]]),
                        }
                    log(f"head on the card against the host's: {report['head_check']}")
                if position >= len(prompt) - 1:
                    tokens.append(chosen)
                    made_at.append(time.perf_counter())  # the token is the host's now
                    log(f"position {position}: token {tokens[-1]}")
            if args.head == "loop":  # every generated token: one GENERATE command
                started = time.perf_counter()
                count = decode.generate(
                    len(want),
                    model.layers,
                    args.start or 0,
                    given,
                    model_end(args),
                    chain=not args.kernels,
                    rows=args.prefill,
                )
                report["loop_seconds"] = round(time.perf_counter() - started, 4)
                log(f"loop: {count} tokens in {report['loop_seconds']} s")
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
        decode.run.stop()
        if args.head == "loop" and "failed" not in report:  # the thread's regions
            start, made = decode.made(count)
            tokens += [token for token, _ in made]
            made_at = [ticks / card.TICKS_PER_MICROSECOND / 1e6 for _, ticks in made]
            report["wall_from"] = "the card's timer at each token's choice"
            report["start"] = args.start
            report["prefill"] = args.prefill
            report["kernels"] = args.kernels
            if args.text:  # the card's bytes against the tables' for its tokens
                out = decode.text_out()
                ids = [token for token, _ in made]
                report["text_equal"] = out == text.joined(*decode.text, ids)
                report["text"] = out.decode("utf-8", errors="replace")
            if made:  # the prompt's commands and the first token's, by the card's timer
                first_ms = (made[0][1] - start) / card.TICKS_PER_MICROSECOND / 1e3
                report["first_token_ms"] = round(first_ms, 2)
            thread = card.decode_slot(decode.stats()[decode.generate_slot])["thread"]
            commands = max(len(given) - 1 + len(made), 1)
            report["loop_us"] = {  # a token's command (layers, norm, head) and more
                name: round(thread[name] / commands, 1) for name in card.THREAD_PARTS
            }
    got = tokens[len(prompt) :]
    # whole tokens by wall clock: from the first generated token to the last
    wall = made_at[-1] - made_at[0] if len(made_at) > 1 else 0.0
    same = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if args.start is not None:  # a cache of zeros: nothing to compare
        same = "not compared"
    report |= {
        "card_tokens": got,
        "reference_tokens": want,
        "first_different": same,
        "per_layer_rms": per_layer,
        "step_us": step_us,
        "tokens_per_second": (
            round(1e6 / float(np.mean(step_us)), 1) if step_us else None
        ),
        "wall_tokens": len(made_at),
        "wall_seconds": round(wall, 4),
        "wall_tokens_per_second": (
            round((len(made_at) - 1) / wall, 1) if wall > 0 else None
        ),
        "context": args.context,
    }
    if decode.head_seconds:  # the head's commands and the logits' read, a token
        commands, reads = np.mean(decode.head_seconds[1:], axis=0) * 1e6
        report["head_us"] = {"commands": round(commands, 1), "read": round(reads, 1)}
    if args.tokens:
        np.savez(args.tokens, card=np.array(got), reference=np.array(want))
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    sys.exit(0 if "failed" not in report else 1)


if __name__ == "__main__":
    main()
