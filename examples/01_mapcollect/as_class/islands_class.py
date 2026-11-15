"""Map/collect written as a class on two cards, the islands of one session: the class
of mapcollect_class.py unchanged, made with the islands' tensors; islands.py's MAP
rounds on each island's group and on island 0 reading island 1's input, checked with
islands.py's sums, beside the hand program (--beside). See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import islands as hand  # noqa: E402  (the hand program's words, rounds and checks)
from mapcollect_class import MapCollect  # noqa: E402

PER = 2  # input tiles a rank
# The hand program's MAP words that the class's map takes, in its order: source,
# tiles, output, first_page, first_tile.
CLASS_WORDS = [0, 1, 2, 3, 6]


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    parser.add_argument("--map-rounds", type=int, default=3)
    parser.add_argument("--map-jobs", type=int, default=2000, help="per round")
    return parser.parse_args()


def hand_program(source, output):
    """islands.py's program: the hand kernels."""
    return km.Program(
        "mapcollect",
        kernels=hand.KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={
            "reader": source.accessor_args(),
            "writer": output.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=hand.KERNELS / "mapcollect.h",
    )


def rounds(session, run, handler, words_of, one_of, tensors, exact, args):
    """MAP rounds of `handler`: each island's own group in turn (words by
    `words_of`), checked; then island 0's group (its arguments by `one_of`) reading the
    local input and the one island 1 holds, what crossed each link, checked."""
    graph = session.graph
    own = [(graph.first(i), graph.count(i)) for i in range(len(session.sessions))]
    out = {}
    own_rounds(session, run, handler, words_of, tensors, exact, args, out)
    group = own[0]
    source, far, output = tensors
    for name, tensor in (("local", source), ("across", far)):
        words = one_of(tensor, output, group)
        before = run.crossings()
        spent, submits, dispatched = [], [], []
        for _ in range(args.map_rounds):
            started = time.perf_counter()
            ticket = run.submit(handler, group, words)
            submitted = time.perf_counter()
            (event,) = run.wait(ticket, timeout=10.0)  # island 0's part alone
            spent.append((time.perf_counter() - started) * 1e6)
            submits.append((submitted - started) * 1e6)
            dispatched.append(event.microseconds)
        moved = run.crossings() - before
        out[f"map_{name}_us"] = hand.median(spent)
        out[f"submit_{name}_us"] = hand.median(submits)
        out[f"dispatcher_{name}_us"] = hand.median(dispatched)
        out[f"spread_{name}_us"] = [round(min(spent), 1), round(max(spent), 1)]
        out[f"crossed_{name}"] = {
            "0->1": moved.total(0, 1),
            "1->0": moved.total(1, 0),
        }
        (got,) = session.read_bf16(output)
        out[f"{name}_check"] = hand.check(exact, got, [group], PER)
        hand.log(f"MAP on {group} reading {name}: {out[f'map_{name}_us']} us")
        hand.log(f"crossed while reading {name}:\n{moved}")
    asks, pages = args.map_rounds * group[1], args.map_rounds * PER * group[1]
    out["crossings_agree"] = (
        out["crossed_across"]["0->1"]["asks"] == asks
        and out["crossed_across"]["1->0"]["pages"] == pages
        and not out["crossed_local"]["0->1"]["asks"]
    )
    return out


def own_rounds(session, run, handler, words_of, tensors, exact, args, out):
    """MAP rounds of each island's own group in turn, checked."""
    source, _, output = tensors
    graph = session.graph
    own = [(graph.first(i), graph.count(i)) for i in range(len(session.sessions))]
    every = [own[job % len(own)] for job in range(args.map_jobs)]
    out["map_own_us"] = hand.median(
        [
            hand.timed(run, handler, every, words_of(source, output, every))
            for _ in range(args.map_rounds)
        ]
    )
    (got,) = session.read_bf16(output)
    out["own_check"] = hand.check(exact, got, own, PER)
    hand.log(f"MAP on {own} in turn: {out['map_own_us']} us, {out['own_check']}")


def zero(session, tensor):
    raw, _ = formats.to_bytes(np.zeros((tensor.shape), np.float32), "bfloat16", "tile")
    session.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))


def hand_one(source, output, group):
    return [int(word) for word in hand.map_words(source, output, [group])[0]]


def class_many(source, output, groups):
    return hand.map_words(source, output, groups)[:, CLASS_WORDS]


def class_one(source, output, group):
    """The class's arguments of one MAP: tensors and ints (submit makes the words)."""
    first, count = group
    return (source, PER * count, output, first, PER * first)


def loaded(session, program, name, tensors, exact, args, words_of, one_of):
    """One load of `program`: the rounds, the stop, the report."""
    run = session.load(program, deadline=1.0)
    hand.log(f"{name}: loaded on every island")
    handler = (  # a class's handler by its class: each island runs a copy
        MapCollect.map
        if isinstance(program, MapCollect)
        else run.program.handler("map")
    )
    report = rounds(session, run, handler, words_of, one_of, tensors, exact, args)
    results = run.stop()
    report["kernel_errors"] = [result["error_name"] for result in results]
    sizes = getattr(run.program, "sizes", None) or getattr(program, "sizes", None)
    if sizes is not None:
        report["kernel_text"] = sizes.total
    report["ok"] = bool(
        all(
            report[f"{n}_check"]["wrong_tiles"] == 0 for n in ("own", "local", "across")
        )
        and report["crossings_agree"]
        and report["kernel_errors"] == ["none"] * len(session.sessions)
    )
    hand.log(f"{name}: ok {report['ok']}")
    return report


def main():
    args = parse()
    report = {"runs": {}}
    generator = np.random.default_rng(1)
    with km.Session(cards=[0, 1], log=hand.log) as session:
        ranks = session.ranks
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        hand.log(f"open: {ranks} ranks, islands {report['islands']}")
        # rank r's input tiles 2r and 2r + 1 are row r of a [ranks, 2] tile matrix
        exact = generator.integers(-2, 3, (PER * ranks, 32, 32)).astype(np.float32)
        matrix = exact.reshape(ranks, PER, 32, 32).transpose(0, 2, 1, 3)
        matrix = matrix.reshape(ranks * 32, PER * 32)
        source = session.tensor(matrix, "bfloat16", "tile")  # split 0 by rank
        far = session.tensor(
            matrix, "bfloat16", "tile", group=tuple(report["islands"][1])
        )
        output = session.tensor(
            np.zeros((ranks * 32, 32), np.float32), "bfloat16", "tile"
        )
        tensors = (source, far, output)
        if args.beside:
            report["runs"]["hand"] = loaded(
                session,
                hand_program(source, output),
                "hand program",
                tensors,
                exact,
                args,
                hand.map_words,
                hand_one,
            )
            zero(session, output)
        program = MapCollect(source=source, output=output, log=hand.log)
        report["may_cross"] = program.may_cross
        report["runs"]["class"] = loaded(
            session,
            program,
            "class",
            tensors,
            exact,
            args,
            class_many,
            class_one,
        )
    report["ok"] = all(each["ok"] for each in report["runs"].values())
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
