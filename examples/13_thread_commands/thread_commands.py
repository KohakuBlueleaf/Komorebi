"""Commands to an X280 thread beside commands to core groups: the host and a graph (fed
by the host, then on the X280) chain a grid command and a thread command round after
round, the thread's word feeding the next round; every word against numpy, times
printed. Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.engines.graph import Graph

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
STARTED = time.monotonic()
M32 = 0xFFFFFFFF
GROUP = (0, 8)  # AFFINE's cores
TABLE = np.random.default_rng(13).integers(0, 1 << 32, 256, dtype=np.uint32)
THREAD_HANDLERS = {"fold": 1, "served": 2}  # kHandlerFold, kHandlerServed in its C
WINDOW = 8  # commands in flight in the mixed waves: a thread holds 8 unanswered


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def affine(a0, a1, count):
    """AFFINE's result word (kernels/writer.cpp)."""
    return (a0 * 0x9E3779B1 + a1 + count) & M32


def fold(x):
    """FOLD's word (thread/fold.c): x xor the sum of 16 table words picked by x."""
    picked = TABLE[[(x + k * 17) % 256 for k in range(16)]]
    return x ^ int(picked.astype(np.uint64).sum()) & M32


def second(word, on_thread):
    """The round's second command: FOLD on the thread, or AFFINE on core 8 alone."""
    return fold(word) if on_thread else affine(word, 0, 1)


def rounds(iteration, carry, count, on_thread=True):
    """The numpy replay of `count` rounds: (iteration, carry) after them."""
    for _ in range(count):
        carry = second(affine(iteration, carry, GROUP[1]), on_thread)
        iteration += 1
    return iteration, carry


def program(cluster):
    """The grid's kernels (AFFINE) and the thread on `cluster` (FOLD, SERVED)."""
    made = km.Program(
        "thread_commands", kernels=KERNELS, handlers=KERNELS / "handlers.h"
    )
    made.thread(
        cluster=cluster,
        sources=[HERE / "thread" / "fold.c"],
        regions={"table": TABLE},
        handlers=THREAD_HANDLERS,
    )
    return made


def median_microseconds(seconds):
    return round(float(np.median(seconds)) * 1e6, 1)


def host_rounds(run, cluster, count):
    """`count` rounds from the host, one command at a time: AFFINE on the group, its
    word to the thread's FOLD, FOLD's word into the next AFFINE."""
    iteration, carry, wrong = 0, 0, 0
    grid_s, thread_s = [], []
    for _ in range(count):
        started = time.perf_counter()
        word = run.wait(
            run.submit("affine", GROUP, args=(iteration, carry), result=True)
        ).result
        grid_s.append(time.perf_counter() - started)
        wrong += word != affine(iteration, carry, GROUP[1])
        started = time.perf_counter()
        carry = run.wait(
            run.submit("fold", cluster=cluster, args=(word,), result=True)
        ).result
        thread_s.append(time.perf_counter() - started)
        wrong += carry != fold(word)
        iteration += 1
    want = rounds(0, 0, count)
    return {
        "rounds": count,
        "wrong": wrong,
        "as_numpy": [iteration, carry] == list(want),
        "grid_command_us": median_microseconds(grid_s),
        "thread_command_us": median_microseconds(thread_s),
    }


def mixed(run, cluster, waves):
    """Waves of WINDOW commands in flight, thread and grid alternating."""
    wrong = 0
    started = time.perf_counter()
    for wave in range(waves):
        handles, words = [], []
        for k in range(WINDOW):
            value = (wave * WINDOW + k) * 0x01000193 & M32
            if k % 2:
                core = (wave + k) % run.ranks  # less a Tensix dispatcher's core
                handles.append(
                    run.submit("affine", (core, 1), args=(value, k), result=True)
                )
                words.append(affine(value, k, 1))
            else:
                handles.append(
                    run.submit("fold", cluster=cluster, args=(value,), result=True)
                )
                words.append(fold(value))
        events = run.wait(handles, timeout=5.0)
        wrong += sum(
            event.kind != "done" or event.result != word
            for event, word in zip(events, words, strict=True)
        )
    seconds = time.perf_counter() - started
    return {
        "commands": waves * WINDOW,
        "wrong": wrong,
        "per_command_us": round(seconds / (waves * WINDOW) * 1e6, 2),
    }


def build_graph(made, cluster, count, on_thread=True):
    """A round: AFFINE on the group, FOLD on the thread after it (else AFFINE on core
    8), a CPU node that counts and goes again while iteration < limit."""
    graph = Graph(max_runs=count)
    iteration = graph.bindings("iteration")  # register 0
    carry = graph.register()  # register 1
    limit = graph.bindings("limit")
    word = graph.register()
    grid = graph.command(
        made.handler("affine"), group=GROUP, args=[iteration, carry], result=word
    )
    if on_thread:
        thread = made.threads[cluster].handler("fold")
        then = graph.command(
            thread, cluster=cluster, args=[word], after=[grid], result=carry
        )
    else:
        then = graph.command(
            made.handler("affine"), group=(8, 1), args=[word, 0], after=[grid],
            result=carry,
        )  # fmt: skip
    step = graph.cpu(after=[then])
    step.addi(iteration, iteration, 1)
    step.lt(graph.again, iteration, limit)
    return graph


def graph_rounds(run, graph, count, where, on_thread=True):
    """The graph's `count` rounds host-fed (`where` "host") or on the X280; registers
    0 and 1 (iteration, carry) are the ones the end event carries."""
    bindings = {"iteration": 0, "limit": count}
    started = time.perf_counter()
    if where == "host":
        got = run.run_hostfed(graph, bindings, timeout=60.0)
        registers = (got.registers[0], got.registers[1])
    else:
        upload = run.upload(graph)
        end = run.wait_engine(run.start(upload, bindings), timeout=60.0)
        registers = end.registers
        run.release(upload)
    seconds = time.perf_counter() - started
    return {
        "rounds": count,
        "as_numpy": list(registers) == list(rounds(0, 0, count, on_thread)),
        "per_round_us": round(seconds / count * 1e6, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cluster", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--rounds", type=int, default=200, help="host rounds")
    parser.add_argument("--graph-rounds", type=int, default=1000)
    parser.add_argument("--waves", type=int, default=200, help="mixed waves of 8")
    args = parser.parse_args()
    made = program(args.cluster)
    graph = build_graph(made, args.cluster, args.graph_rounds)
    report = {"cluster": args.cluster}
    with km.Session(log=log) as session:
        log("device open")
        run = session.load(made)
        log("loaded")
        report["host"] = host_rounds(run, args.cluster, args.rounds)
        log(f"host rounds: {report['host']}")
        report["mixed"] = mixed(run, args.cluster, args.waves)
        log(f"mixed waves: {report['mixed']}")
        report["graph_host"] = graph_rounds(run, graph, args.graph_rounds, "host")
        log(f"graph, host-fed: {report['graph_host']}")
        report["graph_x280"] = graph_rounds(run, graph, args.graph_rounds, "x280")
        log(f"graph on the X280: {report['graph_x280']}")
        grid_only = build_graph(made, args.cluster, args.graph_rounds, False)
        report["graph_x280_grid_only"] = graph_rounds(
            run, grid_only, args.graph_rounds, "x280", on_thread=False
        )
        log(f"the same with AFFINE for FOLD: {report['graph_x280_grid_only']}")
        served = run.wait(run.submit("served", cluster=args.cluster, result=True))
        # every FOLD so far (host, waves, both graphs) and this SERVED
        want = args.rounds + args.waves * WINDOW // 2 + 2 * args.graph_rounds + 1
        report["served"] = {"got": served.result, "want": want}
        result = run.stop()
        report["dispatcher_error"] = result["error_name"]
    report["ok"] = bool(
        report["host"]["wrong"] == 0
        and report["host"]["as_numpy"]
        and report["mixed"]["wrong"] == 0
        and report["graph_host"]["as_numpy"]
        and report["graph_x280"]["as_numpy"]
        and report["graph_x280_grid_only"]["as_numpy"]
        and served.result == want
        and result["error"] == 0
    )
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
