"""Random graphs and mutated uploads through both executors and both checks: the
Python executor and the C engine built for the host must give the same commands,
events and registers, and `check` and `kmg_check` the same answer.
"""

import argparse
import random
import sys

from komorebi.engines import graph as gr
from komorebi.engines.graph_native import NativeExecutor, load, native_check

CPU_CODES = sorted(gr.CPU_OPS)


def random_graph(rng):
    g = gr.Graph(
        max_runs=rng.choice([1, 2, 5]),
        step_limit=rng.choice([1, 8, 64, gr.MAX_STEPS]),
        req=rng.randrange(1 << 32),
    )
    regs = g.regs(rng.randrange(2, 8))
    again = g.again if rng.random() < 0.5 else None
    for _ in range(rng.randrange(1, 40)):
        after = rng.sample(g.nodes, min(len(g.nodes), rng.randrange(0, 5)))
        skip = rng.choice(regs) if rng.random() < 0.15 else None
        kind = rng.random()
        if kind < 0.55:
            count = rng.randrange(1, 40)
            group = None if rng.random() < 0.2 else (rng.randrange(0, 119), count)
            if group and rng.random() < 0.8:
                group = (min(group[0], 119 - count), count)
            args = [rng.choice(regs) if rng.random() < 0.3 else rng.randrange(9)]
            g.command(
                rng.randrange(1, 240),
                group=group,
                count=None if group else count,
                args=args,
                after=after,
                result=rng.choice(regs) if rng.random() < 0.3 else None,
                exclusive=rng.random() < 0.2,
                skip_if_zero=skip,
            )
        elif kind < 0.85:
            e = g.cpu(after=after, skip_if_zero=skip)
            labels = [e.forward() for _ in range(3)]
            for k in range(rng.randrange(1, 10)):
                if k in (1, 4, 7) and rng.random() < 0.3:
                    e.bind(labels[k // 3])
                pick = rng.random()
                if pick < 0.15:
                    e.jlt(rng.choice(regs), rng.choice(regs), rng.choice(labels))
                elif pick < 0.2:
                    e.jnz(rng.choice(regs), rng.choice(labels))
                elif pick < 0.22:
                    e.mod(rng.choice(regs), rng.choice(regs), rng.choice(regs))
                else:
                    e.addi(rng.choice(regs), rng.choice(regs), rng.randrange(-3, 4))
            for label in labels:
                if label.position is None:
                    e.bind(label)
            if again is not None and rng.random() < 0.3:
                e.ge(again, rng.choice(regs), g.const(rng.randrange(1, 6)))
        else:
            g.join(after=after, skip_if_zero=skip)
    return g


def outcome(r):
    return (r.status, r.error, r.commands, r.events, r.regs, r.runs, r.stopped, r.kept)


def compare_runs(rng, lib, n):
    bad = 0
    for i in range(n):
        g = random_graph(rng)
        data = g.encode()
        values = {name: rng.randrange(4) for name in g._names}
        bindings = g.binding_values(**values)
        mod = rng.randrange(1, 6)
        options = dict(
            bindings=bindings,
            delay=(lambda r, m=mod: r.seq % m),
            result=lambda r: r.args[0] * 3 + r.count,
            deadline_passes=rng.choice([None, None, 3, 50]),
            cancel_pass=rng.choice([None, None, None, 7]),
            max_passes=20_000,
        )
        a = gr.simulate(data, **options)
        b = gr.simulate(data, engine=NativeExecutor(lib), **options)
        if outcome(a) != outcome(b):
            bad += 1
            print(f"graph {i}: python {a.status} {a.error}, C {b.status} {b.error}")
    return bad


def compare_checks(rng, lib, n):
    bad = 0
    for i in range(n):
        data = bytearray(random_graph(rng).encode())
        for _ in range(rng.randrange(1, 4)):
            at = rng.randrange(len(data))
            data[at] = rng.randrange(256) if rng.random() < 0.5 else data[at] ^ 1
        if rng.random() < 0.05:
            data = data[: rng.randrange(len(data))]
        data = bytes(data)
        a, b = gr.check(data), native_check(data, lib)
        if a != b:
            bad += 1
            print(f"upload {i}: check {a}, kmg_check {b}")
        elif a is None:
            r1 = gr.simulate(data, max_passes=2000)
            r2 = gr.simulate(data, max_passes=2000, engine=NativeExecutor(lib))
            if outcome(r1) != outcome(r2):
                bad += 1
                print(f"mutated upload {i}: runs differ")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    lib = load()
    runs = compare_runs(rng, lib, args.graphs)
    checks = compare_checks(rng, lib, args.graphs * 5)
    print(f"{args.graphs} graphs: {runs} differ; {args.graphs * 5} uploads: {checks}")
    return 1 if runs or checks else 0


if __name__ == "__main__":
    sys.exit(main())
