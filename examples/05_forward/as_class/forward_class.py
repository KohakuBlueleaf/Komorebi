"""Forward written as a class: the dynamic handler of ../forward.py as a method whose
loop takes each message this core is given, compiled into the program's compute
kernel (build/forward/ beside this file); the reader and writer are the runtime's.
The same runs and checks as ../forward.py (beside it with --beside). See README.md.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import forward as hand  # noqa: E402  (the hand program, its options and its report)

ONE_CORE = 1  # mode 1: every forward to rank target_rank


class Visit(km.Message):
    """A message: its id, the forwards it has left, its generator, its sum."""

    id: int
    forwards: int
    state: int
    sum: int


class Forward(km.Program):
    """forward: `message_count` messages seeded over the ranks, each forwarded
    `forwards` times and then consumed; each core's totals to its results words."""

    @km.handler
    def forward(
        self,
        message_count: int,
        forwards: int,
        mode: int,
        target_rank: int,
        results: int,
    ):
        consumed = 0
        visits = 0
        total_low = 0
        total_high = 0
        if self.seeding():
            part = self.partition(message_count)
            for index in range(part.count):
                state = (part.start + index + 1) * 0x9E3779B1
                if state == 0:
                    state = 1
                seed = Visit(id=part.start + index, forwards=forwards, state=state)
                self.send(self.rank, seed)
        for message in self.arrive():
            visits += 1
            message.state ^= message.state << 13
            message.state ^= message.state >> 17
            message.state ^= message.state << 5
            message.sum += message.state
            if message.forwards == 0:
                consumed += 1
                low = total_low + message.sum
                if low < total_low:
                    total_high += 1
                total_low = low
                self.consume()
            else:
                message.forwards -= 1
                if mode == ONE_CORE:
                    self.forward(target_rank, message)
                else:
                    self.forward(message.state % self.size, message)
        km.cxx(
            """
            UNPACK(({
                volatile tt_l1_ptr uint32_t* words = km::l1(results);
                words[0] = consumed;
                words[1] = total_low;
                words[2] = total_high;
                words[3] = visits;
                words[8] = messages.quota();
                words[9] = command.sequence;
            }));
            """,
            takes=[results, consumed, total_low, total_high, visits],
        )


def run_class(session, args):
    """One load of the class's program: the RUN command, its end, the hand program's
    report on the results words."""
    grid = session.grid
    cores = grid[0] * grid[1]
    count = args.count or cores
    results = hand.l1_blocks(session, grid, hand.FORWARD["kResWords"])
    program = Forward(log=hand.log).dynamic("forward", **hand.options(args))
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    mode = ONE_CORE if args.one_core >= 0 else 0
    words = (args.messages, args.forwards, mode, max(args.one_core, 0), results.address)
    start = time.perf_counter()
    handle = run.submit(Forward.forward, (0, count), args=words, exclusive=True)
    run.drain(timeout=args.deadline + 5.0)
    host_seconds = time.perf_counter() - start
    run.stop()
    event, sequence = run.event(handle), run.sequence(handle)
    result = hand.report(
        args,
        run,
        (event, sequence),
        program.messages,
        (count, cores),
        -1,
        results,
        None,
    )
    result["host_s"] = round(host_seconds, 4)
    result["kernel_text"] = program.sizes.text
    result["ok"] = (
        result["exact"] and result["event"] == "done" and result["gave_core"] == -1
    )
    return result


def main():
    sys.argv, beside = [a for a in sys.argv if a != "--beside"], "--beside" in sys.argv
    args = hand.parse()
    if args.stall >= 0 or args.probe >= 0 or args.engine or args.again:
        raise SystemExit("the class has no stall or probe test, nor the engine run")
    report = {"runs": {"hand": [], "class": []}}
    with km.Session(log=hand.log) as session:
        for _ in range(args.repeat):
            if beside:
                report["runs"]["hand"].append(hand.run_once(session, args, -1))
            report["runs"]["class"].append(run_class(session, args))
    for name, runs in report["runs"].items():
        timed = [r["run_microseconds"] for r in runs if r.get("run_microseconds")]
        if timed:
            report[f"{name}_run_us"] = [
                min(timed),
                round(float(np.median(timed)), 1),
                max(timed),
            ]
    report["ok"] = all(r["ok"] for runs in report["runs"].values() for r in runs)
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
