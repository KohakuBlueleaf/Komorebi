"""A handler that never returns on island 0's link core 0 (CPU 0) while traffic needs
it (--step 1: a MAP reading pages island 1 holds; 2: a command crossed to island 1):
the give-ups name the link, the host calls run.reset_link, the command again is exact.
Two cards as islands; one JSON line; see README.md."""

import argparse
import json
import sys
import time

import numpy as np
from islands import map_words
from ladder import HANDLER_SPIN, KERNELS, SPIN_KERNELS, SPIN_WHERE, STARTED, TILE, log

import komorebi as km
from komorebi.layout import partition

X280_TICKS_S = 50e6


def wrong_ranks(got, values, group):
    """Ranks of `group` whose MAP page is not the sum of their two input tiles (`values`
    in tile order), bit-exact."""
    tiles = values.astype(np.float64)
    got = got.reshape(-1, TILE, TILE).astype(np.float64)
    first, count = group
    return [
        first + rank
        for rank in range(count)
        if not np.array_equal(
            got[first + rank],
            tiles[
                [2 * first + t for t in partition(range(2 * count), rank, count)]
            ].sum(0),
        )
    ]


def timeout_of(wait):
    """('done', None) or ('given up' / 'timeout', the CommandTimeout) of `wait()`."""
    try:
        events = wait()
        if hasattr(events, "kind"):  # one Event (a named tuple)
            events = [events]
        log(f"events: {list(events)}")
        return [event.kind for event in events], None
    except km.CommandTimeout as timeout:
        log(f"timeout: {timeout}")
        return ("given up" if timeout.given_up else "timeout"), timeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(1, 2), required=True)
    step = parser.parse_args().step
    generator = np.random.default_rng(1)
    report = {"cards": [0, 1], "step": step}
    with km.Session(cards=[0, 1], log=log, kernel_bytes=84 * 1024) as session:
        ranks, graph = session.ranks, session.graph
        near = (graph.first(0), graph.count(0))
        far = (graph.first(1), graph.count(1))
        # rank r's input tiles 2r and 2r + 1 are row r of a [ranks, 2] tile matrix
        values = generator.integers(-2, 3, (2 * ranks, TILE, TILE)).astype(np.float32)
        matrix = values.reshape(ranks, 2, TILE, TILE).transpose(0, 2, 1, 3)
        matrix = matrix.reshape(ranks * TILE, 2 * TILE)
        local = session.tensor(matrix, "bfloat16", "tile")
        held = session.tensor(matrix, "bfloat16", "tile", group=far)
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        program = km.Program(
            "mapcollect",
            kernels=SPIN_KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": local.accessor_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            depth=4,
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program, deadline=2.0, wait=4.0)
        handler = run.program.handler("map")
        island0 = run.island(0)
        if step == 1:  # island 0's ranks read what island 1 holds: asks on link 0
            group, words = near, map_words(held, output, near)

            def submit():
                return run.submit(handler, group, words)

            def wait(ticket):
                return lambda: run.wait(ticket, timeout=15.0)

        else:  # island 0's dispatcher crosses a command to island 1's ranks
            group, words = far, map_words(local, output, far)

            def submit():
                return island0.submit(handler, group, words)

            def wait(handle):
                return lambda: island0.wait(handle, timeout=15.0)

        report["group"] = list(group)
        started = time.perf_counter()
        report["before"] = timeout_of(wait(submit()))[0]
        report["before_us"] = round((time.perf_counter() - started) * 1e6, 1)
        log(f"before: {report['before']} in {report['before_us']} us")
        (got,) = session.read_bf16(output)
        report["wrong_before"] = wrong_ranks(got, values, group)[:8]
        log(f"before, wrong ranks: {report['wrong_before']}")
        lines = []  # lines crossed each way so far: before the SPIN, after the reset

        def crossed():
            moved = run.crossings()
            lines.append([moved.total(0, 1)["lines"], moved.total(1, 0)["lines"]])

        crossed()
        session.write(output, np.zeros((output.pages, output.page_bytes), np.uint8))
        log("SPIN on island 0's link core 0 (CPU 0), then the command it holds up")
        link = graph.link_group(0, 0, 1)
        index = (graph.link_cores[0][0] - link[0]) & 0xFFFFFFFF  # reader: me - first
        spin = island0._queue_command(  # the hang injected: a link group, no rank
            HANDLER_SPIN, link, [index, SPIN_WHERE["reader"]]
        )
        held_up = submit()
        run.send()  # both stored now: the command is in flight while the SPIN runs
        report["spin"], spun = timeout_of(lambda: island0.wait(spin, timeout=15.0))
        report["spin_links"] = spun.links if spun else None
        report["command"], lost = timeout_of(wait(held_up))
        report["command_links"] = lost.links if lost else None
        report["command_text"] = str(lost) if lost else None
        report["link_events"] = island0.link_events
        report["links_hung"] = run.links_hung
        deadline = 2.0  # the load's
        for event in island0.link_events:
            event["after_submit_us"] = round(event["ticks"] / X280_TICKS_S * 1e6, 1)
        hung = [e for e in island0.link_events if e["what"] == "hung"]
        if hung:  # the give-up after the SPIN's deadline, by the dispatcher's clock
            report["give_up_after_deadline_us"] = round(
                (hung[0]["ticks"] / X280_TICKS_S - deadline) * 1e6, 1
            )
        log(f"link events: {island0.link_events}; hung: {report['links_hung']}")
        reset = None
        if report["links_hung"] == [(0, 0)]:
            reset = run.reset_link(0, 0)
            report["reset_link"] = {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in reset.items()
            }
            log(f"host reset_link: {report['reset_link']}")
        report["links_hung_after"] = run.links_hung
        crossed()
        session.write(output, np.zeros((output.pages, output.page_bytes), np.uint8))
        started = time.perf_counter()
        report["after"] = timeout_of(wait(submit()))[0]
        report["after_us"] = round((time.perf_counter() - started) * 1e6, 1)
        log(f"after: {report['after']} in {report['after_us']} us")
        crossed()
        report["lines_crossed"] = lines  # [0 -> 1, 1 -> 0] at each read
        log(f"lines crossed (0->1, 1->0): {lines}")
        try:
            results = run.stop()
            report["stop"] = "clean"
            report["kernel_errors"] = [result["error_name"] for result in results]
        except km.KomorebiError as error:
            report["stop"] = str(error)
        (got,) = session.read_bf16(output)
    report["wrong_after"] = wrong_ranks(got, values, group)
    report["exact_after"] = not report["wrong_after"]
    report["wrong_after"] = report["wrong_after"][:8]
    report["ok"] = bool(
        report["before"] == ["done"]
        and not report["wrong_before"]
        and report["spin"] == "given up"
        and report["spin_links"] == [(0, "hung")]
        and report["command"] == "given up"
        and report["command_links"] == [(0, "lost")]
        and report["links_hung_after"] == []
        and reset is not None
        and report["after"] == ["done"]
        and report["exact_after"]
        and report["stop"] == "clean"
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report, default=str), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
