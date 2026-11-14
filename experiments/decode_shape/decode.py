"""The shape of an LLM decode step on the card: a chain of K-split products (seven
D x D products a layer, L layers of random weights, T tokens) driven by the host one
command at a time, by a stream the dispatcher runs, or by an X280 thread. Prints one
JSON line. Usage in README.md."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi, power  # noqa: E402
from komorebi.engines.stream import Stream  # noqa: E402
from komorebi.session import visible_card  # noqa: E402
from projects.dense import bench, card  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

D = 4096  # the model width: every product is [32, D] @ [D, D]
PRODUCTS = 7  # a layer: four attention projections, three MLP products (H = D)
PLANS = {
    "whole": (14, 1, 16, 5),
    "half": (7, 1, 16, 5),
    "row": (2, 1, 16, 5),
}  # s, oh, ow, kc
CLUSTER = 1
STATS_WORDS, STATS_SLOTS = 16, 16  # thread/chain.c's region "stats"
REDUCE = pl.REDUCE_FORMS["tree"]
MODES = {"full": pl.S["kSplitFull"], "parts": pl.S["kSplitParts"]}
EXCLUSIVE = abi.headers().constants["KM_FLAG_EXCLUSIVE"]
W_BYTES = D * D * 2
CHECK_RMS, CHECK_MAX = 5e-2, 1e-1  # a loose bound: seven bf16 products in a row
SECONDS_PER_PRODUCT = 2e-3  # the bound of every wait, per product
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--plan", choices=list(PLANS), default="whole")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["host:full:x:1", "stream:full:x:1", "thread:full:x:1"],
        help="way:mode:flag:window[:chains], way host|stream|thread, mode "
        "full|parts, flag x (exclusive) or - (none), window 1 to 32",
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0, help="x a case's time")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    args.cases = [case_of(text) for text in args.cases]
    return args


def case_of(text):
    way, mode, flag, window, *chains = text.split(":")
    if way not in ("host", "stream", "thread") or mode not in MODES:
        raise SystemExit(f"case {text}: way host|stream|thread, mode full|parts")
    case = {
        "case": text,
        "way": way,
        "mode": mode,
        "exclusive": flag == "x",
        "window": int(window),
        "chains": int(chains[0]) if chains else 1,
    }
    if not 1 <= case["window"] <= 32 or case["chains"] not in (1, 2):
        raise SystemExit(f"case {text}: window 1 to 32, chains 1 or 2")
    if way != "thread" and case["chains"] != 1:
        raise SystemExit(f"case {text}: two chains are the thread's")
    if way == "host" and case["window"] != 1:
        raise SystemExit(f"case {text}: the host submits one at a time")
    return case


def weight(seed, layer, product):
    """W of a layer's product: uniform in +-sqrt(3 / D) (variance 1 / D), bf16."""
    rng = np.random.default_rng([seed, layer, product])
    bound = math.sqrt(3.0 / D)
    return ref.bf16(rng.uniform(-bound, bound, (D, D)).astype(np.float32))


def activation(seed, salt):
    rng = np.random.default_rng([seed, 1 << 20, salt])
    return ref.bf16(rng.uniform(-1, 1, (32, D)).astype(np.float32))


class Chain:
    """The loaded program, its tensors and the three ways of driving the chain."""

    def __init__(self, session, plan, weights, buffers, deadline, depth):
        self.session, self.plan, self.weights = session, plan, weights
        self.cores = plan.cores()
        self.pairs = buffers
        tensors = {"a": buffers[0][0], "w": weights[0], "y": buffers[0][1]}
        program = card.split_program(plan, tensors)
        pair_words = [tensor.address for pair in buffers for tensor in pair]
        program.thread(
            cluster=CLUSTER,
            sources=[HERE / "thread" / "chain.c"],
            regions={
                "weights": np.array([w.address for w in weights], np.uint32),
                "buffers": np.array(pair_words, np.uint32),
                "stats": 4 * STATS_WORDS * STATS_SLOTS,
            },
            words=[pl.S["kHandlerSplit"], REDUCE],
            handlers={"chain": 1},
        )
        self.run = session.load(program, deadline=deadline, wait=1.0, depth=depth)
        self.slot = 0

    def words(self, mode, index, pair, layers):
        """Product `index` of the chain on `pair`: A and Y in turn, W of its layer."""
        a, y = self.pairs[pair]
        if index % 2:
            a, y = y, a
        w = self.weights[(index // PRODUCTS % layers) * PRODUCTS + index % PRODUCTS]
        return pl.split_args(
            self.plan, a.address, w.address, y.address, "tree", MODES[mode]
        )

    def host(self, case, tokens, layers, pair=0):
        """One command at a time, each waited for: each one's microseconds."""
        microseconds = []
        for index in range(tokens * layers * PRODUCTS):
            handle = self.run.submit(
                "split",
                (0, self.cores),
                self.words(case["mode"], index, pair, layers),
                exclusive=case["exclusive"],
            )
            event = self.run.wait(handle, timeout=1.0)
            if event.kind != "done":
                raise km.KomorebiError(f"product {index}: {event}")
            microseconds.append(event.microseconds)
        return {"event_microseconds": microseconds}

    def stream(self, case, tokens, layers, pair=0):
        """A stream: SUBMIT, WAIT window - 1, A and Y swapped, W the next layer's."""
        addresses = np.array([w.address for w in self.weights[: layers * PRODUCTS]])
        steps = np.diff(addresses)
        if steps.size and not (steps == steps[0]).all():
            raise km.KomorebiError("W addresses not equally spaced: no stream chain")
        products = tokens * layers * PRODUCTS
        stream = Stream(
            deadline_microseconds=50_000, max_steps=10 * products + 100, req=7
        )
        a, y, w, swap, index, token = stream.registers(6)
        a_tensor, y_tensor = self.pairs[pair]
        stream.set(a, a_tensor.address)
        stream.set(y, y_tensor.address)
        stream.set(token, 0)
        next_token = stream.label()
        stream.set(w, int(addresses[0]))
        stream.set(index, 0)
        next_product = stream.label()
        stream.submit(
            pl.S["kHandlerSplit"],
            group=(0, self.cores),
            args=[a, w, y, REDUCE, MODES[case["mode"]]],
            exclusive=case["exclusive"],
        )
        stream.wait(case["window"] - 1)
        stream.mov(swap, a)
        stream.mov(a, y)
        stream.mov(y, swap)
        stream.add(w, w, stream.const(int(steps[0]) if steps.size else 0))
        stream.addi(index, index, 1)
        stream.jlt(index, stream.const(layers * PRODUCTS), next_product)
        stream.addi(token, token, 1)
        stream.jlt(token, stream.const(tokens), next_token)
        stream.wait(0)
        stream.halt(token)
        upload = self.run.upload(stream)
        handle = self.run.start(upload)
        end = self.run.wait_engine(handle, timeout=products * SECONDS_PER_PRODUCT + 2)
        self.run.release(upload)
        if end.kind != "end" or end.value != tokens:
            raise km.KomorebiError(f"stream: {end}")
        return {}

    def thread(self, case, tokens, layers, pair=0):
        """One command to the thread; it submits and waits for every product."""
        slot = self.slot
        self.slot += 1
        flags = EXCLUSIVE if case["exclusive"] else 0
        args = [tokens, layers, MODES[case["mode"]], flags, case["window"]]
        args += [self.cores, case["chains"], pair, slot]
        products = tokens * layers * PRODUCTS * case["chains"]
        handle = self.run.submit("chain", cluster=CLUSTER, args=args, result=True)
        event = self.run.wait(handle, timeout=products * SECONDS_PER_PRODUCT + 2)
        if event.kind != "done" or event.result >= 0xFFFFFFF0:
            raise km.KomorebiError(f"thread: {event}")
        return {"thread_ticks": event.result, "stats_slot": slot}


def measure(chain, case, tokens, layers, index):
    """One case under the power and clock samplers: seconds and rates."""
    meter = power.PowerSampler(index)
    with meter, bench.Clock(index) as clock:
        meter.mark("chain")
        started = time.perf_counter()
        got = getattr(chain, case["way"])(case, tokens, layers)
        seconds = time.perf_counter() - started
        meter.mark("after")
    products = tokens * layers * PRODUCTS * case["chains"]
    result = dict(case) | got
    result |= {
        "tokens": tokens,
        "layers": layers,
        "products": products,
        "seconds": round(seconds, 4),
        "ms_per_token": round(seconds * 1e3 / tokens, 3),
        "us_per_layer": round(seconds * 1e6 / (tokens * layers), 2),
        "us_per_product": round(seconds * 1e6 / products, 2),
        "weight_gbs": round(products * W_BYTES / seconds / 1e9, 1),
        "mean_w": meter.result.phases["chain"].mean_w,
        "aiclk_mhz": clock.summary(),
    }
    if "event_microseconds" in result:
        events = np.array(result.pop("event_microseconds"))
        result["event_us"] = {
            "median": round(float(np.median(events)), 2),
            "min": round(float(events.min()), 2),
            "sum_s": round(float(events.sum()) / 1e6, 4),
        }
    return result


def pause(seconds):
    """The power rule's pause after a loop, with a line every 10 s."""
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        time.sleep(min(left, 10.0))
        log(f"pause, {max(end - time.monotonic(), 0):.0f} s left")


def main():
    args = parse()
    plan = pl.split_plan(1, D // 32, D // 32, *PLANS[args.plan])
    if any(case["chains"] * plan.cores() > 120 for case in args.cases):
        raise SystemExit(f"two chains of {plan.cores()} cores leave the grid")
    report = {"plan": args.plan, "split": PLANS[args.plan], "cores": plan.cores()}
    report |= {"layers": args.layers, "tokens": args.tokens, "d": D}
    index = visible_card()
    checks = [] if args.no_check else sorted({case["way"] for case in args.cases})
    with km.Session(log=log) as session:
        weights, upload_s, layer0 = [], [], []
        for layer in range(args.layers):
            for product in range(PRODUCTS):
                matrix = weight(args.seed, layer, product)
                started = time.perf_counter()
                weights.append(card.bf16_tensor(session, matrix))
                upload_s.append(time.perf_counter() - started)
                if layer == 0:
                    layer0.append(matrix)
            log(f"layer {layer} written, {W_BYTES / np.mean(upload_s) / 1e6:.0f} MB/s")
        report["write_mb_s"] = round(W_BYTES / float(np.mean(upload_s)) / 1e6, 1)
        report["write_s"] = round(float(np.sum(upload_s)), 2)
        # pairs (A, Y): two chains' buffers, then each way's check: its start, output
        zero, start = np.zeros((32, D), np.float32), activation(args.seed, 0)
        firsts = [zero, zero] + [start] * len(checks)
        buffers = [
            (card.bf16_tensor(session, first), card.bf16_tensor(session, zero))
            for first in firsts
        ]
        longest = max(
            args.tokens * args.layers * PRODUCTS * case["chains"] for case in args.cases
        )
        deadline = round(longest * SECONDS_PER_PRODUCT + 1.0, 3)
        report["deadline_s"] = deadline
        # ring slots: the program's 4, or the largest window
        depth = max(4, *(case["window"] for case in args.cases))
        report["depth"] = depth
        chain = Chain(session, plan, weights, buffers, deadline, depth)
        log(f"loaded on {plan.cores()} cores, depth {depth}, deadline {deadline} s")
        results, ok = [], True
        try:
            for number, way in enumerate(checks):
                case = case_of(f"{way}:full:x:1")
                getattr(chain, way)(case, 1, 1, pair=2 + number)
                log(f"check chain by {way} done")
            for case in args.cases:
                log(f"case {case['case']}: {args.tokens} tokens of {args.layers}")
                result = measure(chain, case, args.tokens, args.layers, index)
                log(f"case {case['case']}: {json.dumps(result)}")
                results.append(result)
                pause(args.pause * result["seconds"])
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
            ok = False
            result = chain.run.thread_result(timeout=0.5)
            report["thread_result"] = result._asdict()
            log(f"thread: {report['thread_result']}")
        chain.run.stop()
        log("run stopped")
        stats = chain.run.threads[CLUSTER].read("stats").view(np.uint32)
        stats = stats[: STATS_WORDS * STATS_SLOTS].reshape(STATS_SLOTS, STATS_WORDS)
        for result in results:
            if "stats_slot" in result:
                words = stats[result["stats_slot"]]
                result["thread"] = {
                    "status": int(words[0]),
                    "done": int(words[1]),
                    "rejected": int(words[2]),
                    "not_done": int(words[3]),
                    "event_sum_s": round(
                        (int(words[6]) | int(words[7]) << 32) / 50e6, 4
                    ),
                    "event_max_us": round(int(words[8]) / 50, 2),
                    "event_min_us": round(int(words[9]) / 50, 2),
                }
        if "failed" in report and chain.slot:  # the last thread chain's raw slot
            report["thread_slot"] = stats[chain.slot - 1].tolist()
        outputs = {}
        for number, way in enumerate(checks):
            outputs[way] = card.read_rows(session, buffers[2 + number][1], range(32))
            log(f"check output of {way} read")
    want = start
    for matrix in layer0:
        want = ref.bf16(ref.matmul(want, matrix))
    report["checks"] = {}
    for way, got in outputs.items():
        errors = ref.errors(got, want)
        errors["ok"] = errors["rms_rel"] <= CHECK_RMS and errors["max_rel"] <= CHECK_MAX
        report["checks"][way] = errors
        ok &= errors["ok"]
    report["cases"] = results
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = ok
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
