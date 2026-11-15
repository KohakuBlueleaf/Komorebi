"""The attention project's ATTEND as a class, beside the hand program on the same
tensors: the job table's entries, Q's rows, the K / V broadcast in bands and the
output rows as lines, every compute step the hand's (flash.hpp). README.md says what
each line lowers to.
Example: python attention.py --shape t1; python attention.py --shape causal64"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from projects.attention import card  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

STARTED = time.monotonic()
# The plan's sizes for a head dim of 64 (t1, causal64): Q / K and V's tiles a row, the
# rows R a job, the keys KB a block, the window, the broadcast's length.
DT, DV, R, KB, WINDOW, LENGTH = 2, 2, 13, 12, 2, 10
SUB = pl.SUB_CHUNK  # key tiles a compute step
QB = R * DT  # a job's block of Q
OPTIONS = {"math_fidelity": "HiFi2", "fp32_dest_acc_en": True}


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Job(km.Entry):
    """A job of the table (attention.h kJob*): first Q, K, V and O pages, its rows, the
    first row in its head, key tiles, the row step."""

    q: int
    k: int
    v: int
    o: int
    rows: int
    start: int
    keys: int
    stride: int


class Attention(
    km.Program,
    compute=OPTIONS,
    startup=km.Startup("ones", "causal", "p", order="reverse"),
    defines={
        "FLASH_HEAD_TILES": DT,
        "FLASH_VALUE_TILES": DV,
        "FLASH_ROWS": R,
        "FLASH_BLOCK_KEYS": KB,
        "FLASH_SUB_CHUNK": SUB,
    },
):
    """attend: the hand program's ATTEND (a0 q .. a8 job_count): each core's jobs, Q's
    rows against the job's K / V blocks from its band's broadcast, each row of O."""

    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push="pack")
    causal: km.L1[km.bf16, 1, 32, 32] = km.own(push="pack")
    p: km.L1[km.bf16, 2 * SUB, 32, 32] = km.own(push="pack")
    state: km.L1[km.fp32, 2 * R, 32, 32] = km.own(push=None)
    acc: km.L1[km.fp32, R * (DV + 1), 32, 32] = km.own(push=None)
    state_token: km.L1[km.bf16, 4, 32, 32] = km.own(push="pack")
    acc_token: km.L1[km.bf16, R + 4, 32, 32] = km.own(push="pack")

    @km.handler
    def attend(
        self, q: km.Tensor, k: km.Tensor, v: km.Tensor, o: km.Tensor,
        tables: km.Tensor, mode: int, passes: int, first_job: int, job_count: int,
    ):  # fmt: skip
        km.cxx(
            """
            MATH((llk_math_eltwise_unary_sfpu_init<
                  SfpuType::exponential, DST_ACCUM_MODE>()));
            MATH((flash::init(__builtin_bit_cast(float, FLASH_SCALE_BITS))));
            MATH((flash::macro_init()));
            PACK((flash::addr_init()));
            flash_class::constant_tiles<ones, causal, p, acc>();
            cb_reserve_back(state, 2 * FLASH_ROWS);
            cb_reserve_back(acc, FLASH_ROWS * (FLASH_VALUE_TILES + 1));
            """,
            takes=[self.ones, self.p, self.state, self.acc, self.state_token,
                   self.acc_token, self.causal],
            include=["../../../../../experiments/language/attention_class/flash.hpp"],
            at="start",
        )  # fmt: skip
        km.cxx("PACK((flash::restore_constants()));", at="end", on="pack")
        down = km.Broadcast(
            self, along="column", length=LENGTH, mover="A", window=WINDOW,
            unread=mode // 2,
        )  # fmt: skip  # mode 2 (compute only): each block pushed unread
        jobs = tables.pages(self.rank, 1).entries(Job, first=8, size=8, count=0)
        # job_count 0: to the last job (the slice's end clamped to the count)
        end = first_job + job_count + jobs.count * (1 - min(job_count, 1))
        for _ in range(max(passes, 1)):
            for job in jobs[first_job:end]:
                r0 = job.q // DT
                qs = km.Stream(
                    q.tiles[r0 : min(r0 + R * job.stride, r0 + job.rows * job.stride)
                            : job.stride, 0:DT],
                    block=QB,
                )  # fmt: skip
                k0 = job.k // DT
                v0 = job.v // DV
                for b in range((job.keys + KB - 1) // KB):
                    kr = k0 + b * KB
                    vr = v0 + b * KB
                    if down.sender():
                        kv = down.cast(
                            k.tiles[kr : min(kr + KB, k0 + job.keys), 0:DT].T,
                            v.tiles[vr : min(vr + KB, v0 + job.keys), 0:DV],
                        )
                    else:
                        kv = down.arrive()
                    km.cxx(
                        """
                        flash_class::cb_queries = qs_buffer;
                        using Flash = flash_class::Flash<
                            kv_buffer, ones, p, state, acc, state_token, acc_token,
                            causal>;
                        Flash::block(
                            job_rows, job_start, job_keys, job_stride,
                            b * FLASH_BLOCK_KEYS,
                            b * (FLASH_BLOCK_KEYS / FLASH_SUB_CHUNK)
                        );
                        """,
                        takes=[qs, kv, job, b, self.ones, self.p, self.state,
                               self.acc, self.state_token, self.acc_token,
                               self.causal],  # fmt: skip
                    )
                o0 = job.o // DV
                out = km.Stream(
                    o.tiles[o0 : min(o0 + R * job.stride, o0 + job.rows * job.stride)
                            : job.stride, 0:DV],
                    write=True,
                    block=DV,
                )  # fmt: skip
                for row in range(job.rows):
                    km.cxx(
                        """
                        flash_class::cb_output = out_buffer;
                        flash_class::Flash<
                            ones, ones, p, state, acc, state_token, acc_token, causal
                        >::finalize(row);
                        """,
                        takes=[out, row, self.ones, self.p, self.state, self.acc,
                               self.state_token, self.acc_token,
                               self.causal],  # fmt: skip
                    )


def inputs(shape, seed):
    """The call's dims, whether causal, the scale and Q, K, V (the project's run's)."""
    batches, heads, kv_heads, sequence_length, head_dim, causal = pl.SHAPES[shape]
    sizes = (batches, heads, kv_heads, sequence_length, sequence_length, head_dim)
    call = pl.dims(*sizes, head_dim, causal)
    q, k, v = sdpa.inputs(*sizes, head_dim, seed)
    return call, causal, 1.0 / np.sqrt(head_dim), q, k, v


def timed(run, group, args, count):
    """`count` attend commands back to back: each command's microseconds (all
    done)."""
    handles = [
        run.submit(Attention.attend, group, args=args, exclusive=True)
        for _ in range(count)
    ]
    events = run.wait(handles, timeout=60.0)
    kinds = {event.kind for event in events}
    if kinds != {"done"}:
        raise km.KomorebiError(f"commands not done: {sorted(kinds)}")
    return [event.microseconds for event in events]


def hand_attention(session, plan, tensors, scale):
    """The hand program loaded (card.Attention) and its kernel text by kernel."""
    built, load = [], session.load

    def kept(program, **options):
        built.append(program)
        return load(program, **options)

    session.load = kept
    try:
        hand = card.Attention(session, plan, tensors, scale)
    finally:
        session.load = load
    return hand, built[0].sizes.text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=("t1", "causal64"), default="t1")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", help="the two O's pages, .npz")
    parser.add_argument(
        "--only", choices=("hand", "class"), help="one program (a build without a card)"
    )
    args = parser.parse_args()
    call, causal, scale, q, k, v = inputs(args.shape, args.seed)
    report = {"shape": args.shape}
    with km.Session(log=log) as session:
        plan = pl.make(call, session.nocs().grid)
        made = (plan.r, plan.kb, plan.window, plan.length, call.dt, call.dvt)
        if made != (R, KB, WINDOW, LENGTH, DT, DV):
            raise SystemExit(f"plan {made}: the class is of {(R, KB, WINDOW, LENGTH)}")
        report["plan"] = plan.summary()
        zeros = np.zeros((*q.shape[:-1], v.shape[-1]), np.float32)
        tensors = {
            "q": card.bf16_tensor(session, q),
            "k": card.bf16_tensor(session, k),
            "v": card.bf16_tensor(session, v),
            "o": card.bf16_tensor(session, zeros),
            "tables": card.table_tensor(session, pl.tables(plan)),
        }
        if args.only != "class":
            hand, report["hand_text"] = hand_attention(session, plan, tensors, scale)
            _, hand_us = hand.calls(args.calls)
            report["hand_run"] = hand.stop()
            (hand_o,) = session.read(tensors["o"])
        class_o_tensor = card.bf16_tensor(session, zeros)
        scale_bits, threshold_bits = pl.exp_constants(scale)
        defines = {
            "FLASH_SCALE_BITS": scale_bits,
            "FLASH_THRESHOLD_BITS": threshold_bits,
        }
        if causal:
            defines["FLASH_CAUSAL"] = 1
        program = Attention(
            q=tensors["q"], k=tensors["k"], v=tensors["v"], o=class_o_tensor,
            tables=tensors["tables"], defines=defines, session=session, log=log,
        )  # fmt: skip
        loaded = session.load(program)
        try:
            group = (0, len(plan.jobs))
            attend = (
                tensors["q"], tensors["k"], tensors["v"], class_o_tensor,
                tensors["tables"], pl.K["kModeFull"], 1, 0, 0,
            )  # fmt: skip  # the hand's words: one pass of every job
            class_us = timed(loaded, group, attend, args.calls)
        finally:
            loaded.stop()
        report["class_text"] = program.sizes.text
        (class_o,) = session.read(class_o_tensor)
    hand_words = hand_o.reshape(-1).view(np.uint16)
    class_words = class_o.reshape(-1).view(np.uint16)
    report["o_words"] = int(hand_words.size)
    report["o_words_differ"] = int((hand_words != class_words).sum())
    report["hand_us"] = round(float(np.median(hand_us)), 2)
    report["class_us"] = round(float(np.median(class_us)), 2)
    if args.save:
        np.savez(args.save, hand=hand_o, made=class_o)
    report["ok"] = report["o_words_differ"] == 0 and report["hand_run"]["ok"]
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
