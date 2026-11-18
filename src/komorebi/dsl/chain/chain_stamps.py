"""A class's chain stamps (a program made with chain_stamps=True): each core's rows read
out and zeroed by one command, and each record's terms in microseconds: the input's
wait, the overlay's load, the body, the hand-over (the writes acknowledged, or a direct
input's sends), the marks. Model: projects/decode's chain_stamps."""

import numpy as np

from komorebi.chain import calls as kc

READER, WRITER, COMPUTE = 0, 1, 2  # the rows' kernels (km_chain.h kChainStampKernels)
# The stamp words a class's chain takes besides the chain's own (km_chain.h): the
# overlay loaded (a mover's), the record's body done (both movers, before the marks).
STAMP_LOADED, STAMP_BODY = 2, 3
STAMPS_PAGE = 2048  # a core's page of the stamps' tensor (kChainStampBytes fit)


def tensor_for(session, ranks):
    """A DRAM tensor for the read-out: a page a flat rank, ranks 0 to `ranks` - 1
    (a group's first + count: each core writes the page of its flat rank)."""
    return session.empty((ranks, STAMPS_PAGE // 4), "uint32", "row")


def read(session, run, program, tensor, group, timeout=5.0):
    """Every core of `group` (first, count) of `run` on `session` (`tensor` from
    tensor_for): its stamps since the last read, then zeroed: [count, kernels,
    records, words] uint32 sums of AICLK cycles from each chain's start (word 0 a
    row's runs)."""
    if not getattr(program, "chain_stamps", False):
        raise ValueError(
            f"{type(program).__name__}: made without chain_stamps=True: no stamps"
        )
    first, count = group
    if tensor.pages < first + count:
        raise ValueError(
            f"chain stamps: the group ({first}, {count}) needs {first + count} pages "
            f"(a page a flat rank), and the tensor has {tensor.pages}: "
            f"tensor_for(session, {first + count})"
        )
    field = run.programs.index(program)
    handler = run.field_handler(program.chain_handler, field)
    args = [tensor.address] + [0] * 9  # a7 0: no records, the read-out
    event = run.wait(run.submit(handler, group, args, exclusive=True), timeout=timeout)
    if event.kind != "done":
        raise RuntimeError(f"chain stamps: {event}")
    (pages,) = session.read(tensor)
    w = kc.words()
    shape = (w["kChainStampKernels"], w["kChainStampRecords"], w["kChainStampWords"])
    words = pages.view(np.uint32).reshape(pages.shape[0], -1)[first : first + count]
    return words[:, : w["kChainStampBytes"] // 4].reshape(count, *shape)


def terms(rows, mhz, records):
    """Each of `records` records' terms (us, from rows that read() gave, at `mhz`):
    [{term: (median, largest) over the cores that ran it}] in record order. Times are
    each core's means over the chains it ran since the last read."""
    w = kc.words()
    out = []
    for record in range(min(records, w["kChainStampRecords"])):
        writer, reader = rows[:, WRITER, record], rows[:, READER, record]
        ran = writer[:, 0] > 0
        if not ran.any():
            out.append({})
            continue
        runs = writer[ran, 0:1].astype(np.float64)
        at = writer[ran].astype(np.float64) / runs / mhz  # us from the chain's start
        read_at = reader[ran].astype(np.float64) / np.maximum(reader[ran, 0:1], 1) / mhz
        taken = at[:, w["kChainStampTaken"]]
        marked = at[:, w["kChainStampMarked"]]
        loaded = np.maximum(at[:, STAMP_LOADED], read_at[:, STAMP_LOADED])
        body_from = np.maximum.reduce([taken, marked, loaded])
        body = at[:, STAMP_BODY]
        acked = at[:, w["kChainStampLast"]]
        marks = at[:, w["kChainStampMarks"]]
        found = {
            "taken": taken,
            "input wait": np.where(marked > 0, marked - taken, 0.0),
            "overlay load": np.where(loaded > 0, loaded - taken, 0.0),
            "body": body - body_from,
            "hand-over": np.where(acked > 0, acked - body, 0.0),
            "marks": np.where(marks > 0, marks - acked, 0.0),
            "record end": np.maximum(marks, body),
        }
        out.append(
            {
                name: (round(float(np.median(v)), 2), round(float(v.max()), 2))
                for name, v in found.items()
            }
            | {"cores": int(ran.sum())}
        )
    return out
