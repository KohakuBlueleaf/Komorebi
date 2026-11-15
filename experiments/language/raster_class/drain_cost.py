"""What a drained stream's end costs: a command that fills no page, ends its stream and
drains it, beside one with a body alone; then 64 pages filled, drained and put, each
page's record checked. Example: python drain_cost.py"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

import komorebi as km  # noqa: E402
from projects.raster import card  # noqa: E402

RECORDS = 64  # pages a core fills and drains in the second command
REPEATS = 5
# A page's record: its number and the core's, at the slot the mover puts it.
FILL = """
volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(out);
if (out != 0) {
    words[0] = record;
    words[1] = first + record;
    asm volatile("fence" ::: "memory");
}
"""


class Drained(km.Program):
    """drain: `records` pages pack fills, ended, drained by mover B to slots
    first + record of `lists`; idle: a pack body alone (the command's own cost)."""

    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def drain(self, lists: km.Tensor, records: int):
        first = self.rank * RECORDS
        with km.Stream(self.out, mover="B") as out:
            for record in range(records):
                km.cxx(FILL, takes=[out, record, first], on="pack")
        for page in out:
            lists.items(size=128).put(page, at=page[1])

    @km.handler
    def idle(self, lists: km.Tensor, records: int):
        km.cxx("(void)records;", takes=[records], on="pack")


def main():
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        lists = card.tensor(session, np.zeros((workers * RECORDS // 16, 512)))
        program = Drained(log=card.log, session=session, depth=1, lists=lists)
        loaded = session.load(program, deadline=1.0)
        times = {"idle": [], "drain_0": [], f"drain_{RECORDS}": []}
        try:
            for name, handler, records in [
                *[("idle", Drained.idle, 0)] * REPEATS,
                *[("drain_0", Drained.drain, 0)] * REPEATS,
                *[(f"drain_{RECORDS}", Drained.drain, RECORDS)] * REPEATS,
            ]:
                handle = loaded.submit(handler, (0, workers), args=(lists, records))
                times[name].append(loaded.wait(handle, timeout=10.0).microseconds)
        finally:
            loaded.stop()
        status = card.check_run(loaded, workers, 3 * REPEATS, sum(times.values(), []))
        (pages,) = card.read(session, lists)
        slots = pages.reshape(-1, 32)
        want = np.arange(workers * RECORDS)
        wrong = int((slots[:, 1] != want).sum() + (slots[:, 0] != want % RECORDS).sum())
    report = {
        "run": status,
        "microseconds": {name: float(np.median(t)) for name, t in times.items()},
        "all": times,
        "records_wrong": wrong,
    }
    print(json.dumps(report, default=int), flush=True)
    sys.exit(0 if status["ok"] and wrong == 0 else 1)


if __name__ == "__main__":
    main()
