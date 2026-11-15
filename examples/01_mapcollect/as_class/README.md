# 01 map/collect, written as a class

`MAP` of [../mapcollect.py](../mapcollect.py) as one method of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md), experimental).
Made, the class is compiled into an ordinary program's files under `build/mapcollect-<hash>/`
beside the script: `mapcollect.h`, `movement.cpp` (both movers) and `compute.cpp`,
with `program.txt`, the buffers the build chose. The jobs, the checks and the JSON line
are the hand program's.

```bash
python mapcollect_class.py                       # the class, the hand program's defaults
python mapcollect_class.py --beside --timing 1000 --out-pages 256
```

`--beside` runs the hand program first in the same session and from the same seed;
`--timing N` then runs N unchecked jobs on the whole grid, each program in turn. The
other flags are the hand program's.

The class reads `tiles` pages from `first_tile`: the hand program's `a4 slow_cycles`
and `a5 slow_rank` (a test of slow compute) and its other handlers are not in it.

Measured on one Blackhole card (`--beside --timing 1000 --out-pages 256`): both
programs ok, 20 jobs checked on 8 cores, 10 of them exact, the random ones at the
same relative RMS (0.001722); kernel text 33,584 B against the hand program's 34,288
B. On the cores, 1,000 jobs on 120 cores (64 tiles: one tile or none a core), from the
first core seeing a command to the first core seeing the next (trace stamps): 1.77 us
a command against the hand program's 1.91 us. The write of the core's tile is a page
write flushed as the hand writer's (`km::write_pages`, `noc_async_writes_flushed`);
written as a one-page `km::move`, settled by acknowledgement, it took 0.46 us a
command more.

## On two cards

```bash
python islands_class.py --beside   # cards 0 and 1, 236 ranks
```

The same class, unchanged, on `km.Session(cards=[0, 1])`, made with the island
session's tensors; the build prints what may cross a link (both moves of `map`: the
input and the output are split by rank over both islands). The rounds and checks are
[../islands.py](../islands.py)'s: MAP on each island's own group, then on island 0
reading the input it holds and one island 1 holds. Measured on two Blackhole cards,
both programs in one session, 20 rounds of each one-group read: every tile exact;
reading island 1's input, 2,360 asks went to island 1 and 4,720 pages came back, as
the hand program's. MAP on the own groups 2.88 us a command against 3.04 us. A round
is one command timed from the host, its median:

| Island 0 reading | Class | Hand | of it: the submit | the dispatcher |
|---|---|---|---|---|
| its own input | 198.4 us | 190.6 us | 22.6 / 14.5 us | 4.82 / 4.62 us |
| island 1's input | 626.0 us | 615.5 us | 23.2 / 14.7 us | 448.4 / 448.1 us |

The dispatcher's column is each command's submit to its completion; the rest of a
round (171 us reading its own input) is the host's wait over the two islands, the
same in both. The class's extra is its submit: its handler makes the command's words
from the tensors and ints in Python.
