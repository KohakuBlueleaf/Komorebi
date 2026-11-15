# 02 merge tree, written as a class

`MERGE` of [../merge.py](../merge.py) as one method of a class deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md), experimental): each core
adds its part of the tiles, `km.reduce` sums the parts by owner or along a tree, and the
leader writes the sum. Made, the class is compiled into the program's files under
`build/mergetree/` beside the script. The jobs and the checks are the hand program's.

```bash
python merge_class.py --beside                   # 20 groups of 8, owner and tree in turn
```

`--beside` runs the hand program first in the same session and from the same seed.
The class sums parts of one tile (`--part 1`) and stalls no rank (`--stall-rank`
is refused). Its leader returns the hand program's result word, the sum's 1,024
32-bit words added, by a `km.cxx` body on mover B (`gives=int`) and `return`.

Measured on one Blackhole card (`--beside`): both ok, 40 jobs, every exact job equal
to numpy and each owner and tree pair equal bit for bit, the random ones at relative
RMS 8.6e-08, and all 40 result words right. Kernel text 42,240 B against the hand
program's 36,176 B: the class's read is a stream (`km::move`), 9,648 B of it (BRISC
3,288, NCRISC 2,944, TRISC0 2,452, TRISC1 936, TRISC2 44), where the hand program's
reader reads its pages; its write is a page write, as the hand writer's. On the cores
(trace stamps, the same 40 jobs), a command from its first core seeing it to the
leader's sent: 15.33 us against 15.14 us (median, +1.3%); the 40 commands 643.4 us
against 638.5 us. From the host, 49.6 us a job against 29.5 us: the fixed costs of a
40-job loop (the hand program alone gave 43.5 us and 29.5 us in two runs) and the
class's words, 3.2 us a job (made in Python by `submit`).
