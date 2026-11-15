# 01 map/collect, written as a class

`MAP` of [../mapcollect.py](../mapcollect.py) as one method of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md), experimental).
Made, the class is compiled into an ordinary program's files under `build/mapcollect/`
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
same relative RMS (0.001722); kernel text 34,256 B against the hand program's 34,288
B. A job on 8 cores: host 28.8 us against 34.7 us, latency 8.32 us against 7.27 us.
1,000 jobs on 120 cores (64 tiles: one tile or none a core): 6.56 us a command on the
card's clock against the hand program's 2.76 us.
