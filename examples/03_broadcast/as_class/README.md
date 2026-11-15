# 03 broadcast, written as a class

`BROADCAST` of [../broadcast.py](../broadcast.py) as one method of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md), experimental):
the row broadcast runs on mover A (NoC 0) and the column broadcast on mover B (NoC 1);
each core adds every tile of both into one tile. Made with the rectangle and the
grid's NoCs, the class is compiled into the program's files under `build/broadcasts-<hash>/`
beside the script, and each core's program arguments come from
`layout.broadcast_args`. The jobs and the checks are the hand program's.

```bash
python broadcast_class.py --beside --rect 0,0,12,10 --blocks 16 --pages 4 --window 2 \
    --jobs 8 --first-blocks 3 --empty-every 4
```

`--beside` runs the hand program first in the same session and from the same seed.
The class casts (`--read` is refused), is fed by the dispatcher (`--feeder` is
refused) and stalls no core (`--stall-job` is refused).

Measured on one Blackhole card (the line above): both ok, 960 ranks checked (8 jobs on
120 cores), every one bit-exact, 19,920 blocks delivered, no core and no broadcast gave
up; kernel text 48,720 B against the hand program's 53,408 B (mock build). A job, exclusive: host
109.0 us against 99.8 us.
