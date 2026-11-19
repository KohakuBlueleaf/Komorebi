# Commands across islands that do not finish there

Question: when a command crosses from one island's own run (`run.island(0)`, so its
dispatcher sends each other island's part across) and the far part cannot finish, or
the far dispatcher refuses it, does the source get an answer and does the run go on?

`cross_cases.py` opens two cards and runs, with a deadline of 1 s a command:

- example 01's HANG on island 0's last 10 ranks and island 1's first 10, island 1's
  rank 5 waiting for an arrival that never comes: the source must get a TIMEOUT naming
  island 1 (`KM_EVENT_ISLAND + 1`) at the deadline; then a NOP across (island 0 and
  island 1's ranks before the hung one) and a NOP on island 0 alone must be done, and
  the run's stop must be clean;
- example 16's sum and scale, scale on island 1's last 10 ranks, its first 5 switched
  to scale: sum's EMPTY on island 0's last 5 ranks and those 5, sent without the host's
  check, must be rejected by island 1's dispatcher (reason 13: program) and the source
  must get that rejection, not a hang; after sum's load back there the same EMPTY must
  be done.

```
python experiments/islands/cross_cases/cross_cases.py --cards 0,1
python experiments/islands/cross_cases/cross_cases.py --cards 0,1 --dispatcher tensix
```

## Results

Blackhole cards 0 and 1, 2026-11-19: every check holds under both dispatchers (the X280
and the Tensix dispatcher); the run takes 8 s.
