# One card's calls on islands, by value

Two cards opened as the islands of one session (`km.Session(cards=[0, 1])`). Do the
calls that an island run and an island tensor take as one card's give one card's values?
The host-side checks (names, keywords, shapes) are CPU tests
(`tests/core/islands/test_one_unit.py`). This script checks three calls whose values only
the cards can give:

- `run.submit(..., block=)`: 32 DOT commands of example 01's kernels on flat ranks
  114 to 121 (island 0's last four and island 1's first four), each with its own block
  of 64 words. Every core's reader sums the block against the command's word and gives
  up on a mismatch, so a part that read another block, or none, is not done. Each
  part's writer returns the dot of the block, which is checked against numpy.
- `run.threads` and `run.thread_results()`: `thread/island.c` on work clusters 1 and 2
  of both cards returns at once with its island, its cluster and `KMX_THREAD(island,
  cluster)`. The host keys the results by flat thread number (3 x island + cluster), so
  the numbers must be 1, 2, 4 and 5, each matching the thread's own words.
  `run.thread_result()` must return the lowest number's result.
- `session.tensor(..., transposed_tiles=True)`: Y = X W (64 x 256 x 128, bfloat16,
  HiFi4) with W in SrcB (`x @ km.srcb(w)`). W is a copy on every island, stored with
  its tiles transposed, and X and Y are split by rows. One APPLY on the same flat ranks
  computes Y's row 0 on island 0 and row 1 on island 1. Y is checked against float64,
  and W is read back. Setting `transposed_tiles` after the data was written must be
  refused.

    python experiments/islands/islands_calls/calls.py [--cards 0,1] [--jobs 32] [--words 64]

It prints one JSON line and exits 1 when a check fails.

## Result

2026-11-18 00:43, cards 0 and 1 (p150a), host load 69.6, one run, after
`kmr prebuild --cards 0,1` (0 compiled at the run), 4.8 s:

| Check | Result |
|---|---|
| DOT with a block, 2 parts each | 32 of 32 done; 0 wrong results (64 part results, 0 wrong) |
| Threads by flat number | 1, 2, 4, 5: each `done`, code 7, words (island, cluster, number) agreeing; `thread_result()` is thread 1's |
| Turned product, W transposed on both islands | done on 2 parts; relative RMS error 3.11e-4, largest error 1.30e-3 of Y's RMS; W read back exact |
| `transposed_tiles` set after the data | refused |
| Kernels at both stops | `none` on both islands |
