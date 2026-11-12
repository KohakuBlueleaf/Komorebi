# page_index

Can the compute kernel, after waiting for all S pages of a circular buffer, read page
k by its index, with nothing copied or popped before it? This decides how a core
hands values that other cores put into fixed slots of its buffer to its compute
kernel.

Core (1, 0) reads input tile k into its own page 0, puts it into page k of core
(0, 0)'s buffer (same L1 address: one program, one buffer layout), waits for the
write, and bumps a flag on core (0, 0). The receiver's reader fills every other page
from the input, waits (bounded) for the flag, and pushes all S = 4 pages. Compute
waits for 4, copies page k and then page j by index into two output pages, pops 4.
The writer stores both; the host compares them with input tiles k and j.

```
python probe.py --slot 2 --other 1
python probe.py --slot 3 --other 0
```

Result (Blackhole p150a, 2026-11-13): both runs exact for both pages (`page_k_exact`,
`page_j_exact` true; every wait returned in time). Yes: `copy_tile(cb, k, dst)` after
`cb_wait_front(cb, S)` reads page k. The runtime's arrival buffer is handed over whole
(S pages pushed, S popped) and read by slot index.
