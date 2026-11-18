# The test feeder for experiments

One Tensix core plays the dispatcher for the kernels on the others: it reads commands
from DRAM, sends them to every worker and collects their acks in its own L1. The
experiments that drive their kernels this way import it from here:
[wall_clock](../../core/wall_clock/README.md) and
[dram_routes](../../noc/dram_routes/README.md).

These four files are the same as those of `tests/hardware/feeder`, so that an
experiment runs from a checkout without the tests; a change to one copy goes to both.

| File | What |
|---|---|
| `commands.py` | commands, start-up groups and results, as pure functions; constants read from the C headers |
| `fed.py` | runs persistent kernels on every core but one, the last core the feeder |
| `feeder.cpp` | the feeder kernel |
| `feeder.h` | its arguments, buffers and the layout of the results it writes |
