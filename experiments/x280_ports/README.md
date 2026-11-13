# X280 ports

Questions the ports between X280 cores and Tensix cores depend on, asked on a card
before the library served them: how long one ask takes to be answered, how many
messages the X280 cores handle, what a NoC window costs, whether a 4 KiB block is better
sent into L1 or through DRAM, and whether the port stays exact under load and across
a command that times out.

## Results

| Question | Answer |
|---|---|
| An X280 core's tell into a core, a core's tell to its X280 core | Exact over 1,000 commands each way |
| One ask, one X280 core, one Tensix core | 0.94 µs mean (p99 1.14) |
| One ask at full load | 12 X280 cores, 120 Tensix cores (ten each): 9.06 µs mean, p99 9.86; 13.1 M answered asks a second, 1.09 M an X280 core. The X280 core's loop bounds it (about 0.9 µs a served message), not the NoC |
| A NoC window | aiming one again: 230 cycles (450 with the read-back); a 16-byte store through one held: 51. A cluster can hold 168 windows, so the library holds one per served core for the run |
| A look at a mailbox line | 259 cycles (`kmx_port_asked` with nothing new); a LIM word costs 30, a cached region word 3 |
| 4 KiB to a core | into L1 at an address the core publishes: 6.5 µs; through DRAM (write, flush, the core reads): 12.6 µs. 120 cores, 12 X280 cores: 8.1 GB/s against 2.1 |
| A soak | 1,000,800 round trips over 120 cores while the writers read 1.0 M DRAM pages: exact, 9.58 µs mean, p99 9.86. With the X280 cores' stores on NoC 1: the same |
| A command that times out with X280 cores mid-serve | its event ends `kmx_port_run`; the next load in the same session is exact |
| Both ways at full rate | 120 cores sent 76 M tells a second (latest wins: the X280 cores saw 72,542 of 580,409); the X280 cores handled 28.7 M messages a second, 2.4 M an X280 core; asks under that load 10.2 µs mean |

One Blackhole p150a, slow dispatch, AICLK 1350 MHz (read with the grid running), the
X280 at the session's 1000 MHz.

## The probe

`x280/ports.c` is one kmx thread on clusters 1 to 3, word 0 its mode; the grid's
reader owns each core's port (`kernels/reader.cpp`), the writer reads DRAM beside it
when asked. `ports.py` loads the program, waits for the threads and prints one JSON
line per step:

```sh
python ports.py tell --rounds 1000
python ports.py send --rounds 1000
python ports.py ask --cores 120 --clusters 1 2 3 --rounds 10 --asks 1000
python ports.py window --rounds 1000
python ports.py block --cores 120 --clusters 1 2 3 --rounds 3 --asks 20 --kind l1
python ports.py soak --cores 120 --clusters 1 2 3 --rounds 10 --asks 834 --reads
python ports.py timeout --cores 120 --clusters 1 2 3 --rounds 3 --stop-after 500 --then
python ports.py rate --cores 120 --clusters 1 2 3 --rounds 3 --units 10000
```

- `tell`, `send`: one core, a command per round; the core gives the X280 core's word as its
  result, or tells the X280 core one word the X280 core checks.
- `ask`, `soak`: each core asks `--asks` times per command and checks every answer; the
  X280 cores serve through `kmx_port_run` until the command's event. `--reads` has the
  writers read DRAM pages meanwhile; `--noc 1` puts the X280 cores' stores on NoC 1.
- `window`: the window costs, on one X280 core.
- `block`: each core asks for 4 KiB blocks, into its L1 (`--kind l1`) or through the
  cluster's bank (`--kind dram`), and checks every word.
- `timeout`: the X280 cores stop answering after `--stop-after` answers, so the command times
  out mid-serve; `--then` loads the program again in the same session.
- `rate`: tells both ways as fast as each side can for `--units` × 1024 cycles, an
  ask every eighth tell.

Each core keeps a histogram of its round trips in AICLK cycles; `ports.py` turns it
into microseconds. `ok` is false, and the script exits non-zero, when a check fails.
