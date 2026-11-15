# Test feeder

Runs the persistent kernel kernels on a card without the X280: core 119 (the last of the
12 x 10 grid) runs `feeder.cpp` and plays the dispatcher, and cores 0 to 118 run a
program's reader, compute and writer. `test_mapcollect_fed.py` beside this directory
calls `mapcollect_fed.py` in one device open (set `KMR_CARD=1` to run the card tests).
Longer runs are run by hand: `experiments/checks/feeder_runs/`.

| File | What it does |
|---|---|
| `feeder.cpp`, `feeder.h` | Takes commands from a DRAM table, writes each one to a free ring slot of every worker, waits for the leader's acknowledgement in its own L1, and writes every core's stop flag when a command misses its deadline. |
| `fed.py` | `Fed.run(kernels, commands, ...)`: builds the program (start-up arguments, control blocks, buffers), runs it and returns the feeder summary, one row per command, every core's state words and the trace. |
| `commands.py` | Constants read from the headers, command words, and the decoding of the feeder rows and the state words. |
| `mapcollect_fed.py` | The map/collect example's kernels under the feeder. Prints one JSON line. |

## The feeder's rules

- A command goes to every worker; only its group runs it. The feeder writes the ring slot
  into bits 16 to 23 of `hdr`, and a sequence number into `sequence` and the landing words.
  A worker does not run a command whose header names another slot.
- A command with `KM_FLAG_EXCLUSIVE` waits until no command in flight has a group that overlaps
  its own.
- An EXIT command waits until no slot is busy and goes out with a group of every worker.
  Nothing after it is submitted. Without one in the table the feeder sends one at the end.
- A command that is not acknowledged within the deadline (from its submit) stops the run:
  the feeder writes the stop flag of every core and commands the slot, sequence number
  and table index.
- A command with `KM_FLAG_RESULT` is acknowledged with `{ack, result}` in one 8-byte write;
  the feeder keeps the result word in the command's row.

## mapcollect_fed.py

| Option | Meaning |
|---|---|
| `--jobs N`, `--handler map\|nop\|align` | N commands of one handler. |
| `--groups fixed\|random\|pair\|overlap`, `--first`, `--count`, `--group-list f:c,...` | How groups are chosen. |
| `--tiles T`, `--out-pages`, `--out-tensors` | MAP input tiles per job (job j reads tiles j to j + T - 1) and output space. |
| `--depth D` | Ring slots in use (1 to 64). |
| `--wait S`, `--deadline S` | A core's timeout on a wait for another core; the feeder's per-command deadline. |
| `--stall J --stall-rank M` | Job J: rank M waits for the stop flag. |
| `--late J --late-rank M --late-cycles C` | Job J: rank M reports C cycles late. |
| `--bad-item J --bad-side writer\|reader` | Job J: rank 0 asks for an item past the end of its buffer. |
| `--result J --result-value V` | Job J is a RESULT command with `KM_FLAG_RESULT`: the leader acknowledges it with the result word V, shown as `result`. |
| `--slow J --slow-rank M --slow-cycles C` | MAP job J: rank M's compute holds its tile C cycles. |
| `--exit-at J`, `--exit-group f:c` | Job J is an EXIT (with that group in the table). |
| `--raw J`, `--raw-slot S` | Job J is submitted as built: the header names slot S and the group is kept (tests of commands that name the wrong slot, or an EXIT with a partial group). |
| `--row N` | Cores per grid row for completion; 0: one level. |
| `--exclusive` | `KM_FLAG_EXCLUSIVE` on every MAP command. |
| `--trace`, `--dump FILE` | Per-command timestamps; save rows, state and trace (npz). |

The JSON line holds the feeder summary, per-job checks (`jobs_checked`, `jobs_bad`),
the cores that gave up and where, and `ranks_bad`: cores whose runs, completions,
rejected commands or last seen and reported sequence numbers differ from what the groups
of the submitted and acknowledged commands give.

Times in the summary and the trace are Tensix wall-clock cycles compared across cores
and converted with the AICLK the driver reports. Measured with
`experiments/core/wall_clock/` (one Blackhole p150a, core (0, 0) against six cores from the
corners to the centre): the clock is one chip-wide count, every core within 90 cycles
(67 ns) of the first, with no measurable drift over 2 s (under 0.01 ppm); against host
time it runs at 1343 to 1347 MHz with the driver reporting 1350 (within the host
timing's error).
