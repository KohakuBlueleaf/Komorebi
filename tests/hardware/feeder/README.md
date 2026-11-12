# Test feeder

Runs the resident worker kernels on a card without the X280: core 119 (the last of the
12 x 10 grid) runs `feeder.cpp` and plays the dispatcher, and cores 0 to 118 run a
program's reader, compute and writer. `test_mapcollect_fed.py` beside this directory
drives it through `mapcollect_fed.py` (set `KMR_CARD=1` to run the card tests).

| File | What it does |
|---|---|
| `feeder.cpp`, `feeder.h` | Takes records from a DRAM table, writes each one to a free ring slot of every worker, waits for the leader's acknowledgement in its own L1, and writes every core's stop word when a record misses its deadline. |
| `fed.py` | `Fed.run(kernels, records, ...)`: builds the program (start-up arguments, control blocks, buffers), runs it and returns the feeder summary, one row per record, every core's state words and the trace. |
| `records.py` | Constants read from the headers, record words, and the decoding of the feeder rows and the state words. |
| `mapcollect_fed.py` | The map/collect example's kernels under the feeder. Prints one JSON line. |

## The feeder's rules

- A record goes to every worker; only its team runs it. The feeder writes the ring slot
  into bits 16 to 23 of `hdr`, and a sequence number into `seq` and the landing words.
  A worker does not run a record whose header names another slot.
- A record with `KM_F_EXCL` waits until no record in flight has a team that overlaps
  its own.
- An EXIT record waits until no slot is busy and goes out with a team of every worker.
  Nothing after it is issued. Without one in the table the feeder sends one at the end.
- A record that is not acknowledged within the deadline (from its issue) stops the run:
  the feeder writes the stop word of every core and records the slot, sequence number
  and table index.

## mapcollect_fed.py

| Option | Meaning |
|---|---|
| `--jobs N`, `--seg map\|nop\|align` | N records of one segment. |
| `--teams fixed\|random\|pair\|overlap`, `--first`, `--count`, `--team-list f:c,...` | How teams are chosen. |
| `--tiles T`, `--out-pages`, `--out-tensors` | MAP input tiles per job (job j reads tiles j to j + T - 1) and output space. |
| `--depth D` | Ring slots in use (1 to 16). |
| `--wait S`, `--deadline S` | A core's bound on a wait for another core; the feeder's per-record deadline. |
| `--stall J --stall-member M` | Job J: member M waits for the stop word. |
| `--late J --late-member M --late-cycles C` | Job J: member M reports C cycles late. |
| `--bad-item J --bad-side writer\|reader` | Job J: member 0 asks for an item past the end of its buffer. |
| `--slow J --slow-member M --slow-cycles C` | MAP job J: member M's compute holds its tile C cycles. |
| `--exit-at J`, `--exit-team f:c` | Job J is an EXIT (with that team in the table). |
| `--raw J`, `--raw-slot S` | Job J is issued as built: the header names slot S and the team is kept (tests of records that name the wrong slot, or an EXIT with a partial team). |
| `--row N` | Cores per grid row for completion; 0: one level. |
| `--excl` | `KM_F_EXCL` on every MAP record. |
| `--trace`, `--dump FILE` | Per-record stamps; save rows, state and trace (npz). |

The JSON line holds the feeder summary, per-job checks (`jobs_checked`, `jobs_bad`),
the cores that gave up and where, and `membership_bad`: cores whose runs, completions,
refused records or last seen and reported sequence numbers differ from what the teams
of the issued and acknowledged records give.

Times in the summary and the trace are Tensix wall-clock cycles compared across cores
and converted with the AICLK the driver reports. Measured with
`experiments/wall_clock/` (one Blackhole p150a, core (0, 0) against six cores from the
corners to the centre): the clock is one chip-wide count, every core within 90 cycles
(67 ns) of the first, with no measurable drift over 2 s (under 0.01 ppm); against host
time it runs at 1343 to 1347 MHz with the driver reporting 1350 (within the host
timing's error).
