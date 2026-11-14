# 19 thread programs

An X280 thread splits the grid between two programs each round. Two programs in the
runtime's form are loaded combined (every core holds both,
[several programs in one run](../../docs/reference/host.md#several-programs-in-one-run)):

- `spin` (a reader part): each unit waits the cycles its cost line in DRAM names, then
  writes a line of results. What a unit costs depends on the data.
- `scale` (reader, compute and writer parts): each tile is multiplied by a scalar on the
  vector unit. A tile costs about the same every time.

The host puts every round's work in DRAM beforehand: how many spin units and how many
tiles, and the units' costs (their mean swings over the rounds). It sends the thread one
command; the thread then runs every round on the card. Each round it reads the round's
amounts, chooses `k` (spin on cores 0 to k - 1, scale on the rest), submits both commands
([a thread in a run of several programs](../../docs/reference/threads.md#a-thread-in-a-run-of-several-programs)),
waits for both, and keeps what the round told it for the next one.

**How the cores report.** Through nothing of their own: each command's event, which the
thread's `kmx_wait` returns, carries its time from submit to completion. The thread takes
off a fixed part (an empty SPIN on one core, timed once a job), divides by the most units
one core had, and has each program's ticks a unit. The next round's `k` is the one, near
the share of each program's estimated work, that makes the slower command fastest. No
port or extra word was needed.

| Handler | Program | Arguments |
|---|---|---|
| `SPIN` | spin | a0 the first unit, a1 units, a2 the costs, a3 the outs, a4 the job |
| `SCALE` | scale | a0 the input, a1 tiles, a2 the output, a3 the first output tile, a4 the scalar (float32 bits), a6 the first input tile |
| `ROUNDS` | the thread | a0 0 (the thread splits) or 1 (k = a1 in every round), a2 rounds, a3 cores, a4 the job, a5 the scalar |

The host checks every unit of every round of every job against numpy: each spin unit's
line, and each tile (its scalar differs from the job before, so no result is stale).

## The thread, in short

```c
kmx_command(&spin, kProgramSpin, kProgramSpinHandlerSpin, 0, spin_cores);
spin.a0 = first_unit; ...
kmx_command(&scale, kProgramScale, kProgramScaleHandlerScale, spin_cores,
            cores - spin_cores);
scale.a1 = tiles; ...
submit(&spin, &spin_handle);
submit(&scale, &scale_handle);
kmx_wait(spin_handle, deadline, &spin_event);  /* .ticks: submit to completion */
kmx_wait(scale_handle, deadline, &scale_event);
```

`kProgramSpin`, `kProgramSpinHandlerSpin` and the rest are defines the thread's build
gets from the run's programs. The thread is declared by `scale`, the second program: any
one program of a run may declare it.

## Files

| File | What it is |
|---|---|
| `programs.h` | what the thread, the kernels and Python share (plain C): the handlers, buffers, the thread's words, a round's line and a log line |
| `kernels/spin/`, `kernels/scale/` | the two programs, `km_program` in each kernel |
| `thread/rounds.c` | the thread: ROUNDS, the split, the submits and waits |
| `thread_programs.py` | the jobs (the thread's split, fixed splits, the host feeding a fixed split), every result checked, the table and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python thread_programs.py --cores 12   # one row
python thread_programs.py              # the whole grid
```

`--rounds` (24), `--splits` (the fixed splits tried, 8), `--seed`. Each job is a load of
its own. The script exits non-zero if a result is wrong.

## Numbers

A Blackhole p150a at AICLK 1,350 MHz, 24 rounds: spin 2n to 8n units a round (n the
cores) of 200 to 7,800 cycles on average, scale 4n to 16n tiles. Every unit of every job
exact. Times are the sum of the rounds on the thread's clock (for the host, its own
clock around each round).

| | 12 cores (a row) | 120 cores |
|---|---|---|
| the thread's split, every round | 974 µs | 1,023 µs |
| the best fixed split, run by the same thread | 1,001 µs (k 9) | 1,070 µs (k 93) |
| the same fixed split, fed by the host round by round | 7,378 µs | 7,521 µs |
| the thread's time a round outside the grid's work (decision, two submits, the waits) | 3.1 µs | 3.2 µs |

The best fixed split is found by trying 8 values of `k` on the host, each a whole job,
so it knows the whole job in advance. The thread knows only the past rounds and does
3 to 4% better. Its first round has no times yet and splits by unit counts alone. Fed by
the host, a round costs about 300 µs of submits and waits over PCIe against 40 µs of
work.

## What this tier cannot do

- A command's program and its cores are fixed when it is submitted. Units are not moved
  from a slow command to the cores of a fast one while the round runs (that is the
  per-core tier, `18_thread_split`).
- The thread decides from times one round late; a round much unlike the one before is
  split badly.
- Cores that do not hold a program cannot be given its work: the split is over a group
  that holds both, the combined form here.
