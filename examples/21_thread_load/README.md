# 21 thread load

An X280 thread gives cores another program while the run is live. Two programs in the
runtime's form are loaded with a dict, each on its own group
([a group switches program](../../docs/reference/host.md#a-group-switches-program)):

- `sum` (cores 0 to k - 1): each core adds its share of a round's input tiles into one
  bfloat16 tile.
- `widen` (cores k to the last): each core writes its share of the tiles as float32.

The two hold other buffers in other formats (`sum`: bfloat16 at 0 and 16; `widen`:
bfloat16 at 1, float32 at 16), so a core that changes program sets its buffers up again.
Their readers and computes are [16_two_programs](../16_two_programs/README.md)'s. Their
writers do not write the tiles: each core hashes its tiles and writes one 64-byte line
(the hash, its tiles, the round, the program), so the host can check every core of
every round of a job of thousands of rounds.

The host puts every round's amounts in DRAM beforehand: how many tiles each program has,
changing every round. It sends the thread one command; the thread then runs every round
on the card. Each round it reads the amounts, moves the boundary `k` to sum's share of
the tiles, and when `k` changed it loads the cores between the old and the new boundary
with the other program, one call
([a thread gives a group another program](../../docs/reference/threads.md#a-thread-gives-a-group-another-program)):

```c
if (split != k) { /* the cores between the two boundaries change program */
    status = split < k
        ? kmx_load(kProgramWiden, split, k - split, deadline, &loaded)
        : kmx_load(kProgramSum, k, split - k, deadline, &loaded);
}
/* then SUM to (0, k) and WIDEN to (k, cores - k), both waited for */
```

| Handler | Program | Arguments |
|---|---|---|
| `SUM` | sum | a0 the input, a1 tiles, a2 the lines, a3 the round's first line, a4 the job and round, a6 the first input tile |
| `WIDEN` | widen | the same |
| `ROUNDS` | the thread | a0 the first round, a1 rounds, a2 the job, a3 the boundary when it starts |
| `CHECK` | the thread | a0 widen's record, a1 a core of sum, a2 a core of widen, a3 the line its SUMs write |

`CHECK` runs before the rounds and writes what it saw into a region:

- a SUM to a core of widen: refused by the dispatcher, reason 13; the thread's
  `kmx_wait` returns the rejected event and the run goes on;
- a load sent with `kmx_submit` to the core of a SUM of 128 tiles still in flight:
  rejected at once, reason 11 (the dispatcher is busy for it); then `kmx_load` of the
  same load returns `KMX_OK` after that SUM ended (on the card: the SUM took 28 µs, the
  load returned 37 µs after the SUM was sent);
- `kmx_load` of sum back to that core, and a SUM to it: done.

The host then does the same rounds itself: for each round `run.load` and its wait when
`k` changed, then both commands and their waits. It checks every line of every round of
every job against numpy (the hash of each sum tile and of each widened tile, the tiles,
the round, the program each core ran), and each round's `k` the thread chose.

## Files

| File | What it is |
|---|---|
| `thread_load.h` | what the thread, the writers and Python share (plain C): handlers, the line, the hash, the thread's words, a round's line and a log line |
| `kernels/sum/`, `kernels/widen/` | the two programs, `km_program` in each kernel; `kernels/read_tiles.h` and `two_programs.h` as in example 16 |
| `thread/rounds.c` | the thread: ROUNDS (the boundary, the load, both commands, both waits) and CHECK |
| `thread_load.py` | the thread's jobs, the host's job, every line checked, the table and one JSON line |

## What it needs

A Komorebi install and a card prepared as in [Running on a card](../../docs/install.md#running-on-a-card).

```
python thread_load.py --cores 12                       # a row
python thread_load.py                                  # the whole grid, 40 rounds
python thread_load.py --rounds 2100 --jobs 5 --no-host  # 10,000 loads and more
```

`--rounds` (40), `--jobs` (thread jobs, each a load of its own, 1), `--no-host`,
`--no-check`, `--seed`. The script exits non-zero if a line or a `k` is wrong or CHECK
did not see what it must.

## Numbers

A Blackhole p150a at AICLK 1,350 MHz before and after, 2026-11-14, host load 4 to 6;
`python thread_load.py --rounds 40` (120 cores) and `--cores 12`. Every line of every
round exact. Times of the thread on its clock, of the host on its own around each step.
A load from issue to done: `kmx_load`'s call, or the host's `run.load` and its wait; in
the dispatcher: the load's event, submit to completion.

| | 12 cores, thread | 12 cores, host | 120 cores, thread | 120 cores, host |
|---|---|---|---|---|
| the whole job, 40 rounds | 1,709 µs | 15,670 µs | 1,825 µs | 16,950 µs |
| a load, issued to done | 6.02 µs | 133.92 µs | 6.19 µs | 134.85 µs |
| a load in the dispatcher | 3.88 µs | 3.92 µs | 4.11 µs | 3.96 µs |
| loads a round | 0.775 | 0.775 | 1.0 | 1.0 |

`--rounds 2100 --jobs 5 --no-host` on 120 cores: 10,345 loads issued by the thread, a
SUM and a WIDEN between each two, every line of the 10,500 rounds exact; each job of
2,100 rounds took 94.2 ms on the thread's clock (44.9 µs a round).

What the numbers say:

- The load is the same command from either side: the dispatcher takes 4 µs for it
  whoever sent it. Issued by the thread it returns 6 µs after the call; from the host
  each load and its wait cost a round trip over PCIe, about 134 µs, as every command
  the host waits for does.
- So the rounds that move the boundary run about nine times faster from the thread
  (1.8 ms against 17 ms for 40 rounds), the host sending one command.

## What this example does not do

- A command's program and cores are fixed when it is submitted: the load moves cores
  between rounds, not while a round runs.
- The boundary follows the tile counts alone, not measured times (that is
  [19_thread_programs](../19_thread_programs/README.md)'s split, over a group that holds
  both programs).
- A load issued by a cast or a pull, a switch in the combined form, a switch of a
  dynamic program: not possible.
