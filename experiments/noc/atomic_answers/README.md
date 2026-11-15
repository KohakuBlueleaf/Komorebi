# Atomic answers

When is the returned value of a non-posted NoC atomic add lost? Routed messages
reserve queue positions with such adds and waits for each old value; at 10^6 forwards on
119 cores some runs ended with reason 32 (`lost`): the add had landed and was counted
as answered, but the old value never arrived.

## Result

**The answer is lost only when the asked tile submits atomics of its own on the same
NoC while it answers, with a return address that changes from call to call.** A tile
that only answers never lost one; neither did a tile whose own adds keep one return
address, or go out on the other NoC. The asker's side does not matter. The add itself
always lands (every tail exact), no other answer is ever wrong, and a lost answer is not
late (waits up to one second). This is an empirical condition of the atomic unit; its
cause is not known.

The runtime therefore keeps every tail on counter tiles, group cores that only answer
([dynamic handlers](../../../docs/reference/dynamic.md#counter-tiles)).

## The probe

`stress.py` runs `writer.cpp` on cores 0 to `cores - 1` (RISCV_1, NoC1). Each core
makes K reservations, one at a time, on random asking cores: an add of 1 (sometimes 2)
to the destination's tail with its old value returned to one of 32 lines of the
asker's own L1, then a timed wait for the atomic's count and for the old value to
replace a sentinel. An answer that does not come within the timeout is counted lost and
its line is not used again; a core whose 32 lines all hold a lost answer stops with
`fail` 4. The host checks that every tail equals what was added to
it. Every wait has a timeout, the start barrier's one second.

```
python stress.py                          # tails on the asking cores themselves
python stress.py --counters 1             # tails on core 119, which only answers
python stress.py --cores 118 --counters 2 # tails on cores 118 and 119
python stress.py --fixed-return           # one return line for every add
```

Card 0, Blackhole p150a, 2026-11-13, AICLK 1350 MHz, K = 20000 (one JSON line each):

| Run | Reservations | Lost | Latency mean / max (cycles) | M per second |
|---|---|---|---|---|
| 119 askers, tails on the askers (2 runs) | 2.38 M each | 13, 11 | 379 / 589, 776 | 257 |
| 119 askers, tails on 1 counter tile | 2.38 M | 0 | 1179 / 2279 | 113 |
| 118 askers, tails on 2 counter tiles | 2.36 M | 0 | 609 / 1142 | 214 |
| 119 askers, tails on the askers, one return line | 2.38 M | 0 | 379 / 631 | 340 |

## The investigation's table

Measured with a larger form of the same probe (more kernels and switches), card 0,
119 cores, one add at a time per core, 2.38 M reservations per run unless noted.

| Run | Lost per run | Reading |
|---|---|---|
| all 119 ask random cores, RISCV_1/NoC1 (6 runs) | 21 to 35 | reproduces, at 1e-5 per answer (the runtime: 2.7e-7) |
| same on RISCV_0/NoC0 | 24, 32 | not NoC1 specific |
| + DRAM message writes / + reader (DRAM reads, tail polls) / both | 14 and 4 / 12 and 17 / 8 to 17 | they neither cause nor cure it |
| every core asks itself | 0 | |
| fan-in to one word (all ask core 0) | 82 | |
| 60 askers, 59 idle answerers (2 runs) | 0 | needs more than answering |
| same, answering cores spin on loads, stores or reads (1.2 M each) | 0 | core activity and reads of the answerer are not it |
| 60 askers, 60 answerers that also ask (1.2 M) | 11 | |
| A asks B; B asks idle C with a constant return address and increment, 2 x 2.4 M | 0 | the answerer submitting atomics is not enough |
| A asks B; B (writer code) asks idle C | 18, 21; every loss an A to B ask | the loser is the asked-and-asking tile |
| same, increment constant 1 on all | 2, 3 | increment variation contributes |
| same, return line fixed on all | 0, 0 | |
| same, return line fixed on B only | 0, 0 | the answerer's return address is the factor |
| same, return line fixed on A only | 21, 14 | not the asker's |
| B's atomics with varying return line and increment, on NoC1 | 34, 24 | |
| the same on NoC0 (the other data-movement core) | 0, 0 | per NoC interface |
| return line fixed, all-to-all, 12 runs | 0 of 28.56 M | |
| every target address distinct per core / all tails at one word | 25, 18 / 16, 10 | address equality is not it |
| adds through command buffer 2 instead of 3 | 38, 27, 26 | not the command buffer |

| Counter tiles | Lost | Latency mean / max (cycles) | Reservations per second |
|---|---|---|---|
| 1 counter tile, 119 askers, 3 runs | 0 of 7.14 M | 1180 / 2324 | 126 M |
| 2 counter tiles, 118 askers, 2 runs | 0 of 4.72 M | 611 / 1122 | 229 M |
| 1 tile, every asker also reading its tail at full rate | 0 of 4.76 M | same | 126 M |
| (tails on the askers) | 21 to 35 per 2.38 M | 359 / 668 | 326 M |

In the investigation's runtime runs on two earlier builds (10^6 forwards on 119 cores, the
forward example under the test feeder, the tails on the feeder's own tile, which submits no
atomics): 0 lost in 111 runs, against 2 and 12 failing runs of 50 with the tails on the
receivers. The counter tiles of the released runtime are measured in the
[forward example](../../../examples/05_forward/README.md#measured).

Not tried: posted atomics as the disturbing traffic (Blackhole submits atomics
non-posted), and atomics aimed at DRAM or the X280.

## Files

| File | What it is |
|---|---|
| `stress.py` | the host: launch on the asking cores, read back, one JSON line |
| `writer.cpp` | the asking kernel: K timed reservations |
| `stress.h` | the per-core L1 block |
