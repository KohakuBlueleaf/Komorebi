# Dynamic crossing probes

Three questions asked of the card before the dynamic crossing was built on them: can a
core reserve positions in another core's queue with one atomic add, how long does an
item take from one core to another through DRAM, and how many items per second move
between cores.

## Result

**Reservation: yes.** A non-posted NoC atomic increment (`noc_fast_atomic_increment`
with the return value asked for) returns the value before the add into a return address
the caller programs for that call. 2, 8 and 119 cores adding to one word at once got
disjoint positions that covered the range exactly, and no response was acknowledged
before its returned value had landed.

| Adders | Reservations each | Covered exactly | Cycles per reservation, min / mean / max |
|---|---|---|---|
| 2 | 4 | yes | 256 / 258.8 / 266 |
| 8 | 64 | yes | 256 / 256.2 / 273 |
| 119 | 256 | yes (30464 reservations) | 266 / 1221.6 / 2596 |

**Round trip and rate.** The sender (RISCV_1, NoC1) reserves positions, then writes the
64-byte items into the receiver's queue in DRAM (interleaved, 32 items a page). The
receiver (RISCV_0, NoC0) polls its tail word, reads up to 32 items and takes the prefix
whose marks have landed. Data movement only: no compute kernel touches the items.

| Case | Items | Result |
|---|---|---|
| one item, one pair | 1 | 1256 cycles (0.93 µs) from reserve to accepted |
| 32 items, one pair | 32 | 4585 cycles (3.4 µs) |
| one pair, packets of 1 / 8 / 32 | 32768 | 3.2 / 14.6 / 14.7 M items/s |
| 120 cores to random cores, packets of 1 / 8 / 32 | 983040 | 180 / 849 / 1211 M items/s |

Every run was exact: counts and payload sums per receiver, no item accepted torn, no
gap. Reads that found fewer landed items than reserved were common at packets of one
(54016 of 163517 reads with 120 cores), so the receiver's prefix rule matters. A
mark in word 0 only is not enough: the chunks of one write land in no defined order, so
the mark is in word 0 of every 16-byte chunk, salted per process so that a stale item
of an earlier run never matches.

Measured on one Blackhole p150a, slow dispatch, AICLK 1350 MHz, other jobs on the host
but not on this card. Cycles are AICLK cycles.

## What runs

| File | What it is |
|---|---|
| `reservation.cpp`, `reservation.py` | adder cores issue K atomic adds each to one word on core 0 and record the returned values; the host checks the cover |
| `hop.h`, `hop_send.cpp`, `hop_recv.cpp`, `hop.py` | senders reserve and write items to the receivers' DRAM queues; receivers read and check them |

```
python reservation.py --adders 2 --k 4
python reservation.py --adders 119 --k 256
python hop.py --mode pair --n 1 --packets 1
python hop.py --mode pair --n 32 --packets 1024
python hop.py --mode random --n 32 --packets 256
```

Each prints one JSON line. Every wait in the kernels is bounded; a core that gives up
says where.
