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
before its returned value had landed in these runs. Two rules came out of it:

- The return address is programmed on every call, to a 16-byte line of the sender's
  own. The firmware's default return slot is one shared L1 word.
- A start barrier between cores must allow more than the launch skew of slow dispatch:
  a bound of 10 million cycles let about 47 of 119 cores give up before the last one
  started; the probe allows 1.35 billion.

In the runtime, at 10^6 hops on 119 cores, a reservation's answer has been seen to
stay away although its count moved: about one run in four before the packet commit,
one in eighteen after it, and one gap in one rate run with the commit below. The
runtime waits for the answer to replace a sentinel and reports reason `lost`.

| Adders | Reservations each | Covered exactly | Cycles per reservation, min / mean / max |
|---|---|---|---|
| 2 | 4 | yes | 256 / 258.8 / 266 |
| 8 | 64 | yes | 256 / 256.2 / 273 |
| 119 | 256 | yes (30464 reservations) | 266 / 1221.6 / 2596 |

AICLK was not sampled around the reservation runs; their cycles are wall-clock cycles.

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
(54016 of 163517 reads with 120 cores), so the receiver's prefix rule matters. The
probe marks word 0 of every 16-byte chunk, salted per process.

**Packet commit.** The runtime instead marks word 0 of an item only and commits each
packet: the items go out with the first item's mark 0, and after their write barrier a
16-byte write gives the first item its mark. `--commit` measures that cost (rerun of the
same cases on 2026-11-13, AICLK 1350 MHz read before and after every run):

| Case | Without commit | With commit |
|---|---|---|
| round trip, 1 item | 1260 cycles | 1318 cycles |
| round trip, 32 items | 4590 cycles | 5068 cycles |
| one pair, packets of 1 / 32 | 3.1 / 14.7 M items/s | 1.7 / 14.7 M items/s |
| 120 cores random, packets of 1 / 8 / 32 | 178 / 838 / 1214 M items/s | 93 / 677 / 1151 M items/s |

The commit costs one write round trip per packet: half the rate at packets of one,
5 % at packets of 32. The random run with packets of one and the commit had one gap
(position 5064 on one receiver) out of 983040 items.

Measured on one Blackhole p150a, slow dispatch, other jobs on the host but not on this
card. AICLK read 1350 MHz before and after each round-trip and rate run (the first
runs read it after only). Cycles are wall-clock cycles at that clock.

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
python hop.py --mode random --n 32 --packets 256 --commit
```

Each prints one JSON line. Every wait in the kernels is bounded; a core that gives up
says where.
