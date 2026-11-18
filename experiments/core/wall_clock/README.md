# The Tensix wall clock across cores

Whether the wall-clock count each Tensix core reads is one count for the whole chip,
and at what rate it runs against host time. Core (0, 0) sends a timestamped ping to each
peer core with a small NoC write; the peer timestamps the arrival and replies. Two rounds
a spin apart give each peer's offset, its round trip and its drift; spins of 0.5 s and
2 s against the host's clock give the rate.

## Result

One chip-wide count. On one Blackhole p150a (slow dispatch, core (0, 0) against six
cores from the corners to the centre, rounds 2 s apart): every core within 90
cycles (67 ns) of the first, offsets -79 to +88 cycles with round trips of 245 to 424
cycles (the offset includes the NoC path's asymmetry), drift under 0.01 ppm. Against
host time the clock ran at 1343 to 1347 MHz with the driver reporting 1350 MHz, within
the host timing's error (about 0.5 %, an estimate).

The far cores start up to about a million cycles after core (0, 0), so the probe waits
4 million cycles before the first ping and drops a round a peer saw at its first look.

## The rate by chip (rate.py)

The ratio of wall-clock ticks to the clock the driver reports (`tt_aiclk`), which
`komorebi.device.chips` keeps per chip (`tick_ratio`) and uses wherever ticks become
time (`chips.tick_mhz`: the feeder's `mhz`, its limits, a session's run). Core (0, 0)
spins 0.5 s and 2.5 s of ticks; the 2 s between them over the host's time between them,
three times, the median beside the table's:

    python experiments/core/wall_clock/rate.py --runs 3

## Running it

On a card, with tt-metal's environment set:

    python experiments/core/wall_clock/probe.py --cores 2 --smoke   # two cores, 0.1 s
    python experiments/core/wall_clock/probe.py                     # seven cores, 3 x 2.5 s

It prints one JSON line: per peer the offset, the round trip and the rate, and the
AICLK the driver reported.

## The 64-bit read across a carry (tear.py)

`km::wall64` reads the low word and then a high word. On Blackhole the high word is a
live one, read before and after the low word (`wall64_of`). Wormhole names no live high
word: the low word's read latches `WALL_CLOCK_H`, read next. If the two movers of a
core shared that latch, one mover's low-word read between the other's two reads at a
carry would tear the other's value by 2^32. `tear.py` checks it: both movers of each
core read `km::wall64()` in a tight loop for `--window-s` at the driver's AICLK,
starting half that before a carry of the low word, and count reads that went back or
jumped by 2^31 or more.

On one Wormhole chip of a Galaxy (slow dispatch, both movers of all 80 cores, three
runs of 500 million cycles each from 250 million before a carry, about 0.51 s at the
chip's 985.5 million ticks a second): 480 records, every one across a carry, 7.10
billion reads, none back, none jumping; the largest step between two reads 42 cycles.
The latched read is kept.

    python experiments/core/wall_clock/tear.py --runs 3 --all   # every core, about 15 s
