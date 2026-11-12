# The Tensix wall clock across cores

Whether the wall-clock count each Tensix core reads is one count for the whole chip,
and at what rate it runs against host time. Core (0, 0) sends a stamped ping to each
peer core with a small NoC write; the peer stamps the arrival and replies. Two rounds
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

## Running it

On a card, with tt-metal's environment set:

    python experiments/wall_clock/probe.py --cores 2 --smoke   # two cores, 0.1 s
    python experiments/wall_clock/probe.py                     # seven cores, 3 x 2.5 s

It prints one JSON line: per peer the offset, the round trip and the rate, and the
AICLK the driver reported.
