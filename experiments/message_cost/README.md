# message_cost

What a value crossing to one core costs on this chip: one put by distance (the hop
table), and many senders incrementing one word of one core at the same moment (the
bump table). These are the two wire steps of `km::emit` and `km::arrive`.

```
python message_cost.py hops     # 1024 repetitions per target, size and NoC
python message_cost.py bumps    # 1, 8, 32, 60, 118 senders, 1024 repetitions each
```

Every wait in the kernels is bounded (10 ms; 2 s for start-up); a repetition that
passes its bound is counted as failed. Each script samples AICLK and card power while it
runs and prints one JSON line with every row.

## Hop table

One sender, logical core (6, 5), puts 64 bytes or 2 KiB into the same L1 address of
each other core of its row and its column, 1024 times per target, one put at a time:
cycles from the put to its write flush (the acknowledgement is back). The NoC0 kernel
runs first, then the NoC1 kernel. Hops are counted on the NoC torus (17 x 12) in NoC0
coordinates: NoC0 goes +x then +y, NoC1 -x then -y, so a target on one side is 1 to 5
hops away on one NoC and 12 to 16 on the other.

Blackhole p150a (card 1), AICLK 1350 MHz throughout (min = mean), card power 61 to 64 W,
host load 6.3. Cycles min / median / p99 over 1024 repetitions; every target of a group
gives the same three numbers to within one cycle:

| NoC | Bytes | Targets in the sender's row (1 to 16 hops) | Targets in its column (1 to 16 hops) |
|---|---|---|---|
| 0 | 64 | 228 / 229 / 229 (0.170 µs) | 186 / 186 / 186 (0.138 µs) |
| 0 | 2048 | 250 / 251 / 273 (0.186 µs) | 207 / 207 / 207 (0.153 µs) |
| 1 | 64 | 228 / 228 / 228 (0.169 µs) | 186 / 186 / 186 (0.138 µs) |
| 1 | 2048 | 270 / 270 / 270 (0.200 µs) | 207 / 207 / 228 (0.153 µs) |

- Distance costs nothing measurable. A straight-line fit of the median against the hop
  count gives 0.37 cycles per hop on NoC0 and 2.7 to 4.0 on NoC1, but within each group
  the median does not change with the distance at all; the fit only sees the
  difference between the two groups, which do not differ in hops.
- What does differ: a target in the same row costs 42 to 63 cycles more than a target
  in the same column, on both NoCs. The cause is not known.
- "The long way round the torus" is therefore not a reason to choose a NoC for one put;
  a put of 2 KiB costs 20 to 40 cycles more than one of 64 bytes.

## Bump table

Core 0 clears a word, writes a release time (its clock + 40 µs) to every sender, and
measures from the release until the word counts every sender; senders 1 to N (core
order, as the members of an owner reduce) wait for the release and increment the word
by an atomic. Senders first check in, so all of them run before the first repetition.
A sender that saw its release late marks its increment; none did.

Same card and conditions (AICLK 1350 MHz throughout, 63 to 64 W, load 6.2), 1024
repetitions each, cycles min / median / p99:

| Senders | Cycles | µs (median) |
|---|---|---|
| 1 | 75 / 95 / 96 | 0.070 |
| 8 | 199 / 208 / 225 | 0.154 |
| 32 | 446 / 463 / 474 | 0.343 |
| 60 | 783 / 803 / 811 | 0.595 |
| 118 | 1479 / 1496 / 1505 | 1.108 |

The receiver's word takes about 12 cycles (9 ns) per increment beyond the first: the
increments are served one after another. An owner reduce over 119 members spends about
1.1 µs only in counting its arrivals.

The release relies on the cores' wall clocks agreeing, as the runtime's other aligned
tests do.
