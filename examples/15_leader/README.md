# 15 leader

A leader picks what its group does next. The first core of each broadcast reads a value
from DRAM, picks one of four steps from it, and casts its choice as a word; every core of
the broadcast takes that step. The four steps are compiled into the kernels; the word
only selects one. No core but the leader reads the data, and nothing but the word
travels: the broadcast has no buffer.

## What each core does

Each command casts `casts` words along every broadcast of the rectangle (a column, or a
row inside one panel). For each one:

1. The leader (rank 0) reads value `index` (from a page of 256 it keeps), picks
   `value >> 30`, one of four, and casts the word `choice | (index + 1) << 2`.
2. Every other core arrives and reads the word with `broadcast.word()`.
3. Every core, the leader too, checks that the word names `index`, then takes step
   `choice` on its state: `state * 33 + index`, `state ^ state << 7`, `state +
   0x9E3779B9`, or a rotation right by 3.

After the last cast each core writes its state, the count of words that named the wrong
index, the first of them and its cast count to its row of the results.

With `CARRY` each cast also carries a tile (2 KiB, `index % tiles` of the tile data)
through a second broadcast, which has a buffer; compute adds the tiles it receives and
the writer writes the sum. The words then ride in the same VALID write as the tiles.

## How it maps onto the runtime

| Step | Kernel | Call (`komorebi/device/km_broadcast.hpp`) |
|---|---|---|
| the leader casts its choice | reader, NoC0 | `broadcast.cast(cast, word)` (with `CARRY`: `reserve`, the tile's read, then the same cast) |
| the others take it | reader | `broadcast.arrive(cast)`, `broadcast.word()` |
| the step | reader | `step(word, state, index)` from `kernels/leader.h` |
| the tiles (`CARRY`) | compute, writer | `km::broadcast_receive`, `km::broadcast_done`, `km::broadcast_end`; the sum written |

| Handler | Arguments | What a core does |
|---|---|---|
| `LEAD` | a0 values, a1 casts, a2 results, a3 first value index, a4 results page of rank 0 | words alone, broadcast 0 (no buffer) |
| `CARRY` | the same, a5 tiles, a6 tile count, a7 sums | a tile and a word in each cast, broadcast 1 (window `--window`) |

The host builds broadcast 0's words with `layout.broadcast_args(..., (0, 0, 0, 0))`: window
and pages 0 make a broadcast without a buffer.

## Run

```
python leader.py --rect 0,0,1,10 --casts 25000 --jobs 40
python leader.py --rect 0,0,12,1 --along row --casts 25000 --jobs 40 --carry alternate
python leader.py --rect 0,0,12,10 --casts 25000 --jobs 40
```

`--rect first_x,first_y,columns,rows` places the rectangle (its group is the smallest
range of the core order that covers it), `--along column` or `row` the broadcasts.
`--carry always` sends a tile with every word, `alternate` in every second command. The
script replays every command in numpy: each core's end state, its count of wrong words
(must be 0) and, for `CARRY`, its tile sum bit for bit. It prints one JSON line and
exits non-zero when a core is wrong.

## Measured

One p150a, AICLK 1,350 MHz, host load about 6, commands through the dispatcher. 10^6
casts per row of the table (40 commands of 25,000), every word of every core checked:

| Rectangle | Broadcasts | Cores | Words checked | Wrong | µs a cast |
|---|---|---|---|---|---|
| a column of 10 (9 receivers) | 1 | 10 | 10,000,000 | 0 | 0.595 |
| row 0, two panels | 2 of 6 | 12 | 12,000,000 | 0 | 0.535 |
| the grid, twelve columns at once | 12 of 10 | 120 | 120,000,000 | 0 | 0.603 |
| row 0, a tile with each word, window 2 | 2 of 6 | 12 | 12,000,000 | 0 | 1.31 |
| row 0, words alone and tiles in alternate commands | 2 of 6 | 12 | 12,000,000 | 0 | 0.92 |

A word alone costs one round trip to the farthest receiver: the leader casts the next
word only when every receiver marked the last. Twelve broadcasts at once cost no more
than one.

## The arrival mark (a probe)

The question: written as one 16-byte multicast (marker, word, whether there is a word),
does the mark land whole on every receiver, so that a receiver that sees a new marker
also sees its word? A receiver that read the marker before the word landed would take
the word of an earlier cast, whose index differs. In the runs above no core took a
wrong word, along a row, down the longest column a broadcast reaches and with every
column of the grid casting at once: the 16-byte mark lands whole. The cheaper form
(a 16-bit word in the marker's low half instead of the block count) is not needed.

## Files

| File | What it is |
|---|---|
| `kernels/leader.h` | handlers, buffers, arguments, the leader's choice and the four steps |
| `kernels/reader.cpp` | the broadcast: the leader reads, picks and casts; every core takes the step |
| `kernels/compute.cpp`, `writer.cpp` | `CARRY`: the sum of the tiles, written to DRAM |
| `leader.py` | builds the broadcasts and commands, runs them, replays and checks every core |
