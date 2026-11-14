# 17 leader overlay

A leader picks which code its group runs next. The first core of each broadcast reads a
value from DRAM, picks one of four **overlays** from it and casts the overlay's number as
a word; every core of the broadcast, the leader too, loads that overlay when it does not
hold it and calls it. The operations are not in the kernels: each one is an overlay,
kept in DRAM and read into a slot of a core's L1 when a cast names it
([Overlays](../../docs/reference/overlays.md), [Broadcast](../../docs/reference/broadcast.md)).
The same program with the four operations compiled into the kernels runs beside it.

## What each core does

Each core holds 32 words of data. Each command runs `rounds` rounds along every
broadcast of the rectangle (a column, or a row inside one panel). For each round:

1. The leader (rank 0) reads value `index`, picks `value >> 30` and casts the word
   `choice | (index + 1) << 2` on a broadcast without a buffer.
2. Every other core arrives and reads the word with `broadcast.word()`; every core checks
   that it names `index`.
3. Every core takes overlay `choice + 1`: `load()` reads what the tile does not hold, then
   `call()` runs it on the core's data.

| Overlay | Parts | What it does to the data |
|---|---|---|
| `mix` | reader | each word times 33, plus the round's index and the word's position |
| `turn` | reader | the words turned by 1 to 16 places, each rotated left by 5, xor the index |
| `scale` | reader, compute | the reader fills a tile from 3 bits of each word, compute multiplies it by 1 to 4 on the vector unit, the reader folds the tile back into the words |
| `shift` | reader, compute | the same with an addition of 1 to 4 |

Each kernel has one slot fewer than it has overlays: 3 slots of 1,216 B on the reader
(4 overlays) and 1 on each compute core (2 overlays), so loads really happen. After the
last round each core writes its data, its count of wrong words and its loads to its row
of the results.

```c++
// reader.cpp: the round
if (broadcast.sender()) {
    const uint32_t choice = pick(page[index % kValuesPage]);
    broadcast.cast(cast, choice_word(choice, index));  // the overlay, to every core
} else {
    broadcast.arrive(cast);
}
const uint32_t word = broadcast.word();       // on the leader too
if (has_compute(word)) hand(reader, word);    // compute takes the word from the reader
km::Overlay overlay(reader, overlay_of(word));
overlay.load();                               // reads what the tile lacks
overlay.call(DataArgument{data, index_of(word)});

// compute.cpp: the words its reader hands it, until 0
km::Overlay overlay(compute, overlay_of(word));
overlay.load();                               // parks until the reader has its parts in place
overlay.call(TileArgument{scalar});
```

### Two things the pieces did not give by themselves

- **Compute does not see the word.** A cast arrives on the reader; compute's `load()`
  must name the same overlay in the same command. The reader hands the word to its own
  compute through a one-page buffer (`kCbChoice`) on the rounds whose overlay has compute
  parts, and a 0 ends the command. Unpack reads it and gives it to math and pack
  (`read_tile_value`).
- **The leader runs up to one round ahead.** A member marks a word as taken in `arrive`,
  before its `load()` and `call()`, so the leader's next cast does not wait for the
  members' calls. A member that is still calling takes the next word when it is done.
  Neither header needed a change.

## Run

```
python leader_overlay.py --rect 0,0,1,10 --choices cycle            # a column of 10
python leader_overlay.py --rect 0,0,12,1 --along row --choices blocks
python leader_overlay.py --rect 0,0,12,10 --choices cycle --compiled
```

`--choices cycle` picks 0, 1, 2, 3 in turn: with 3 reader slots every round loads.
`--choices blocks` picks among mix, turn and one of scale and shift, which changes every
`--period` (1,000) rounds: loads are rare. `--rounds` (100,000) runs in commands of
`--per` (2,000) rounds. `--timed` commands (10) record every 15th round on every core:
the leader's cast, `load()`, `call()` and the wall clock at the cast and after the call.
The wall clocks of the cores agree, so the time from the leader's cast to the last call
done in its broadcast is a difference of two clocks. `--compiled` builds the kernels with
the four operations compiled in, chosen by the word as in
[15_leader](../15_leader/README.md), and no overlays.

The script replays every command in numpy: each core's data after each command, its count
of wrong words (must be 0) and its rounds. It prints the medians and one JSON line and
exits non-zero when a core is wrong.

## Measured

One p150a, AICLK 1,350 MHz before and after every run, host load about 5 to 10, commands
through the dispatcher, 2026-11-14, on the commit that moves the slots under 1 MB of L1.
100,000 rounds per run; every core's data after every command was exact in all 12 runs. Medians of the timed rounds (12,800 to 153,600 a run),
in µs:

| | Row 0 (2 broadcasts of 6) | Column of 10 | Grid (12 columns of 10) |
|---|---|---|---|
| leader's `cast` | 0.32 | 0.38 | 0.38 |
| `load()`, loading the reader part | 0.88 | 0.88 | 0.88 |
| `load()`, loading the reader and compute parts (meeting) | 1.82 | 1.88 | 1.85 (2.11 when every core loads at once) |
| `load()`, held | 0.07 | 0.07 | 0.07 |
| `call()` of mix: overlay / compiled in | 0.41 / 0.36 | 0.40 / 0.36 | 0.41 / 0.36 |
| `call()` of scale: overlay / compiled in | 16.25 / 16.16 | 16.25 / 16.16 | 16.25 / 16.17 |
| cast to last call done, mix loading | 2.12 | 2.23 | 2.23 |
| cast to last call done, mix held | 1.19 | 1.31 | 1.31 |
| cast to last call done, mix compiled in | 1.00 | 1.13 | 1.13 |
| cast to last call done, scale loading / held / compiled in | 19.11 / 17.29 / 16.91 | 19.28 / 17.41 / 17.02 | 19.27 / 17.41 / 17.03 |

A round on average (command time / rounds), overlays against compiled in:

| Choices | Loads per core | Row 0 | Column of 10 | Grid |
|---|---|---|---|---|
| cycle | 100,000 reader, 50,000 with compute | 10.33 / 8.70 | 10.43 / 8.76 | 10.43 / 8.76 |
| blocks | 143 reader, 100 with compute | 6.32 / 6.08 | 6.38 / 6.13 | 6.38 / 6.13 |

Commands: `python leader_overlay.py --rect R --along A --rounds 100000 --per 2000 --timed
10 --choices C [--compiled]`, R and A `0,0,12,1 row`, `0,0,1,10 column`, `0,0,12,10
column`, C `cycle` and `blocks`.

L1 of a core, overlays against compiled in:

| | Overlays | Compiled in |
|---|---|---|
| kernel text: reader, writer, unpack, math, pack | 19,376, 2,988, 3,460, 2,104, 2,496 B | 16,552, 2,988, 2,992, 1,436, 1,956 B |
| kernel buffer (text and configuration) | 31,168 B | 26,672 B |
| slots: reader 3 x 1,216, unpack 1,472, math 896, pack 1,088 | 7,104 B | 0 |
| overlay area beside the control block | 3,328 B | 0 |
| total | 41,600 B | 26,672 B |

The parts are 128 B (mix), 192 B (turn), and 1,216, 1,472, 896 and 1,088 B (scale and
shift: reader, unpack, math, pack); 117,760 B of DRAM hold a copy of each in every bank.

What the numbers say:

- From the leader's cast to the last member's call done, an overlay that every core
  loads costs about 0.9 µs more than one every core holds, and a held one 0.19 µs more
  than the same code compiled in. The load is the same on a row, a column and the grid:
  each core reads its own copy, so 120 cores loading at once add little, except when
  every core reads five parts at once (2.11 µs against 1.74).
- A held `load()` costs 0.07 µs, so when loads are rare the call itself is what loading
  on demand costs a round.
- A loop runs from L1 at or above 1 MB about 4 cycles an iteration slower than from
  below it, with the same instructions. The slots were there, and an overlay's calls
  cost 0.13 µs more than the same code compiled in on mix's 32-word loop and 3.3 µs
  (20 %) on scale's tile, almost all of it that. The slots now sit just under 1 MB, and
  a call costs 0.05 µs (mix) and 0.1 µs (scale) more: the call's record of the
  calls running (about 20 cycles), the entry's zeroing of the part's `.bss` and, on
  the compute cores, the part's address table (about 50 cycles a call). A part that has
  just been loaded runs 10 to 20 cycles slower, with its instruction cache empty.
- With four small operations, loading on demand costs L1 instead of saving it: the load
  code, the slots and the overlay area take 14.9 KB more than the operations compiled
  in: the kernels with the load code and without the operations are 4.5 KB larger than
  the kernels with the operations. Loading on demand saves L1 only when the operations
  a group may run are larger than the slots kept for them, the 3 KB area and the load
  code together.

## Files

| File | What it is |
|---|---|
| `kernels/leader_overlay.h` | the handler, the overlays' numbers, buffers, the leader's choice and word, the parts' arguments |
| `kernels/ops.h`, `kernels/tile_ops.h` | the four operations, used by the parts and by the compiled-in build |
| `kernels/reader.cpp` | the broadcast: the leader reads, picks and casts; every core loads and calls |
| `kernels/compute.cpp` | takes each word from its reader, loads and calls the compute part |
| `kernels/writer.cpp` | no part: returns at once |
| `kernels/mix`, `turn`, `scale`, `shift` | the overlays' parts |
| `leader_overlay.py` | builds the broadcasts and commands, runs them, replays and checks every core, prints the medians |
