# The same programs on one island and on many

Question: does a program written over flat ranks run unchanged on any number of islands,
giving the words it gives on one island, with only its time and its L1 differing?

`accept.py` runs four programs that were written before this page, each from its own
source with no line about islands, on one session shape at a time, and saves every
output's words or compares them with a saved one-island run:

| Program | Source | What runs |
|---|---|---|
| Attention, the class | [attention_class](../../language/attention_class/README.md) | one ATTEND over every job's core, the plan made on the session's grid |
| Attention, the project's program | [projects/attention](../../../projects/attention/README.md) | the hand kernels; on islands its own plan over the islands as one grid (`card.island_plan`), as its README's `--cards 0,1` |
| SwiGLU, the class | [31_swiglu](../../../examples/31_swiglu/README.md) | `up` on 12 ranks and `down` on 4 as commands, placed around island 1's first rank so the groups cross the edge; then its chain as written (cores 0 to 11) with every tensor held by the last island |
| The decode layer class | [decode_class](../../language/decode_class/README.md) (`across.py`) | its layer as one chain, then the same five calls as a command each (`--steps-only`), at `across.py`'s first ranks |

The shapes: card 0 alone (`one`), two cards (`two`), card 0 split as a line of three
islands (`line3`), as a ring of four (`ring4`) and as the 2 x 2 grid with wrap
(`grid4`, two wires a pair).

```
python experiments/islands/acceptance/accept.py --shape one --save one.npz
python experiments/islands/acceptance/accept.py --shape two --kernel-bytes 163840 --against one.npz
python experiments/islands/acceptance/accept.py --shape two --links 1 --programs attention_islands --against one.npz
python experiments/islands/acceptance/accept.py --shape line3 --kernel-bytes 262144 --against one.npz
python experiments/islands/acceptance/accept.py --shape ring4 --kernel-bytes 262144 --against one.npz
python experiments/islands/acceptance/accept.py --shape grid4 --kernel-bytes 262144 --against one.npz
```

**The measure.** Equal math is equal words: every output's 16-bit words (bf16) or 32-bit
words (SwiGLU's float32 y) against the one-island run's, counted; where any differ, the
relative RMS and largest error of the values as well. The one-island outputs were
checked first: the attention class's O equals the project program's word for word; the
SwiGLU's hidden is within 6.6e-3 relative RMS of float64 (y 9.4e-3); the decode layer's
seven tensors equal the decode class's own one-card words, and its chain's equal its
steps'. Each program's tensors are freed after it (an island session's tensor table
holds 32 at once).

## Results

Main at 8e828d52, 2026-11-17, Blackhole p150a cards, AICLK 1350 MHz, the dispatcher
image built from that commit, every load built before the run (0 compiled on the card).
Times are the median from submit to event, microseconds: attention 20 commands; SwiGLU
and the decode layer 5 and 10 after a first. "Equal" is 0 words differing in every
output.

| Program | One island (card 0) | Two cards | Line of 3 (card 0) | Ring of 4 (card 0) | 2 x 2 with wrap (card 0) |
|---|---|---|---|---|---|
| Attention class, t1 | 14.36 | refused (1) | refused (2) | refused (2) | refused (2) |
| Attention project program, t1 | 14.54 | equal, 110.29 (238 cores, one link) | refused (3) | refused (3) | refused (3) |
| SwiGLU, steps across the edge | 104.4 | equal, 3,233.86 | **2 of 96 hidden tiles wrong** (4), 3,420.36 | equal, 3,765.34 | **does not end** (5) |
| SwiGLU, its chain | 106.7 | equal, 6,424.76 | equal, 17,218.56 | equal, 7,491.24 | **does not end** (5) |
| Decode layer, chain | 346.02 | equal, 7,580.47 | refused (6) | refused (6) | refused (6) |
| Decode layer, steps | 369.52 | equal, 6,958.63 | refused (6) | refused (6) | refused (6) |

Every two-card row is word-equal to one island, and so is the SwiGLU on the ring of
four and its chain on the line of three. The times are stated, not tuned: on two cards the SwiGLU reads most of its
weight pages across the link (its tensors placed by the default hint, split along rows
over every rank, or held by the last island for the chain), and the decode layer's
crossings are its own (its README).

### What did not run, and why

1. **The attention class on two cards** is refused at submit: `the broadcast's rectangle
   (0, 0, 12, 9) is the group (0, 108), fixed at load; submit it to that group (not (0,
   120)), or make the program with rect=; a broadcast stays on one island`. Documented
   ([islands, classes on islands](../../../docs/reference/islands.md#classes-on-islands)):
   a class's broadcast stays inside one island, and an island of a card has 9 whole
   rows of ranks (its last two cores are link cores), where the class's column broadcast
   is 10 long. The project program, which plans its broadcasts inside each island, runs.
2. **The attention class on a split card** is refused at the load with `Attention uses
   self.x, self.y or a broadcast: load it with Session.load, or make it with
   nocs=session.nocs()`, although it was loaded with `Session.load`. The message names
   the wrong remedy: a split card does not hand a class's rectangle its NoCs. The
   limit behind it is that the islands of a split card differ in rows and rank counts
   (47, 34 and 35 ranks on the line of three). Open.
3. **The attention project program on a split card**: `ValueError: island 0: 47 ranks`
   (34 on the 2 x 2). Its plan over the islands is written for whole cards with one
   link core each (its README runs two cards); this is the program's own host code,
   not the framework's.
4. **SwiGLU's `up` on a line of three** (ranks 41 to 52; island 1 starts at 47) wrote
   two of its 96 hidden tiles wrong, those of ranks 47 and 51, every row, values off by
   up to 0.028 (hidden against float64: 1.06e-2 relative RMS, against 6.6e-3 on one
   island). No error was raised and every kernel ended clean. Open: a defect.
5. **SwiGLU on the 2 x 2** does not end: a command whose pages are held two links away
   (`down`, whose W2 rows lie on all four islands, and the chain with its tensors on
   island 3) waits in `ask_count` until its deadline (`CommandTimeout`); the cores'
   `bad_item` give-ups that follow are recorded after the abort. The same chain with its
   tensors one link away (islands 1 and 2) is done. On a line of three given two wires
   a pair (`graph=[(0, 1), (0, 1), (1, 2), (1, 2)]`) an ask two links away is not
   answered either, while with one wire a pair it is (the table's line of three, and
   the ring of four, where island 2 is two links from island 0).
   Open: a defect, a page ask two links away when pairs have two wires.
6. **The decode layer on a split card** does not load: `OverlayError: overlay qkv, reader
   part: not position-independent` (its handlers are overlays, built with a split card's
   defines). Open. It would not run in any case: its steps take 120 ranks, and a split
   card has 116 (line of three) or 112 (2 x 2).

Two cards open with `kernel_bytes=163840` here (the islands' default) because the script
is run with different programs at different shapes, and a script's remembered buffer is
the last run's. A split card opens with `kernel_bytes=262144`: its session does not yet keep the
kernel buffer a refused load needed for the next run, as two cards do, so SwiGLU's
kernels (with the wires' link code, 207,808 B) are refused at the default 163,840 B.

## Example 26 over many rounds

[26_islands_sum](../../../examples/26_islands_sum/README.md) once gave one wrong sum on
two cards, before a far line's confirmation named the queue that holds it. On the same
commit and cards, `sum.py --cards 0,1 --every --rounds 1000` (each round's total checked,
the output zeroed before it): 0 wrong rounds of 1,000, 181.57 µs a round.
