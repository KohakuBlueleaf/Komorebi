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
| SwiGLU, placed island-aware (`swiglu_islands`) | the same class | `up` on island 0's first 12 ranks with x, W1, W3 and hidden held by that group, `down` on island 1's first 4 with W2 and y held by its group (`session.tensor(..., group=)`): the only crossing is `down` reading hidden |
| The decode layer class | [decode_class](../../language/decode_class/README.md) (`across.py`) | its layer as one chain, then the same five calls as a command each (`--steps-only`), at `across.py`'s first ranks |

The shapes: card 0 alone (`one`), two cards (`two`), card 0 split as a line of three
islands (`line3`), as a ring of four (`ring4`) and as the 2 x 2 grid with wrap
(`grid4`, two wires a pair).

The commands are under the results.

**The measure.** Equal math is equal words: every output's 16-bit words (bf16) or 32-bit
words (SwiGLU's float32 y) against the one-island run's, counted; where any differ, the
relative RMS and largest error of the values as well. The one-island outputs were
checked first: the attention class's O equals the project program's word for word; the
SwiGLU's hidden is within 6.6e-3 relative RMS of float64 (y 9.4e-3); the decode layer's
seven tensors equal the decode class's own one-card words, and its chain's equal its
steps'. Each program's tensors are freed after it (an island session's tensor table
holds 32 at once).

## Results

Main at eb2860f3, 2026-11-17, Blackhole p150a cards, AICLK 1350 MHz, the dispatcher
image built from that commit, every load built before the run (0 compiled on the card).
Times are the median from submit to event, microseconds: attention 20 commands; SwiGLU
and the decode layer 5 and 10 after a first. "Equal" is 0 words differing in every
output against the one-island run of the same commit, which is itself word-equal to the
one-island run of 8e828d52 (every program, every output).

| Program | One island (card 0) | Two cards | Line of 3 (card 0) | 2 x 2 with wrap (card 0) |
|---|---|---|---|---|
| Attention class, t1 | 14.39 | refused (1) | refused (2) | refused (2) |
| Attention project program, t1 | 14.56 | equal, 108.59 (238 cores, one link) | refused (3) | refused (3) |
| SwiGLU, steps across the edge | 104.42 | equal, 3,206.66 | equal, 3,517.70 | equal, 3,517.52 |
| SwiGLU, steps placed island-aware | | equal, 790.44 | equal, 774.14 | equal, 718.32 |
| SwiGLU, its chain | 107.46 | equal, 6,389.22 | equal, 17,526.76 | equal, 15,902.58 |
| Decode layer, chain, its placement | 344.27 | equal, 7,536.94 | refused (2) | refused (2) |
| Decode layer, steps, its placement | 369.48 | equal, 6,929.56 | refused (2) | refused (2) |
| Decode layer, first ranks 0, 0, 0 | | gave up in 2 of 4 runs on 2026-11-17; under hunt (4) | | |

The commands, each in its own session (the kernel buffer given, since a script's
remembered buffer is its last run's; the attention class and SwiGLU take up to
209,296 B of kernel text on a split card, which leaves the decode layer too little L1):

```
python experiments/islands/acceptance/accept.py --shape one --kernel-bytes 163840 --save one.npz
python experiments/islands/acceptance/accept.py --shape two --kernel-bytes 163840 --programs swiglu,swiglu_islands,decode --against one.npz
python experiments/islands/acceptance/accept.py --shape two --links 1 --kernel-bytes 212992 --programs attention,attention_islands --against one.npz
python experiments/islands/acceptance/accept.py --shape line3 --kernel-bytes 212992 --programs attention,attention_islands,swiglu,swiglu_islands --against one.npz
python experiments/islands/acceptance/accept.py --shape line3 --kernel-bytes 159808 --programs decode --against one.npz
python experiments/language/decode_class/across.py --cards 0 --steps --save decode_one.npz
python experiments/language/decode_class/across.py --cards 0,1 --firsts 0,0,0 --steps --against decode_one.npz
```

and the same two `line3` lines with `--shape grid4`.

Every row that runs on two cards or on a split card is word-equal to one island. The
decode layer's two-card rows hold for its placement (`across.py`'s first ranks 6, 103,
70: no float reduce group across the edge). Earlier the same day, with its groups
started at rank 0, groups of 15 crossed rank 118 (attend's merge, `o`, `down`) and were
added in another order than one island's tree: qkv equal, the attention output 366 of
65,536 words differing (largest 0.0050 relative), the layer's output 996 (largest
0.0065). The ring of four was not run on eb2860f3; on 8e828d52 its SwiGLU steps and
chain were equal (3,765.34 and 7,491.24 µs).

**Where the time goes.** One `up` across the edge (its tensors placed by the default
hint, rows split over every rank) made 480 far page asks for 3,282 pages (6.8 MB) and 12
far puts on two cards (3,312 pages on a split card); placed island-aware, `up` crossed
nothing and `down` asked 192 times for the 384 hidden pages, and the two steps take
790.44 µs against 3,206.66 µs on two cards. So most of the time across islands is where
the program's tensors are placed, not the islands themselves.

### What does not run, and why

1. **The attention class on two cards** is refused at submit: `the broadcast's rectangle
   (0, 0, 12, 9) is the group (0, 108), fixed at load; submit it to that group (not (0,
   120)), or make the program with rect=`. Its default rectangle is an island's rank
   rows (9 whole rows: each island's last cores are link cores), where its plan has 120
   jobs. A class rectangle may span islands (a `rect=` taller than island 0 is one
   rectangle over them); this class is made without one.
2. **The attention class and the decode layer on a split card** cannot run: each submits
   to 120 ranks, and a split card has 116 (line of 3) or 112 (2 x 2) ranks, its link
   cores taken: `ValueError: group (0, 120) of 116 ranks`. On the 2 x 2 the decode layer
   is refused one step earlier, by name, for L1 (`L1BudgetError`: the 72,704 B the
   session keeps at the top of every core's L1 on four islands). Both are the programs'
   sizes, not a defect.
3. **The attention project program on a split card**: `ValueError: island 0: 47 ranks`
   (34 on the 2 x 2). Its plan over the islands is written for whole cards with one link
   core each; this is the program's own host code, not the framework's.
4. **The decode layer on two cards at first ranks 0, 0, 0** gave up on 2026-11-17 in 2 of
   4 runs (the other two ended): in the run of eb2860f3 its first chain was done and the
   second given up, rank 105's writer waiting in `ask_place`, ranks 84 to 94 in
   `ask_confirm`, island 1's ranks 118 and 119 in `port`. Open: a defect, under hunt.

No longer so (2026-11-17): SwiGLU on the 2 x 2 with wrap ends, word-equal, steps and
chain (it waited in `ask_count` while a page was held two links away, before forwarding
on a 2 x 2 was fixed); its two wrong hidden tiles on a line of three (seen once on
8e828d52) did not come back; the decode layer on a split card no longer reaches the
launch's circular buffer clash (it is refused before, by its group, as above).

## Example 26 over many rounds

[26_islands_sum](../../../examples/26_islands_sum/README.md) once gave one wrong sum on
two cards, before a far line's confirmation named the queue that holds it. On the same
commit and cards, `sum.py --cards 0,1 --every --rounds 1000` (each round's total checked,
the output zeroed before it): 0 wrong rounds of 1,000, 181.57 µs a round.
