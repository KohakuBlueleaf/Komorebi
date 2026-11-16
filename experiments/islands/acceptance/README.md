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

```
python experiments/islands/acceptance/accept.py --shape one --save one.npz
python experiments/islands/acceptance/accept.py --shape two --kernel-bytes 163840 --against one.npz
python experiments/islands/acceptance/accept.py --shape two --links 1 --programs attention_islands --against one.npz
python experiments/islands/acceptance/accept.py --shape two --kernel-bytes 163840 --programs swiglu,swiglu_islands --against one.npz
python experiments/islands/acceptance/accept.py --shape line3 --against one.npz
python experiments/islands/acceptance/accept.py --shape ring4 --against one.npz
python experiments/islands/acceptance/accept.py --shape grid4 --against one.npz
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
Cells marked * are from main at 8e219975 (split cards keep their kernel buffer and hand
a class its rectangle), run the same way.
Times are the median from submit to event, microseconds: attention 20 commands; SwiGLU
and the decode layer 5 and 10 after a first. "Equal" is 0 words differing in every
output.

| Program | One island (card 0) | Two cards | Line of 3 (card 0) | Ring of 4 (card 0) | 2 x 2 with wrap (card 0) |
|---|---|---|---|---|---|
| Attention class, t1 | 14.36 | refused (1) | refused (2)* | refused (2)* | refused (2)* |
| Attention project program, t1 | 14.54 | equal, 110.29 (238 cores, one link) | refused (3) | refused (3) | refused (3) |
| SwiGLU, steps across the edge | 104.4 | equal, 3,233.86; equal, 3,208.84* | **2 of 96 hidden tiles wrong** (4), 3,420.36; equal in 400 turns* | equal, 3,765.34 | **does not end** (5) |
| SwiGLU, steps placed island-aware | | equal, 792.40* | | | |
| SwiGLU, its chain | 106.7 | equal, 6,424.76 | equal, 17,218.56 | equal, 7,491.24 | **does not end** (5) |
| Decode layer, chain | 346.02 | equal, 7,580.47 | refused (6)* | refused (6)* | refused (6)* |
| Decode layer, steps | 369.52 | equal, 6,958.63 | refused (6)* | refused (6)* | refused (6)* |

Every two-card row is word-equal to one island, and so is the SwiGLU on the ring of
four and its chain on the line of three. The times are stated, not tuned. The decode
layer's two-card rows hold for its placement (`across.py`'s first ranks 6, 103, 70: no
float reduce group across the edge). With its groups started at rank 0, groups of 15
cross rank 118 (attend's merge, `o`, `down`) and are added in another order than one
island's tree: qkv equal, the attention output 366 of 65,536 words differing (largest
0.0050 relative), the layer's output 996 (largest 0.0065), chain and steps alike
(2026-11-17).

**Where the two-card time goes.** One `up` across the edge (ranks 112 to 123, its
tensors placed by the default hint, rows split over every rank) made 480 far page asks
for 3,282 pages (6.8 MB) and 12 far puts, read from the links' counts around that one
command; placed island-aware, `up` crossed nothing and `down` asked 192 times for the
384 hidden pages, and the two steps take 792.40 µs against 3,208.84 µs. So most of the
two-card time is where the program's tensors are placed, not the islands themselves;
the decode layer's crossings are its own (its README).

### What did not run, and why

1. **The attention class on two cards** is refused at submit: `the broadcast's rectangle
   (0, 0, 12, 9) is the group (0, 108), fixed at load; submit it to that group (not (0,
   120)), or make the program with rect=; a broadcast stays on one island`. Documented
   ([islands, classes on islands](../../../docs/reference/islands.md#classes-on-islands)):
   a class's broadcast stays inside one island, and an island of a card has 9 whole
   rows of ranks (its last two cores are link cores), where the class's column broadcast
   is 10 long. The project program, which plans its broadcasts inside each island, runs.
2. **The attention class on a split card** is refused by name: `ValueError: group (0,
   120) of 116 ranks` (112 on four islands). Its plan, made on the session's grid, has
   120 jobs, and a split card has fewer ranks (its link cores). Before the split loads
   were fixed it was refused with a remedy that did not apply (`load it with
   Session.load`).
3. **The attention project program on a split card**: `ValueError: island 0: 47 ranks`
   (34 on the 2 x 2). Its plan over the islands is written for whole cards with one
   link core each (its README runs two cards); this is the program's own host code,
   not the framework's.
4. **SwiGLU's `up` on a line of three** (ranks 41 to 52; island 1 starts at 47) once
   wrote two of its 96 hidden tiles wrong, those of ranks 47 and 51, every row, values
   off by up to 0.028 (hidden against float64: 1.06e-2 relative RMS, against 6.6e-3 on
   one island), with no error and every kernel ended clean. Later, run right after a
   two-card job as then, its hidden read after every `up`: equal in 400 of 400 turns.
   Open: a defect not reproduced since.
5. **SwiGLU on the 2 x 2** does not end: a command whose pages are held two links away
   (`down`, whose W2 rows lie on all four islands, and the chain with its tensors on
   island 3) waits in `ask_count` until its deadline (`CommandTimeout`); the cores'
   `bad_item` give-ups that follow are recorded after the abort. The same chain with its
   tensors one link away (islands 1 and 2) is done. On a line of three given two wires
   a pair (`graph=[(0, 1), (0, 1), (1, 2), (1, 2)]`) an ask two links away is not
   answered either, while with one wire a pair it is (the table's line of three, and
   the ring of four, where island 2 is two links from island 0).
   Open: a defect. A fix that had the island between forward the ask was tried and
   taken back: with it the steps ended with equal words (3,547.54 µs) but the chain
   ended with wrong ones (98,234 of 98,304 hidden words and every y word differing,
   relative RMS 1.08 and 1.22); this script is the check a new fix must pass.
6. **The decode layer on a split card** passes the class's own L1 check (`the buffers
   take 848640 B of 1142976 B left for buffers`) and then fails at the launch with the
   runtime's message, not a named refusal: `Statically allocated circular buffers ...
   clash with L1 buffers on core range [0-0 - 11-9]. L1 buffer allocated at 1500736 and
   static circular buffer region ends at 1560832`. The session holds 120,832 B of L1
   tensors there, which the check does not count. Open: a defect. (Before the split
   loads were fixed it did not load: its overlay part was not position-independent.)

Two cards open with `kernel_bytes=163840` here (the islands' default) because the script
is run with different programs at different shapes, and a script's remembered buffer is
the last run's. The first split-card runs opened with `kernel_bytes=262144`, because a
split card did not then keep the buffer a refused load needed; the runs marked * open
with the buffer it keeps.

## Example 26 over many rounds

[26_islands_sum](../../../examples/26_islands_sum/README.md) once gave one wrong sum on
two cards, before a far line's confirmation named the queue that holds it. On the same
commit and cards, `sum.py --cards 0,1 --every --rounds 1000` (each round's total checked,
the output zeroed before it): 0 wrong rounds of 1,000, 181.57 µs a round.
