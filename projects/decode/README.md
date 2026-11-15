# Decode: a decoder layer at the DRAM ceiling

One decode step of a Llama-style decoder (one token's row of activations through every
layer) on one Blackhole p150a: the weights are read once a token, so the step's floor is
the time the card's DRAM takes to read them. The question is whether the programming
model lets a decode step run at that floor, and where every microsecond above it goes.

## What a layer is here

The shapes are Qwen3-1.7B's (`model.py`, from its `config.json`): hidden 2048, MLP 6144,
16 query heads and 8 KV heads of 128, 28 layers, RMS norm with eps 1e-6. The weights are
random (bf16, variance 1 / K; the two output projections also / sqrt(2 x layers) so the
residual keeps its scale; the norm weights in [0.5, 1.5)). A layer, in order, as the
card runs it since stage 2 (one product for q, k and v, one for gate and up; the norms
inside the products since stage 2 (4)):

| Step | What | Weight (K x N) |
|---|---|---|
| norm1 | h = rmsnorm(x) w1 | 2048 |
| qkv | [q k v] = h [Wq Wk Wv] | 2048 x 4096 |
| o | x = x + a Wo | 2048 x 2048 |
| norm2 | h = rmsnorm(x) w2 | 2048 |
| gate_up | g = silu(h Wg) (h Wu) | 2048 x 12288 |
| down | x = x + g Wd | 6144 x 2048 |

Attention over the cache is not here yet: `a` (attention's output) is q itself, a marked
gap. The activation is one tile row (32 rows, the first one the token's), as the card's
tiles hold it.

A layer is 100,671,488 bytes of weights, a token 2.819 GB over 28 layers: at the card's
DRAM read of 495.6 GB/s (120 cores on both movers, the dense project's ceiling) that is
203.1 us a layer and 5.688 ms a token, 175.8 tokens a second.

## How it runs

- Every product is the dense project's K split (`../dense/kernels/split`): a group of s
  cores per 1 x 8 block of output tiles, each core reading 1/s of K in chunks of 2 tiles
  on both movers (AB, gate_up AAAB), the parts summed by a tree over the cores' ports. The shapes are plans
  in one table in the program's arguments; a command's a5 picks one. The plans share
  every buffer; a group has at most 16 cores (a sum of 4 rounds, half of a core's 8
  ports): qkv 16 groups of 7 (112 cores), o and down 8 of 15 (120), gate_up 48 of 2
  (96); for q, k, v apart, q as o, k and v 4 groups of 16 (64 cores).
- What a layer does besides the products is an overlay (`kernels/`), called once at the
  hook after the sum (`split_op.hpp`): `add` (the residual: the sum plus a block of x,
  and the block's sums of squares for the next norm), `scale` (the norm after qkv's
  product: the sum times each row's 1 / rms) and `gate` (the same scale, then: a
  gate_up block is 4 tiles of the gate and 4 of up, silu of the first times the second,
  written by its own writer part). Stage 1 ran each norm as its own command.
- An X280 thread (`thread/layers.c`) walks the layers: one host command (`layers`) runs
  T tokens of L layers, each layer's six commands in order, and records each command's
  time from its event. Every command is waited for before the next, except with q, k
  and v apart and the overlap on (stage 2): q, k, v and o two in flight, each sum in its
  own set of ports and arrival slots.

## Run

```
python projects/decode/bench.py --layers 1 --tokens 10 --json out.json
python projects/decode/bench.py --tokens 10 --json out.json
python projects/decode/bench.py --tokens 10 --cases overlap --no-check
python projects/decode/bench.py --tokens 1 --cases parts --no-check --trace stamps.npz
```

`--chunk` and `--pattern` set every plan's K chunk and pattern (the default is
`model.PATTERNS`).

It writes each layer's weights through the host channel (6.2 GB/s), checks one layer
from a known x against `model.layer()` (numpy, float64, each output rounded to bf16 as
the card stores it), twice from the same x, fused and waited, then apart and overlapped
(the second run after every core's change of role), then runs each case as one `layers` command and prints one JSON line
with the thread's times and the account. The bound is loose (rms error / rms <= 2e-2,
largest error / largest value <= 5e-2 a tensor); bf16 alone gives 5e-3 a product and
1e-2 for the gate. The cases: `full`, the layer; `bare`, the products only (no
operation, so no norm); `parts`, the products' parts only (each core reads and multiplies
its share, nothing is summed or written: the split's mode 1); `overlap`, the layer
with q, k and v apart and q, k, v and o two in flight; `parts_apart`, the parts with q, k
and v apart.

The thread builds each command from its words and the layer's row of addresses
(`thread/layers.c`):

```c
command->a4 = job->parts ? KM_SPLIT_PARTS : KM_SPLIT_FULL;
command->a5 = words[plan];
command->a6 = operation == 0u || job->bare ? 0u : words[operation];
command->a7 = other;
```

and the root of each group calls the operation where it would pack the sum
(`../dense/kernels/split/compute.cpp`):

```cpp
if (ok && full && group.rank == 0 && split_op_part(command, kOpCompute)) {
    ok = split_op(compute, command);         // the operation takes the sum
```

## Stage 1: the standard layer

The check (`results/qwen3-1.7b-layer0-x10.json`), the larger error of the two runs: rms
5.5e-3 (q, k, v), 4.8e-3 (h), 6.0e-3 (up), 1.06e-2 (gated), 4.2e-3 (x); largest 9.8e-3
(x) and 1.4e-2 (gated).

All 28 layers, distinct random weights each (2.8 GB in DRAM), 10 tokens from one
command, card clock 1,350 MHz (`results/qwen3-1.7b-x10.json`, and the parts case
`results/qwen3-1.7b-x10-parts.json`): **10.75 ms a token, 93.0 tokens a second, 262 GB/s
of weights**, against 5.69 ms (176 tokens a second) at the ceiling. A layer, in
microseconds, from the thread's clock and each command's event (submit to completion);
"the read below the ceiling" is a product's parts-only time less its weight's time at
495.6 GB/s, "the boundary" its time with the sum and Y's write less its parts-only time:

| Term | Kind | us a layer |
|---|---|---:|
| the weights read at 495.6 GB/s (7 products) | HARDWARE | 203.1 |
| the read below the ceiling inside each product: q 7.5, k 5.8, v 5.8, o 7.5, up 12.1, gate 12.1, down 11.9 | OURS | 62.5 |
| the boundary of each product (the tree's sum, Y's write, the end of a command that sums): q 7.8, k 10.4, v 10.4, o 7.8, up 7.3, gate 7.3, down 8.6 | OURS | 59.6 |
| the operations at the hook (full less bare): o's add 1.1, gate's silu(g) u 11.6, down's add 0.8; q, k, v, up 0.3 (none: noise) | OURS | 13.8 |
| norm1, norm2 (16 cores, a command each) | OURS | 27.9 |
| the thread between commands (submit, wait: 9 a layer) | OURS | 16.9 |
| total | | 383.9 |
| unaccounted | | 0.0 |

383.9 us a layer is 1.89 times the ceiling. Each product carries 13 to 24 us above its
read whatever its size (k reads 4 MB in 24.7 us, up 25 MB in 70.4 us): a cost a command,
not a byte. The read below the ceiling grows with the shape (5.8 us for 4 MB, 12 us for
25 MB: the dense project's read, which this project takes as it is); the boundary does
not (7 to 10 us; the 30-core groups of k and v sum over 5 rounds, the others over 3 or
4). The gate's 11.6 us is silu and the product on 8 tiles at each root, with both
inits a tile. Code placement is inside the operations' terms: after the first token every
call finds its overlay held (three compute slots for add, gate and norm), a check of
about 72 cycles, 0.05 us, five calls a layer. The unaccounted line is zero by
construction: the thread's term is the layer less its commands; 0.3 us of noise sits in
the operations' line.

The first run of this layer was wrong in x, and only there (rms 4.2e-2): down's sum
lacked one core's part in every group. The split's sum buffer has two consumers on a
core: a group's root pops it on unpack, which writes its own count of pops into the
buffer's shared word, and every other core's writer pops it by adding to that word.
With four plans a core is a root in one command (up, gate: groups of 5) and a sender in
the next (down: groups of 15); the counts parted, the sender saw a stale item and sent
its part before it was summed. Unpack now counts a sender's pop as its own
(`compute.cpp`); with one plan the count it keeps is never used on a sender.

## Stage 2: removing what is ours

### The boundary: two products in flight

A product's boundary is paid because the next product waits for the whole command,
its sum and its write, before its first weight is read. q, k and v read h and o reads q,
which is written two commands earlier; so with the overlap the thread submits each of
them before it waits for the one ahead (`thread/layers.c`, `run_attention`):

```c
status = submit_step(job, step, &handle);
if (status == KMX_OK) {
    status = wait_step(job, step - 1u, ahead);
```

A core then takes the next product as soon as its own part is sent, while its group's
root still sums. The two products' parts must not meet: k and o carry
`kSplitSecondSet` in a3, and their sums use the upper four ports and the upper half of
the arrival slots (`../dense/kernels/split/sets.hpp`); the arrival buffer holds both
sets (64 pages). They go without the exclusive flag. The cost: a sum may take 4 rounds,
so k and v moved from 4 groups of 30 cores to 4 groups of 16 (64 cores).
up, gate and down read what the step before writes (A, or the gate's u at its root),
so they stay waited.

All 28 layers, 10 tokens, card clock 1,350 MHz (`results/qwen3-1.7b-x10-overlap.json`,
all four cases in one run; both check runs, waited and overlapped, as exact as stage 1):

| Term | Kind | waited | overlapped |
|---|---|---:|---:|
| the weights read at 495.6 GB/s | HARDWARE | 203.1 | 203.1 |
| the read below the ceiling (k, v on 64 cores: 6.8 each, was 5.8) | OURS | 64.5 | 64.5 |
| q, k, v, o: the boundary | OURS | 34.2 | 8.6 |
| up, gate, down: the boundary | OURS | 23.2 | 23.2 |
| the operations at the hook (o 1.1, gate 11.6, down 1.0, noise 0.3) | OURS | 13.9 | 13.9 |
| norm1, norm2 | OURS | 28.0 | 28.0 |
| the thread between commands | OURS | 18.7 | 10.3 |
| total | | 385.6 | 351.6 |
| unaccounted | | 0.0 | 0.0 |

The overlapped layer: 351.6 us, **9.84 ms a token, 101.6 tokens a second, 286 GB/s** of
weights. Of q, k, v and o's 34.2 us of boundary 25.6 are hidden; what is left (8.6 us)
is about o's own (7.8 us waited), which nothing follows inside the span. The thread's
term falls by 8.4 us: three of its submits a layer happen while a product runs. The "q,
k, v, o" rows come from the span the thread measures (q's submit to o's event, 88.8 us)
less those products' parts and o's add; the other rows as in stage 1.

### The fused products

q, k and v read the same h, gate and up too: as one product each they pay one
boundary, one command and one thread round trip instead of three and two
(`model.fused`: [Wq Wk Wv] side by side; gate_up takes 4 tiles of the gate's columns,
then the same 4 of up's, so a 1 x 8 block holds both halves of its 4 output tiles and
the gate needs nothing from another group). The gate's compute part packs up's half in
bf16 (as the reference stores u), then silu of the gate's half times it; its writer
part writes the 4 tiles where Y's write would have written 8 (`kernels/gate/`). gate_up
is 384 tiles wide: 48 blocks of 1 x 8 leave groups of 2, 96 cores; the plans share
their buffers, so a wider block is not open to it. qkv cannot overlap o, which reads
q: the fused layer is waited throughout. The program holds both forms, so one run
measures both (`results/qwen3-1.7b-x10-fused.json`; both check runs exact):

| Term | Kind | apart, overlapped (before) | fused |
|---|---|---:|---:|
| the weights read at 495.6 GB/s | HARDWARE | 203.1 | 203.1 |
| q, k, v (or qkv) and o: the read below the ceiling | OURS | 28.3 | 16.2 |
| q, k, v and o: the boundary left (overlapped); qkv 6.8 + o 7.9 (fused) | OURS | 8.6 | 14.7 |
| gate_up: the read below the ceiling (96 cores; up and gate apart: 24.3) | OURS | 26.3 | 26.3 |
| gate_up: the boundary (one round; up and gate apart: 14.6) | OURS | 4.6 | 4.6 |
| down: the read below the ceiling, the boundary | OURS | 20.4 | 20.4 |
| the operations: o's add, the gate (5.7; apart 11.6), down's add | OURS | 8.2 | 8.0 |
| norm1, norm2 | OURS | 28.0 | 28.0 |
| the thread between commands | OURS | 8.1 | 12.2 |
| total | | 335.5 | 333.4 |
| unaccounted | | 0.0 | 0.0 |

The fused layer: 333.4 us, **9.34 ms a token, 107.1 tokens a second, 302 GB/s** of
weights (stage 2 (1) gave 351.6 us with up and gate apart). gate_up saves 14.0 us
against up and gate (138.1 against 70.4 + 81.7): one boundary of 7.3 and a sum of 3
rounds become a sum of one (4.6), the gate's op halves (each of 48 roots takes 4 tiles,
not each of 24 roots 8), and the command and its round trip go; its read on 96 cores
costs 2.0 us more than on 120. qkv against q, k, v overlapped: the three reads' fixed
costs go (16.2 against 28.3 below the ceiling) but the overlap of o goes too; fused
is 2.1 us ahead. The account takes the fused case's own bare and parts runs, and for
the overlapped column the parts of q, k, v apart (`parts_apart`).

### The read below the ceiling

`bench.py --cases parts --trace stamps.npz` saves the runtime's stamps of every core and
command: when the reader took it, when the writer (mover B) ended, when the core reported.
Over 28 layers of parts, medians in us from the first core's start:

| Product (cores) | start spread | first core ends | last core ends | W and A at 495.6 GB/s |
|---|---:|---:|---:|---:|
| qkv (112) | 0.19 | 13.9 | 39.3 | 38.1 |
| o (120) | 0.18 | 5.8 | 19.5 | 19.0 |
| gate_up (96) | 0.19 | 67.6 | 125.7 | 114.3 |
| down (120) | 0.18 | 22.5 | 60.2 | 57.1 |

The cores start together, and they do not end together: each core of gate_up has 32 K
tiles, and they end from 68 to 125 us in the order of their place on the grid. So it is
neither compute (the same work on every core) nor the partition. The read also moves
more than the weights: every group reads all of A, and A's tiles are 32 rows (N/8
groups x K tiles x 2 KB: qkv 2.1 MB, o 1.05, gate_up 6.3, down 3.1, together 25.3 us at
the ceiling). On the bytes it moves, the read ends at the ceiling for qkv and o, 5% above
it for down and 10% above it for gate_up. Two runs tested the size of a block. With
chunks of 4 tiles (twice the bytes in flight on a mover), gate_up took 144.9 us (was
128.0). With chunks of 1, the cores' ends drew closer (the first at 94 us) and the last
did not move (127.8). The end is set by the whole grid's rate, not by one core's reads in
flight. On 96 cores, AB gives what gate_up gets: about 445 GB/s (the movement probe: AB
375 GB/s on 60 cores, 485 on 120; AAAB 500 and 503). Under AB, mover B carries half of
the bytes on NoC 1, which is the slower NoC.

The change is the chunks' pattern, a plan word (`kSpPattern`; `../dense/plan.py`
`pattern`, `depth`). Under AAAB mover A takes three chunks of four. Mover A's half holds
three chunks, so A does not wait for compute's pop. With one chunk a half, AAAB had gained
nothing. On every product, AAAB took gate_up from 128.0 to 121.2 us (parts). It cost qkv
1.2 us, o 1.8 and down 2.6, because on 112 and 120 cores AB is already near its rate. So
gate_up runs on AAAB and the rest on AB (`model.PATTERNS`). Every plan's mover A half is 3
chunks deep, because the plans share buffers; AB is unchanged by the depth. The dense
project's plans stay AB with a depth of 1: dec4096's single call took 91.78 us after the
change and 91.82 before, and its parts 81.84 (exact). The layer, all cases, both check
runs exact (`results/qwen3-1.7b-x10-pattern.json`):

| Term | Kind | before (fused) | gate_up on AAAB |
|---|---|---:|---:|
| the weights read at 495.6 GB/s | HARDWARE | 203.1 | 203.1 |
| qkv: the read below the ceiling | OURS | 8.8 | 9.2 |
| o: the read below the ceiling | OURS | 7.4 | 6.8 |
| gate_up: the read below the ceiling | OURS | 26.3 | 19.7 |
| down: the read below the ceiling | OURS | 11.9 | 11.8 |
| the boundaries: qkv 7.0, o 8.4, gate_up 6.2, down 8.5 (before 6.8, 7.9, 4.6, 8.5) | OURS | 27.8 | 30.1 |
| the operations: o's add, the gate (4.7; before 5.7), down's add, qkv noise | OURS | 8.0 | 6.7 |
| norm1, norm2 | OURS | 28.0 | 28.0 |
| the thread between commands | OURS | 12.2 | 12.2 |
| total | | 333.4 | 327.5 |
| unaccounted | | 0.0 | 0.0 |

The layer now takes 327.5 us: **9.17 ms a token, 109.0 tokens a second, 307 GB/s** of
weights. gate_up's full command went from 138.1 to 132.2 us. Its boundary and its
operation together took 10.9 us (before 10.3): the split between those two terms is
within a run's noise. Of the 47.4 us still below the ceiling, 25.3 is A's re-read, a cost
of the 1 x 8 blocks, which every plan shares. The rest is each product's fixed start and
end (about 5 us; o's parts end 3.7 us before its completion) and gate_up's and down's
remaining rate.

### The norms inside the products

A norm command ran on 16 cores. Its stamps (`bench.py --cases full --trace`, medians over
28 layers):

- the cores start within 0.14 us of each other;
- the first core's writer ends at 10.5 us and the last at 12.6;
- the command completes at 13.1, and the thread's step takes 14.1;
- the command before it completes 3.0 us before it starts (that gap is in the thread's
  term).

So a norm is about 12.6 us in its kernels: each of the 16 cores reads x's whole row (64
tiles, 128 KB, 2 MB in all, at one core's read rate) for the sums of squares, then
computes and writes its 4 tiles of h. Its own start and end add about 1.5 us, and the
thread about 3 us more.

Folded, the norm needs nothing to move the row:

- The norm's weight goes into the rows of the W after it, on the host: qkv (and q, k, v)
  take diag(w1) W, and gate_up takes diag(w2) W, in bf16 (`model.fused`; against the
  reference's rounding of h, an error of 3e-3 rms on q).
- Each root of o and down holds an 8-tile block of the new residual. Its `add` also
  computes that block's sums of squares, the diagonal of the sum of y y^T over the
  block's tiles, and writes them as one tile into a row of 8 (`kernels/add/`).
- The next product's root reads the 8 tiles, adds them, takes the mean, adds epsilon,
  takes the rsqrt, and multiplies its sum by the result (`ops.hpp` `rms_scale`;
  `kernels/scale/` after qkv, and inside `gate` after gate_up).
- o now writes x1, no longer x in place: in the overlap, v still reads x.
- A host write of x comes with its squares.

The check compares x1 where it compared h, and h no longer exists. Both runs are exact:
rms 6.4e-3 (q, k, v), 3.3e-3 (x1), 1.03e-2 (gated), 4.2e-3 (x)
(`results/qwen3-1.7b-x10-norms.json`).

| Term | Kind | before | norms folded |
|---|---|---:|---:|
| the weights read at 495.6 GB/s | HARDWARE | 203.1 | 203.1 |
| the read below the ceiling: qkv, o, gate_up, down | OURS | 47.4 | 47.3 |
| the boundaries: qkv, o, gate_up, down | OURS | 30.1 | 30.2 |
| qkv's operation: none, then the scale | OURS | -0.2 | 3.4 |
| o's add (now with its squares) | OURS | 1.2 | 1.5 |
| gate_up's operation: the gate, then the scale and the gate | OURS | 4.7 | 9.3 |
| down's add (now with its squares) | OURS | 1.1 | 1.4 |
| norm1, norm2 (a command each) | OURS | 28.0 | 0 |
| the thread between commands (6 commands a layer, then 4) | OURS | 12.2 | 8.4 |
| total | | 327.5 | 304.6 |
| unaccounted | | 0.0 | 0.0 |

The layer now takes 304.6 us: **8.53 ms a token, 117.2 tokens a second, 330 GB/s** of
weights (overlapped: 313.6). The 28.0 us of norm commands and 3.8 us of the thread
become 3.4 + 4.5 us at two roots, and the squares cost 0.7. The scale runs after the
sum, in series at each root: its 8 tiles are read only once the root's own read is
done, then it computes 1 / rms on one tile. That is most of the scale's time, and it is
left in the operations' term.

## What the framework lacked

- A buffer read by two kinds of kernel on one core (unpack on some commands, the writer
  on others) has no single count of pops; the program has to take the shared one back.
- A program in the runtime's form cannot hold overlays, so the operations hang on the
  product's kernels at one hook instead; and a core that changes program keeps its
  launch arguments, so the four shapes are a table of plans in those arguments.
- Two commands that use ports cannot be in flight on the same cores: a port and an
  arrival slot do not tell two commands apart. The split splits them in two sets
  itself (`sets.hpp`), which halves what a sum may use: 4 rounds of the tree.
- The read takes a chunk of A with its chunk of W, so a product whose A is the step
  before's output cannot start its weight's read early; only q, k, v and o overlap.
