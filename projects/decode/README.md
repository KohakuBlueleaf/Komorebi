# Decode: a decoder layer at the DRAM ceiling

One decode step of a Llama-style decoder (one token's row of activations through every
layer) on one Blackhole p150a: the weights are read once a token, so the step's floor is
the time the card's DRAM takes to read them. The question is whether the programming
model lets a decode step run at that floor, and where every microsecond above it goes.

## What a layer is here

The shapes are Qwen3-1.7B's (`model.py`, from its `config.json`): hidden 2048, MLP 6144,
16 query heads and 8 KV heads of 128, 28 layers, RMS norm with eps 1e-6. The weights are
random (bf16, variance 1 / K; the two output projections also / sqrt(2 x layers) so the
residual keeps its scale; the norm weights in [0.5, 1.5)). A layer, in order:

| Step | What | Weight (K x N) |
|---|---|---|
| norm1 | h = rmsnorm(x) w1 | 2048 |
| q, k, v | q, k, v = h Wq, h Wk, h Wv | 2048 x 2048, 2048 x 1024 (twice) |
| o | x = x + a Wo | 2048 x 2048 |
| norm2 | h = rmsnorm(x) w2 | 2048 |
| up | u = h Wu | 2048 x 6144 |
| gate | g = silu(h Wg) u | 2048 x 6144 |
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
  on both movers, the parts summed by a tree over the cores' ports. The four shapes of a
  layer are four plans in one table in the program's arguments; a command's a5 picks one.
  The plans share every buffer; a group has at most 16 cores (a sum of 4 rounds, half
  of a core's 8 ports): q, o and down 8 groups of 15, up and gate 24 of 5 (120 cores),
  k and v 4 groups of 16 (64 cores).
- What a layer does besides the products is an overlay (`kernels/`), called once at the
  hook after the sum (`split_op.hpp`): `add` (the residual: the sum plus a block of x),
  `gate` (silu of the sum times a block of u) and `norm` (on 16 cores in the split's
  operation-only mode: each core reads the whole row for the sums of squares and writes
  its 4 tiles of h).
- An X280 thread (`thread/layers.c`) walks the layers: one host command (`layers`) runs
  T tokens of L layers, each layer's nine commands in order, and records each command's
  time from its event. Every command is waited for before the next, except q, k, v and o
  with the overlap on (stage 2): two of them in flight, each sum in its own set of
  ports and arrival slots.

## Run

```
python projects/decode/bench.py --layers 1 --tokens 10 --json out.json
python projects/decode/bench.py --tokens 10 --json out.json
python projects/decode/bench.py --tokens 10 --cases overlap --no-check
```

It writes each layer's weights through the host channel (6.2 GB/s), checks one layer
from a known x against `model.layer()` (numpy, float64, each output rounded to bf16 as
the card stores it), twice from the same x, waited and then overlapped (the second run
after every core's change of role), then runs each case as one `layers` command and prints one JSON line
with the thread's times and the account. The bound is loose (rms error / rms <= 2e-2,
largest error / largest value <= 5e-2 a tensor); bf16 alone gives 5e-3 a product and
1e-2 for the gate. The cases: `full`, the layer; `bare`, the products only (no
operation, no norm); `parts`, the products' parts only (each core reads and multiplies
its share, nothing is summed or written: the split's mode 1); `overlap`, the layer
with q, k, v and o two in flight.

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
