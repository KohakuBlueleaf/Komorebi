# Decode: a decoder layer at the DRAM ceiling

One decode step of a Llama-style decoder (one token's row of activations through every
layer) on one Blackhole p150a: the weights are read once a token, so the step's floor is
the time the card's DRAM takes to read them. The question is whether the programming
model lets a decode step run at that floor, and where every microsecond above it goes.

## What a layer is here

The shapes are Qwen3-1.7B's (`model.py`, from its `config.json`): hidden 2048, MLP 6144,
16 query heads and 8 KV heads of 128, 28 layers, RMS norm with eps 1e-6, rotary base
1e6. The bench's weights are random (bf16, variance 1 / K; the two output projections
also / sqrt(2 x layers) so the residual keeps its scale; the norm weights in [0.5,
1.5)); `real.py` runs the model's own (stage 4). A layer, in order, as the
card runs it since stage 2 (one product for q, k and v, one for gate and up; the norms
inside the products since stage 2 (4)):

| Step | What | Weight (K x N) |
|---|---|---|
| norm1 | h = rmsnorm(x) w1 | 2048 |
| qkv | [q k v] = h [Wq Wk Wv]; q and k: each head's norm, the rotation (stage 4) | 2048 x 4096 |
| attend | a = softmax(q K^T / sqrt(128)) V over the cache (stage 3) | the cache |
| o | x = x + a Wo | 2048 x 2048 |
| norm2 | h = rmsnorm(x) w2 | 2048 |
| gate_up | g = silu(h Wg) (h Wu) | 2048 x 12288 |
| down | x = x + g Wd | 6144 x 2048 |

Until stage 3, `a` (attention's output) was q itself. Since stage 3 the token's query
attends over the layer's cache of keys and values on the card; since stage 4 with
Qwen3's norm of each head of q and k and the rotary position applied first. The
activation is one tile row (32 rows): the first M rows are M tokens (1 to 32, a word of
every command, stage 4); the rest are carried but not computed by the vector unit.

A layer is 100,671,488 bytes of weights, a token 2.819 GB over 28 layers: at the card's
DRAM read of 495.6 GB/s (120 cores on both movers, the dense project's ceiling) that is
203.1 us a layer and 5.688 ms a token, 175.8 tokens a second.

## How it runs

- Every product is the dense project's K split (`../dense/kernels/split`): a group of s
  cores per 1 x 8 block of output tiles, each core reading 1/s of K in chunks of 2 tiles
  on both movers (AB, gate_up AAAB), the parts summed by a tree over the cores' ports.
  The shapes are plans in one table in the program's arguments; a command's a5 picks one. The plans share
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
python projects/decode/bench.py --tokens 1 --cases parts --no-check --no-chain --trace stamps.npz
python projects/decode/bench.py --tokens 10 --no-check --loop-check --json out.json
python projects/decode/bench.py --layers 1 --tokens 10 --staged --cases --json out.json
python projects/decode/bench.py --tokens 10 --no-check --commands --json out.json
```

Since stage 6 a layer is one command (the chain); `--no-chain` builds the program of
stage 5 (a command a step), `--commands` also times every case a command a step on the
chain's program, and `--staged` checks the layer a command a step, then qkv and attend
as a chain of two, before the chain's own checks.

`--context` sets the cache's rows (4096), `--lengths` the cache lengths T timed and
checked (4096, 1024, 256: the last token at row T - 1); `--loop-check` checks one token
through every layer against the chain of `model.layer()`.

`--chunk` and `--pattern` set every plan's K chunk and pattern (the default is
`model.PATTERNS`). `--bodies` names the operations whose vector bodies are emitted
(stage 5; none by default; `python projects/decode/bodies.py` first).

The model's own weights and a prompt: `torch_reference.py` (in an environment with
torch and transformers, on the host CPU) tokenizes the prompt and writes the prompt's
tokens, torch's greedy continuation and its hidden states at the first positions;
`real.py` runs them on the card from the checkpoint's safetensors and compares:

```
python projects/decode/torch_reference.py --hub <hub cache> --out ref.npz
python projects/decode/real.py --snapshot <the model's snapshot> --reference ref.npz
```

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

### What is left: the thread and the operations

These are the layer's terms after (4) (`results/qwen3-1.7b-x10-norms.json`; the same at
the head after the merge of main, 304.7 us). None of them is changed here.

- The thread, 8.4 us: 2.1 us for each of the 4 commands. A norm's stamps showed 3.0 us
  from one command's completion to the next command's first core: the event reaching
  the X280, the thread's wait returning, its submit, and the ring to the cores. The 1.0
  us between the last core's report and the thread's event is in each product's term.
  Removing these gaps needs the next command in the cores' queue before the one ahead
  completes. That works only where A is not the one ahead's output, which is the
  overlap of stage 2 (1); in the fused layer every product reads what the one ahead
  writes.
- The operations, 15.6 us: qkv's scale 3.4 and gate_up's 9.3 (the gate 4.7 and the
  scale 4.5), o's add 1.5 and down's 1.4 (each with 0.35 of squares). They run at the
  root after its sum, in series. The scale reads its 8 tiles of squares only after the
  root's own read, then computes 1 / rms on one tile, though none of that depends on
  the sum. The hook after the sum is the only place an operation is called (a
  condition of the design), so this is left as it is.

## Stage 3: attention over the cache

### The design

- **The cache** is two tensors a layer in DRAM, keys and values, each [8 KV heads x
  4096 rows, 128]; a tile holds 32 rows of a head. One card's DRAM pages are
  interleaved over its banks, so no placement is nearer to a core than another.
- **A KV head's group** is o's plan: 8 groups of 15 cores, all 120. qkv's own groups
  (16 of 7) would leave 8 cores without a share and give each of the others twice the
  compute; the cache read reaches the ceiling only on 120 cores on both movers.
- **ATTEND** is one more command a layer, after qkv: the K split's command in its
  operation mode, every core calling the `attend` overlay (`kernels/attend/`). The
  group's full key tiles 0..t/32 - 1 are split over its 15 cores, the larger shares on
  the higher ranks; the root (rank 0) also takes tile t/32, where the token's row goes.
  Each core reads its query tiles (both query heads of the KV head, 8 tiles) and its
  keys on its reader and its values on its writer, in batches of 4 key tiles.
- **The append**: the root's reader reads the cache's tile t/32 and the token's key from
  qkv into its L1, the core puts the key in row t mod 32, writes the tile back whole
  and keeps it as its last key tile; its writer does the same with the value.
- **The state**: for each query head, the scores (q against each key tile on the matrix
  unit, the root's last tile starting from a mask of the rows past t), each row's
  maximum m (a reduce, then broadcast to every column by a product), P = exp((s - m) /
  sqrt(128)) in bf16, the row sums l = P times ones and O = P V, kept in fp32: 12 tiles.
- **The merge**: the states go up the group's ports as the split's sum does (4 rounds
  of a 48 KB message); a core merges each state it receives into its own: m = max(m1,
  m2), then l and O as diag(a1) X1 + diag(a2) X2, a_i = exp((m_i - m) / sqrt(128)), two
  products a tile on the matrix unit. The root divides O by l and writes its head's
  block of `a`; o reads `a` as it read q before.
- **Code placement**: the attention body and the merge are an overlay like the other
  operations; the product's loop stays resident and its text is unchanged (62,160 of
  70,656 bytes). The overlay's parts: reader 7,744, writer 7,616, unpack 12,544, math
  12,288, pack 16,256 bytes; the slots grow to 221,760 bytes of L1, with 962,560 of
  circular buffers, of 1,461,120. After the first token every call finds the overlay
  held (one more check of about 72 cycles); no program change (`kmx_load`) is needed.
- **Not done**: the cache read starting when qkv is issued.
  A core runs one command at a time, and ATTEND's cores are qkv's.

### The check

One layer from a known x against `model.layer()` with `model.attention()` (float64,
the same expressions) over a random cache, the last token at row T - 1, twice
(`results/qwen3-1.7b-attend-layer0-x10.json`):

| T | a rms | a largest | x rms | x largest |
|---:|---:|---:|---:|---:|
| 4096 | 1.29e-2 | 1.98e-2 | 2.8e-3 | 6.1e-3 |
| 1024 | 1.31e-2 | 2.21e-2 | 2.7e-3 | 9.1e-3 |
| 256 | 1.28e-2 | 1.53e-2 | 2.9e-3 | 6.1e-3 |

q, k and v carry 6.4e-3 each (their product in bf16), and the scores carry q's and k's
error into the exponent: a's error is about twice theirs. Attend alone from the host on
a random qkv (so q, k, v exact) gives a rms 5.1e-3 to 7.8e-3 at positions 5, 1000, 2500,
4095, one group and all 8; a model of the kernels' roundings (bf16 scores and P, fp32
states) gives 2.4e-3 to 4.6e-3, the rest is the products' fidelity (HiFi2) on the scores
and P V. The cache's new row equals the token's k and v bit for bit at every check.

One token through all 28 layers from one command, against the float64 chain of the 28
layers each over its own cache (T 4096): x rms 2.5e-2, largest 4.0e-2.

### The account

All 28 layers, 10 tokens from one command, card clock 1,350 MHz
(`results/qwen3-1.7b-attend-x10.json`). The product terms are stage 2's (unchanged, 0.1
us of noise); attend's cases: `attend_read` (the reads and the append only),
`attend_local` (no merge: the root ends on its own state), `attend_exp_twice` (P's
exponent taken twice: its time).

| Term | Kind | T 4096 | T 1024 | T 256 |
|---|---|---:|---:|---:|
| the weights at 495.6 GB/s | HARDWARE | 203.1 | 203.1 | 203.1 |
| the products' read below the ceiling, boundary, operations (stage 2's) | OURS | 93.2 | 93.2 | 93.1 |
| the cache's K and V at 495.6 GB/s (4,096 bytes a row) | HARDWARE | 33.8 | 8.4 | 2.1 |
| the busiest core's tile products (164, 56, 20) at 39.2 M a second | HARDWARE | 4.2 | 1.4 | 0.5 |
| the busiest core's exponent tiles (18, 6, 2), measured | HARDWARE | 21.9 | 7.3 | 2.4 |
| the cache read below the ceiling (q on every core, the append, the command) | OURS | 13.3 | 13.8 | 14.4 |
| attend's compute not under the read | OURS | 24.5 | 18.7 | 17.3 |
| the merge's 4 rounds | OURS | 39.7 | 37.4 | 38.9 |
| the thread between commands (5 a layer) | OURS | 10.6 | 10.5 | 10.6 |
| total a layer | | 444.1 | 393.8 | 382.4 |
| unaccounted | | 0.0 | 0.0 | 0.0 |
| a token (28 layers) | | 12.44 ms | 11.03 ms | 10.71 ms |
| tokens a second | | 80.4 | 90.7 | 93.4 |

Stage 2's layer without attention was 304.7 us, 8.53 ms a token. Attention costs 139 us
a layer at T 4096 and 78 at T 256, of which the cache's bytes at the ceiling are 34 and
2: the rest is a cost a layer that does not shrink with T. The merge is the largest:
about 10 us a round, its 4 exponents (1.2 us a tile, the rate P's exponents show) half
of it. The first merge computed the rescale on the SFPU, 4 fp32 copies to DEST and 3
products and sums a tile: 69.9 us; on the matrix unit the rounds take 39.7. The
exponent's rate is the next term: 1.2 us a tile at fp32 (not the approximate form),
over all 32 rows of the tile though only the first is the token's.

## Stage 4: the model itself

### Tokens a command

The rows that hold tokens, M (1 to 32), are a word of every command: a3's bits 16 to
21, and the `layers` command's a6 (M a step, positions a5 + step x M). Tokens t to t +
M - 1 are rows 0 to M - 1 of the activation; ATTEND's root takes the tokens' one or two
cache tiles (t/32 and the next when they cross it), each with its mask (63 masks: row i
keeps the columns up to i + k - 31), and the append puts the M rows in.

The vector unit runs only the bands of 4 rows that hold tokens (2 iterations a band in
faces 0 and 1; the whole tile past 16 rows): P's subtract, scale and exponent, the
merge's maximum, subtracts, exponents and diagonal products, the root's division, the
norms' epsilon and rsqrt (scale, gate, the heads' norms) and the gate's silu. The
matrix unit's products stay whole tiles. P's rows past M are 0 (DEST is filled with 0
first, the subtract writes the live bands only), so nothing that is not finite reaches
the merge's diagonal products, which mix rows.

One layer at T 4096, one token, before and after (`results/qwen3-1.7b-rows-layer0-x10.json`):

| Term | before | after |
|---|---:|---:|
| P's exponent tiles (18) | 21.9 | 3.9 |
| the merge's 4 rounds | 39.7 | 21.5 |
| attend's compute not under the read | 24.5 | 19.6 |
| gate_up's operation | 9.2 | 4.0 |
| qkv's operation | 3.4 | 2.6 |
| the layer | 444.1 | 398.7 |

At T 1024 and 256 the layer is 360.8 and 351.5 us (393.8 and 382.4 before). The check
with 32 tokens a command at T 4096 gives a rms 1.31e-2, x 2.8e-3, the append exact.

### The norms of q and k and the rotation

The model's attention as its library computes it (`transformers`, Qwen3's modeling
file: `Qwen3RMSNorm` on each head of q and k, then `apply_rotary_pos_emb` with
`rotate_half`; the table's frequencies 1 / 1e6^(2i/128) and angles in float32, cos and
sin cast to bf16). On the card it runs at the hook the other operations use: qkv's
`scale` overlay, on the roots of the blocks of q's and k's columns (a3's bit 22, set by
the thread on qkv), after the norm's scale and before attend appends k. A root holds 2
heads; for each: the squares of its 4 tiles summed on the live rows, a product with a
tile of 1/128 (each row's mean in every column), epsilon and rsqrt; each tile times it
and times the head's weight (a tile row of it); then tile j becomes tile j times cos_j
plus tile (j + 2) mod 4 times sin_j, with the sin's first half negated as `rotate_half`
takes it.

The table comes from the host, as one tile tensor: the identity and mean tiles, a tile
of 1/128, each layer's q and k norm weights (4 tiles each), then a 512-byte row a
position (cos's 128 values, then the signed sin's), 4 rows a page: 2.1 MB for 4,128
positions. The host computes it once with the reference's own float32 expression; on
the X280 the thread would compute cos and sin each step in another float library. It
differs from torch's table in 10 of 528,384 values, by one bf16 step. The core's
reader reads the live rows' band of 4 (one 512-byte read a row, the rows of a page in
one), puts each row's 16 pieces of 32 bytes into row i of 8 tiles (4 of cos, 4 of sin)
and hands them to compute. The command's words are full with it: a8 is the qk tensor
for qkv, a9 the position and the layer, so the norm's epsilon became the overlay's
compile-time value.

The plan, off the card, against `model.heads_norm_rotated` (float64, the same
expressions): 1.3e-16 at 6 positions and token counts, q and k; with the card's two
bf16 roundings (the squares and the normed tiles) 1.7e-3. On the card
(`results/qwen3-1.7b-qk-layer0-x10.json`), against `model.layer()` with them:

| T | q rms | k rms | v rms | a rms | x rms |
|---:|---:|---:|---:|---:|---:|
| 4096 | 6.1e-3 | 6.6e-3 | 5.9e-3 | 1.24e-2 | 2.6e-3 |
| 1024 | 6.4e-3 | 6.5e-3 | 5.9e-3 | 1.67e-2 | 2.8e-3 |
| 256 | 6.1e-3 | 6.6e-3 | 5.9e-3 | 1.01e-2 | 2.8e-3 |
| 4096, 32 tokens | 6.0e-3 | 6.1e-3 | 6.4e-3 | 1.11e-2 | 2.8e-3 |

The append is exact in every case. qkv's operation takes 13.8 us (2.6 before): a q or
k root runs per head about 20 copies to DEST, 17 vector steps, 1 product, 8
multiplies, 6 packs and the reconfiguration between each kind, about 7,500 cycles; its
reads (5 tiles and the table's rows) are under the sum.

### Real weights and a prompt

`real.py` reads the checkpoint (two safetensors files, bf16) with numpy, transposes
each W to K x N and folds the norm weights into the products as stage 2 does (W' =
diag(w) W, in bf16). The embedding, the final norm and the head run on the host: the
head is 151,936 x 2,048 (622 MB in bf16), one product of 311 M multiply-adds a token,
about 0.1 s in numpy; on the card it would read 1.6 times a layer stack's weights a
token. The prompt's tokens are fed one a step (the cache starts empty), each step one
`layers` command of 28 layers, the largest logit taken greedily.

The prompt "The tallest mountain in the world is" (7 tokens), 16 tokens continued
(`results/qwen3-1.7b-real-prompt.json`): the card's are torch's, all 16: " Mount
Everest, which is 8848 meters high. The second tallest". Torch runs the model in bf16
on the CPU (its eager attention). The hidden state after each layer at the first
positions, against torch's (rms error over rms; the last after the final norm):

| Position | layer 0 | the largest (layer) | layer 27 | numpy, the same expressions: the largest |
|---:|---:|---:|---:|---:|
| 0 | 1.24e-2 | 4.95e-2 (27) | 4.95e-2 | 1.56e-2 |
| 1 | 1.03e-2 | 3.52e-2 (27) | 3.52e-2 | 3.86e-2 |
| 2 | 1.14e-2 | 6.06e-2 (14) | 4.34e-2 | 3.97e-2 |
| 3 | 1.05e-2 | 4.27e-2 (14) | 1.53e-2 | 3.76e-2 |

The numpy column is `model.layer()` (float64, bf16 outputs, the same folds) run as the
card: it and the card differ from torch by the same order, the folds and the roundings
torch does in other places. A token takes 10.08 to 10.11 ms at the cache lengths 7 to
22 the prompt reaches: 99.0 tokens a second.

### The account with all of it in

All 28 layers, 10 tokens from one command, random weights with random head norm
weights (`results/qwen3-1.7b-qk-x10.json`):

| Term | Kind | T 4096 | T 1024 | T 256 |
|---|---|---:|---:|---:|
| the weights at 495.6 GB/s | HARDWARE | 203.1 | 203.1 | 203.1 |
| the products' read below the ceiling and boundary | OURS | 77.4 | 77.3 | 77.1 |
| the products' operations (qkv's with the heads' norms and rotation: 13.8) | OURS | 21.0 | 21.2 | 21.1 |
| the cache's K and V at 495.6 GB/s | HARDWARE | 33.8 | 8.4 | 2.1 |
| the busiest core's tile products | HARDWARE | 4.2 | 1.4 | 0.5 |
| the busiest core's exponent tiles (live rows only) | HARDWARE | 4.0 | 1.3 | 0.5 |
| the cache read below the ceiling (q on every core, the append, the command) | OURS | 14.3 | 14.8 | 14.9 |
| attend's compute not under the read | OURS | 19.3 | 12.4 | 10.9 |
| the merge's 4 rounds | OURS | 21.6 | 20.9 | 21.5 |
| the thread between commands (5 a layer) | OURS | 10.9 | 10.9 | 10.9 |
| total a layer | | 409.7 | 371.7 | 362.6 |
| unaccounted | | 0.0 | 0.0 | 0.0 |
| a token (28 layers) | | 11.47 ms | 10.41 ms | 10.15 ms |
| tokens a second | | 87.2 | 96.1 | 98.5 |
| stage 3 | | 80.4 | 90.7 | 93.4 |

One token through all 28 layers against the float64 chain (T 4096): x rms 2.6e-2,
largest 3.8e-2.

## Stage 5: the operations as emitted bodies

### The bodies

`bodies.py` writes each hand-written vector operation as a tile function of
`komorebi.dsl.tile` and emits them into `kernels/generated/` (not in the repository;
`python projects/decode/bodies.py` before a build: the card refuses headers built from
other sources than the tile package's and that file's of the moment).
`kernels/emitted.hpp` is the operations' compute on them: each operand tile copied into
DEST with its own buffer's format, a body, a pack. Each operation is a switch of its
overlays (`DC_EMIT_ADD`, `_NORM`, `_GATE`, `_QK`; `card.BODIES`; `bench.py` and
`real.py --bodies add norm gate qk`). Each body runs the live rows only: it is emitted
with the tile compiler's band (`rows=True`), `<name>(dst, rows)`, and the operations
pass the command's live rows (`decode::live_rows`), as the hand bodies do:

| Body | Expression (operand k in DEST tile dst + k) | Form | Math cycles a tile | At 1 live row |
|---|---|---|---:|---:|
| add | a + b | stream, load macro | 91.6 | 24.2 |
| square | a a | stream | 118.3 | 24.0 |
| square_add | a + b b | stream, load macro | 91.6 | 24.2 |
| squares3 | a a + b b + c c | stream | 278.8 | 44.1 |
| sum3 | a + b + c | stream | 214.7 | 36.0 |
| mul | a b | stream, load macro | 91.6 | 24.2 |
| mul3 | a b c | stream | 214.7 | 36.0 |
| mad | a + b c | stream, load macro | 123.7 | 28.2 |
| gate | silu(a b) c | stream | 1,635.6 | 222.3 |
| rms_row, rms_head | each row's 1 / sqrt(sum / n + 1e-6), n = 2048, 128 | row | 425.0 | 107.0 |

The cycles are the compiler's estimates (a body that runs one band took about 10 cycles
more on the card than its estimate; see the tile compiler's reference). The operations
on them:

- **add** (o's and down's): y = s + b (the sum in fp32, the residual's block), packed to
  bf16; the block's squares as one tile, the elementwise sum of y^2 over its 8 tiles,
  whose row sums are the row's sums of squares (the host's diagonal form has the same
  row sums, so the norm after reads either).
- **norm** (qkv's and gate_up's): the 8 parts summed (sum3 three times, add), rms_row
  into DC_CB_SCALE, in fp32 (unpacked to DEST) when the norm, the gate and the heads are
  all emitted (a hand body reads it on the matrix unit's source B, in bf16 there); the
  scale is s_t R (mul).
- **gate**: up = s R into DC_CB_UP (bf16, as the reference rounds u), then
  silu(g R) u in one body.
- **qk** (a head, DEST holding 4 fp32 tiles): the squares of its 4 tiles (squares3,
  square_add), rms_head into DC_CB_SCALE; n_j = y_j w_j R two a section (mul3 at DEST 0
  and 1) into DC_CB_NORMED; then tiles j and j + 2 in one section, n_(j+2) s_j +
  n_j c_j and n_j s_(j+2) + n_(j+2) c_(j+2) (mul, mad). 28 copies, 9 packs and 5
  sections a head.

The emitted path sets unpack A's format before each copy's init itself
(`reconfig_data_format_srca` from the buffer last copied) and never uses source B, so it
does not rely on `state_configure` (tt-metal's, a no-op). The hand path still does: every
buffer it reads on source B is bf16 like the product's, so no format is wrong there.

### What the card showed

The first runs gave q, k, a and the gate wrong, the same to every digit run after run.
Run one operation at a time (one session, a program a subset), the norm alone was
right and the add alone gave a wrong gate (its squares), the heads alone wrong q and k.
The cause is in the load macro's setup: a template is written through the back door
(an instruction with VD 12 + k stores its word), and when the template's instruction
writes a plain operand's register (square_add's and mad's: VD 4 and 5), the setup's
renaming of the plain registers renames VD too, so the word is executed instead of
stored and the macro runs the template the previous body left (add's or mul's).
`bodies.py` kept a load macro only when every template write had VD 12 or more;
square_add and mad ran plain in the results here. The tile compiler's setup now keeps
VD, and `emit` refuses a macro whose template its setup does not store: the other
bodies' headers are unchanged, square_add and mad take the load macro again; with it
every check below is the plain form's to the last digit. Each body is followed by the matrix
unit's wait for the vector unit (`DC_VECTOR_DONE`; the headers leave the wait to the
caller); on the card it changed no value.

### The check and the account

The layer against `model.layer()`, rms error over rms (`results/qwen3-1.7b-emitted-layer0-x10.json`
against `results/qwen3-1.7b-qk-layer0-x10.json`):

| T | | q | k | v | a | x1 | gated | x |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | hand | 6.1e-3 | 6.6e-3 | 5.9e-3 | 1.24e-2 | 1.2e-3 | 1.22e-2 | 2.6e-3 |
| 4096 | emitted | 5.2e-3 | 4.9e-3 | 4.6e-3 | 9.7e-3 | 4.5e-4 | 7.9e-3 | 2.5e-3 |
| 1024 | emitted | 5.3e-3 | 5.1e-3 | 4.6e-3 | 1.38e-2 | 9.6e-4 | 8.8e-3 | 2.8e-3 |
| 256 | emitted | 5.2e-3 | 5.0e-3 | 4.6e-3 | 9.2e-3 | 8.8e-4 | 8.7e-3 | 2.8e-3 |
| 4096, 32 tokens | emitted | 5.1e-3 | 5.1e-3 | 5.0e-3 | 9.5e-3 | 4.9e-4 | 8.2e-3 | 2.6e-3 |

The largest errors over the largest values are 1.1e-2 (a) or less at T 4096; the append
is exact in every case. With the band (`results/qwen3-1.7b-band-x10.json`) every error
of the table, rms and largest, is the whole tiles' to the last digit: the band computes
the live rows' words as the whole tile does and leaves the others. The emitted bodies are closer: the scale and the products by
1 / rms are fp32 vector multiplies where the hand ones are the matrix unit's (tf32
operands), and DC_CB_SCALE is fp32.

All 28 layers, 10 tokens from one command, the operations' term (us a layer) and the
layer: the hand bodies, the emitted ones on whole tiles
(`results/qwen3-1.7b-emitted-x10.json`) and on the live rows' band
(`results/qwen3-1.7b-band-x10.json`):

| | T 4096 hand | whole | band | T 1024 hand | whole | band | T 256 hand | whole | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv's operation (norm, scale, heads) | 13.8 | 16.6 | 15.2 | 13.8 | 16.7 | 15.1 | 13.8 | 16.6 | 15.2 |
| o's (the add, its squares) | 1.6 | 3.5 | 2.3 | 1.6 | 3.4 | 2.3 | 1.6 | 3.5 | 2.4 |
| gate_up's (norm, gate) | 4.1 | 9.0 | 4.0 | 4.2 | 8.9 | 4.1 | 4.2 | 8.9 | 3.9 |
| down's (the add, its squares) | 1.5 | 3.5 | 2.2 | 1.6 | 3.3 | 2.3 | 1.5 | 3.4 | 2.2 |
| the operations | 21.0 | 32.5 | 23.7 | 21.2 | 32.3 | 23.8 | 21.1 | 32.4 | 23.7 |
| a layer | 409.7 | 421.1 | 412.4 | 371.7 | 383.1 | 374.5 | 362.6 | 373.9 | 365.4 |
| tokens a second | 87.2 | 84.8 | 86.6 | 96.1 | 93.2 | 95.4 | 98.5 | 95.5 | 97.7 |

The rest of the account is unchanged to 0.5 us a term. The band takes 8.8 of the
whole tiles' 11.5 us a layer back; gate_up's operation is now under the hand's. What
is left against the hand, 2.7 us a layer, is in qkv's (1.4), o's (0.7) and down's
(0.8) operations. In cycles a tile of a block (8 tiles a core, 1,350 MHz), hand
against band: o 273 against 395, down 245 against 373, qkv 2,332 against 2,562,
gate_up 695 against 673. The terms:

- **The copies into DEST.** Each operand of a body is a copy (`copy_tile`, and an
  unpack reconfiguration where the buffer changes); the hand operations copy the first
  operand and read the others on the matrix unit's source (an elementwise add or
  multiply onto DEST). The emitted ones copy 3 tiles a tile in the add (the hand 1), 11
  in qkv's operation, 3.5 in gate_up's. The model does not price them. In the add, with
  the hand's vector work taken as the bodies' 48.4 cycles a tile and the loop's term
  below, the 2 extra copies are about 100 cycles a tile, about 50 a copy (an inference
  from the totals, not a measurement).
- **The loop.** A body that runs one band took about 10 cycles more than its words on
  the card (the add 35.0 against 24.2, the gate 220 against 210, the norm 183 against
  173): 2 bodies a tile in the add (about 22 cycles), 5.4 in qkv's (about 58). The
  model has no such term; it prices the band's loop and tests at nothing, as they ran.
- **The gate's silu.** 222.3 cycles a tile at one row (the card's 1-row silu(a) b
  220), 4 a block: with the band the gate is no longer a term (gate_up's operation is
  0.1 us under the hand's).

The default stays the hand bodies (`card.DEFAULT_BODIES` is empty): the emitted
operations are 2.7 us a layer over them.

With the model's weights and the prompt (`results/qwen3-1.7b-emitted-real-prompt.json`):
16 of 16 tokens torch's; the hidden state's rms error against torch's at positions 0 to
3 at most 2.4e-2, 3.3e-2, 7.3e-2 (layer 14), 5.3e-2 (14) (hand: 4.95e-2, 3.5e-2, 6.1e-2,
4.3e-2); a token 10.41 to 10.43 ms, 96.0 tokens a second (hand 99.0). With the band
(`results/qwen3-1.7b-band-real-prompt.json`, a cache of 4,096 rows): 16 of 16 tokens
torch's, the errors against torch's the whole tiles' bit for bit, a token 10.16 to 10.19
ms, 98.2 tokens a second.

## Stage 6: a core goes on to the next kernel by itself

Until stage 5 the X280 thread submitted a layer's five commands (qkv, attend, o,
gate_up, down) one by one and waited for each: every core waited for the slowest of
the command before it, then for the thread's next submit. Now a core holds what the
current command is, its weight, and where the next one is, and goes on by itself.

### The design

- **The records.** For each command of a layer the thread writes a record of 64 bytes
  (`../dense/kernels/split/chain.h`): the command's ten words (a1 its weight), its
  cores, an info word (whether its input is marked, whether its parts go up a tree,
  how it reads the output before it) and the next record's address (0: the last). They
  sit in a region of the thread's (DRAM, the bank its command blocks use), written
  before each layer: the position changes every token and the layer every command, so
  a table the host writes once would need words patched on the core.
- **One command a layer.** The thread submits one SPLIT command on all 120 cores in the
  chain's mode (a4 3), a7 and a8 the first record as a block, and waits for it once
  (`run_chain` in `thread/layers.c`). The dispatcher starts the layer and nothing else.
- **The core goes on.** Its reader reads record r + 1 into a ring of two slots at r's
  start, once the writer and compute have taken r - 1 (`chain.hpp`); the writer takes
  each record from the ring, compute's unpack takes it and hands its words to math and
  pack. A core outside a record's cores skips it. The operation of a record is an
  overlay, loaded per call as before.
- **B's weight first.** The weight's read is a stream of its own, not paired with the
  activation's (`chunks.hpp`): once the mover's W half is empty of A (compute has
  taken every chunk), it reads B's W into it until B's input is marked, then reads A's
  chunks with the rest of W. A core that A does not use (gate_up takes 96 of 120)
  reads its share of down's W while gate_up runs elsewhere.
- **B's input.** Each root of A, once its block is written (the writes acknowledged),
  adds one to a marks word on every core of B whose part of B's input that block
  meets: by K range for a product, by the three ranges of qkv attend reads (a KV
  head's q rows, its k and v); a core waits until its marks reach its total. Only
  compute's input waits; nothing waits for the thread.
- **The ports.** Two records' sums meet in one port and arrival slot (a port does not
  tell two commands apart), so a tree's receiver sends a credit to each of its senders
  when it starts a record; a sender emits after the credit.
- **The end.** The last record has no next; the command completes as any command does,
  and a give-up in the middle records its where code (0xC0 to 0xC3 the chain's own
  waits) as before.

Every wait has a releaser: the marks (A's roots), the ring (the reader; the writer's
and compute's takes), the credits (the receivers). The chain's words and the two slots
sit in buffer 31, the runtime's trace buffer's index (0 to 27 are the program's, 28 the
overlay slots', 29 and 30 the runtime's), so a chain's program runs untraced. The chain's kernels take 78,128 B of code
against the runtime's 70,656 B kernel buffer, so the program takes 80 K (from the
212 K no buffer uses). Off (`--no-chain`), the program is stage 5's byte for byte: its
five kernel images (and the dense project's K split's) are main's.

### The check

The layer against `model.layer()` (`results/qwen3-1.7b-chain-layer0-x10.json`), staged:
a command a step on the chain's program, then qkv and attend as a chain of two, then
the layer as one command. Every tensor of the chain equals the commands' to the last
digit (the sums are the same in the same order): at T 4096 q 6.1e-3, k 6.6e-3, v
5.9e-3, a 1.24e-2, x1 1.2e-3, gated 1.22e-2, x 2.6e-3 (rms), the hand's of stage 5;
T 1024, 256 and 32 tokens alike; the append is exact.

With the model's weights and the prompt (`results/qwen3-1.7b-chain-real-prompt.json`):
16 of 16 tokens torch's, every layer's error against torch's the hand's of stage 4 to
the last digit; a token 8.78 to 8.85 ms at cache lengths 7 to 22, 113.4 tokens a
second (stage 4: 10.08 to 10.11 ms, 99.0).

### The account

All 28 layers, 10 tokens, the same program a command a step and as one command
(`results/qwen3-1.7b-chain-x10.json`; us a layer). In one command the steps have no
events of their own, so its rows are the cases' differences as before (operations:
full less bare; exponents: exp_twice less full) and the rest of the command as one
row: what it holds cannot be split without stamps of each record.

| Term | Kind | T 4096 commands | chain | T 1024 commands | chain | T 256 commands | chain |
|---|---|---:|---:|---:|---:|---:|---:|
| the weights at 495.6 GB/s | HARDWARE | 203.1 | 203.1 | 203.1 | 203.1 | 203.1 | 203.1 |
| the cache's K and V at 495.6 GB/s | HARDWARE | 33.8 | 33.8 | 8.4 | 8.4 | 2.1 | 2.1 |
| the busiest core's tile products | HARDWARE | 4.2 | 4.2 | 1.4 | 1.4 | 0.5 | 0.5 |
| the busiest core's exponent tiles | HARDWARE | 4.0 | 3.5 | 1.3 | 1.1 | 0.5 | 0.2 |
| the products' operations | OURS | 21.3 | 18.2 | 21.2 | 17.7 | 21.1 | 17.8 |
| the reads below the ceiling, the boundaries, attend's read, compute and merge | OURS | 132.6 | 82.7 | 126.2 | 83.5 | 125.7 | 82.8 |
| the thread (submit, wait) | OURS | 11.2 | 11.1 | 11.2 | 11.1 | 11.2 | 11.1 |
| total a layer | | 410.3 | 356.6 | 373.0 | 326.4 | 364.2 | 317.6 |
| unaccounted | | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| a token (28 layers) | | 11.49 ms | 9.98 ms | 10.44 ms | 9.14 ms | 10.20 ms | 8.89 ms |
| tokens a second | | 87.1 | 100.2 | 95.7 | 109.4 | 98.1 | 112.4 |

A command a step on the chain's program is stage 5's within 1.6 us (409.7 / 371.7 /
362.6). The commands' large row is, at T 4096, the products' reads below the ceiling
47.7, their boundaries 30.5, attend's read below the ceiling 14.0, its compute not
under the read 19.3 and its merge 21.3. The chain's is 82.7 to 83.5 at every T: 49.9 /
42.7 / 42.9 us less, more than the four boundaries: most likely a core reads B's weight
while the slower cores finish A, and o's while attend's merge runs on others. The thread
did not move: one submit and wait a layer takes what five did (11.1 against 11.2), so
it is not a cost a command. The operations' row is 3.1 to 3.5 smaller (likely their
tails under the next record's reads on other cores). The chain's unaccounted is zero by
its form: its large row is the rest.

## Stage 7: the chain further

### Where the chain's time is

With `--stamps` (`SPLIT_CHAIN_STAMPS`, off by default) each kernel adds the wall
clock at a record's edges into words of its own in the chain's buffer, from its
chain's start (`chain.h`'s `kChainStamp*`): the reader and the writer the record
taken, W's first block issued, the time W's half held the next weight before the
mark, the mark seen, the moves done, the root's output ready and its writes
acknowledged, the next record's fetch; compute's pack its first chunk, its last
product, the sum, the output. A SPLIT in `kChainStampsMode` writes each core's page
to DRAM and zeroes it; the thread sums its own parts (records written, submit,
wait). Stamped, a layer takes 0.6 us more (357.2 against 356.6).

Each record's span runs from the record before's end (its slowest root's writes
acknowledged) to its own; its terms are its median root's, and the tail is the
slowest root's end against it, so the spans and the end add to the command exactly
(`results/qwen3-1.7b-chain-stamps-x10.json`, 28 layers, 10 tokens, us a layer):

| Record | Term | Kind | T 4096 | T 1024 | T 256 |
|---|---|---|---:|---:|---:|
| qkv | W at the ceiling | HARDWARE | 33.9 | 33.9 | 33.9 |
|  | W's rate below the ceiling | OURS | -2.7 | -2.7 | -2.8 |
|  | W read under the record before | OURS | 2.7 | 2.6 | 2.6 |
|  | compute after the last chunk | OURS | 1.4 | 1.4 | 1.4 |
|  | the tree's rounds | OURS | 6.0 | 6.2 | 6.2 |
|  | the operation and the output | OURS | 14.8 | 14.8 | 14.8 |
|  | the root's write and acknowledge | OURS | 1.2 | 1.2 | 1.2 |
|  | the tail: the slowest root against the median | OURS | 16.2 | 7.1 | 5.7 |
| | **span** | | **73.5** | **64.4** | **63.0** |
| attend | its input's mark after the record before's end | OURS | -3.4 | -0.7 | -0.8 |
|  | the cache's K and V at the ceiling | HARDWARE | 33.8 | 8.4 | 2.1 |
|  | the busiest core's tile products | HARDWARE | 4.2 | 1.4 | 0.5 |
|  | the exponent tiles (measured, d5) | HARDWARE | 3.5 | 1.1 | 0.2 |
|  | attend's read, compute and merge above them | OURS | 41.8 | 43.8 | 45.8 |
|  | the root's write and acknowledge | OURS | 1.0 | 1.0 | 1.2 |
|  | the tail: the slowest root against the median | OURS | 6.0 | 2.0 | 2.1 |
| | **span** | | **86.8** | **57.1** | **51.1** |
| o | W at the ceiling | HARDWARE | 16.9 | 16.9 | 16.9 |
|  | W's rate below the ceiling | OURS | -1.2 | 3.9 | 3.9 |
|  | W read under the record before | OURS | -9.3 | -12.3 | -13.2 |
|  | compute after the last chunk | OURS | 1.1 | 2.3 | 2.2 |
|  | the tree's rounds | OURS | 5.1 | 9.0 | 8.8 |
|  | the operation and the output | OURS | 4.5 | 6.5 | 6.5 |
|  | the root's write and acknowledge | OURS | 0.7 | 0.7 | 0.7 |
|  | the tail: the slowest root against the median | OURS | 0.1 | 4.5 | 4.0 |
| | **span** | | **17.9** | **31.4** | **29.7** |
| gate_up | W at the ceiling | HARDWARE | 101.6 | 101.6 | 101.6 |
|  | W's rate below the ceiling | OURS | 6.8 | 4.4 | 4.0 |
|  | W read under the record before | OURS | -18.1 | -20.1 | -19.2 |
|  | compute after the last chunk | OURS | 2.2 | 2.2 | 2.2 |
|  | the tree's rounds | OURS | 2.7 | 1.2 | 1.2 |
|  | the operation and the output | OURS | 9.6 | 7.9 | 7.8 |
|  | the root's write and acknowledge | OURS | 1.4 | 1.4 | 1.4 |
|  | the tail: the slowest root against the median | OURS | 33.1 | 37.4 | 37.9 |
| | **span** | | **139.2** | **136.0** | **136.8** |
| down | W at the ceiling | HARDWARE | 50.8 | 50.8 | 50.8 |
|  | W's rate below the ceiling | OURS | -26.5 | -27.2 | -27.1 |
|  | W read under the record before | OURS | -13.9 | -15.7 | -15.9 |
|  | compute after the last chunk | OURS | 1.0 | 1.0 | 1.0 |
|  | the tree's rounds | OURS | 9.3 | 10.4 | 10.4 |
|  | the operation and the output | OURS | 2.8 | 2.8 | 2.8 |
|  | the tail: the slowest root against the median | OURS | 2.1 | 2.6 | 2.6 |
| | **span** | | **25.5** | **24.6** | **24.5** |
| | the command's end (the last writes, the report) | OURS | 3.0 | 2.8 | 2.8 |
| | the thread: the records written | OURS | 8.8 | 8.8 | 8.8 |
| | the thread: the submit | OURS | 1.1 | 1.1 | 1.1 |
| | the thread: its wake past the event | OURS | 1.1 | 1.1 | 1.1 |
| | the thread: the rest | OURS | 0.3 | 0.3 | 0.3 |
| | **a layer** | | **357.2** | **327.5** | **319.1** |
| | unaccounted | | 0.0 | 0.0 | -0.0 |

| Record | cores waiting with W's half full | their wait (us) | the next record's fetch (us) |
|---|---:|---:|---:|
| qkv | 0 | 0.0 | 2.14 |
| attend | 0 | 0.0 | 0.95 |
| o | 120 | 38.3 | 0.90 |
| gate_up | 91 | 19.4 | 0.82 |
| down | 90 | 54.4 | 0.02 |

Read together:

- **The weights are read at about the ceiling overall.** The DRAM terms net -4.8
  us at T 4096: the tails (gate_up 33.1, qkv 16.2) are the record still reading
  while cores that are done read the next one's W (the credits under the record
  before, -38.6, and the median root's read faster than its share, -23.6). qkv's
  tail grows with T: the groups attend marks first read the cache meanwhile.
- **DRAM idles while cores wait.** o's, gate_up's and down's cores hold the next W in
  a full half and wait for their mark (o: all 120 cores for 38.3 us, under attend):
  the command is 346 us, its DRAM reads at the ceiling 237.
- **The thread's 11.3 us is writing the records** (8.8): 80 words a layer from the
  X280 into DRAM. Submit and wake are 2.2.
- What is ours and not about DRAM: attend above its hardware 41.8, the operations
  31.7 (qkv's norms and rotation 14.8, the gate 9.6), the tree's rounds 23.1.

### A token as one command

`--token` (LAYERS' a4 512): the thread writes the records of all 28 layers once, 140
of them, the last of each layer naming the next layer's qkv, whose input is marked by
the previous layer's down like any other record's. They hold everything that does not
change from token to token; they are written again only when a job's form changes
(its flags, first layer, layer count, rows). The one word a token changes is the
position, in qkv's a9 (its rotation's place) and attend's a9 (the cache's new row; the
length it attends over follows from it): the token's one command carries it in its a0,
and a record's info word names the word of its command the core ORs it into
(`kChainTokenShift`). The thread submits once a token and waits once.

The check: one token through 2 layers and through 28 as one command, against the
chain a layer a command from the same x: every tensor the card keeps (qkv, a, x1,
gated, x) equal to the last digit, both times. The real prompt: 16 of 16 of torch's
tokens, every layer's error equal to the commands' to the last digit
(`results/qwen3-1.7b-real-prompt-token.json`), 117.2 tokens a second at T <= 256.

Beside each other in one job (`results/qwen3-1.7b-chain-token-x10.json`, stamped, 28
layers, 10 tokens; "layer": the chain a layer a command; a token's spans folded over
its 28 layers, qkv's from the previous layer's down):

| us a layer | T 4096 token | T 4096 layer | T 1024 token | T 1024 layer | T 256 token | T 256 layer |
|---|---:|---:|---:|---:|---:|---:|
| qkv | 47.1 | 73.5 | 43.6 | 64.5 | 43.1 | 63.3 |
| attend | 94.6 | 86.5 | 58.6 | 56.8 | 52.0 | 51.3 |
| o | 31.0 | 18.0 | 37.8 | 31.1 | 36.2 | 29.1 |
| gate_up | 147.7 | 138.9 | 147.5 | 136.5 | 148.2 | 138.2 |
| down | 24.8 | 25.7 | 24.6 | 24.5 | 24.5 | 24.4 |
| the end (the last record to the event) | 0.2 | 3.0 | 0.2 | 2.8 | 0.2 | 1.2 |
| **the command** | **345.4** | **345.6** | **312.3** | **316.2** | **304.2** | **307.6** |
| the thread: records written | 0.81 | 12.06 | 0.81 | 12.05 | 0.82 | 12.05 |
| submit | 0.04 | 1.05 | 0.04 | 1.05 | 0.04 | 1.05 |
| wake after the event | 0.04 | 1.13 | 0.04 | 1.12 | 0.04 | 1.11 |
| the rest | 0.11 | 2.96 | 0.11 | 2.97 | 0.11 | 2.96 |
| **a layer** | **346.4** | **362.8** | **313.3** | **333.4** | **305.2** | **324.8** |
| tokens a second | 103.1 | 98.4 | 114.0 | 107.1 | 117.0 | 109.9 |

- **The thread's 11 to 17 us a layer are gone.** The 140 records are written once a
  job (227 us; over 10 tokens 0.81 a layer, and none when the form is the one already
  written); submit and wake are 2.2 us a token, the rest 3.1.
- **The command itself gained almost nothing** (T 4096: 345.4 against 345.6; T 1024 and
  256: 3.5 to 3.9). qkv's span fell by 26 us, its W read under the previous down's
  tail, but its cores now wait 35 us with the W half full (112 of them), and attend,
  o and gate_up took the time back: the layer's DRAM reads still stop while cores wait
  for their marks (o 120 cores, 40 us; gate_up 94, 20; down 95, 56).
- The layer chain beside it here has a slower thread than in the table above (17.2
  against 11.3 us a layer): its rest, 3.0 against 0.3, is the command's cores computed
  once more from five commands; its records, 12.1 against 8.8, are not explained yet.

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
- A runtime trace gives a core four stamps a command (its reader's take, its writer's
  end, its report), so a command's read cannot be told from its compute. A norm's 12.6
  us was taken apart by its sizes, not by stamps.
- The movement engine moves one block a stream on a mover in a pass, then waits at a
  barrier. A pattern that gives one mover more chunks (AAAB) gains only when that
  mover's half is deep enough not to wait on compute, so the split sizes it by hand
  (the plan's depth).
- An operation is called only after the sum. A norm's scale needs only the squares, not
  the sum, so it could run while the root waits for its parts if there were a hook
  before the sum.
- A command cannot start on a core before the one ahead completes, so attention's
  cache read (which needs nothing from qkv but the token's row) waits for qkv's end.
- A buffer has one unpack format for every operation: the merge's fp32 states unpack
  to DEST for its copies, and the matrix unit reads them as tf32 only after the
  overlay reprograms the unpackers itself (`rescale_init` in `kernels/attend/compute.cpp`).
- The math library's matmul init, outlined by the compiler, reads the buffers' tile
  shapes from a table the persistent kernel holds, which an overlay may not copy: the
  overlay gives the shape (32 x 32) itself.
- A command has ten words and qkv's uses them all (the inputs, the plan, the operation,
  the squares, the qk tensor, the position and the layer); the norm's epsilon became a
  compile-time value of the overlay to make room.
- A program has 29 buffers of its own (0 to 28; the runtime's are above). The
  operations' new buffers are the ones other operations use, taken while those are
  idle, each still filled by the same kind of kernel (a clear after a give-up looks up
  a buffer's producer).
- A first form of the rotation's table tilized the rows on compute (tt-metal's tilize
  inside the overlay); its first command did not end and the cause was not located.
  The reader now puts the rows into the tiles itself.
- The tile compiler's forms ran whole tiles: neither form could be told to run the
  first M rows (the live rows' bands), most of why the emitted operations were slower
  than the hand ones. The band (`rows=True`) now takes the live rows at run time.
- The tile compiler's load macro stored no template whose instruction writes a plain
  operand's register (square_add, mad); its setup now keeps the template's VD.
- An emitted body reads every operand from DEST, so each is a copy, where the hand
  operations read all but the first on the matrix unit's source: the tile compiler has
  no body that takes an operand there, most of what is left against the hand.
- A chain of commands is the decode project's by hand: the record's form, the reader's
  ring and the hand-off of its words to the compute threads, the marks (who meets
  which block of whose input, counted per core) and the ports' credits are in the
  split's kernels (`chain.hpp`), and the thread builds the records. The playbook and
  the compiler could derive each from the products' inputs and outputs.
- A program has no buffer index left for words a core keeps across commands: the
  chain's take the trace buffer's, so a chain's program cannot be traced; and in one
  command the steps have no events of their own.
