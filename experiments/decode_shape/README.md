# decode_shape

Where the time of an LLM decode step goes on one Blackhole p150a, and which way of
driving it leaves the least time outside the weight read. Not a model: no tokenizer, no
real weights, no attention over a cache, no new kernel math. A step is a chain of matrix
products whose weights a transformer layer reads, each fed by the one before it, run
with the dense program of [`projects/dense`](../../projects/dense/README.md) as it is.

## The stand-in

One token row (M = 32 rows, one tile) at width D = 4096. A layer is seven products
`[32, 4096] @ [4096, 4096]` in bf16 (32 MiB of weights each, 224 MiB a layer): the four
attention projections and the MLP's three with H = 4096. L = 32 distinct layers of
random weights in DRAM (7 GiB), cycled for T = 20 tokens (4,480 products a chain, 7.5
GB of weights a token). Weights are uniform in ±sqrt(3/D), so the activation keeps
its scale along the chain; the activation goes back and forth between two
`[32, 4096]` tensors.

- **H = 4096, not 8192.** The dense program is in the kernel form, so a run holds one
  program, and a K-split program's plan (K and N tiles, group size, chunk) is fixed when
  it is loaded: its kernels' arguments and buffers. D x 8192 and 8192 x D are two more
  plans, which one run cannot hold beside D x D (D x 8192 could be two D x D commands;
  8192 x D could not: its two halves of K need a sum that no handler gives).
- **No gate.** `silu(a) * b` is not among the K split's handlers; it is left out.
  What is measured is the bytes read and the commands issued.
- **Every product** is the dense page's fastest form for `dec4096`: the K split
  14,1,16,5 with the tree reduce, 8 groups of 14 cores = 112 cores (the plan the page
  measured, not 120).
- **L = 32** from the host's write rate, measured first on one layer: 105 to 130 MB/s
  (`session.tensor` of a 32 MiB matrix, tile layout), so the 7 GiB take 58 to 93 s.
- **The check.** Before the timed chains each way runs one layer (seven products) from
  the same start; its last output is compared with numpy (float64 products, each output
  rounded to bf16): relative RMS 2.13e-2 and largest 2.84e-2 of the largest output,
  identical for the three ways, inside the loose bound of 5e-2 and 1e-1.

## Run

```
python experiments/decode_shape/decode.py --layers 32 --tokens 20 \
    --cases host:full:x:1 stream:full:x:1 thread:full:x:1
python experiments/decode_shape/decode.py --layers 32 --tokens 20 --no-check \
    --cases host:parts:x:1 stream:parts:x:1 stream:parts:-:4 thread:parts:x:1
python experiments/decode_shape/decode.py --plan half --layers 32 --tokens 20 \
    --cases thread:full:x:1 thread:full:x:1:2
python examples/11_movement/movement.py --pattern A AB AAAB --cores 12
```

A case is `way:mode:flag:window[:chains]`: way `host`, `stream` or `thread`; mode
`full` (the product) or `parts` (the dense page's parts-only cut: each core's part
read and computed, no reduce, nothing written); flag `x` (`KM_FLAG_EXCLUSIVE`) or `-`
(none); window, the commands in flight (1: each waited for before the next); chains 1,
or 2 on two groups of the `half` plan (7,1,16,5: 56 cores each, cores 0 to 55 and 56
to 111). `--plan row` (2,1,16,5, 16 cores) is the first rung for anything new. Each
case runs under the power and clock samplers and is followed by a pause as long as it
took. The script prints one JSON line (`--json` writes it too); the runs behind the
tables are in `results/`. Setup is that of
[01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs).

| Way | What drives the chain |
|---|---|
| `host` | the host: `run.submit` of one SPLIT, `run.wait`, the next |
| `stream` | one stream the dispatcher runs: `SUBMIT` with A, W and Y from registers, `WAIT window - 1`, A and Y swapped, W the next matrix's address (the weights' addresses are equally spaced, checked) |
| `thread` | one command from the host to an X280 thread on cluster 1 (`thread/chain.c`, declared on the dense program in `decode.py`): it submits every product of every layer of every token with `kmx_submit` and waits with `kmx_wait`, at most `window` in flight |

A stream rather than a graph: a graph takes at most 64 distinct command templates and
256 nodes, so a chain of products with constant arguments has at most 9 layers; a
stream's registers and loop take any L.

## A. The ceiling

DRAM to L1, no compute, GB/s over all cores
(`examples/11_movement/movement.py --pattern A AB AAAB --cores N`, 256 tiles of 2 KB a
core in blocks of 16; 2026-11-14 15:13, card clock 1,343 to 1,350 MHz under load, host
load 3 to 6; `results/movement.jsonl`). `A`: one data-movement CPU (mover A, NoC 0) reads;
`AB`, `AAAB`: both on one stream.

| Cores | A | AB | AAAB |
|---|---|---|---|
| 12 | 344.7 | 219.6 | 436.6 |
| 60 | 426.5 | 374.8 | 500.4 |
| 120 | 428.2 | 485.2 | 503.3 |
| 120, from [11_movement](../../examples/11_movement/README.md#numbers) | 427.8 | 485.9 | 502.6 |
| 120, from [movement](../../docs/reference/movement.md#choosing-a-pattern) (32 tiles a block) | 422.9 | 477.8 | 498.9 |

Beside it the dense page's `dec4096` K split: 92.4 µs a product in a loop, 363 GB/s of
W; parts only 81.7 µs, 410 GB/s. Here, one product at a time: 90.3 µs (parts only
79.8). The parts are 88% of a D x D product's time, and the weight read alone at the
ceiling (32 MiB at 503 GB/s, 66.7 µs) is 72% of it.

## B. Who drives the chain

L = 32, T = 20 (4,480 products), 112 cores, every product with the flag and waited for
(2026-11-14 15:26, card clock 1,350 MHz in every sample, host load 5;
`results/whole-l32.json`; the thread's chain again with T = 40, twice:
`results/whole-twice-l32.json`). Outside: the chain's time less the products' own
times, 7 x 90.3 µs a layer (the host way's median submit-to-completion time, each
product alone on the grid).

| Way | ms a token | µs a layer | GB/s of W | µs a layer outside the grid's work |
|---|---|---|---|---|
| host, one command at a time | 52.63 | 1,644.6 | 142.8 | 1,012.5 |
| stream | 20.87 | 652.3 | 360.1 | 20.2 |
| X280 thread | 20.86 | 651.9 | 360.3 | 19.8 |
| X280 thread, T = 40, twice | 20.91, 20.90 | 653.5, 653.2 | 359.4, 359.6 | 21.4, 21.1 |

The host pays 145 µs of its own a product (the round trip of `submit` and `wait`);
the stream and the thread pay 2.8 to 3.1 µs a product. Mean card power: 77.6 W with
the host driving, 87.8 W with the thread.

## C. The boundary between products

The same chain, 112 cores (2026-11-14 15:26 and 15:30, card clock 1,343 to 1,350 MHz,
host load 5 to 10; `results/whole-l32.json`, `results/whole-parts-l32.json`). Parts
only (no reduce, nothing written) may go without the flag: no ports, as the dense page
runs them. Own time of a parts-only product: 79.8 µs, 558.6 µs a layer.

| Mode, flag, window | Way | ms a token | µs a layer | GB/s of W |
|---|---|---|---|---|
| full, flag, 1 | stream | 20.87 | 652.3 | 360.1 |
| full, no flag, 1 | stream | 20.84 | 651.1 | 360.7 |
| parts, flag, 1 | stream | 19.18 | 599.3 | 391.9 |
| parts, no flag, 4 | stream | 16.44 | 513.7 | 457.3 |
| parts, no flag, 8 | stream | 16.42 | 513.2 | 457.7 |
| parts, flag, 1 | X280 thread | 18.51 | 578.4 | 406.1 |

In a waited chain the flag costs nothing (652.3 against 651.1 µs a layer); our
reading is that the 16 µs the dense page measures back to back are the wait for the
group to drain, which a dependent chain pays anyway. Without the boundary, parts-only products take 73.3 µs
each, below the 79.8 µs of one alone: a core that finished its part takes the next
command and reads the next product's weights while the others of its group still
work. A layer gains 86 µs (14%), at 458 GB/s, 91% of the ceiling.

Which of a layer's products may overlap the next:

- In this stand-in none: every product's A is the Y before it.
- In a real layer, Q, K and V read the same input, as do the MLP's gate and up; the
  output projection waits for attention and the down projection for the gate. With
  the dense program as it is, no full K split may overlap another: its parts travel
  between the cores of a group over their ports, so the commands carry the flag.
- What would have to change (not built): the W chunks are read on their own, before
  the A chunk they meet, so a core reads the next product's weights while the last
  one's Y is still being summed; A (256 KB against 32 MiB of W) is then the only
  wait, on the Y tiles its K partition needs rather than on the whole command. The
  ports' arrival slots would need to tell two commands' parts apart, so that a
  group's sum of one product cannot meet a part of the next. Then the boundary is a
  wait on data, and the measured ceiling of the parts-only chain without the flag
  (73.3 µs a product) is what a full layer could reach.

The X280 thread without the flag did not run: see "What did not run".

## D. Two halves

Two independent chains at once, each on 56 cores (the `half` plan), driven by one X280
thread, against one chain on 112 cores run twice (2026-11-14 15:38 and 15:33, card
clock 1,343 to 1,350 MHz, host load 5 to 6; `results/half-l32.json`,
`results/whole-twice-l32.json`). Every product full, with the flag, waited for.

| Chains | Cores | ms a token (each chain) | Tokens a second (all) | GB/s of W |
|---|---|---|---|---|
| one | 56 | 24.18 | 41.4 | 310.8 |
| two at once | 56 + 56 | 37.47 | 53.4 | 401.2 |
| one, run twice | 112 | 20.91, 20.90 | 47.8 | 359.5 |

A product alone takes 105.0 µs on 56 cores; with the other chain beside it 144 to
169 µs. Two halves give 1.12 times the tokens of the whole grid. With part A (60
cores read 500 GB/s, 120 cores 503): the DRAM read does not scale with the cores; it
is shared. One chain on 56 cores reads 311 GB/s, below what 60 cores can read: the
rest of each product (the sum, the write of Y, the command's start and end) leaves
the read idle part of the time, and the second chain fills some of it.

## What did not run

X280 thread chains failed three times, each ended by its bound without a reset: the
chain did not return and its command passed its deadline. Two had commands without
the flag (full, window 1; parts, window 4) and one had commands with the flag (parts,
56 cores). All three started 9.8 to 11.7 s after the thread started; every thread
chain that passed started within 9 s of it. The same commands from a stream passed.
The cause was in the library, not in the chains: one of the thread's X280 cores 1 to 3
kept the thread's 10 s deadline (of `kmx_main`, which this thread does not have) and
left its serve loop when it passed, so the chain's first `kmx_invalidate`, which runs on
all four cores, never returned. That is fixed; the thread chains without the flag, and
part D's parts-only pair, were not run again.

## What this says for a real decode

- The bound is the weight read: a product's parts are 88% of its time, and the chain
  driven from the card reads W at 360 GB/s, 72% of the 503 GB/s ceiling.
- Drive from the card: a stream or an X280 thread costs about 3 µs a product (20 µs a
  layer), the host 145 µs (a token takes 2.5 times as long).
- What is left is the completion boundary: without it the parts reach 458 GB/s. A
  K split whose W read does not wait for A, and ports that tell commands apart, would
  let a layer's products overlap.
- Two chains on halves of the grid share the DRAM read; they gain 12% by hiding each
  other's per-command time, not by reading faster.
- The framework lacks for it: one run holding several plans of the kernel form (the
  MLP's D x 8192 and 8192 x D beside D x D) and the gate at this width.
