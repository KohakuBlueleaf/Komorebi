# nn/tinygrad: tinygrad's schedule as a list of kernel calls

tinygrad already hands a device its work as a list of kernels. Under `TinyJit` the list
is captured once and then replayed call after call. This folder reads that list for the
two models of the nn package and sets each kernel in it against what Komorebi has: the
decode program's five calls a layer, and the tile compiler. One of tinygrad's own
kernels also runs on the card as a program of its own. tinygrad is not a dependency of
the package: `schedule.py` runs where tinygrad is installed, and `epilogue.py` reads
only what it writes.

## How Kohaku-TT's tinygrad backend works

Kohaku-TT registers the card as the tinygrad device `"KTT"`
(`src/kohakutt/tinygrad_backend/`, `docs/tinygrad.md`). Paths below are in Kohaku-TT's
tree.

- **What tinygrad hands it.** tinygrad's scheduler cuts the graph into kernels. Each
  kernel is one AST: a `SINK` of stores whose values are trees of loads, elementwise
  operations and `REDUCE`s over ranges. `compat.set_seam` replaces
  `tinygrad.engine.realize.to_program`, the function that turns an AST into a program,
  so the device is handed the AST before any code is generated (`compat.py:267`). The
  device's renderer renders nothing (`device.py:109`).
- **What it makes of a kernel.** `device.seam` plans each AST once, cached by the AST's
  key in `PLANS` (`device.py:163`). Library-level calls go to Kohaku-TT's own kernels
  through `overrides.make_lib_plan`: attention, layer and RMS norm, conv2d, pooling,
  random numbers (`docs/tinygrad.md:16-28`). Every other kernel is read as a graph of
  bridge operations: a `DevKernel` evaluates the AST node by node (`executor.py:326`).
  A load becomes a view, a memory read or a gather. Elementwise nodes become TT-NN
  unary and binary operations. A `REDUCE` of `ADD` over a `MUL` becomes a matmul
  (`core.mm`), and other reductions become `sum` or `amax`. These are calls through the
  C functions of `_ktt_native.so` (`core.py`). Nothing is compiled per kernel: the
  operations are TT-NN's and Kohaku-TT's prebuilt programs. A kernel the reader cannot
  take runs on the host in numpy and is counted in `executor.FALLBACKS`
  (`executor.py:277`). The plan caches the pure parts of an evaluation (`pure_memo`,
  `const_cache`) for the next launch.
- **How it runs the stream.** tinygrad's own runtime calls each kernel's program in
  schedule order. `TinyJit` replays the captured list of calls. Each call is
  `KTTProgram.__call__` (`device.py:129`), which runs the plan's evaluation on the host
  thread: one host-issued TT-NN operation after another, queued on the card. On a replay
  the evaluation runs again; only the plan is kept. Nothing waits for the card until a
  buffer is read back, or until tinygrad times a launch with `wait`, as `DEBUG=2` does
  (`device.py:140-151`; `docs/tinygrad.md`, "DEBUG report").
- **Where its recorded numbers put the time.** `docs/reference/results.md:303` records a
  GPT of d 256 with 4 layers at 2132 ms a step eager and 148 ms with `TinyJit`. The
  replay saves tinygrad's own scheduling and lowering, not the bridge's per-operation
  issue from the host. `results.md:304` records a convnet at 560 to 600 ms a step,
  against 1250 to 1700 on tinygrad's CPU backend. The precision is fixed by the
  operations: fp32 matmul at HiFi2 by default, about 1e-2 relative
  (`docs/tinygrad.md`, "Numerics").

In Komorebi's terms there is no list on the card. The host is the executor, and it
submits several operations for each kernel that tinygrad hands it. There is also no
compiler: each tinygrad kernel becomes a fixed composition of library operations. The
nn package goes the other way. The list is in DRAM, the X280 thread submits each entry,
and the entry's kernel is one of the program's own, fused as the decode layer's are.

## The two models as tinygrad schedules them

`schedule.py` builds both models with tinygrad's own operations on its CPU device,
under `TinyJit`, and calls each one three times. It then records the captured list
(`results/schedule.json`): for each kernel, its buffers, operations, reductions and
stored expression, taken from the AST at the same seam Kohaku-TT uses. The weights are
bf16, as the decode program's are, and the activations are fp32.

```
DEV=CPU NOOPT=1 PARALLEL=0 python -I projects/nn/tinygrad/schedule.py \
    --json projects/nn/tinygrad/results/schedule.json --gate gate.npz
PYTHONPATH=src python projects/nn/tinygrad/epilogue.py renders
```

The classes: (a) the item is a kernel the decode program has; (b) it is an expression
the tile compiler renders today, with the tile function's cost by the cost model in
cycles a tile; (c) it only moves data; (d) nothing renders it today. For a matmul kernel,
the "tile function" column renders its stored expression after the reductions (the
epilogue). `epilogue.py renders` gives that column. tinygrad writes sigmoid out as
`1 / (1 + 2^(x * -log2 e))`, and the translator maps it, and `x * sigmoid(x)`, back to
`trace.sigmoid` and `trace.silu`.

No item of either model is class (c). tinygrad folds every reshape, transpose, slice,
the `cat` of the rotary embedding and the cache's read into the kernels' indexing.

**Model A**: `g = silu(x @ w1) * (x @ w3)`, `y = x + g @ w2`, x 32 x 2048. Two kernels
are replayed:

| # | tinygrad kernel | What it computes | Class: what it becomes | Tile function, cycles a tile |
|---|---|---|---|---|
| 0 | `r_32_6144_2048_2048` | both products over K 2048 (bf16 weights cast in the loop), stored `a * 1/(1 + 2^(-a log2 e)) * b` | (a) gate_up, the gate overlay, without the norm's scale: a SPLIT call | `trace.silu(a) * b`, 1539.6 (run below) |
| 1 | `r_32_2048_6144` | `g @ w2` over K 6144, stored `x + acc` | (a) down, the add overlay: a SPLIT call | `a + b`, 150.4 |

**Model B**: one decode token through the layer at cache positions 100 to 102. It is
tinygrad's `TransformerBlock` (`tinygrad/llm/model.py`) with its own `nn.RMSNorm`,
`linear`, `apply_rope`, its fp16 cache and `scaled_dot_product_attention` with GQA, at
the Qwen3-1.7B shapes. The position is a bound variable, so one list serves every
position. Eighteen kernels are replayed:

| # | tinygrad kernel | What it computes | Class: what it becomes | Tile function, cycles a tile |
|---|---|---|---|---|
| 0 | `r_2048` | x's rms: `sqrt(mean(x^2) + 1e-6)` | (d) no sqrt in the stream form, and a reduction over a row of 64 tiles (the row form takes 1 or 2); ours folds the norm into qkv | refused: sqrt |
| 1 | `r_1024_2048` | k = (x / rms * w_norm) Wk | (a) part: qkv's K columns, the norm folded as ours is | an operand stored |
| 2 | `r_2048_2048` | q = (x / rms * w_norm) Wq | (a) part: qkv's Q columns | an operand stored |
| 3 | `r_8_128` | each K head's rms | (d) sqrt, and a row of 4 tiles; ours: qkv's scale operation | refused: sqrt |
| 4 | `r_16_128` | each Q head's rms | (d) as 3 | refused: sqrt |
| 5 | `r_2_8_2_64_2048` | v = x-hat Wv, and k's head norm and rope; both cast to fp16 and stored at start_pos in the cache | (d) sin (cos and sin come from the position inside the kernel), a store at a variable row; ours: qkv's V and k's rope, and attend's append | an operand stored |
| 6 | `E_16_64` | q's head norm times its weight, rope's second half `x2 cos + x1 sin` | (d) sin; ours: qkv's scale operation | refused: sin |
| 7 | `E_16_64` | the same, rope's first half `x1 cos - x2 sin` | (d) as 6 | refused: sin |
| 8 | `r_16_(start_pos+1)_2_64` | scores q K^T / sqrt(128) over the cache (head h reads KV head h / 2) | (a) part: attend's scores; the reduction over a variable K is not lowered | `a * 0.0884`, 118.3 |
| 9 | `r_16_(start_pos+1)` | each head's max over the positions | (d) a reduction over up to 128 tiles (the step form takes 1 to 3 a step); ours: attend's online step | an operand stored |
| 10 | `r_16_(start_pos+1)` | sum of 2^((s - m) log2 e) | (d) as 9 | an operand stored |
| 11 | `E_16_(start_pos+1)` | P = 2^((s - m) log2 e) * 1/l | (d) no reciprocal in the stream form (m and l would be .col() operands) | refused: recip |
| 12 | `r_16_128_(start_pos+1)` | P V over the cache | (a) part: attend's P V | an operand stored |
| 13 | `r_2048_2048` | a Wo + x | (a) o, the add overlay | `a + b`, 150.4 |
| 14 | `r_2048` | h's rms | (d) as 0; ours: gate_up's norm scale | refused: sqrt |
| 15 | `r_6144_2048` | silu((h / rms * w_norm) Wg) | (a) part: gate_up's gate half | `trace.silu(a)`, 1475.6 |
| 16 | `r_6144_2048` | (h / rms * w_norm) Wu times 15's output | (a) part: gate_up's up half | `a * b`, 150.4 |
| 17 | `r_2048_6144` | g Wd + h | (a) down, the add overlay | `a + b`, 150.4 |

**How tinygrad's fusion splits the layer against ours.** tinygrad ends a kernel after
every reduction that a later one reads. Of the decode layer's five calls, only o (13)
and down (17), the products with their adds, are one tinygrad kernel each. In model A,
gate_up and down are one kernel each too. tinygrad's boundaries fall inside the other
three of ours:

- **qkv is 8 kernels (0 to 7):** the norm's rms, each product, each head's rms and the
  rope. tinygrad also fuses across our boundary. Kernel 5 holds V's product, K's rope
  and the cache write, which in ours is attend's append.
- **attend is 5 kernels (8 to 12):** the softmax is three passes over the scores, which
  go to DRAM and back between kernels (16 x 4095 fp32). Ours is the online step, one
  pass over the cache.
- **gate_up is 3 kernels (14 to 16):** the rms, then the `.contiguous()` that tinygrad's
  own model puts after the silu, which splits gate from up. The 6144-wide hidden vector
  goes to DRAM and back. Model A has no such `.contiguous()`, and there it is one
  kernel, as ours is.

None of our kernels is smaller than one of tinygrad's. As a list, the layer is 18
entries against the decode program's 5. At the list's measured 3.99 us a call when
closed (1.89 of it the command; `../README.md`, "What an entry costs"), the 13 extra
entries are about 27 us of the thread's own time a layer. That does not count
their trips through DRAM, nor the kernels behind them, which today are not ours.

## One of tinygrad's kernels on the card

Model A has no class (b) item of its own: both of its kernels are products. The
smallest expression in it that the tile compiler renders and that is not a decode
kernel is kernel 0's epilogue, tinygrad's `a * 1/(1 + 2^(-a log2 e)) * b`. It runs as a
program of its own in its own session: `epilogue.py card`, kernels in `kernels/`. The
form is the tile experiments' (`experiments/language/tile_rules`): a stream of two operand tiles
in from DRAM, the emitted body, and one tile out. The operands are tinygrad's own
products a = x W1 and b = x W3, 32 x 6144 fp32 (192 tiles), each computed by its own
tinygrad kernel. Fed to tinygrad's elementwise kernel, they give the fused kernel's
output bit for bit. The tile function is what the translator makes of the stored
expression, `trace.silu(a) * b`. `build` emits its header and builds the program on the
mock device into the kernel cache that the card job reads, so the job compiles nothing.

Card 0, one core, the program kept and run 4 times (`results/epilogue_card.json`):

| | |
|---|---|
| against float64 | max rel 2.40e-7, mean 5.2e-8 (tolerance 1e-6, the tile compiler's gate) |
| against tinygrad's output | max rel 3.23e-7, mean 3.9e-8 (tinygrad's own: 2.15e-7 of float64) |
| the body, a tile | 1568 cycles less the empty span (the cost model: 1539.6) |
| the stream, 192 tiles | 355.5k cycles on the math CPU, 1851 a tile, 263 us at 1350 MHz |
| a call, run to finish on the host | 2.30 ms (2329, 2302, 2297 us; the first, with the load, 24.2 ms), under the environment's slow dispatch |

The 4 calls give the same errors to every digit. This proves the path from a tinygrad kernel, to
a tile function, to the card. tinygrad's own epilogue runs inside its product, so in the
list it belongs inside the gate_up call's overlay. It stands alone only where tinygrad
stores an elementwise kernel of its own (model B's 11, once the stream form has a
reciprocal).

What the list entry for it will need:

- The executor places a second program beside the decode program. The entry's header
  names a program, but the executor runs only the one its session loaded.
- The program becomes a handler that takes a, b, y and the tile count from the
  command's words (a0 to a9), as SPLIT does, instead of each kernel's runtime arguments.
  The nonce and the info page are the experiment's and go away: the entry reports by its
  event.
- Its cores in the entry's cores word, with the tiles split among them. One core takes
  263 us for 192 tiles; spread over 120 cores, that is two tiles a core.
- a, b and y as DRAM tensors that the host package holds in the tile layout the
  neighbouring calls use. Here a tile is any 1024 consecutive values, because the
  expression is elementwise.

## What a tinygrad front end needs

From the host package:

- An op table from a kernel AST to calls, planned once per AST key, as Kohaku-TT's
  `PLANS` does. A product kernel becomes a SPLIT call with the overlay that its
  epilogue names: none, add, gate, silu, scale. Any other kernel becomes a
  tile-compiled program.
- The epilogue translator (`epilogue.py`) as part of it. Two refusals have to be added
  to the stream form or rewritten away: reciprocal and sqrt (kernel 0's sqrt is read
  only as `1/rms` by 1, 2 and 5, so it is an rsqrt across the boundary). Sin is needed
  for tinygrad's rope.
- A grouping pass over the schedule that recognizes our fused kernels: qkv from
  tinygrad's 8, attend from its 5, gate_up from its 3. Without it, a layer is 18
  entries.
- tinygrad's buffers as DRAM tensors in our layouts: bf16 weights in SPLIT's layout,
  and activations as rows of tiles. The JIT's scratch buffers are fixed for its life,
  so a replay is a closed list built once.
- Bound variables (start_pos) as entry words that the host changes per call. tinygrad's
  fp16 cache has to become our bf16 cache, or else be a cast.

From the executor:

- Entries of a second program: the header's program field honoured.
- A handler form for tile-compiled programs: operands from the command's words.
- A replayed closed list whose variable words (the position, the cache length) are read
  from one DRAM slot that the host writes once per step, instead of rewriting entries.
