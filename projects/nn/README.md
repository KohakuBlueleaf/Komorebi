# nn: a list of kernel calls and one X280 thread that runs it

The first piece of a small network runtime on Komorebi: the host describes work as a
list of kernel calls in DRAM, and one generic X280 thread walks the list and submits
each call, so the host is not on the path between two kernels. The question is what a
call costs this way against the host submitting every call itself, and whether a real
model runs through it as fast as the hand-written thread that knows the model.

The calls here are the decode project's own kernels (`../decode`): its program, its
plans and its words, called and not copied. The thread knows no model.

## The entry

A list is entries of 128 bytes (two 64-byte lines) in a region of the thread's DRAM
bank, each one kernel call (`nn.h`). The first line is the split's chain record
(`../dense/kernels/split/chain.h`) as it is, so a run of entries is passed to the chain
unchanged; the second line is the thread's.

| Words | Line | What |
|---|---|---|
| 0 to 9 | record | the command's a0 to a9 |
| 10 | record | its cores: first \| count << 16 |
| 11 to 14 | record | the chain's info word and three range words (a chain's record only) |
| 15 | record | the next record of the same chain, its bank address; 0: the chain's last |
| 16 | thread | the header: handler \| flags << 8 \| program << 24 |
| 17 | thread | the kind: COMMAND, CHAIN, END, NONE, TRANSFER (a command moving pages to or from the host channel, its nonce word XORed with RUN's a3) |
| 18 | thread | the list's next entry, its bank address; 0: the end |
| 19, 20 | thread | a chain: its command's a4 (the kernel's chain mode) and its cores |
| 21 | thread | the operation overlay the call names (kept for the host; the kernel reads its own word, SPLIT's a6) |
| 22, 23 | thread | the sequence (an open list's number; 0 in a closed list) and the check |
| 24, 25 | thread | a chain: its command's a0 (the token's word, ORed into the word a record's info names) and a1 (`kChainAheadOff` or 0) |

The chain's next (word 15) and the list's next (word 18) are two words because a chain
ends where the list goes on. The check is the sum of word i x (2i + 1) over the entry,
words 18 and 23 left out, modulo 2^32.

## The thread

`thread/executor.c`, on work cluster 2 beside the decode's own thread on cluster 1 (one
command ring each; the same program, so both run in one session). For each entry it
builds the command from the words (a COMMAND), or one command in the chain's mode whose
block is the entry's first line (a CHAIN, the records after it read by the cores), then
submits it and waits for its event. It is driven three ways:

- **Closed list** (`run`: a0 the first entry, a1 the times, a2 a stats slot, a3 the
  nonce of its transfers): the host
  writes the whole list once with a DRAM barrier, the thread drops its cached lines of
  the region once and walks the list from the first entry to a next of 0, N times.
- **Open list** (`drain`: a0 an anchor entry, a1 a stats slot, a2 the ticks it waits
  idle at most, a3 the open list's number): the host appends entries while the thread
  runs; the thread polls the next word of the last entry it ran (the line dropped from
  its cache, read again), then reads the linked entry, and a chain's records, until their
  sequence and check hold, runs it, and polls its next. An END entry ends the command, so
  the host waits only when it asks for a result.
- **The host's calls**, for comparison: the host submits each call and waits for it
  (`run.submit`, `run.wait`), as every example does.

Each run leaves a stats slot (its status, entries, polls, stale reads, each term's
ticks, and each entry's command ticks by its index; a command not done leaves its
event's kind | reason << 8 there), read after `run.stop()`.

## The host calls (the interface)

`nn.py`:

- `Call(handler, cores, args, flags, program, overlay, record)`: one kernel call; `record`
  the chain's four words when the call can be a chain's record.
- `Entries(first)`: a list from entry `first` of the region. `command(call)` appends a call
  as one command; `chain(calls, mode, flags, token, ahead)` a run of calls as one chain
  command (a0 `token`; a record's weight read in the record before with `ahead`).
  `words(base, sequence)` gives the entries' words with the links and the checks.
- `executor_thread(program, cluster=2)` adds the thread to any program.
- `Executor(session, run)`: `write(entries)` (the host's DRAM write and a barrier),
  `closed(entries, times)` (one `run` command, waited for: the thread's ticks and the
  slot), `open(at, idle, fence)` (an `OpenList`), `stats()` after the stop.
- `OpenList`: `append(call)`, `append(chain=calls, mode=, flags=)`, `sync()` (an END,
  then the event).
- `drive(run, calls)`: the host's own calls.
- `layer_calls(decode, layer, position, rows)`: the decode layer's five calls (qkv,
  attend, o, gate_up, down) with their chain records, from `Decode`'s words and tensors,
  as the decode thread builds them; `token_calls(decode, layers)` a token's records
  through L layers, the position from the chain's a0; `decode_entries(...)` a closed
  list of T tokens of L layers (a command a call, a layer or a token as one chain); `nop_call(decode, cores)` the decode program's cheapest call. `Decode` is the
  decode project's with the executor added.

## Run

```
python projects/nn/bench.py --check --costs --decode --json out.json
python projects/nn/bench.py --layers 1 --check --costs
```

`--check` runs one token through layer 0 at T 4096 eight ways from the same x (the
decode thread a command a step, a layer and a token as a chain; the list closed the
three ways; the open list; the host's calls) and compares qkv, a, x1, gated, x and the cache's new row with the first,
bit for bit. `--costs` times 200 cheapest calls a list on one core and on all 120,
`--decode` 10 tokens of 28 layers at T 4096, 1024 and 256: the list a command a call,
a layer and a token as one chain (the token also with the weight read a record early,
at T 1024 and up), the decode thread's token, the host's calls.

## The check

Every tensor and the cache row of every way equal the decode thread's commands bit for
bit, at T 4096 (before and after the removal below).

## The decode layer through the list

Qwen3-1.7B, 28 layers, random weights as the decode bench's, 10 tokens a run; us a
layer, from the thread's clock (the host's for the host's calls); the hand's columns are
the decode project's `results/qwen3-1.7b-chain-x10.json`. Run 2026-11-15 18:18 on one
p150a, AICLK 1,350 MHz (one sample of 1,343), host load 1.2, `results/qwen3-1.7b-list-x10.json`.

| | T 4096 | T 1024 | T 256 |
|---|---:|---:|---:|
| (a) closed list, a command a step | 410.94 | 373.50 | 364.85 |
| hand, a command a step | 410.26 | 373.00 | 364.22 |
| (a) its commands' events | 399.19 | 361.80 | 353.09 |
| hand's events | 399.02 | 361.78 | 353.00 |
| (a) the thread (5 entries) | 11.75 | 11.70 | 11.76 |
| (a') closed list, the layer as a chain | 347.28 | 318.22 | 307.92 |
| hand, the chain | 356.57 | 326.41 | 317.61 |
| (a') the chain command | 344.74 | 315.69 | 305.38 |
| hand's chain command | 345.42 | 315.27 | 306.47 |
| (a') the thread (1 entry) | 2.54 | 2.53 | 2.54 |
| hand's thread, the chain | 11.15 | 11.14 | 11.14 |
| (c) the host's calls | 1,168.05 | 1,148.62 | 1,115.85 |
| (c) its host submit and wait less the events | 767.5 | 785.0 | 761.3 |

The hand's run-to-run spread over its recorded runs is 0.1 to 0.3 us a layer for the
chain and 0.6 to 1.6 for a command a step. (a) is the hand within it: its events within
0.2, its thread's 2.35 us a command (fetch 0.55, submit 0.51, wake 1.01, the rest 0.28)
against the hand's 2.26. (a') is 8.2 to 9.7 us under the hand: the chain command takes
the same time (within 1.1), and the thread's share is 2.5 us where the hand's is 11.1,
most of which is the hand writing the five records itself before each layer; here the
host wrote them once. (c) runs the same events and spends 153 to 157 us a call on the
host's submit and wait.

### A token as one command

The decode chain now takes up to 256 records and a word of the chain command (a0) that a
record's info names (qkv's and attend's position), so a token's 140 records are one
entry of the list; the program's weight halves are the chain's deeper ones. Run
2026-11-15 18:55 on one p150a, AICLK 1,343 to 1,350 MHz, beside the decode thread's own
token in the same job; us a layer, the host's wall of one run of 10 tokens x 28 layers
(the run's stats were not read: its last case failed, below), the decode thread's from
its clock.

| | T 4096 | T 1024 | T 256 |
|---|---:|---:|---:|
| list, a command a step | 416.1 | 378.9 | 370.0 |
| list, a layer as a chain | 353.2 | 327.5 | 318.9 |
| list, a token as a chain | 341.1 | 308.2 | 300.0 |
| decode thread, a token (its clock) | 339.95 | 307.53 | |
| decode thread, a token (the host's wall) | 341.1 | 309.3 | |
| list, a token, the next weight early | 316.1 | 287.9 | |
| decode thread, the same (its clock) | 314.92 | 288.12 | |
| the host's calls | 1,175.2 | 1,147.2 | |

The list's token is the decode thread's to 0.1 us at T 4096 in the same wall (one
submit and one wait a token either way). Every tensor and the cache row of the eight
ways of the check equal bit for bit. With the next weight early the chain hangs at T
256, on the decode thread as well (the decode project's README); the bench runs that
form at T 1024 and up.

## What an entry costs

200 calls of the decode program's cheapest kernel, SPLIT in its operation mode with no
operation (each core of the call enters the handler and returns), on one core / on all
120. us a call, same run as above.

| Term | (a) closed | (b) open, fenced | (b) open, checked | (c) host |
|---|---:|---:|---:|---:|
| host: building the entry | | 14.0 / 13.9 | 12.4 / 12.4 | |
| host: the entry's write | | 4.1 / 4.1 | 4.0 / 4.1 | |
| host: the DRAM barrier | | 51.7 / 51.9 | | |
| host: the link's write | | 4.2 / 4.3 | 4.1 / 3.8 | |
| host: submit | | | | 11.8 / 11.7 |
| host: wait, less the command | | | | 126.7 / 126.6 |
| thread: polls for the link (idle) | | 71.8 / 71.5 | 18.0 / 17.5 | |
| thread: reading the entry until new | | 1.3 / 1.3 | 1.3 / 1.3 | |
| thread: fetch and build | 0.31 / 0.31 | 0.27 / 0.27 | 0.27 / 0.27 | |
| thread: submit | 0.51 / 0.51 | 0.52 / 0.52 | 0.51 / 0.51 | |
| the command (the dispatcher's submit to completion) | 1.89 / 2.25 | 1.94 / 2.30 | 1.89 / 2.27 | 1.86 / 2.26 |
| thread: the wake (the wait less the command) | 1.01 / 1.05 | 0.98 / 1.00 | 0.97 / 1.02 | |
| thread: the rest | 0.27 / 0.28 | 0.32 / 0.32 | 0.32 / 0.32 | |
| a call | 3.99 / 4.40 | 76.7 / 76.8 | 23.1 / 23.0 | 140.6 / 140.9 |

(a) is the replays of a written list (5 x 200); its first run fetches the dropped lines,
0.54 us an entry, and takes 4.36 / 4.52. In (b) the host appends as fast as it can and
the thread follows: a call takes the host's append, and the thread waits in polls of
about 1.03 us each (a line dropped and read from DRAM). Not one entry was read before it
was new. From the host's link write to the kernel's start, summed from the terms (the
host's and the thread's clocks are not one, so this is not measured end to end): the
write 4.1, half a poll 0.5, the entry's read 1.3, fetch and submit 0.8, and the
dispatcher's start, a part of the wake's 1.0: 6.7 to 7.7 us.

A sync and a read back on the host after the last entry: the open list's END and its
event 131 to 190 us; one tile of x read 213 to 239 us, x's 64 tiles 360 to 409 us. A
closed list's write: 200 entries in 0.22 to 0.30 ms.

## What was removed and what it gave

The largest term on the package's own paths was the open list's DRAM barrier, 51.7 us
of an append of 76.7 (the barrier waits on all eight DRAM channels): it kept the thread
from reading an entry before it was in DRAM. It is gone: each entry carries its open
list's number and a check, and the thread, once it sees a link, reads the entry (and a
chain's records) again until both hold, so the host writes the entry and the link with
no barrier between them. An append went from 76.7 to 23.1 us, a sync from 190 to 132 us;
the thread's read of a linked entry until it is new, 1.3 us, is the same in both forms.
The largest term of all is still the host's own wait in (c), 127 us
a call above its command; the lists exist to keep it off the path between kernels. In
the open list the next largest is now the host's building of an entry in Python (12.4
us), then its two writes (4.0 and 4.1 us).

## What the next packages take from this

- The entry (the table above, `nn.h`): a call is ten words, its cores, its header and an
  overlay note; a chainable call also carries its chain record words. An op table maps an
  operation to a `Call` (or a run of them for a chain) from its tensors and shapes, as
  `layer_calls` does for the decode layer.
- The host calls: `Entries` for a closed list built once and run N times (a compiled
  graph), `OpenList.append` / `sync` for eager calls, `drive` to compare.
- An open list's entries take a slot each of 2,048 in the region, in order; a new open
  list starts at a new anchor. A closed list is written whole before its run.
- Not here: placing a program that is not on a call's cores (the entry names its program
  in the header; the decode program is one program), several calls in flight, a ring
  that reuses the region's slots, a value read on the host without stopping the list.

## torch on the list

The package's torch side, in four files; torch is not a dependency of the framework
(nothing under `src/` imports it).

- `table.py` (no torch): the `Tensor` (shape, its DRAM tensor or the host's values,
  whether a call not yet run makes it, its rows' sums of squares when a norm after it
  needs them); the op table, a `Descriptor` a kernel call, each planner building a
  `Call` from the decode program's own plans and words; the `Device` (the decode
  program with the executor and one more plan, an open list the calls are appended to,
  synced only on a host read, the fallbacks counted, the host's time by term); the
  chain of a list's calls.
- `front.py`: the custom op `komorebi::attend`; the eager tensor (`CardTensor`, a
  `__torch_dispatch__` subclass); the torch.compile backend `komorebi`.
- `models.py`: model A and model B as torch modules.
- `torch_bench.py`: both models eager and compiled against torch float32 on the CPU.

### The op table

| Descriptor | Serves | Kernel | Shapes |
|---|---|---|---|
| `matmul` | `a @ W` | SPLIT, full, no operation | a [r, K] with r <= 32, W [K, N] of a plan |
| `matmul_add` | `r + a @ W`, and the sum's squares | SPLIT, full, add | as matmul, N = 2048 (a plan of 8 groups) |
| `norm_gate` | `silu(norm(x) Wg) * (norm(x) Wu)` | SPLIT, full, gate | x [r, 2048] with its squares, W' [2048, 2M] |
| `norm_matmul` | `norm(x) @ W` | SPLIT, full, scale | x [r, 2048] with its squares, W' [2048, N] |
| `norm_qkv` | `norm(x) @ [Wq Wk Wv]`, q's and k's heads normed and rotated | SPLIT, full, scale with the heads | x [1, 2048] with its squares, W' [2048, 4096] |
| `attend` | `komorebi::attend`, by name | SPLIT, its operation mode, attend | qkv [1, 4096], caches [8 x 4096, 128] |

Every tensor is bf16 in tiles; a call takes one tile row (the rows that hold values are
its rows field). The tiny ops of the table are the product alone. The add, the gate and
the scale exist only after a product (they work on the product's sum as it leaves the
cores), and the operation mode alone has no sum to work on; so `a + b`, `a * b`, a
scale, silu and the norm's pieces have no descriptor and run on the host (nothing new
was written for them). The decode layer's plans hold 2048 x 6144 only as gate and up
side by side; `x @ w1` alone takes a sixth plan, made by the decode project's planner
with its constants (groups of 5 cores, 24 groups) and added to the program's table of
plans before the load.

### Eager

`card_module(module, device)` puts every parameter and buffer on the card;
`to_card(x, device)` an input. An op with a descriptor is planned and appended to the
open list and returns at once; nothing waits until a host read (`.numpy()`, `.item()`,
printing). An op without one, or whose shapes the descriptor does not take, is a
fallback, counted by op: the open list is synced, its inputs read, torch runs it on the
CPU in float32, and its output is written back to the card (a [32, 1] mean, which has
no tile form, stays on the host).

This is Python dispatch on a CPU tensor's key. A real device key (as Kohaku-TT's: its
backend renamed from PrivateUse1, a device module, the generated tensor methods, and
aten kernels in C++) would change three things: a tensor would be on a device torch
knows (`.to("...")`, its storage and allocator the card's), so no wrapper and no host
copy of the metadata; the dispatch, the planner and the entry's build would run in C++
instead of Python per op (here 1.7 to 4 ms a call of the times below); and the
fallbacks would become torch's own CPU fallback with a copy each way, as here.

### torch.compile

`torch.compile(module, backend="komorebi", options={"device": device, ...})`. The
backend goes through AOT autograd's inference path to an aten graph and maps it onto the
same op table, in stages, each switched by an option:

- `fuse`: fusion rules as data (`front.RULES`): a pattern written as a torch function
  (the same functions the modules call), traced to aten once, matched in the graph, and
  what a match becomes (descriptor calls, the weights prepared once: the norm's weight
  folded into W, the gate's and up's columns in turns of 4 tiles). Three rules: norm,
  q k v, the heads and attend -> `norm_qkv` + `attend`; `r + a @ W` -> `matmul_add`;
  norm and the gated MLP -> `norm_gate`. Off: every product is a `matmul`, the rest
  host ops.
- `once`: every tensor between calls is given its DRAM tensor at the first call and
  kept, and each segment's closed list is written then; off, both every call.
- `chain`: runs of calls where each takes the call before's output become one chain
  entry; the rest one command each.
- `joint` (the removal below, off by default): an input a norm reads and its squares
  as one DRAM tensor, written in one host write.

The compiled function writes the inputs (with their squares when a norm reads them),
runs each segment's closed list, runs the host ops between segments (their inputs read,
their outputs written when a later call takes them), and reads the outputs. The first
call traces, prepares and writes the weights and the lists: 0.2 to 1.2 s.

### Model A: the MLP half

`y = x + (silu(h w1) * (h w3)) w2` with `h = x * rsqrt(mean(x^2) + eps) * wn`, x 32 x
2048, w1 and w3 2048 x 6144, w2 6144 x 2048, the decode bench's random weights. The norm
is part of the model because the card's only fused gate is the decode layer's, which
applies the norm's 1 / rms. Right against torch float32 on the CPU (the bound fixed
before the runs, the decode project's: rms error / rms <= 2e-2, largest error / largest
value <= 5e-2): eager and without fusion rms 2.46e-3, largest 5.9e-3; fused and chained
3.18e-3, 7.5e-3. One p150a, AICLK 1,350 MHz, 8 calls after the first, us a call
(`results/torch-model-a.json`):

| Term | eager | compile | fused | fused + chain |
|---|---:|---:|---:|---:|
| input written (x) | 343.6 | | 359.2 | 356.5 |
| its squares written | | | 259.5 | 240.5 |
| dispatch (its own Python) | 2,568.8 | | | |
| planner | 23.5 | | | |
| tensors given | 194.5 | | | |
| entry built | 60.7 | | | |
| append (store, link) | 35.0 | | | |
| open list started | 413.3 | | | |
| sync | 428.5 | | | |
| a closed list run (its RUN, the card's time in it) | | 616.7 | 414.5 | 401.6 |
| fallbacks / host ops: reads | 2,411.3 | 1,332.1 | | |
| fallbacks / host ops: torch on the CPU | 439.0 | 921.9 | | |
| fallbacks / host ops: writes back | 2,560.9 | 1,013.8 | | |
| output read | 307.8 | (in the reads) | 314.3 | 312.2 |
| the rest (Python around them) | 502.2 | 387.4 | 346.5 | 340.2 |
| **a call** | **10,289.1** | **4,271.9** | **1,694.1** | **1,651.0** |
| the card: the commands' events | 207.0 | 206.5 | 213.6 | 203.2 |
| the executor's thread (idle within it) | 347.5 (124.1) | 214.1 | 218.5 | 206.1 |
| lists a call | 3 open | 2 closed | 1 | 1 |
| fallbacks a call | 9 | 9 | 0 | 0 |

Beside them: the decode thread's own gate_up and down, a command a step, 131.8 + 72.6 =
204.4 us of events (T 4096); torch float32 on the CPU, 16 threads at nice 10, 7,535 us a
call (for scale). The card's time is the hand's in every way; what a call adds is the
host's. The chain gains 10 us here: gate_up is its first record, so no record before it
reads its weight early.

### Model B: the decode layer

The decode layer for one token as a torch module (`models.ModelB`): norm, q, k and v, q's
and k's heads normed and rotated in torch, `komorebi::attend` over the cache (the token
at row 4,095 of 4,096), o with its add, norm, the gated MLP, down with its add; the
model's own weights of layer 0 (bf16), a random cache. Right: eager and without fusion
rms 6.25e-3, largest 1.53e-2; fused and chained 4.67e-3, 7.4e-3. us a call
(`results/torch-model-b.json`):

| Term | eager | compile | fused | fused + chain |
|---|---:|---:|---:|---:|
| input written (x) | 417.3 | | 380.1 | 367.7 |
| its squares written | | | 239.9 | 233.3 |
| dispatch (its own Python) | 3,956.6 | | | |
| planner | 60.2 | | | |
| tensors given | 441.1 | | | |
| entry built | 169.0 | | | |
| append (store, link) | 95.5 | | | |
| open lists started | 965.2 | | | |
| sync | 1,027.1 | | | |
| closed lists run | | 1,223.9 | 617.4 | 552.5 |
| fallbacks / host ops: reads | 5,797.2 | 2,587.5 | | |
| fallbacks / host ops: torch on the CPU | 551.7 | 1,668.8 | | |
| fallbacks / host ops: writes back | 5,332.5 | 1,911.2 | | |
| output read | 367.2 | (in the reads) | 312.1 | 307.9 |
| the rest | 1,422.9 | 436.6 | 221.6 | 214.4 |
| **a call** | **20,603.4** | **7,828.1** | **1,771.0** | **1,675.8** |
| the card: the commands' events | 421.1 | 420.5 | 406.4 | 349.9 |
| the executor's thread (idle within it) | 932.7 (471.0) | 440.4 | 418.4 | 352.8 |
| lists a call | 7 open | 4 closed | 1 | 1 |
| fallbacks a call | 45 | 45 | 0 | 0 |

Beside them, the list's own decode layer on the same program (above, us a layer at T
4096): a command a step 416.1, the layer as a chain 353.2, the token as one 341.1; the
decode thread's token 339.95. Before the chain's change: a command a step 410.9, the
layer as a chain 347.3. Fused, the compiled layer's five commands take what the list's
take; with the chain, its one command takes the list's layer chain (349.9 against
353.2 with its thread). torch float32 on the CPU: 11,408 us.

The fallbacks, by op, a call: eager and compiled without fusion, add 8, mul 13, pow 4,
mean 4, rsqrt 4 (the two norms and q's and k's head norms), slice 4 and cat 3 (the
rotation, and q, k, v side by side), view 4, silu 1. With fusion: none.

### What was removed and what it gave

- Eager: the largest term was the fallbacks' traffic over the host channel (model B:
  reads 5,797, writes 5,333 us a call): a fallback's output went back to the card and
  the next fallback read it again. Now the host keeps a fallback's output
  (`Device(mirror=True)`) and writes it to the card only when a call with a descriptor
  takes it. Model A 10,289.1 -> 6,309.3 us a call (reads 2,411 -> 1,336, writes 2,561
  -> 938); model B 20,603.4 -> 12,830.0 (reads 5,797 -> 2,503, writes 5,333 -> 1,870);
  both right as before (`results/torch-removals.json`). Next: the dispatch's own Python
  (1.7 / 3.7 ms), then the reads left (a product's output each fallback after it needs).
- Compile (fused and chained): the largest term was the input's write (356.5 / 367.7)
  with its squares written apart (240.5 / 233.3). Now x and its squares are one DRAM
  tensor (the squares' pages a whole round of the eight banks after x's 64), written in
  one host write: 597.0 -> 566.3 us (model A), 601.0 -> 576.1 (B); a call 1,651.0 ->
  1,547.1 and 1,675.8 -> 1,580.2, right as before. The write was not a fixed cost a
  write: 72 pages take what 64 and 8 took apart, less 31 us. Next: the host channel's
  write itself (566 us for 144 KB), then the RUN's round trip above the card's time
  (195 to 200 us), then the output's read (310 us).

### One submission a call

The compiled call above goes to the card three times: the host writes x and its
squares, sends RUN and waits for its event, then reads y. In the stage `one` the list
moves the pages itself, so a call is one host submission:

- The list's first entry is the transfer that pulls x's pages (and its squares', one
  DRAM tensor) from the session's region of the host channel; its last entry is the
  transfer that pushes y's pages into the region. They are the same PULL and PUSH a
  session's write and read send, issued by the executor thread as entries.
- The host copies x's pages into the region (memory, no card call), sets RUN's nonce,
  submits RUN once and waits for y's sequence words in the region (host memory), then
  copies y out. RUN's own event is collected after the call; a call whose words do
  not arrive in time reads it and raises with it.
- Everything else (the entries, the transfers' words but the nonce, the views of the
  region) is made at compile.
- The dispatcher takes a thread's PUSH or PULL only when its region word is the
  session's region, which the load hands to the dispatcher (0: no channel); with any
  other word it refuses it as a handler it does not serve (reason 1). The kernel's own
  bound keeps the offsets inside that region.
- A transfer's wait for its sequence words now looks without a pause for its first
  300 us, then every 200 us as before (every session transfer has it).

Measured on one card (2026-11-15, 8 calls a model, `results/torch-one-submission.json`),
the call before (stage `joint`, two host writes in one, RUN, a read) and after, both
right (model A rms 3.18e-3, B 4.67e-3, as before). Microseconds a call:

| Term | A joint | A one | B joint | B one |
|---|---:|---:|---:|---:|
| x and its squares as pages (Python) | 608.7 (with the write) | 440.2 | 573.1 (with the write) | 363.0 |
| copy into the region | | 18.8 | | 19.2 |
| RUN: submit and store | 405.6 (to its event) | 58.7 | 553.1 (to its event) | 57.2 |
| the wait for y's words | | 281.6 | | 567.7 |
| y read and untilized | 310.6 | 13.6 + 93.8 | 310.7 | 13.2 + 88.4 |
| Python of the call | 246.5 | 92.5 | 136.5 | 94.4 |
| a call | 1,571.5 | 999.2 | 1,573.4 | 1,203.1 |

What is the card in a one-submission call (the thread's clock): A 237.6 us, the pull
14.2, the commands 203.6, the push 12.3, the thread's own 7.5; B 386.0 (pull 14.4,
commands 351.7, push 12.4). What is the system: the rest.

Where a host command's wait above its command went (127 us in "What an entry costs",
about 200 above RUN here): the host does not wait on the device. A wait (the run's,
then the host module's) stores the command's 64-byte line and the read count through the
write-combining window into the dispatcher's memory, one fence, then spins on the event
ring and the state word, both in pinned host memory; no device read, no sleep, no poll
interval. The time lies between the store and the event's return through the
dispatcher. In a one-submission call that path is gone from the call: in A's wait of
281.6 us the thread runs 237.6, so 44 us remain from the host's store to the thread's
first entry and from the push's last word to the host seeing it; the submit itself
(Python, the store) is 58.7. The one sleep on a host command's path was the sequence
words' wait (200 us a look): a plain write or read of 72 pages went from 261.0 / 262.6
us (mean; medians 221.5 / 224.5) to 214.6 / 219.5 (210.9 / 216.7) with the spin, of 64
pages 211.6 / 215.6 to 211.4 / 216.2 (the words most often land before the event).

Then four removals, in order, each measured on the card (8 calls a model, both right
as before; `results/torch-one-submission-spin.json` for the first three,
`results/torch-one-submission-removals.json` for the copies). Microseconds a call:

| Term | A first | A after | B first | B after |
|---|---:|---:|---:|---:|
| x and its squares into the region | 440.2 + 18.8 | 184.3 | 363.0 + 19.2 | 131.8 |
| RUN: submit and store | 58.7 | 13.5 + 36.3 + 8.5 | 57.2 | 13.2 + 35.6 + 7.4 |
| the wait for y's words | 281.6 | 281.6 | 567.7 | 428.9 |
| y read and made a tensor | 13.6 + 93.8 | 73.1 | 13.2 + 88.4 | 72.2 |
| Python of the call | 92.5 | 8.8 | 94.4 | 8.7 |
| outside it (torch's wrappers, the bench) | (in the above) | 80.4 | (in the above) | 70.6 |
| a call | 999.2 | 686.6 | 1,203.1 | 768.4 |

- The pages of x and y. The slow part was this project's own path, not the framework's
  formats (whose rounding, tiling and untilizing are whole-array already): x went from
  torch to float numpy, was rounded to bf16 again and joined with its squares in float
  before tiling. Now x's own bf16 bits are tiled into a host copy, the squares made from
  the bits (a bf16 product is exact in float32, the sum is taken in float64: the same
  bytes, compared on 300 random inputs of 32, 7 and 1 rows), and the copy goes into the
  region whole; y's pages come out whole and their bits become the bf16 tensor. A
  strided numpy copy straight into or out of the channel's memory was slower than the
  old path (y 113 us): a copy of the channel's memory is made whole, then tiled.
- The long spin. The call waits for y's words spinning 5 ms before it sleeps (a plain
  transfer still spins 300 us): B's last sleeping look cost 134.8 us (921.3 with the
  short spin against 786.5); A's call is under 300 and does not change.
- The submit. What a call built (RUN's handler number and words) is made at compile;
  a call sets the slot and the nonce. Of the 58.7: the queue 13.5, the hand of the
  command to the dispatcher in Python 36.3, the native store 8.5.
- Other Python (92.5 first). It is mostly outside the call, about 75 us: torch's
  compile wrappers (dynamo and AOT, 17 us a call measured off the card), the bench's
  conversion of y to float numpy (18), and the bench's loop. The call's own is 8.8, its span computation now done after the
  calls.

What is left, largest first: the card's own time in the wait (A 238 of 282, B 384 of
429; the 44 us between them is the host's store to the thread's first entry and the
push's last word to the host seeing it); tiling x on the host (A 184, B 132; 130 off the
card); y's 64 pages out and untilized (73); the hand of RUN in Python (36).

### What a kernel had to provide by hand

- A plan for each product shape in the program's table (at most 8, fixed at the load):
  a shape not planned before the load cannot run; `x @ w1` alone needed a sixth.
- Its weights in the operation's layout: the norm's weight folded into W (diag(w) W),
  the gate's and up's columns in turns of 4 tiles a block, q, k and v side by side.
  The fusion rule prepares them once from the module's own parameters.
- The rows' sums of squares beside its input: made only by an add after a product, or
  by the host for an input; a norm before a product cannot run without them. The add
  only where the product's plan has 8 groups (N = 2048): its roots write a tile each.
- The operations only after a product: no element-wise kernel stands alone, so the
  norm's pieces, silu, a residual add without its product and the rotation fall back.
- The heads' norms and rotation: their weights in a slot of the qk tensor (a slot a
  layer, written once), the rotation's table for the cache's rows, epsilon compiled
  in; the position and the slot in a word. The rule checks the module's cos and sin
  against that table.
- attend: the cache's layout (heads x rows), one token's row, q, k and v as one
  tensor with q and k already normed and rotated, and its own write of the token's k
  and v into the card's cache (a side effect torch does not see; the same row each
  call here).
- For the chain: each call's record words (sends, marked, what it reads of the
  output before: its K range or attend's three ranges, the gate's output width), and
  only where each call takes the one before's output.
- The rows that hold values, in a word of every call.

### Run

```
python projects/nn/torch_bench.py --prepare --data ref.npz --snapshot <model snapshot>
python projects/nn/torch_bench.py --data ref.npz --snapshot <model snapshot> --models a b
python projects/nn/torch_bench.py --data ref.npz --snapshot <model snapshot> \
    --ways eager joint --mirror
python projects/nn/torch_bench.py --data ref.npz --snapshot <model snapshot> \
    --ways joint one --channel
```

`--channel` first measures a plain write and read with the old wait and with the spin,
and the dispatcher's refusal of a thread's transfer outside the session's region.

`--prepare` runs off the card: the inputs, torch float32's outputs and its CPU time.
