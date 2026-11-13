# Remote loading: one core programs another

Whether a kernel body can be put into a Tensix core's slot by ANOTHER core, and what
it costs: who decides (another Tensix core, or an X280 core sending a message) and who
moves the bytes (the target itself from DRAM, another Tensix core, or the X280 core).
It builds on [overlay_probe](../overlay_probe/README.md) (a core loading itself): the
same builder, the same chain bodies, the same slot rules.

## Results

One Blackhole p150a, slow dispatch, AICLK 1350 MHz (driver, every run), X280 at the
session's 1000 MHz, host load about 7. Tensix times are the cores' wall clock; X280
times are its 50 MHz timer. Every result is checked on the host against a reference:
the body's result, the target's call counter, the body and argument named.

| Question | Answer | Number |
|---|---|---|
| A Tensix core stores a body into another's slot, then sends it a message | Yes: 1,000 swaps exact; the target invalidates its own i-cache and calls | store 0.27 µs (4 KiB), 0.47 µs (20 KiB), 1.20 µs (80 KiB): 299 cycles + 16.6 a KiB; command to the target's result 1.62 µs (3.5 KiB body, 0.89 µs of it the body) |
| One Tensix core feeds many | Yes: every target exact | 20 KiB to 8 cores: stored in 2.2 µs, last result 8.0 µs; to 119 cores: stored in 29.2 µs (83 GB/s), last result 42.3 µs |
| An X280 core decides, the target loads itself | Yes, as a handler of the framework's persistent kernel, no framework change: 1,000 swaps exact; 12 X280 cores to 120 Tensix cores, each choosing from its own schedule, 12,000 swaps exact | 3.48 µs a swap on one core (load 0.40 µs, body 0.95 µs, two port messages the rest); 120 cores: 6.2 M swaps a second |
| The X280 core moves the bytes | Yes: 1,000 swaps exact | X280 store 1.84 µs (4 KiB), 3.30 µs (20 KiB), 20.7 µs (80 KiB), against the core's own load of 0.42, 0.68, 1.64 µs |
| The X280 core specialises a template | Yes: 1,000 values of a 32-bit constant, each result exact | two instruction words stored in 0.92 µs; 3.66 µs a specialised call |
| The writer invalidates the target's i-cache from outside | **No.** The register is in the tile's configuration space, which the chip's documentation lists as not mapped for the NoC; a NoC write to it was never acknowledged and left the card unresponsive to the host until a reset | one run; not run from the X280 core |

## What it means for the design

- Any core can program a slot; the slot's own core always invalidates its own i-cache
  (one register store) before it calls, so every load ends in a message to that core.
- The cheapest bytes are the target's own read from DRAM; a Tensix writer is as fast
  (a body in its L1, 62 bytes a cycle to 119 cores), the X280 core 4 to 13 times slower.
- An X280 core is a good decider (a message a swap, 3.5 µs) and a good specialiser
  (two words a value), not a good mover of whole bodies.
- A body that names no symbol of its persistent kernel (counter and helpers passed as
  pointers, `body.h`) runs under any persistent kernel, the framework's included.

## Running it

On a card with tt-metal's environment set (`--card`, default 1):

    python experiments/remote_load/remote.py capture      # remote.cpp's build commands
    python experiments/remote_load/remote.py swap         # core to core, 1,000 swaps
    python experiments/remote_load/remote.py sizes        # its store, 4, 20, 80 KiB
    python experiments/remote_load/remote.py fan --targets 119
    python experiments/remote_load/remote.py x280 --mode self --swaps 1000
    python experiments/remote_load/remote.py x280 --mode self --swaps 100 --cores 0 \
        --clusters 1 2 3                                   # 12 X280 to 120 Tensix cores
    python experiments/remote_load/remote.py x280 --mode store --swaps 20 --sizes 4 20 80
    python experiments/remote_load/remote.py x280 --mode patch --swaps 1000
    python experiments/remote_load/remote.py stale        # without the remote write

`stale --remote 1` makes the writer write the target's invalidate register through
the NoC: it wedges the card (a reset is needed). It is kept only to reproduce the
answer above. Each job prints progress lines and one JSON line, also kept in
`build/remote_load/`.

## How

- **Tensix to Tensix** (`remote.cpp`, one persistent kernel on every core used): the
  writer reads two bodies once from DRAM into its L1. On a command it checks that each
  target is parked (its last reply echoes the last message), stores the body with
  `noc_async_write`, waits until every write is acknowledged, then sends each target one
  16-byte message (id, bytes, argument, then the go word). The target polls the go
  word in its loop, invalidates its i-cache, calls the slot, writes a 64-byte reply
  into the writer's table.
- **From an X280 core** (`kernels/reader.cpp`, `thread/remote.c`): a program of the
  framework. Its reader has one handler that serves the port's messages: {DRAM address,
  bank and bytes, argument} for a body the core reads itself, or {bytes, argument} for
  a body already stored. The thread runs one command on the grid (`kmx_port_run`);
  each X280 core decides its cores' next body when a reply comes, from its own
  schedule. The slot is an ordinary L1 buffer allocated before the load, its address in
  the program's arguments; the images are an ordinary DRAM tensor (a copy in every
  bank).
- **The X280 core's store**: a 2 MiB NoC window of its own onto the target tile (the
  first window after the ports' ones), posted 512-byte vector stores, then the last
  64-byte line again through the window set non-posted; its acknowledgement follows the
  posted stores on the same path (`kmx_push`'s method). kmx has no call for a store into
  a Tensix core's L1, so the thread opens the window itself.
- **The template**: `body.cpp` with `OV_MARK` loads a 32-bit constant with one inline
  `lui` / `addi` pair; `remote.py` finds the one pair that loads the built value,
  asserts it unique and its encoding equal to the X280 core's (`lui_addi`), and the
  thread rewrites those two words for each value (fields masked to their widths).

## Files

- `remote.cpp`: the Tensix-to-Tensix persistent kernel (fetch, feed, stale, clock, exit).
- `kernels/`: the framework program (reader with the handler; writer, compute idle).
- `thread/remote.c`: the X280 thread (modes self, store, patch).
- `remote_x280.h`: the message, the words and the regions shared by both sides.
- `body.h`, `body.cpp`: a body that names no symbol of its persistent kernel.
- `remote.py`: the jobs; it builds the bodies with `../overlay_probe/builder.py`.
