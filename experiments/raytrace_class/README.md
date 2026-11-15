# The ray tracer's trace pass, as a class

Question: can a program of the ray tracer's kind be written as a class today, and where
not, which general line is missing? Its work comes in packets of rays, each walked down
a tree for as many steps as its rays need, on the pack CPU's vector unit. The
[ray tracer](../../projects/raytrace/README.md) is not rewritten here.

The pass is the trace (`TRACE`, the whole tree in L1), and the scene is the tiny one:
1,048,576 primary rays in 32,768 packets on the 120 cores. This pass carries the most
of the program's movement. Each core reads its directory entry and its packets, walks
each packet, and writes a hit page for it and its profile page. The other passes are
stages (`GEN`, `SHADE1`, `SHADE2`, `IMAGE`) or the same trace with the tree read in
treelets.

`trace_class.py` makes the class `TracePass`:

- `trace_rays(directory, packets, hits, mode, profile, image)` is the hand program's
  `TRACE`. Its command words are in the hand's order (a0 to a4), then the tree's
  tensor (a5).
- `self.tree.fill(km.Stream(image.pages(0, TREE)))` is a class line: the tree's 16
  pages, a field of the class, read by each command.
- The rest is three inline bodies: mover A's reads, compute's walk and mover B's
  writes. The compute body includes the hand program's headers as they are:
  `raytrace.h`, `timed.hpp`, `dir_noc.hpp` and `walk.hpp` (with `sfpu.hpp`). The pack
  CPU's walk is `walk::load_packet`, `walk::packet` and `walk::store_hits`, as the hand
  program's `trace_list` calls them. The code around them is the hand program's
  `reader.cpp`, `compute.cpp` and `writer.cpp`, with the hand's buffer numbers
  replaced by the class's buffers.
- `fill_tree(image)` is the tree's fill alone, a term of the account below.

Each use of something no line gives is marked `temporary` in the code and is a row of
the table below. The class's files are under `build/tracepass/` beside the script.

```bash
python prepare.py                              # the inputs, off the card
python run_trace.py --program hand --reps 5    # the project's own driver of the pass
python run_trace.py --program class --reps 5   # the class, then fill_tree
python check.py --reference                    # class against hand, word for word, and against numpy
```

`prepare.py` makes the inputs as the project's `run.py` does: every pixel's camera ray,
the host's split of tiles over the 120 cores, the packet pages, the directory and the
tree image. `run_trace.py --program hand` runs them through the project's
`card.run_handler` (`LOAD`, then the `TRACE` commands, one at a time). The class takes
the same tensors in one load, also one command at a time (`depth=1`).

## Results

Card 1, a Blackhole p150a, AICLK 1,350 MHz; the median of 5 commands of each.

| | Class | Hand |
|---|---|---|
| Hit pages | equal to the hand's word for word: 0 of 18,874,368 words differ (each packet's cycles word apart); every core's counts equal (packets, node pops, inner nodes, leaves, pages read and written) | |
| Against numpy | 658,018 hits of 1,048,576 rays; triangle, t, u and v: 0 differ | the same (the hand's pages are the class's) |
| Time a command | 3,305.1 us | 3,259.9 us |
| Kernel text, B (mover A, mover B, unpack, math, pack) | 13,008, 8,720, 5,004, 1,000, 13,012: 41,440 | 23,188, 10,312, 3,936, 1,448, 46,220: 85,808 (every handler of the program) |

The hand's 3,259.9 us is today's; the project's README records 3.007 ms (2026-11-13) and
3.263 in its frame. The class's kernels hold the trace alone, the hand's all of its
handlers.

The 45.2 us between them, on the slowest core (core 41 in both):

| Term | us | |
|---|---|---|
| the tree read again by each command, before compute starts | 21.2 | a copy the hand does not make (its `LOAD` reads the tree once; here a body cannot take a field another handler fills). `fill_tree` alone takes 22.8 us a command |
| the walk on the pack CPU | 24.0 | box tests +14.6, triangle tests +4.5, packet loads +1.2, hit pages out +1.1, pops +0.4, the rest +2.3; the same source, built into another kernel. Over all cores +14.2 us on the mean, -14.1 to +26.3 by core: not resolved further |
| the argument words, the streams' blocks | 0 | one word more (the tree's tensor); packets and hits go one to a buffer page, as the hand's |

## What the class could not say

Each row is a part of the pass kept inline or changed, with the general line that would
say it and an estimate of its size in compiler lines. None is built here.

| The hand program | Why no line says it | The general line | Size | Elsewhere |
|---|---|---|---|---|
| reads each packet, 1,408 B, at its place inside a DRAM page of 64 packets (page `p / 64`, offset `p % 64 x 1,408`) into a buffer page, one packet a page | a stream reads whole pages of a tensor, and a page of this tensor is 64 packets; a core's packets start at any packet | a stream of items of a page: `km.Stream(packets.items(first, count))`, the item size the tensor's (made with 64 items a page) | about 40, and 30 in the device's stream | every stage of the ray tracer (`GEN`, `SHADE1`, `SHADE2`, `IMAGE` read and write packets the same way); two cards: "no call moves a part of a page across cards yet" (the project's README) |
| writes each hit page, 2,432 B, at its place inside a DRAM page of 64 | the same, for a write stream | `km.Stream(hits.items(first, count), write=True)` | with the row above | the same stages |
| reads its directory entry, a page of 16 words: the first packet and the count; gives compute the count in a header page, and mover B reads the entry again | `pages.read()` gives a page's first word, on one mover; no line reads word i, and none hands a mover's word to compute | a read page's words: `first, count = directory.pages(self.rank, 1).read()[0:2]`, its words on every CPU that uses them | about 45 | the rasterizer's reader and writer (`raster/kernels/reader.cpp`, `writer.cpp`: a directory page per core); the frame planned on the X280 writes these entries |
| unpack waits for each packet page and hands its L1 address to the pack CPU by mailbox; the pack CPU walks it in place, releases it, and unpack frees it | a compute body runs on every compute CPU; a stream's tile is a buffer and an index, and only unpack holds the read pointer | a compute body on one CPU with a stream's item: `km.cxx(..., takes=[packet], on="pack")`, binding the item's address, with unpack's wait, hand-off and free | about 50 | the rasterizer's compute (`raster/kernels/compute.cpp`: the same mailbox), attention's compute (`attention/kernels/compute.cpp`) |
| includes `walk.hpp` (and so `<riscv_vector.h>`) on the pack CPU alone | a body's `#include` lines go to the top of every compute CPU's kernel | with the row above, a body's includes under that CPU's guard; here a three-line header, `kernels/pack_headers.hpp` | about 5 | the ray tracer's stages, the rasterizer |
| loads the tree once (`LOAD`) and keeps its address for every later `TRACE` | a body takes a field only in the handler that fills it; a field kept between commands is read by a product line only | a kept field taken by a body: `km.cxx(..., takes=[self.tree])` in a handler that does not fill it, waited for once and not freed | about 15 | the ray tracer's top tree and its treelets' slots; the linear layer's kept field (read by a product line today) |
| sets the compute kernel's `enable_trisc2_rvv` and `dst_full_sync_en` (the walk runs on the pack CPU's vector unit and holds DST whole) | a class sets the compute options its own lines need, and no others | the class's compute options: `class TracePass(km.Program, compute={"enable_trisc2_rvv": True, "dst_full_sync_en": True})`; here the run script sets them on the made program | about 10 | the rasterizer (`raster/card.py` sets the same) |
| its buffers: the packet pages (4 of 4 KiB), the hit and profile pages (4 of 4 KiB), the reader's and writer's directory pages | no line gives a body a buffer of its own; here four fields that no line fills, named in the bodies by their buffer names (`kFieldPacketsIn`, ...) | none once the item streams and the read page exist; else a body's buffer: `inbox: km.L1[km.fp32, 4, 32, 32]` taken by bodies on any CPU, `takes=[self.inbox]` | about 25 | |
| names its tensors by their command words and accessors (`command.a1`, `Directory`, `Profile`) | `takes=` refuses a tensor, and a tensor's word is dropped when only a body uses it | `km.cxx(..., takes=[packets])`, binding the tensor's word and its accessor | about 10 | the rasterizer's and attention's bodies, should they be classes |
| writes one profile page a core: compute's counters, then the reader's and the writer's, put together by the writer | a word field is the unpack CPU's alone, in L1, read by the host; no line puts words of three CPUs into one page of a tensor | a profile field: words each CPU sets, written by mover B to page `rank` of a tensor at the command's end | about 35 | the rasterizer's and attention's profile pages (`raster/kernels/writer.cpp`, `attention/kernels/writer.cpp`); the class's stamps (`@km.handler(stamps=)`) are the math CPU's alone |

Seen besides, not a line: a class's compute kernel starts with
`compute_kernel_hw_startup` of buffers 0 and 16, which this class does not declare (the
hand program names its state and output buffers). The walk uses neither the unpacker
nor the packer, so it does no harm here.
