# The ray tracer's trace pass, as a class

Question: can a program of the ray tracer's kind be written as a class, with nothing but
the class's own lines and its arithmetic in inline bodies? Its work comes in packets of
rays, each walked down a tree for as many steps as its rays need, on the pack CPU's
vector unit. The [ray tracer](../../projects/raytrace/README.md) is not rewritten here.

The pass is the trace (`TRACE`, the whole tree in L1), and the scene is the tiny one:
1,048,576 primary rays in 32,768 packets on the 120 cores. This pass carries the most
of the program's movement. Each core reads its directory entry and its packets, walks
each packet, and writes a hit page for it and its profile page. The other passes are
stages (`GEN`, `SHADE1`, `SHADE2`, `IMAGE`) or the same trace with the tree read in
treelets.

`trace_class.py` makes the class `TracePass`, with the walk's compute options on the
class (`compute={"enable_trisc2_rvv": True, "dst_full_sync_en": True}`):

- `load_tree(image)` fills the field `self.tree` with the tree's 16 pages. No other
  handler fills it, so it stays in L1 between commands, as the hand program's `LOAD`.
- `trace_rays(directory, packets, hits, mode, profile)` is the hand program's `TRACE`,
  its command words in the hand's order (a0 to a4):

```python
first, count = directory.pages(self.rank, 1).read()[0:2]
hit = km.Stream(hits.items(first, count, size=HIT), write=True, pattern="B")
image = km.cxx("image = tree;", takes=[self.tree], gives=int, on="pack")
km.cxx("...setup...", takes=[image], on="pack", include=[".../walk.hpp"])
for rays in km.Stream(packets.items(first, count, size=PACKET)):
    km.cxx("...the walk of one packet...", takes=[rays, hit, mode], on="pack")
report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
km.cxx("...finish, the profile page...", takes=[report], on="pack")
```

The bodies are the hand program's pack-CPU code as it is: `walk::load_packet`,
`walk::packet` and `walk::store_hits` from the hand's headers (`raytrace.h` on every
compute CPU, `walk.hpp` on the pack CPU alone), the walk's state set up before the
loop and read in it. No body names a buffer, a command word or a generated type: the
packets and hit pages are items of the tensors' pages (1,408 B and 2,432 B, 64 to a
page), the directory's first two words are ints of every kernel, and each packet's L1
address reaches the pack CPU by mailbox from unpack, as in the hand program. The
class's files are under `build/tracepass/` beside the script.

```bash
python prepare.py                              # the inputs, off the card
python run_trace.py --program hand --reps 5    # the project's own driver of the pass
python run_trace.py --program class --reps 5   # the class: load_tree once, then trace_rays
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
| Hit pages | equal to the hand's word for word: 0 of 18,874,368 words differ (each packet's cycles word apart); the walk's counts equal on every core (packets, node pops, empty pops, inner nodes, leaves, errors) | |
| Against numpy | 658,018 hits of 1,048,576 rays; triangle, t, u and v: 0 differ | the same (the hand's pages are the class's) |
| Time a command | 3,294.9 us | 3,259.9 us |
| The tree's load | 23.7 us, once | once (`LOAD`) |
| Kernel text, B (mover A, mover B, unpack, math, pack) | 15,168, 11,328, 6,464, 1,224, 13,368: 48,240 | 23,188, 10,312, 3,936, 1,448, 46,220: 85,808 (every handler of the program) |

The hand's 3,259.9 us is today's; the project's README records 3.007 ms (2026-11-13) and
3.263 in its frame. The class's kernels hold the trace alone, the hand's all of its
handlers. The class's profile page holds the pack CPU's counters; the reader's and the
writer's counters are not in it (the last section).

The 35.0 us between them. On the slowest core (core 41 in both, 288 packets) the pack
CPU's span is 29.4 us longer:

| Term | us | |
|---|---|---|
| the walk (packet loads, pops, box and triangle tests, the vector unit's drain) | -5.1 | load -0.2, pops -3.3, box tests -1.3, triangle tests -0.3: the same work, as fast as the hand's |
| the hit page out (`walk::store_hits`, the fence) | +13.8 | 48 cycles a packet more; the push now follows the body, outside this term |
| the loop's own lines | +20.7 | 97 cycles a packet: unpack's wait and its answer to math and pack, the packet's address handed to pack and its release handed back (the hand hands the same two words), the hit page's reserve before the walk, the push and the pops; it is the class's 36.7 us outside the named terms less the 16.0 us the hand books as its waits for input and output pages (the class does not time those waits apart) |

The other 5.6 us are outside the pack CPU's span: the command's start (mover A's
directory page pushed to compute and its two words handed to math and pack, the kept
tree waited for and its address handed to pack; the hand pushes a header page) and its
end (the profile page, 1,024 words written by the pack CPU where the hand writes 32
and its writer adds its own, then mover B's write).

The class before these lines (bodies naming the generated buffers, the tree filled by
each command) took 3,305.1 us: the tree's read again by each command, 21.2 us, is gone;
so are the 24.0 us its walk took over the hand's.

### The walk's 24.0 us in the class before these lines

That class's walk, on core 41, took 24.0 us more than the hand's: box tests +14.6,
triangle tests +4.5, the rest under 2 us each; the class now takes 5.1 us less. The
three pack-CPU builds, disassembled off the card: the nearest walk's path from its
leaf test to the wall-clock read after an inner node's two box tests, and from the
leaf test to the read after a leaf's triangle test:

| Build | box tests of an inner node: instructions (loads, stores, stack) | a leaf's triangle test: instructions | box tests on core 41 | triangle tests |
|---|---|---|---|---|
| hand | 237 (18, 69, 4) | 567 | 356.8 cycles an inner node | 742.6 cycles a leaf |
| the class before | 238 (20, 69, 6) | 573 | +3.0 cycles an inner node (+14.6 us over 6,545) | +10.1 cycles a leaf (+4.5 us over 600) |
| the class now | 235 (20, 69, 6) | 566 | -0.3 cycles | -0.6 cycles |

What a build difference can explain: the walk is the same source inlined into another
function (the hand calls `walk::packet` as a function of its own; both classes inline
it into the command's loop), so its code sits at other addresses, with other
registers, two more loads and two more stack accesses on the box path. What it cannot
explain is the instructions themselves: the class before ran one more instruction on
the box path and 6 more on the triangle test than the hand, the class now 2 and 1
fewer, and the box path's 237 instructions take 356.8 cycles, a third of them stalls.
So the +3.0 cycles an inner node came from where the code and its data sat (a taken
branch's target, a load's distance to its use, a buffer's place in L1), not from what
was compiled; the streams alone cannot say which, and it was not chased on the card.

## What each line lowers to

| Line | Lowered to | Before it |
|---|---|---|
| `t.items(first, count, size=B)` in `km.Stream(...)`, read or `write=True` | the stream of `km::items(T(t), B, P)`: item i is page i / P, byte (i % P) x B; the pair's pages are B bytes; a size that does not divide the page is refused at its line | the reader's `noc_async_read` at `get_noc_addr(packet / 64, ..., packet % 64 x 1,408)`, into a field named by its buffer |
| `first, count = directory.pages(self.rank, 1).read()[0:2]` | mover A reads the page into a page it pushes to compute, mover B into its own; unpack takes the two words and hands them to math and pack by mailbox | the reader's header page to compute, the writer's second read |
| `km.cxx(..., takes=[rays], on="pack")` in the stream loop | unpack hands the item's L1 address to pack before the body, pack hands a release back after it, then unpack frees the block | the reader's own loop and unpack's own hand-off loop |
| `takes=[hit]`, a write stream, on pack in the loop | the loop's page of `hit`: reserved with the block (0 when compute gave up), pushed after it, moved with the read | the writer's own loop over a field named by its buffer |
| `takes=[report]`, a write stream, outside a loop | one block reserved, pushed after the body, moved by the movers at that line | the writer's profile page |
| `takes=[self.tree]`, a field `load_tree` keeps | unpack's wait for it (never freed), its address handed to pack | a fill of the tree in each command |
| `include=[".../walk.hpp"]` with `on="pack"` | the header under `#if defined(TRISC_PACK)` at the kernel's top | a three-line header of the experiment's own |
| `class TracePass(km.Program, compute={...})` | the compute kernel's options; an unknown key is refused by name | the options set on the made program by the run script |

`takes=` of a tensor (a `km::Tensor` of its command word on a mover) is a line too, and
this class does not need it: its tensors are moved by streams and a word read. A
class's compute kernel now starts the unit with its own first read and write buffers
(this class: its packets' and its hit pages'), not buffers 0 and 16.

## What is still not said

The reader's and the writer's counters in the profile page (their waits, their reads,
their pages): the hand puts three CPUs' words into one page, compute's from its output
page, the reader's from its own L1, and the writer writes it. A class line for it needs
a word handed from mover A to mover B and a body on mover B over a write stream's page
before the movers write it, more than 40 lines of the compiler: not built. The class's
page holds the pack CPU's words, so `check.py` compares the walk's counts alone.

## The lines in the rasterizer

Not written here; where each line would go in [the rasterizer](../../projects/raster/README.md):

| Line | File | What it does there |
|---|---|---|
| items, a write stream | `kernels/writer.cpp` (FILL) | a band of an output tile: rows `first_row` to `end_row` of each plane, 128 B a row, written into the frame's pages by address (`frame_noc(page + plane, byte_offset)`); a stream of items of 128 B, the band's rows |
| items, a read stream | `kernels/setup_writer.cpp` (SUM) | each core's 64 B slice of the counters pages (`noc_async_read(address, staging + 64 * batch_core, 64)`): items of 64 B, one of each core's page, so one stream a core until items take a step |
| a page's words | `kernels/reader.cpp`, `kernels/writer.cpp` (FILL, ARITH) | each core's directory page: the reader's first list page and their count (`directory[kDirFirst]`, `directory[kDirPages]`), the writer's items and frame pages (`kDirItems`, `kDirFrame`) |
| a body on pack with a stream's page | `kernels/compute.cpp` (FILL, ARITH) | unpack hands each list page's L1 address to the pack thread and frees it on its release (`mailbox_write(PackThreadId, get_tile_l1_byte_address(kCbIn, 0))`, then `mail_from(PackThreadId)`); `fill_list` walks the slots in a body on pack |
| `include=` on pack | `kernels/compute.cpp`, `kernels/frame_compute.cpp` | `fill.hpp` and the vector unit's headers, today under `#ifdef TRISC_PACK` |
| a body taking a kept field | `kernels/writer.cpp` (FILL) | the empty tile's planes (`kCbEmpty`, 5 pages) built by the writer in every FILL command; a field a load handler fills once, taken by a body on mover B |
| `compute={...}` | `card.py`, `card_passes.py` | the programs' compute options (`math_fidelity`, `fp32_dest_acc_en`, `enable_trisc2_rvv`) |
| `takes=` of a tensor | `kernels/setup_writer.cpp` (HIST, SUM, HSUM) | the counters, histograms and prefixes tensors named by their command words (`command.a1`, ...), read and written by address in bodies on mover B |
| the start-up from the declared buffers | `kernels/compute.cpp`, `kernels/frame_compute.cpp` | `compute_kernel_hw_startup(kCbIn, kCbOut)`: its list pages and its output |
