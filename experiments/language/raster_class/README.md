# The rasterizer's passes, as classes

Question: can the rasterizer's passes be written as classes, with the class's own lines
for their movement and the hand program's C++ in inline bodies? The work is per
triangle and per tile. The pack CPU's vector unit does the arithmetic, and mover B
scatters its results to places a table names. The
[rasterizer](../../../projects/raster/README.md) is not rewritten here: both classes run
beside its hand programs and are compared with them word for word.

Three classes, on the tiny scene (13 mesh blocks, 356 list pages, 1,024 tiles on the
120 cores):

- `fill_class.py` makes `FillPass`, the hand's `FILL` (`kernels/reader.cpp`,
  `compute.cpp`, `writer.cpp`). `load_empty(planes)` fills the field `self.empty`
  with the empty tile's 5 planes, kept between commands. `fill_tiles(directory, lists,
  frame, profile)` is the hand's command, its words in the hand's order.
- `passes_class.py` makes `SetupPasses`, the hand's `COUNT`, `HIST` and `SETUP`
  (`kernels/setup_reader.cpp`, `setup_compute.cpp`, `setup_writer.cpp`):
  `count_tiles(mesh, counters, blocks, camera)`, `hist_tiles(heavy, histograms,
  page_bytes)` and `setup_lists(mesh, lists, blocks, camera, slot_pages, list_slots,
  item_words)`.
- `region_class.py` makes `RegionPasses`, the hand's `SUM`, `HSUM` and `PLACE` (in
  `setup_writer.cpp`), which write and read an X280 thread's region:
  `sum_tiles(counters, prefixes, sums, counting)`, `hsum_tiles(heavy_tiles,
  histograms, hprefix, sums, counting, step)` and `place_items(items, directory,
  fill_cores, lists)`, `sums` and `items` regions (`km.Region`).

```python
for head in directory.pages(self.rank, 1).entries(DirHead, size=3, fixed=1):
    km.cxx("...the walk's state...", on="pack", include=[".../fill.hpp"])
    for page in km.Stream(lists.pages(head.first, head.pages), pattern="A"):
        km.cxx("...the hand's fill_list...", takes=[page, self.out], on="pack")
    for item in directory.pages(self.rank, 1).entries(
        DirItem, first=3, size=1, count=2
    ):
        km.cxx(ITEM, takes=[item, frames, frame, self.out, self.empty], on="mover_b")
report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
```

```python
first = self.rank * blocks // self.size
end = (self.rank + 1) * blocks // self.size
km.cxx(HEADERS + GIVE_UP, on="mover_b", at="start")  # once a kernel
km.cxx("setup_pass::setup(); ...", on="pack")
for camera_page in km.Stream(camera.pages(0, 1), pattern="A"):
    km.cxx(CAMERA, takes=[camera_page], on="pack")
for block in km.Stream(mesh.pages(first, end - first), pattern="A"):
    km.cxx(BLOCK, takes=[block, first, self.work], on="pack")
km.cxx(END, on="pack")
km.cxx("...the hand's count_stage...", takes=[...], on="mover_b")
```

The pack bodies are the hand's code as it is. FILL's are `fill_list`, resumed at each
list page, with `fill.hpp` included on pack alone. The passes' are the vector unit's
setup, the camera page, and one block of 32 triangles. Their helpers are the hand's
`setup_compute.cpp` itself, included on pack with the class's defines
(`RASTER_PASS_HELPERS`, `RASTER_NO_ENTRY`, `RASTER_OUT_CB`): its helpers alone, its out
buffer the class's field. `CAMERA`, `BLOCK` and `END` are module constants that
`count_tiles` and `setup_lists` share. The mover B bodies are the hand's writer
stages. The classes' files are under `build/<name>-<hash>/` beside the scripts.

```bash
python run_fill.py --program both --scene tiny --reps 5   # hand, then class, one session
python check.py                                           # frame words, counts, against the reference
python run_passes.py --program both --scene tiny          # COUNT, HIST, SETUP: hand, then class
python check_passes.py --scene tiny                       # counters, histograms, lists
python run_region.py --program both --scene tiny          # SUM, HSUM, PLACE: hand, then class
python drain_cost.py                                      # a drained stream's end, timed
```

Each script writes every tensor before its first load, then runs the hand program, then
the class, each in one load, one command at a time (`depth=1`). HIST and SETUP take the
host's middle pass on the reference's counters (`passes.middle`), as
`card_passes.py --frame` takes it on the card's. Tiny has no heavy tile, so HIST
writes the first 256 tiles that have entries.

`run_region.py` loads each program with an idle X280 thread (`thread/idle.c`, which
returns at once) whose regions are `sums` and `items`. The hand's commands name them by
bank and address (`command.a8` the bank + 1), the class's by name. SUM takes the
reference's counters and HSUM its histograms of the same 256 tiles. PLACE takes an
items region laid out as the thread's middle pass leaves it (`setup.h`), made by the
host from its middle pass of the reference's counters; the hand's directory and list
marks equal that middle pass's (0 words differ), which checks the layout.

## Results

Card 1, a Blackhole p150a. FILL is the median of 5 commands; each pass is one command.

| | Class | Hand |
|---|---|---|
| FILL: frame | 0 of 5,242,880 words differ; tiles, triangles, rows, reader pages and writer items equal on every core; the image exact against the reference | |
| FILL: time a command | 88.52 us (the empty planes loaded once, 13.58 us) | 118.90 us |
| FILL: kernel text, B (mover A, mover B, unpack, math, pack) | 15,812, 13,164, 7,032, 1,236, 8,176: 46,128 | 12,040, 9,496, 2,160, 884, 7,604: 32,896 (FILL, ARITH, SPIN) |
| COUNT: counters | 0 of 245,760 words differ from the hand's; both equal to the reference's | |
| HIST: histograms | 0 of 491,520 words differ; both equal to the reference's | |
| SETUP: lists | 0 of 182,272 words differ; both equal to the reference's lists | |
| COUNT, HIST, SETUP: time | 85.20, 24.80, 113.40 us | 87.16, 24.18, 124.34 us |
| Passes: kernel text, B | 15,468, 14,044, 6,452, 1,268, 20,924: 58,864 | 14,144, 16,680, 3,252, 832, 19,812: 55,424 (all seven handlers) |
| SUM: slots prefixes, totals | 0 of 245,760 and 0 of 10,240 region words differ | |
| HSUM: heavy prefixes, totals | 0 of 983,040 words differ; the totals in the same 10,240 | |
| PLACE: directory, list marks | 0 of 122,880 and 0 of 182,272 words differ | equal to the host's middle pass |
| SUM, HSUM, PLACE: time | 84.88, 235.16, 112.62 us | 57.90, 188.78, 94.54 us |
| Region passes: kernel text, B | 20,168, 19,052, 5,696, 1,248, 7,964: 54,816 | (the passes' 55,424 above) |

FILL's 30.6 us. On the hand's slowest core (core 105) the pack CPU waits 50,057
cycles for out pages. Its writer spends about 37,000 cycles before its first item:
the directory's read, and the empty tile's planes built in every command (5,120 word
stores). The class's planes are a kept field, so its writer drains the first item at
once. On its slowest core (core 19) pack waits 716 cycles. The class's writes are also
cheaper per item (median 3,286 cycles against 4,869); that was not chased. SETUP's
11 us was not chased either.

HIST was 33.06 us when its tile list was read on every CPU; the table is now read
only where its words are used (mover B here), 24.80 us. SUM and HSUM are slower than
the hand's: the hand sums on mover B, 32 or 16 cores' 64 B at a time, and the class
streams each core's item to the pack CPU, two mailbox words an item (classes.md,
Inline bodies). PLACE's 18 us more is the region read as 9 whole pages by mover A
into a field (the hand reads 28,608 B of it), waited for by mover B.

## What each line lowers to

| Line | Lowered to | Before it |
|---|---|---|
| `km.Stream(lists.pages(first, count), pattern="A")` with a body on pack | mover A reads each list page; unpack hands its L1 address to pack by mailbox and frees it on pack's release | the hand's reader loop and unpack's hand-off loop |
| `for head in directory.pages(self.rank, 1).entries(DirHead, size=3, fixed=1):` | each mover reads the page; compute waits for mover A's and reads it in place | the reader's header page to compute |
| `first = self.rank * blocks // self.size` | an int of every kernel | the reader's `reader.me() * blocks / cores` |
| `empty: km.L1[...]`, filled by `load_empty` | a field waited for once a command, never freed | the writer's planes, built each command |
| `out = km.own(push="pack")` | a buffer pack pushes and mover B's body pops, no wait of the compiler's | `kCbOut` |
| `self.next_slots.fill(km.Stream(slot_pages.pages(2 * self.rank, 2)))` | mover A's read into the field; mover B's body waits for it and frees it after | the reader's pages `kCbSlots` and `kCbItems`, waited for by the writer |
| `km.cxx(HEADERS + GIVE_UP + "...")` | the sum's text, its `#line` the call's (Inline bodies, classes.md) | |
| `takes=[counters]`, a tensor, on mover B | a `km::Tensor` of its command word | `command.a1` |
| `report`, a write stream on pack and then mover B | the profile page from three CPUs' words, as the ray tracer's | the writer's profile page |

## The lines these classes use, and the pass each serves

| Line, as a reader would copy it | Pass |
|---|---|
| `def sum_tiles(self, ..., sums: km.Region, ...):` and `run.submit(RegionPasses.sum_tiles, (0, cores), args=(..., "sums", ...))` | SUM, HSUM, PLACE: a thread's region by its name |
| `km.Stream(counters.items(first=part, count=counting, size=64, step=128))`: one 64 B item of each core's page | SUM, HSUM |
| `directory.items(size=4096).put(self.page, at=self.rank)` and `lists.items(size=128).put(self.slots, at=end_slot)`: an own field written by mover B | PLACE |
| `slot = self.next_slots.bump(page[1])` and `lists.items(size=128).put(page, at=slot)` | SETUP's record to its list's next slot (not run here) |
| `pages = km.cxx(..., gives=int, on="mover_a")` and `km.cxx(..., takes=[pages], on="mover_b")` | FILL's reader counters (not run here) |
| `with km.Stream(self.out, mover="B") as out:` (or `out.end()`) and `for page in out:` | SETUP's and COUNT's loop over the pages pack pushes (not run here; its cost in `drain_cost.py`) |
| `for cut in heavy.pages(0, 1).entries(Heavy, first=1, size=1, count=0):` read on mover B alone | HIST |

## Moves still bodies

| Move | Body | Why |
|---|---|---|
| SUM's and HSUM's sums and prefixes | pack: 16 or 32 words an item | arithmetic on words is a body's (no line computes on a page's words) |
| PLACE: each item's tile, the directory page, the headers | mover B: the hand's loops over the region's words, the headers written 32 at a time | the headers' loop is bounded by a mover B word (`item_count`), and a range of a mover's word is no line |
| SETUP's and COUNT's writer stages, and FILL's writer loop | mover B: wait, invalidate, pop, the 128 B writes | the drained stream, bump and put now say them; the classes were not moved to them in this package |
| FILL's reader counters to mover B | the own field `handoff` | the hand-over line now says it; not moved here |
| COUNT's counts and HIST's 64 B a tile written by mover B | `km::write_pages`, `noc_async_write` | a put from an own field says it; not moved here |

## What a class cannot say of the frame

In the hand's frame, the work cluster's thread (`thread/frame.c`) runs seven commands
in order: COUNT, SUM, HIST, HSUM, PLACE, SETUP, FILL. Between them it reads the
totals, picks the heavy tiles, sorts the tiles by cost, assigns them to cores, and
writes the items, the slots and the directory. That sequence stays the hand's: a
class is a set of handlers that the host submits one command at a time, and the
classes here are run that way, the region passes on an idle thread's regions. COUNT's
balanced ranges (`blocks_word | ranges`, read from a region) and SETUP's slots from
SUM's prefixes (`RF_SETUP_PREFIX`) are the thread's forms of the same commands; these
classes take the host's forms: the even partition, and next slots given.
