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
- `region_timed.py` holds the same three passes in other forms, to see where their
  time goes (the section "Where SUM, HSUM and PLACE spend their time").

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
python run_timed.py --scene tiny                          # SUM, HSUM, PLACE: stages, routes
python timed_tables.py build/timed.npz                    # the stage tables below
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
11 us is likely its item table, read by every core at its start: the hand reads it as
one 8,192 B page, so from one DRAM bank (120 x 8 KB), the class as two 4,096 B pages
from two banks. At the 62 GB/s one bank gave PLACE below, that is about 16 us against
8 (not measured on the card).

HIST was 33.06 us when its tile list was read on every CPU; the table is now read
only where its words are used (mover B here), 24.80 us. COUNT and HIST are level with
the hand's: the same reads, the hand's pack code, and a class's command costs what the
hand's does (below). SUM, HSUM and PLACE are slower by the class's route, not by the
framework: the next section.

## Where SUM, HSUM and PLACE spend their time

`run_timed.py` runs, in one session, the hand and each class of `region_timed.py`, each
load traced (the runtime's per-core take, outputs-written and completion clocks), each
command three times and once more with stamps (each CPU's wall clock at its stages,
written to a tensor): `RegionRoute` is `RegionPasses` with stamps on mover A, pack and
mover B, and an empty handler; `RegionHand` the hand's writer text as one body on mover
B, stamped; `RegionPlain` the same text without stamps, and PLACE by class lines that
read the hand's two ranges; `RegionBlock` and `RegionBlockTimed` the class's route in
the hand's batches (blocks of 32 and 16 items); `RegionFn` the hand's PLACE text in a
function of its own inside the body. Every form's words equal the hand's.

What any class pays: an empty class command is 2.12 us on the host, the hand's command
with no cores 2.10 (on the card 1.28 against 1.22 from the first take to the
completion); mover B starts its handler 0.4 us after the take in both. The hand's text
as a class body: SUM 57.20 us against the hand's 57.52, HSUM 189.24 against 188.56,
PLACE 100.18 against 94.34. PLACE's 5.8 us is the compiler's: the body's directory
scan (once an item of the region, on every core) compiles with one stack load and one
store an item, where the hand's `place()` keeps every value in a register. Every
handler's lines are compiled into one `movement.run` lambda; the same text in a
function of its own (`RegionFn`) is 97.04 us against 94.92. Compiling each mover case
as a function of its own was tried and not kept: plain PLACE 98.90, the class's PLACE
by the hand's ranges 94.92 (from 100.14), but plain HSUM 223.28 and the region class's
PLACE 115.40 (from 112.02).

The stages, on the core that finished last, for the region class's route (stamped)
and the hand's text (stamped); each column adds up to its command.

| SUM stage | CPU | Class us | Hand us | Difference |
|---|---|---|---|---|
| submit to the first take, completion back | host | 0.82 | 0.90 | -0.08 |
| the reader takes the command | mover A | 0.07 | 0.05 | +0.03 |
| mover B's handler starts | mover B | 0.43 | 0.42 | +0.00 |
| prefixes written (hand: reads issued, sums, writes issued) | mover B | 80.38 | 49.68 | +30.70 |
| read wait | mover B | 0.00 | 4.56 | -4.56 |
| flush wait | mover B | 0.00 | 1.50 | -1.50 |
| totals written | mover B | 0.71 | 0.00 | +0.71 |
| the handler's other lines | mover B | 0.08 | 0.02 | +0.07 |
| handler end to outputs written | mover B | 1.12 | 0.51 | +0.60 |
| slowest core done to the completion | leader | 0.65 | 0.64 | +0.01 |
| command | | 84.26 | 58.28 | +25.98 |

Beside mover B in the class: mover A moves the 120 item pairs in 79.25 us, one pair a
pass (0.66 us, a DRAM round trip each); pack sums them in 26.49 us and waits 50.51.

| HSUM stage | CPU | Class us | Hand us | Difference |
|---|---|---|---|---|
| submit to the first take, completion back | host | 0.90 | 0.87 | +0.04 |
| the reader takes the command | mover A | 0.09 | 0.15 | -0.06 |
| mover B's handler starts | mover B | 0.47 | 0.41 | +0.05 |
| prefixes written (hand: reads issued, sums, writes issued) | mover B | 235.99 | 169.86 | +66.13 |
| read wait | mover B | 0.00 | 10.90 | -10.90 |
| flush wait | mover B | 0.00 | 7.03 | -7.03 |
| totals written | mover B | 1.86 | 0.00 | +1.86 |
| the handler's other lines | mover B | 0.21 | 0.01 | +0.19 |
| handler end to outputs written | mover B | 1.05 | 0.50 | +0.55 |
| slowest core done to the completion | leader | 0.55 | 0.52 | +0.03 |
| command | | 241.12 | 190.26 | +50.86 |

Beside mover B: pack's sums take 163.52 us of its 360 items (0.45 us an item), its
waits and hand-overs 70.86; mover A moves for 156.00 of its 233.27 us and waits for
room the rest.

| PLACE stage | CPU | Class us | Hand us | Difference |
|---|---|---|---|---|
| submit to the first take, completion back | host | 0.88 | 0.84 | +0.05 |
| the reader takes the command | mover A | 0.08 | 0.16 | -0.07 |
| mover B's handler starts | mover B | 0.43 | 0.39 | +0.04 |
| the region's reads land (class: mover A's fill) | mover B | 71.57 | 54.00 | +17.58 |
| the body after the reads | mover B | 38.67 | 43.59 | -4.92 |
| directory and end slot written | mover B | 0.13 | 0.50 | -0.37 |
| the handler's other lines | mover B | 0.03 | 0.10 | -0.07 |
| handler end to outputs written | mover B | 0.53 | 0.51 | +0.02 |
| slowest core done to the completion | leader | 0.51 | 0.48 | +0.03 |
| command | | 112.84 | 100.56 | +12.28 |

To each pass's difference (class 83.52, 235.42, 112.30 us against the hand's 57.38,
188.86, 94.64 in the same session) the table adds the stamps' cost on the class
(-0.74, -5.70, -0.54: the stamped route against the class) and the hand's text in a
class against the hand (+0.90, +1.40, +5.92). HSUM's -5.70 is not the stamps alone:
the same class's HSUM was 240.68 us after the runtime grew by 700 B, the pack CPU's
loop placed elsewhere.

The class's route, by pass:

- SUM: one item in flight a stream. Mover A's 120 item pairs are 120 DRAM round
  trips, and pack and mover B follow them; the hand issues 64 reads to a barrier. The
  same lines with `block=32`: 45.4 us, 12 us under the hand (reads, sums and writes on
  three CPUs at once).
- HSUM: the sums on pack, 0.45 us an item, and with `block=1` mover A one item ahead.
  With `block=16`: 186.5 us, level with the hand; pack still the slowest CPU (163.4 us
  of sums and 0.042 us an item for the address and release mailbox words).
- PLACE: the fill reads all 9 pages of the region (36,864 B) where the hand reads its
  two ranges (28,608 B); 120 cores read the thread's one DRAM bank at about 62 GB/s.
  The class's lines reading the two ranges (`place_ranges`, two fills): 100.14 us,
  the same text as the hand's then (and 94.92 with each case a function of its own).

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
