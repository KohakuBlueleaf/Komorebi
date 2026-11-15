# The rasterizer's passes, as classes

Question: can the rasterizer's passes be written as classes, with the class's own lines
for their movement and the hand program's C++ in inline bodies? The work is per
triangle and per tile. The pack CPU's vector unit does the arithmetic, and mover B
scatters its results to places a table names. The
[rasterizer](../../../projects/raster/README.md) is not rewritten here: both classes run
beside its hand programs and are compared with them word for word.

Two classes, on the tiny scene (13 mesh blocks, 356 list pages, 1,024 tiles on the 120
cores):

- `fill_class.py` makes `FillPass`, the hand's `FILL` (`kernels/reader.cpp`,
  `compute.cpp`, `writer.cpp`). `load_empty(planes)` fills the field `self.empty`
  with the empty tile's 5 planes, kept between commands. `fill_tiles(directory, lists,
  frame, profile)` is the hand's command, its words in the hand's order.
- `passes_class.py` makes `SetupPasses`, the hand's `COUNT`, `HIST` and `SETUP`
  (`kernels/setup_reader.cpp`, `setup_compute.cpp`, `setup_writer.cpp`):
  `count_tiles(mesh, counters, blocks, camera)`, `hist_tiles(heavy, histograms,
  page_bytes)` and `setup_lists(mesh, lists, blocks, camera, slot_pages, list_slots,
  item_words)`.

```python
first, count = directory.pages(self.rank, 1).read()[0:2]
km.cxx("...the walk's state...", on="pack", include=[".../fill.hpp"])
for page in km.Stream(lists.pages(first, count), pattern="A"):
    km.cxx(
        "...the hand's fill_list over the page's 16 slots...",
        takes=[page, self.out],
        on="pack",
    )
km.cxx(
    "...the hand's writer loop...",
    takes=[directory, frame, me, self.out, self.empty, self.table],
    on="mover_b",
)
report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
```

```python
first = self.rank * blocks // self.size
end = (self.rank + 1) * blocks // self.size
km.cxx("setup_pass::setup(); ...", on="pack", include=["../../passes_pack.hpp"])
for camera_page in km.Stream(camera.pages(0, 1), pattern="A"):
    km.cxx(CAMERA, takes=[camera_page, self.out], on="pack")
for block in km.Stream(mesh.pages(first, end - first), pattern="A"):
    km.cxx(BLOCK, takes=[block, first, self.work], on="pack")
km.cxx(END, on="pack")
km.cxx(HEADERS + GIVE_UP + "...the hand's count_stage...", takes=[...], on="mover_b")
```

The pack bodies are the hand's code as it is. FILL's are `fill_list`, resumed at each
list page, with `fill.hpp` included on pack alone. The passes' are the vector unit's
setup, the camera page, and one block of 32 triangles. Their helpers are in
`passes_pack.hpp`, a copy of the hand's `setup_compute.cpp` without its entry, with its
out buffer a member: the hand's file included whole makes 72,544 B of kernel text,
over the 70,656 B limit. `CAMERA`, `BLOCK` and `END` are module constants that
`count_tiles` and `setup_lists` share. The mover B bodies are the hand's writer
stages. The classes' files are under `build/<name>-<hash>/` beside the scripts.

```bash
python run_fill.py --program both --scene tiny --reps 5   # hand, then class, one session
python check.py                                           # frame words, counts, against the reference
python run_passes.py --program both --scene tiny          # COUNT, HIST, SETUP: hand, then class
python check_passes.py --scene tiny                       # counters, histograms, lists
```

Each script writes every tensor before its first load, then runs the hand program, then
the class, each in one load, one command at a time (`depth=1`). HIST and SETUP take the
host's middle pass on the reference's counters (`passes.middle`), as
`card_passes.py --frame` takes it on the card's. Tiny has no heavy tile, so HIST
writes the first 256 tiles that have entries.

## Results

Card 1, a Blackhole p150a. FILL at AICLK 1,350 MHz, the median of 5 commands of each.
The passes are one command of each; the clock was not sampled after the open.

| | Class | Hand |
|---|---|---|
| FILL: frame | equal to the hand's word for word: 0 of 5,242,880 words differ; tiles, triangles, rows, reader pages and writer items equal on every core; the image exact against the reference | |
| FILL: time a command | 88.34 us (the empty planes loaded once, 13.58 us) | 118.90 us |
| FILL: kernel text, B (mover A, mover B, unpack, math, pack) | 15,256, 13,288, 5,392, 1,188, 8,164: 44,000 | 12,040, 9,496, 2,160, 884, 7,604: 32,896 (FILL, ARITH, SPIN) |
| COUNT: counters | 0 of 245,760 words differ from the hand's; both equal to the reference's | |
| HIST: histograms | 0 of 491,520 words differ; both equal to the reference's | |
| SETUP: lists | 0 of 182,272 words differ; both equal to the reference's lists | |
| COUNT, HIST, SETUP: time | 85.34, 23.96, 112.92 us | 87.16, 24.18, 124.34 us |
| Passes: kernel text, B | 15,468, 14,360, 6,452, 1,268, 22,352: 60,608 | 14,144, 16,680, 3,252, 832, 19,812: 55,424 (all seven handlers) |

FILL's 30.6 us. On the hand's slowest core (core 105) the pack CPU waits 50,057
cycles for out pages. Its writer spends about 37,000 cycles before its first item:
the directory's read, and the empty tile's planes built in every command (5,120 word
stores). The class's planes are a kept field, so its writer drains the first item at
once. On its slowest core (core 19) pack waits 716 cycles. The class's writes are also
cheaper per item (median 3,286 cycles against 4,869); that was not chased. SETUP's
11.4 us was not chased either.

## What each line lowers to

| Line | Lowered to | Before it |
|---|---|---|
| `km.Stream(lists.pages(first, count), pattern="A")` with a body on pack | mover A reads each list page; unpack hands its L1 address to pack by mailbox and frees it on pack's release | the hand's reader loop and unpack's hand-off loop |
| `first, count = directory.pages(self.rank, 1).read()[0:2]` | each mover reads the page; unpack hands the two words to math and pack | the reader's header page to compute |
| `first = self.rank * blocks // self.size` | an int of every kernel | the reader's `reader.me() * blocks / cores` |
| `empty: km.L1[...]`, filled by `load_empty` | a field waited for once a command, never freed | the writer's planes, built each command |
| `out = km.own(push="pack")` | a buffer pack pushes and mover B's body pops, no wait of the compiler's | `kCbOut` |
| `self.next_slots.fill(km.Stream(slot_pages.pages(2 * self.rank, 2)))` | mover A's read into the field; mover B's body waits for it and frees it after | the reader's pages `kCbSlots` and `kCbItems`, waited for by the writer |
| `km.cxx(HEADERS + GIVE_UP + "...")` | the sum's text, its `#line` the call's (Inline bodies, classes.md) | |
| `takes=[counters]`, a tensor, on mover B | a `km::Tensor` of its command word | `command.a1` |
| `report`, a write stream on pack and then mover B | the profile page from three CPUs' words, as the ray tracer's | the writer's profile page |

## Moves still bodies

Each is a body in these classes because no line says it yet. With each is the line
that would replace it:

| Move | Body | The line |
|---|---|---|
| a record to its list's next slot (SETUP) | mover B: per item, `next_slot[item]++`, one 128 B NoC write | a scattered write stream whose place is a field's word, bumped |
| an item of each of several pages (SUM) | not run here (X280 path) | items with a step |
| mover A's counters to mover B (FILL's profile) | the own field `handoff` | a mover A word taken on mover B |
| mover B's loop over the pages pack pushes, to the last-page mark | in each writer body: wait, invalidate, pop | a loop over an own field's pages on mover B |
| an own field written to a page by mover B (COUNT's counts, HIST's 64 B a tile) | `km::write_pages`, `noc_async_write` | a write stream mover B fills |
| the directory's words past the first two (FILL's items and frame bases, HIST's tiles) | read into `table` on mover B | `.entries(...)` |
| the hand's pack code included as it is | `passes_pack.hpp`, a copy | defines on an include |

## What a class cannot say of the frame

In the hand's frame, the work cluster's thread (`thread/frame.c`) runs seven commands
in order: COUNT, SUM, HIST, HSUM, PLACE, SETUP, FILL. Between them it reads the
totals, picks the heavy tiles, sorts the tiles by cost, assigns them to cores, and
writes the items, the slots and the directory. That sequence stays the hand's. A class
is a set of handlers that the host submits one command at a time. It has no line for
a command whose words come from the X280's own work. SUM, HSUM and PLACE write a region
of one DRAM bank that the thread reads (`bank - 1` and an offset), not a tensor, and no
class parameter names one. COUNT's balanced ranges (`blocks_word | ranges`, read from
that region) and SETUP's slots from SUM's prefixes (`RF_SETUP_PREFIX`) are the thread's
forms of the same commands. These classes take the host's forms: the even partition,
and next slots given.
