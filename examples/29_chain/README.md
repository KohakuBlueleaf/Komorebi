# 29 chain

Four calls of three handlers of one small program (SCALE, ADD, SUM), each reading what the call before it wrote, run two
ways: a command a step (each submitted and waited for by the host), and one chain (the
four calls written once as records in DRAM, then one command that every core walks by
itself, each record's input waiting for marks from the cores that wrote it, not for
the host). The same kernels serve both. Every output is exact against numpy either way.

| Call (handler) | Cores | Reads | Writes |
|---|---|---|---|
| scale x (SCALE) | 8, one a group | its 2 tiles of `x` | `y` = factor x `x`; the factor is the chain's per-submit word |
| scale y (SCALE) | 8, one a group | its 2 tiles of `y` (the call before) | `z` = 3 x `y` |
| add (ADD) | 8, one a group | its 2 tiles of `z` (the call before) and of `y` (earlier: proven) | `s` = `z` + `y` |
| sum (SUM) | 8, one group | its share of the 16 tiles of `s` (the call before) | `t`, one tile: every tile of `s` added up the group's tree to its root |

The data is int32, worked on in L1 by the movers, so every result is exact.

## The calls, from the host

`chain_steps.py` describes the calls with `komorebi.chain` and gives them to
`run.chain`, which proves the earlier input, writes the records once and returns the
chain; `run.submit(chain, factor=5)` runs it.

```python
own = kc.Range(0, width, width)  # group g: tiles g x width .. + width
add = kc.Call(
    "add",
    (0, t["z"], t["s"], t["y"], 0, 0, width),
    (0, cores),
    1,
    width,
    "s",
    "z",
    ranges=(own,),
    reads=(kc.Read("y", ranges=(own,)),),
    name="add",
)
```

Each call states its handler and ten words, its cores, its groups (`size` cores, here
1) and the block each group's root writes (`width` tiles of `s` from g x width), the
output of the call before that it reads (`z`, by ranges here) and its other inputs
(`y`, written two calls back). The sum states its input as a share of K tiles
(`k=(0, 16)`, shared out over the ranks of its group) and `sends=True`: its parts go up
the group's tree, so the chain gives each sender a credit from its receiver.

`y` is an earlier input: no mark tells `add`'s cores that `y` is written. `run.chain`
proves it from the descriptions: core g's mark from `scale y`'s core g came after that
core waited for `scale x`'s core g, which wrote block g of `y`. A call reading
another core's tiles of `y` is refused with the call, the tensor and the core (the
script builds one on purpose; the text is in the output below).

## On the card

| Kernel | What it does |
|---|---|
| `kernels/reader.cpp` | `km::chain_start()` once; for a chain command `km::chain_reader` walks the records and calls the step for each one this core is in; a step waits for its marks (`km::chain_input`), reads its tiles, scales or adds them in L1, hands them to the writer; the sum also adds the part of each tree round it receives |
| `kernels/writer.cpp` | for a chain command `km::chain_writer`; a step writes its tiles and marks the cores of the next record that read them (`km::chain_mark`); in the sum a rank other than the root waits for its receiver's credit (`km::chain_credit`), then sends its part |
| `kernels/compute.cpp` | the movers do the work; in a chain command compute takes each record in turn (`km::chain_compute`), as every kernel of a core does |
| TOTAL (`kernels/reader.cpp`) | `km::all_reduce` of a copy's page 0 over the islands, then a token to the writer, whose root marks the next record |
| `kernels/runtime/` | the same program in the runtime's form: each part's `km_program` calls the kernel form's function for a command; the runtime's reader zeroes the chain's words before its first command |
| `--form merged` | the runtime's form as two programs merged into one (`km.merge`): `sums` holds SUM (its 3 kept), `maps` SCALE and ADD (1 and 2 become 4 and 5); the chain's records name handlers of both ([the chain](../../docs/reference/chain.md#a-record-of-another-programs-handler)) |

Each record names its handler (the info word's top byte), so a core goes from SCALE to
ADD to SUM by itself; the chain command is sent to the first call's handler with a4 the
chain mode, and every kernel hands a chain command of any of the three handlers to the
walk. The record's group, its info word (marked, sends, the group's size, the output
width) and its ranges are all the device calls use, so the same calls serve this
program and the decode project's.

## Running it

```
python chain_steps.py                 # 8 cores, 2 tiles a core, 200 timed runs a way
python chain_steps.py --form runtime  # the same as parts the runtime's kernels call
python chain_steps.py --form merged   # SUM one program, SCALE and ADD another, merged
```

The script runs a command a step with factor 2, the chain with 5, then 7 and 3, and
checks every word of `y`, `z`, `s` and `t` after each; then 200 runs a way timed, and
the outputs checked once more.

## What it measured

One Blackhole p150a, card 1, 2026-11-15, AICLK 1350 MHz; 200 runs a way. On the host:
from the first submit to the last wait's return, a run; on the card: the dispatcher's
submit-to-done time of each command (a command a step: the four added), the median.

| | A command a step | One chain |
|---|---:|---:|
| on the host, us a run | 703.6 | 266.9 |
| on the card, us a run | 130.6 | 126.3 |
| every word of y, z, s, t | exact | exact |

The card does the same work either way: about 31 us a call, most likely the movers'
integer loops over 2,048 to 5,120 words a core (not timed apart here). What the chain
removes is the host's part: three of the four round trips from a submit to its event,
about 145 us each here (host less card: 573.0 us for four, 140.6 for one).
The four records cost the card 4.3 us less than four commands.

In the runtime's form (card 1, 2026-11-16) every word is exact again; the card takes
131.5 / 126.7 us a run (a command a step / one chain, the kernel form 130.6 / 126.1 in
the same job), the host 838.9 / 302.8 (the kernel form 704.5 / 266.9). In the decode project
the same chain runs a whole token (140 records) as one command; its layer is 315 us
at T 4096, within the spread of its own chain before the move to the framework.

Merged (`--form merged`, card 1, 2026-11-16): every word exact in all five checks, and
the chain's four records name handlers of both sources (SUM of `sums`, SCALE and ADD of
`maps`). The card takes 127.7 / 127.1 us a run, the host 824.9 / 305.9: within the
spread of the runtime's form above, so the dispatcher's switch and the 64-B copy of a
command for the second source do not show at this size.

## As a class

`as_class/steps_class.py` writes the same three handlers as a `km.Program` class (on
float32 tiles: SCALE an inline body, ADD a tile function, SUM a tree's `km.reduce`; the
factor a float word, `factor: float`, which the body takes as a C++ float), and
states the chain without one `kc.Call`: each call's description (its groups, the block
it writes, what it reads and from which call) comes from its handler's lines.

```python
class Steps(StepHandlers):
    @km.chain
    def steps(self, x, y, z, s, t, factor: float, width: int, tiles: int):
        self.scale(x, y, factor, width)
        self.scale(y, z, 3.0, width)
        self.add(z, km.past(y), s, width)  # y: written two calls back
        self.sum(s, t, tiles)


chain = run.chain(program.steps(x, y, z, s, t, 5.0, 2, 16), cores=(0, 8))
listed = run.chain(
    [
        Steps.scale(x, y, 5.0, 2),
        Steps.scale(y, z, 3.0, 2),
        Steps.add(z, km.past(y), s, 2),
        Steps.sum(s, t, 16),
    ],
    cores=(0, 8),
)
```

Both forms give the same records. A class with a `km.chain` method (or made with
`chain=True`) has the walk in its kernels; `StepHandlers`, the same handlers without
one, has none of it. The walk costs kernel text: 74,160 B against 58,240 B (BRISC
+4,332, NCRISC +7,880, each TRISC about +1,220), past the default buffer of 70,656 B, which
the session follows (it opens the card again at the first load).

Card 0, 2026-11-16, AICLK 800 MHz, the machine under load (21.7), 200 runs a way, every
word of y, z, s and t exact after each run below:

| | A command a step | One chain (list) | One chain (method) |
|---|---:|---:|---:|
| on the host, us a run | 946.1 | 160.7 | 160.9 |
| on the card, us a run | 21.25 | 20.86 | 20.88 |

The card's work is float32 tiles on the compute kernel here, not the movers' integer
loops of `chain_steps.py`, so its times are not comparable to the table above. A
command a step of the handlers built without the walk took 20.62 us on the card (966.8
on the host): the walk's checks cost a plain command about 0.16 us (0.63 us over four).

```
python as_class/steps_class.py      # 8 cores, 2 tiles a core, 200 timed runs a way
```

`as_class/overlay_steps.py` makes the three handlers overlays (the same bodies, each
built apart and loaded into a slot when a record needs it) and runs the chain at each
slot count:

```python
class OverlaySteps(Steps):
    scale = km.handler(overlay=True)(Steps.scale.function)
    add = km.handler(overlay=True)(Steps.add.function)
    sum = km.handler(overlay=True)(Steps.sum.function)
```

A record of an overlay handler loads its overlay, waits for the record's input whole,
then calls the part; SUM's tree waits for its credit in the part as in the resident
case. Card 0, 2026-11-16, 8 cores, 200 timed chains a form, every word of y, z, s and t
exact after each chain:

| The chain | us on the card | us from the host | Slots a part | AICLK |
|---|---:|---:|---:|---:|
| resident | 20.88 | 161.25 | | 800 MHz |
| overlays, the default slots | 21.86 | 159.15 | 3 | 800 MHz |
| overlays, 1 slot | 30.06 | 169.82 | 1 | 1,350 MHz |
| overlays, 2 slots | 30.70 | 168.65 | 2 | 1,350 MHz |

The first two rows ran in one job, the last two in another, with the clock read before
each: compare within a pair only. At the default slots every part is held after the
first command, and the 0.98 us is the four records' checks and their whole-input waits.
With fewer slots than handlers (1 or 2 for three) records load in line.

```
python as_class/overlay_steps.py                         # resident, then default slots
python as_class/overlay_steps.py --skip-resident --slots 1,2
```

### The chain run by the class's thread

[as_class/thread_steps.py](as_class/thread_steps.py) gives the class an X280 thread
that runs the chain itself. `km.x280_thread("thread/steps.c", chains={"steps": {...}})`
names the `km.chain` method and its arguments: a tensor of the make by its name (`"x":
"a"`), a value, and `cores=(0, 8)`. A tensor the make lacks is passed with
`thread_tensors={"z": z}`. The thread's header then has
`threadsteps_chain_steps_write(region)`, which writes the four records into the
thread's region `records`, and `threadsteps_chain_steps_command(&command, region,
word)`, which fills the chain command. [thread/steps.c](as_class/thread/steps.c) calls
the two and submits. The host then runs the same method through `run.chain` with
another factor. On card 0, at 1,350 MHz:

| Who ran the chain | Factor | Wrong words (y, z, s, t) | us |
|---|---:|---|---:|
| the thread | 7 | 0, 0, 0, 0 | 43.16 on the thread, from load to return |
| the host | 5 | 0, 0, 0, 0 | 20.82 on the card |

The factor is fixed in the dict: a float parameter is no `km.at_submit` word.

```
python as_class/thread_steps.py --dump outputs.npz
```

## Over two cards

`islands.py` runs the same program on two cards as two islands (236 ranks, 2 tiles a
rank), each island walking the chain on its own cores. Its SUM is a group an island
(`a7` the group's size, read by `ranges` of its island's tiles), each root writing its
island's sum into its own island's copy of `u`; a tree over every rank would need a
credit from one island to another, which `run.chain` refuses (the script prints that
refusal). A second chain adds TOTAL, `km::all_reduce` of `u` over the islands (as
example 28's reader calls it), and ADD of `u`'s page 0 to every tile of `z` into `w`,
which island 0's ranks hold alone, so island 1's ranks put theirs across.

```
python islands.py                 # cards 0 and 1, 50 timed runs a way
```

Two p150a cards, 2026-11-16, the median of 50 runs, every word exact both ways (on the
card: submit to done of the slowest island):

| | a command a step | one chain |
|---|---:|---:|
| four calls, on the card, us | 187.5 | 252.6 |
| four calls, on the host, us | 1,292.1 | 498.3 |
| six calls (TOTAL, ADD into w), on the card, us | 535.7 | 603.0 |
| six calls, on the host, us | 2,181.5 | 841.3 |

The host saves its round trips as on one card. These are the times before the island fix
in the [reference](../../docs/reference/chain.md#over-islands). On the card, each record
after the first took about 30 us more than its command, in a root's walk over every group
of the next record. Since the fix a root walks its own island's groups only (three calls
148.4 -> 108.6 us). The remaining 12 us is a known cost: about 226 ns a group of the
next record. Built with `KM_CHAIN_STAMPS`, STAMPS (handler 5) writes each core's stamps
to page `me` of the bank address in a1.

## Output

```
{"cores": 8, "width": 2, "tiles": 16, "form": "kernel", "refused": "call 2 (add): core 0 reads y tile 2, written by call 0 (scale x) (block 1), and nothing it waits for tells it that block is complete", "kernel": "none", "checks": {"steps x2": {"y": 0, "z": 0, "s": 0, "t": 0}, "chain x5": {"y": 0, "z": 0, "s": 0, "t": 0}, "steps x7": {"y": 0, "z": 0, "s": 0, "t": 0}, "chain x3": {"y": 0, "z": 0, "s": 0, "t": 0}, "after timing": {"y": 0, "z": 0, "s": 0, "t": 0}}, "microseconds_a_run": {"steps": 704.49, "chain": 266.91}, "card_microseconds_a_run": {"steps": 130.58, "chain": 126.12}, "earlier_proven": true, "ok": true, "wall_s": 15.3}
```
