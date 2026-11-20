# One load for every box piece (Wormhole islands)

A token runs many programs on the same cores: the attention's projections, window and
attention, the experts' tree and class, and the small parts. A load on 30 islands takes
tens of seconds, so nothing can be loaded between calls. Every piece's programs must sit
in ONE load that stays for the whole run. `shared.py` makes that load.

The shared load is the form that makes the whole model RUN in one session. It does not
carry chains today: the framework refuses a program that runs chains in a load whose
groups switch program (`src/komorebi/program/several.py`), so a layer cannot yet be one
chain inside it.

## How a piece takes part

A piece's factory takes one more argument, the shared load:

```python
def make(session, weights, layers, run=None):
    ...
    # where the piece made its own load:
    if run is not None:
        run.take(self.groups)  # {program: group}, the hand program first
        self.run = run  # load / submit / wait / store go through it
```

With `run=None` the piece makes a load of its own, exactly as before. The driver makes
every piece with the same `Shared`, then starts it once:

```python
run = Shared(session)
pieces = [make(session, W, layers, run=run) for make in factories]
run.start()
```

`start` takes the first program as the dict's program, on every rank. A hand program (a
kernel set in the runtime's form) must be that one, because a hand program cannot be
added. `start` names every other program ahead and gives one slot a kernel, with the
service part sharing the reader's slot. It then loads, and adds the other programs on
every island. A piece then switches its cores with `run.load(program, group)` and
submits its commands as it did on its own run. Before a switch it asks
`run.holds(program, group)` rather than remembering what it loaded last, because
another piece may have switched those cores since; `Shared` keeps what every rank holds.
`driver/verify.py` makes its pieces this way when `--piece` is given more than once.
With no hand program, `hold(session)` gives a small dict's program
(`start(hold(session))`).

A switch touches only the group it names. Every class is added on every island (its
parts in each island's room, whatever group was taken), so the groups given to `take`
only order the programs; what a piece switches is the group it passes to `run.load`.
Under the home placement (every weight but the routed experts on island 0) the attention,
the global attention and the parts switch only island 0's ranks (their command groups
are at most 73 cores); the experts' classes switch every island's.

## What every program in the load must agree on

- Compute: HiFi2, float32 in Dst, Dst in full-sync mode (`math_fidelity`,
  `fp32_dest_acc_en`, `dst_full_sync_en`). A subclass's `compute=` replaces its
  parent's, so name all three.
- Buffer indices: an added class takes its own formats at its own indices, even where
  the dict's program has a buffer of another format. In a switching load a program's
  buffers stay off 25 and 26 to 31, the runtime's own (the switch, messages, chain
  words, tokens, trace); a class built `beside` another can be pushed onto 25 and is
  refused, so build it beside only for a list load of its own.
- `kernel_bytes`: one value for the session. The attention needs at most 104,448 B.
- At most 75 ranks a command on a joined island, and `km.reduce(..., local=True)` for
  every reduce whose group stays on one island.
- At most 62 programs a load; `Shared(session, reserve=)` is the DRAM room for the
  added programs' parts.
- Nothing kept in a buffer across a switch: state lives in DRAM tensors.

## The check

Every piece in one session through one load, against golden, on four joined islands:
the attention's `attn`, the experts' `ffn`, the parts' norms, mixing, Engram, head and
choice, with no load between any two calls.

```
python -m projects.galaxy.driver.verify --golden G --cards 0-3 --kernel-bytes 104448 \
    --piece projects.galaxy.experts.piece:make \
    --piece projects.galaxy.attention_local.piece:make \
    --piece projects.galaxy.parts.piece:make \
    --load "0,1,3,none@ffn.,engram.,head." --layers 0,1,3,none --prompts en \
    --sites attn,ffn,attn_norm,hc_attn_mixes,hc_attn_pre,hc_attn_post,engram_mix,head,choose
```

Every row equals the same piece's run alone (each in a load of its own) through the
same verify.

## On more islands

`check.py` loads the four pieces' classes in one load at the attention's kernel_bytes
and prints the bytes of the load's kernels (what the kernel buffer holds: the dict's
program's build; the added classes' code is in their parts), the load's phases, a
switch's time on the first island and on every island, and whether a host write on
every island after a switch reads back equal:

```
python -m projects.galaxy.load.check --cards 0-7 --ckpt C
```

On four islands, the build cache warm: the dict's load 9.8 s, the ten added classes
0.2 to 0.6 s each (13 s in all); a switch 0.2 ms on the first island's ranks, 0.4 ms on
every island's; the write reads back equal. Each island's added classes took the same
room, 3.4 MB of the 23 MB reserved (read once from each island run's room, which no
public call reports; the load's log names each added class's parts): an island has its
own room, so the count of islands does not change it.

What grows with the islands is the session's own L1 on every core: the island table
(128 B on 4 islands, 192 B on 8, 768 B on 31) and a cached tensor table (8,896 B, 12,992
B, 33,472 B), 25 KB more on 31 islands than on 4. The attention's Projections, the
largest program, use 1,239,120 of 1,359,520 B on 4 islands in the shared load.
