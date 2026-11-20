# One load for every box piece (Wormhole islands)

A token runs many programs on the same cores: the attention's projections, window and
attention, the experts' tree and class, and the small parts. A load on 30 islands takes
tens of seconds, so nothing can be loaded between calls. Every piece's programs must sit
in ONE load that stays for the whole run. `shared.py` makes that load.

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

## What every program in the load must agree on

- Compute: HiFi2, float32 in Dst, Dst in full-sync mode (`math_fidelity`,
  `fp32_dest_acc_en`, `dst_full_sync_en`). A subclass's `compute=` replaces its
  parent's, so name all three.
- Buffer formats: at every index that the dict's program uses, an added class holds the
  same format or no buffer (`run.addon` refuses otherwise, naming the index). Classes use
  indices 0 to 25; 26 to 31 are the runtime's own (messages, chain words, tokens, trace).
  Added classes may differ from one another.
- `kernel_bytes`: one value for the session. The attention needs at most 104,448 B.
- At most 75 ranks a command on a joined island, and `km.reduce(..., local=True)` for
  every reduce whose group stays on one island.
- At most 62 programs a load; `Shared(session, reserve=)` is the DRAM room for the
  added programs' parts.
- Nothing kept in a buffer across a switch: state lives in DRAM tensors.

## The check

The attention's `attn` of layers 0 and 3 and the experts' `ffn` of layer 3, in one
session through one load, against golden, on four joined islands:

```
python -m projects.galaxy.driver.verify --golden G --cards 0-3 --kernel-bytes 104448 \
    --piece projects.galaxy.experts.piece:make \
    --piece projects.galaxy.attention_local.piece:make \
    --load "0,3@ffn." --sites attn,ffn --layers 0,3 --prompts en
```
