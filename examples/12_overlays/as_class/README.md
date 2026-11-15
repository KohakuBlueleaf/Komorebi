# 12 overlays, written as a class

`mix` and `scale` of [../overlays.py](../overlays.py) as two handlers of a class
deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md), experimental), each
marked `@km.handler(overlay=True)`: its code is an overlay, built apart, loaded into a
slot when a command needs it, and called. Made, the class is compiled into the
program's files under `build/overlays/` beside the script: the kernels, whose cases
load and call, and each overlay's parts (`mix/movement.cpp`; `scale/movement.cpp` and
`scale/compute.cpp`). When the session builds the parts it prints where each handler's
code lives.

```bash
python overlays_class.py --beside   # the hand program first, then the class
```

The turns are the hand program's: mix and scale in turn, then each again and again.

- `mix(seed, steps)`: a `km.cxx` body on mover B mixes the core's word; the leader
  returns it. A class has no line that writes a core's own word to DRAM, so a mix turn
  is a command to each core alone, its word the command's result (the hand program's
  one command writes every core's line).
- `scale(source, output, tiles, scalar)`: the core's tiles of `source` are read by a
  stream, multiplied by `scalar` by a `km.cxx` body on compute (`gives=km.Tile`) and
  pushed into a write stream of `output`. The host writes the input values the hand
  program's reader makes.

Measured on one Blackhole card (`--beside`): both exact, the hand program's 40
commands and the class's 2,420 (every mix word and every scaled value). Kernel text
31,088 B against 27,324 B. The parts: mix reader 64 B, writer 192 B; scale reader
3,968, writer 3,968, unpack 2,624, math 1,664, pack 1,664 B (the hand program's: mix
reader 128 B; scale 1,088, 1,408, 1,728, 1,088, 1,728 B); the class has a slot for
each overlay with a part, the hand program's run one a kernel. A turn from the host:
scale 1,379 us against 1,717 us (the hand program's turn also reads its lines), mix
2,794 us for 120 commands against 2,123 us for one.
