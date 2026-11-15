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

The turns are the hand program's: mix and scale in turn, then each again and again;
each turn is one command to every core.

- `mix(words, stamps, seed, steps)`: a `km.cxx` body on mover B mixes the core's word,
  and `words.pages(self.rank, 1).write(word)` writes it to the core's page of `words`.
- `scale(source, output, stamps, tiles, scalar)`: the core's tiles of `source` are
  read by a stream, multiplied by `scalar` by a `km.cxx` body on compute
  (`gives=km.Tile`) and pushed into a write stream of `output`. The host writes the
  input values the hand program's reader makes.

Both are `@km.handler(overlay=True, stamps="stamps")`: each mover's case stamps the
load and the call and writes the cycles to its page of `stamps` (page 2 x core +
mover), read after the turn's time.

Measured on one Blackhole card (`--beside`): both exact, 40 commands each (every mix
word and every scaled value). Kernel text 33,296 B against 28,016 B (mock build). The parts: mix
reader 64 B, writer 384 B; scale reader 3,968, writer 3,968, unpack 2,624, math 1,664,
pack 1,664 B (the hand program's: mix reader 128 B; scale 1,088, 1,408, 1,728, 1,088,
1,728 B); the class has a slot for each overlay with a part, the hand program's run
one a kernel. A turn from the host, median: mix 1,940 us against 2,095 us, scale
1,353 us against 1,731 us (the hand program's turn also reads its lines). The stamps,
median cycles of a held overlay (load, call):

| | Class, mover A | Class, mover B | Hand, reader |
|---|---|---|---|
| mix | 79 / 60 | 67 / 967 | 79 / 595 |
| scale | 108 / 1,531 | 86 / 2,485 | 89 / 21,860 |

Mix's call on mover B is the hand program's loop (595 cycles) and the word's page
written and settled. Scale's calls are not the same work: the hand program's reader
part makes its input tile on the core, value by value, the class's reads it from
DRAM. Loading an overlay (the first turn after the other one): mix 2,019 and 1,779,
scale 5,460 and 5,234 cycles on movers A and B, against the hand program's 1,096 and
1,448; the class's parts are larger.
