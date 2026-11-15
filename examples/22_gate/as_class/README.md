# 22 gate, written as a class

Experimental. The gate of [../gate.py](../gate.py), `out = silu(a) * b`, as a class
deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md#tile-functions)):
the arithmetic is a tile function, `silu_gate`, which the tile compiler
([Tile functions](../../../docs/reference/tile.md)) traces and emits for compute; the
streams and the handler are compiled into the program's files under `build/gate/`
beside the script (`movement.cpp`, `compute.cpp`, `gate.h`, and the emitted body
`tile_silu_gate.hpp` with its unit program as text, `tile_silu_gate.txt`).

```bash
python gate_class.py --beside --cores 12   # the hand program first, then the class
```

- `gate(a, b, out, tiles, pattern)`: core k takes the pages from k x `tiles` for
  `tiles` pages, as the hand program's. `zip` of two streams reads a and b in step,
  one `movement.move` of both reads and the write; compute copies the two tiles into
  DST 0 and 1, runs the emitted body on the math CPU, and pushes DST 0.
- `GateTimed` is the same handler with `stamps="stamps"`: the math CPU reads the wall
  clock around each call between two `tensix_sync()`, and the page of each rank is
  written to `stamps` (the calls, 16 empty spans, each call's span).

The build prints the body's cost by the tile compiler's model:

```text
tile functions (the math CPU's vector-unit cycles a tile by the tile compiler's cost model; the copies into DST and the pack not priced):
  gate, line 54, `results.push(silu_gate(x, g))`: silu_gate on a, b (tile_silu_gate.hpp): 1614 instructions, 1657.6 cycles a tile; load_macro: refused: the row computes with 46 instructions (17 MAD-class); a macro arms at most one a sub-unit (simple, MAD, round); replay: 32 of 50 instructions of each row recorded once; tolerance 1e-06 relative against float64
```

The class runs with the hand program's bfloat16 values and check, and with float32
values (normal numbers times 8) against float64 of the same inputs, the tile
function's stated tolerance being 1e-6.

## Numbers

A Blackhole p150a, card 1, `python gate_class.py --beside --cores 12 --tiles 64`
(pattern `AAAB`, seed 1, host load 1.9), one run. Card time is the command's, from its
issue to its event; kernel text the packed text of the five CPUs.

| Program | Values | Values wrong | Largest relative error | Card time | Kernel text |
|---|---|---|---|---|---|
| hand (`../gate.py`) | bfloat16 | 0 of 786,432 | 1.45e-2 (values of 2^-4 or more) | 76.06 µs | 38,640 B |
| class | bfloat16 | 0 of 786,432 | 3.89e-3 (values of 2^-4 or more) | 86.68 µs | 37,712 B |
| class | float32 | 0 of 786,432 | 2.46e-7 against float64 (every value) | 97.74 µs | 37,712 B |
| class, timed | float32 | 0 of 786,432 | 2.46e-7 | 102.46 µs | 40,176 B |

The math CPU's span of a call, 768 calls on 12 cores: median 1,666 cycles less the
empty span (10) is 1,656 cycles a tile, against the 1,657.6 the build prints and the
1,656.5 of the same body in [tile_rules](../../../experiments/tile_rules/README.md)
(the hand body there: 1,901.5). The class's bfloat16 command is 14 % longer than the
hand program's: it holds DST for one tile at a time and packs it, the hand program
copies, computes and packs eight tiles an acquire. Its error is smaller: the hand
program multiplies by b on the matrix unit at its default fidelity, the emitted body
in fp32 on the vector unit.
