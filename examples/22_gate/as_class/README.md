# 22 gate, written as a class

Experimental. The gate of [../gate.py](../gate.py), `out = silu(a) * b`, as a class
deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md#tile-functions)):
the arithmetic is a tile function, `silu_gate`, which the tile compiler
([Tile functions](../../../docs/reference/tile.md)) traces and emits for compute; the
streams and the handler are compiled into the program's files under `build/gate-<hash>/`
beside the script (`movement.cpp`, `compute.cpp`, `gate.h`, and the emitted body
`tile_silu_gate.hpp` with its unit program as text, `tile_silu_gate.txt`).

```bash
python gate_class.py --beside --cores 12   # the hand program first, then the class
```

- `gate(a, b, out, tiles, pattern)`: core k takes the pages from k x `tiles` for
  `tiles` pages, as the hand program's. `zip` of two streams reads a and b in step,
  one `movement.move` of both reads and the write; compute holds DST for 2 tiles (4
  float32 tiles, half of DST), copies each tile's a and b into its 2 DST tiles, runs
  the emitted body on the math CPU for each, and pushes the 2 results.
- `GateTimed` is the same handler with `stamps="stamps"`: the math CPU reads the wall
  clock around each call between two `tensix_sync()`, and the page of each rank is
  written to `stamps` (the calls, 16 empty spans, each call's span).

The build prints the body's cost by the tile compiler's model:

```text
tile functions (the math CPU's vector-unit cycles a tile, or a row of the row form, by the tile compiler's cost model; the copies into DST and the pack not priced):
  gate, line 54, `results.push(silu_gate(x, g))`: silu_gate on a, b (tile_silu_gate.hpp): 1426 instructions, 1597.6 cycles a tile; load_macro: refused: the row computes with 40 instructions (17 MAD-class); a macro arms at most one a sub-unit (simple, MAD, round); replay: 32 of 44 instructions of each row recorded once; tolerance 1e-06 relative against float64
```

The class runs with the hand program's bfloat16 values and check, and with float32
values (normal numbers times 8) against float64 of the same inputs, the tile
function's stated tolerance being 1e-6.

## Numbers

A Blackhole p150a, card 1, pattern `AAAB`, seed 1, 12 cores, 64 tiles a core, one
run of each. Card time is the command's, from its issue to its event; kernel text the
packed text of the five CPUs. The tile compiler's body changed once between the runs:
first `python gate_class.py --beside --cores 12 --tiles 64` (host load 1.9), the body
of 1,614 instructions (1,657.6 cycles printed); then, with the tile compiler's replay
rule and step form, `python gate_class.py --cores 12 --tiles 64 --formats float32`
(host load 1.1), the body above.

| Program | Body | Values | Values wrong | Largest relative error | Card time | Kernel text |
|---|---|---|---|---|---|---|
| hand (`../gate.py`) | | bfloat16 | 0 of 786,432 | 1.45e-2 (values of 2^-4 or more) | 76.06 µs | 38,640 B |
| class | first | bfloat16 | 0 of 786,432 | 3.89e-3 (values of 2^-4 or more) | 86.68 µs | 37,712 B |
| class | first | float32 | 0 of 786,432 | 2.46e-7 against float64 (every value) | 97.74 µs | 37,712 B |
| class, timed | first | float32 | 0 of 786,432 | 2.46e-7 | 102.46 µs | 40,176 B |
| class | above | float32 | 0 of 786,432 | 2.46e-7 | 94.46 µs | 37,536 B |
| class, timed | above | float32 | 0 of 786,432 | 2.46e-7 | 99.46 µs | 40,000 B |

The math CPU's span of a call, 768 calls on 12 cores, less the empty span (10): the
body above 1,596 cycles a tile (median 1,606), against the 1,597.6 the build prints;
the first body 1,656 (median 1,666), against 1,657.6 printed and the 1,656.5 of the
same body in [tile_rules](../../../experiments/language/tile_rules/README.md) (the hand body
there: 1,901.5). With one tile an acquire, the class's bfloat16 command was 14 % longer than the
hand program's. Its error is smaller: the hand program multiplies by b on the matrix
unit at its default fidelity, the emitted body in fp32 on the vector unit.

### Two tiles an acquire

Then the class held DST for 2 tiles a time: `python gate_class.py --beside --cores 12
--tiles 64` on card 1 (host load 2.7), the hand program first, each class in bfloat16
and float32, plain and timed, the body above.

| Program | Values | Values wrong | Largest relative error | Card time | Kernel text |
|---|---|---|---|---|---|
| hand (`../gate.py`) | bfloat16 | 0 of 786,432 | 1.45e-2 (values of 2^-4 or more) | 76.24 µs | 38,640 B |
| class | bfloat16 | 0 of 786,432 | 3.89e-3 (values of 2^-4 or more) | 83.70 µs | 37,632 B |
| class, timed | bfloat16 | 0 of 786,432 | 3.89e-3 | 88.84 µs | 39,512 B |
| class | float32 | 0 of 786,432 | 2.46e-7 against float64 | 94.76 µs | 36,984 B |
| class, timed | float32 | 0 of 786,432 | 2.46e-7 | 100.16 µs | 39,504 B |

The bfloat16 command is 9.8 % longer than the hand program's, against 14 % (the float32
one 94.76 µs against 94.46: the same). A tile is 1,765 cycles of the command at 1,350
MHz against the hand's 1,608: the body's 1,597 (the timed spans, median less the empty
span) and 168 for the copies into DST, the acquire and the pack. Holding DST for 2
tiles took nothing measurable off those 168; what is left of the difference is the
body, the emitted silu and product in float32 on the vector unit against the hand's
library silu and its product by b on the matrix unit, which copies no b into DST.
