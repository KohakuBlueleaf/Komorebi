# 33 kernel handler

A hand-written kernel as a handler of a class, beside a handler written in the DSL
([A handler that is a hand-written kernel](../../docs/reference/classes.md#a-handler-that-is-a-hand-written-kernel)).
`gate` is [example 22](../22_gate/README.md)'s two kernels as functions, named by
`km.kernel`; it streams through three pairs the class declares with `km.Pair`. `add`
is written in the DSL (`a + b`, a tile function). The class builds one program of
both, and the same two handlers split into two classes merged by `km.merge` build the
same kernels.

A hand kernel in a class runs under the class's compute options (float32 Dst, which
holds half the tiles a 16-bit Dst does), so a kernel written for other options is
adapted, as this gate was: blocks of 4 tiles where example 22 has 8.

```python
class Mixed(km.Program):
    BLOCK = km.constant(4)  # kConstBlock in the header
    a_in = km.Pair(km.bf16, pages=BLOCK * 4)  # kPairAIn
    b_in = km.Pair(km.bf16, pages=BLOCK * 4)  # kPairBIn
    result = km.Pair(km.bf16, pages=BLOCK * 4, write=True)  # kPairResult

    @km.handler(
        km.kernel(
            movement="kernels/gate_movement.hpp",
            compute="kernels/gate_compute.hpp",
            entry="gate_k::gate",
        )
    )
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        """out = silu(a) * b."""
```

## Files

- `kernel_handler.py`: the classes (`Mixed`; `GateOnly` and `AddOnly` for
  `--merged`), the values and the check of `../22_gate/gate.py`, the runs.
- `kernels/gate_movement.hpp`: `gate_k::gate_movement`, example 22's
  `movement.cpp` as a function template over the tensors' types, called on both
  movers.
- `kernels/gate_compute.hpp`: `gate_k::gate_compute`, example 22's `compute.cpp` as
  a function, called on the three compute CPUs.
- `kernels/hand/`: the same two functions as a hand program of their own (`--hand`):
  `hand.h` holds the names the class's header gives them, at the class's values;
  `movement.cpp` and `compute.cpp` each a `kernel_main` calling them. It is built
  with the class's compute options (HiFi4, `fp32_dest_acc_en`).

```bash
python kernel_handler.py --cores 12 --hand       # the hand program, then the class
python kernel_handler.py --cores 12 --merged     # two classes, km.merge
python kernel_handler.py --cores 12 --add-alone  # also add in a class of its own
```

Each core takes 64 tiles (`--tiles`) from rank x 64. The script runs `gate` and `add`
`--calls` times each (5), each command's output cleared before it and judged after
it, then the two alternately 20 times, each judged. With `--hand` the hand program
runs first, and every `gate` command of the class must give its output word for word
(`hand_words_exact`). Both are judged by the hand
program's check against float64 (relative 2^-6 plus 2^-10). A command's time is its
event's, and the median of `--calls` is printed. The JSON line's `ok` keys say each
run was right.

## Numbers

A Blackhole p150a, card 1, 2026-11-25, 12 cores, 64 tiles a core (786,432 values), pattern
`AAAB`, seed 1. A command's time is its event's, the median of 5. Kernel text is the
packed text of the five CPUs.

```bash
python ../22_gate/gate.py --cores 12
python kernel_handler.py --cores 12 --hand --add-alone
python kernel_handler.py --cores 12 --hand --merged
```

| Program | Handler | Values wrong | Words as the hand program's | Card time | Kernel text |
|---|---|---|---|---|---|
| example 22's own program (a different program: 16-bit Dst, blocks of 8) | gate | 0 | | 76.72 µs | |
| the hand program (`--hand`: float32 Dst, blocks of 4) | gate | 0 | | 94.28 µs | 40,068 B |
| class `Mixed` | gate | 0 | every word, 5 of 5 commands | 94.44 µs | 50,668 B |
| class `Mixed` | add | 0 | | 23.76 µs | |
| class `AddOnly` (`--add-alone`) | add | 0 | | 23.32 µs | |
| `GateOnly` and `AddOnly` merged | gate | 0 | every word, 5 of 5 | 94.46 µs (hand 94.22) | 50,668 B |
| `GateOnly` and `AddOnly` merged | add | 0 | | 23.74 µs | |

The 20 alternations of gate and add were right in both forms (40 commands, 0 wrong,
gate's words the hand program's). The class's gate is 0.17 % over the hand program's
(0.25 % merged). Add pays for the state set again at its entry beside a hand kernel
(the start-up's formats, Dst cleared): 23.76 µs against 23.32 µs in a class of its
own. The largest relative error of the gate is 1.11e-2 (values of 2^-4 or more), and
of add 3.89e-3.
