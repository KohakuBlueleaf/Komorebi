# Several programs with overlays of their own in one run

Whether programs in the runtime's form that each declare overlays numbered from 1 can
share a run: each program's own overlays are numbered after those of the programs before
it, and the runtime adds the program's shift (`KM_OVERLAY_SHIFT`) to every number its
parts name ([overlays, several programs](../../../docs/reference/overlays.md#several-programs-with-overlays)).
An overlay added to the live run (`run.addon`) takes the lowest number left.

The two programs, `first` and `second`, run the same kernels (`kernels/tag/`): TAG calls
the overlay its a0 names, in the program's own numbers, with the word a1 xor the core's
rank. It writes the result, the number named, the shift its build was given and whether
the overlay was held to the core's line. `first` has `times` (1) and `xor` (2), `second`
has `plus` (1, which the run numbers 3), and `added` (`kernels/added/`) is added after the
load, as the run's 4. If a shift were missing, second's TAG naming 1 would call `times`,
and the results would be wrong.

## Results

One Blackhole p150a, card 0, 2026-11-16, AICLK 1350 MHz, host load 8.3, 120 cores, every core holding
both programs (`session.load([first, second])`):
`python experiments/code/overlay_shift/shift.py --rounds 4`.

| | |
|---|---|
| run numbers | first: times 1, xor 2; second: plus 3 (shift 2); added 4 |
| commands | 16 (each overlay 4 times, in the order times, xor, plus, added), 0 wrong in the result, the number named and the shift on every core |
| a command after one of the same program | 4.04 to 4.22 µs (the first one 6.52 µs: every core loads) |
| a command after one of the other program | 5.50 to 5.68 µs: the program's switch, about 1.5 µs |
| the add of `added`, its part cached | build 14.2 ms, write 0.6 ms, entry commands 4.9 ms |

## Running it

On a card, with tt-metal's environment set:

    python experiments/code/overlay_shift/shift.py --rounds 4    # --cores n: fewer cores

It prints a progress line a command and one JSON line: the commands, the wrong words,
each overlay's times in µs, the run's numbers and `ok`.
