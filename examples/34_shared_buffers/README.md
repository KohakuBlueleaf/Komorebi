# 34 shared buffers

A handler's buffers live for one call of it, so two handlers need not hold their
streams' buffers at once
([When a class's buffers do not fit](../../docs/reference/classes.md#when-a-classs-buffers-do-not-fit)).
`Three` has three handlers, `first`, `second` and `third`, of one form: each reads
items of three `uint32` tensors and writes two, in one body on the pack CPU: the sum of
the three word by word into the first, their exclusive-or into the second. The items
are 256, 512 and 1,024 B by handler, so no two handlers' pairs can share an index: as
one set the class needs 30 buffer indices, and a class has 0 to 25. Neither the first
way nor sharing by shape fits, so the build takes the third: each handler's ten buffers
at the same indices (0 to 5 and 16 to 19), each index declared once as large as its
largest use, and a core sets a handler's pages at its entry when they differ from the
ones it is set for.

```python
class Three(km.Program):
    @km.handler
    def first(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, s: km.Tensor,
              x: km.Tensor, items: int):  # fmt: skip
        base = self.rank * items
        total = km.Stream(s.items(first=base, count=items, size=256), write=True)
        mixed = km.Stream(x.items(first=base, count=items, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=base, count=items, size=256)),
            km.Stream(b.items(first=base, count=items, size=256)),
            km.Stream(c.items(first=base, count=items, size=256)),
        ):
            km.cxx(
                MIX_256,
                takes=[ia, ib, ic, total, mixed],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )

    # second and third: the same lines with items of 512 and 1,024 B


program = Three(a=a, b=b, c=c, s=s, x=x, session=session)
```

The build report (`program.txt`) gives the way taken, each index's bytes and each
handler's pages at it, and each set of pages:

```text
buffers: at shared indices of other pages (way 3 of 3); commands of handlers that give an index other pages do not overlap on a core
buffer 0: 4096 B; first 4 x 256 B, second 4 x 512 B, third 4 x 1024 B
...
buffer 16: 2048 B; first 2 x 256 B, second 2 x 512 B, third 2 x 1024 B
...
pages 1 (first): 10 buffers
pages 2 (second): 10 buffers
pages 3 (third): 10 buffers
```

`Orders` has three handlers that read items of 256, 512 and 1,024 B from three tensors
in three orders, each item times 3 plus the handler's key into an output of its size. By
a stream's place in its handler they take 36 indices; a stream takes a pair another
handler has of the same role, format and page bytes, so the class is built by shape
(the second way), and nothing runs for it at a handler's entry:

```text
buffers: shared by shape (way 2 of 3)
kWrite0 (buffers 16, 17, 512 B + 512 B): shared by turn0, turn1, turn2
...
kRead2 (buffers 4, 5, 4096 B + 4096 B): shared by turn0, turn1, turn2
```

## Files

- `shared_buffers.py`: both classes, the values, the checks and the timed runs.
- `kernels/shared_buffers.hpp`: `shared_k::mix`, `Three`'s pack body (three items into
  two, their sum and their exclusive-or), and `shared_k::scale`, `Orders`' (an item
  times 3 plus a key).

```bash
python shared_buffers.py --cores 12
```

Each core takes 64 items of each tensor from rank x 64. The script runs `first`,
`second` and `third` once each and checks every word of both outputs against numpy (the
words past the cores' items must stay 0). Then 5 rounds of three timed runs of 200
commands each: `first` alone, `second` alone, and `first` and `second` alternating, each
run's commands made before, queued at once (`run.submit_many`) and waited, its time over
200; the median of the 5 is kept. `second` moves twice the bytes of `first`, so the
change's cost is the alternating command against the mean of the two alone
(`change_us`); the JSON line also gives it against `first` alone (`against_first_us`).
After the timed runs both outputs are checked again. Last, `Orders` runs each handler
once and checks every word of its three outputs. The script prints both classes' report
lines and exits 1 on a wrong word or a command not done.

## Numbers

A Blackhole p150a, card 1, 2026-11-25, AICLK 1,350 MHz, seed 1, 64 items a core, five
runs at each size, every word right in each; each figure the median of the five. A
command's time is the host's, from the queueing of a run to its last event, over 200. A
core reads its items one after another, so a command takes as long on 120 cores as on
12.

```bash
python shared_buffers.py --cores 12
python shared_buffers.py --cores 120
```

| Run | Words wrong (12 / 120 cores) | A command, 12 cores | A command, 120 cores |
|---|---|---|---|
| `first` (256 B items) | 0 / 0 | | |
| `second` (512 B items) | 0 / 0 | | |
| `third` (1,024 B items) | 0 / 0 | | |
| `Orders`: `turn0`, `turn1`, `turn2` | 0 / 0 | | |
| 200 x `first` | 0 / 0 | 183.48 µs | 183.60 µs |
| 200 x `second` | 0 / 0 | 255.42 µs | 255.76 µs |
| 100 x (`first`, `second`) | 0 / 0 | 225.02 µs | 224.84 µs |
| a change of pages (`change_us`) | | 5.88 µs | 6.14 µs |
| against `first` alone (`against_first_us`) | | 41.74 µs | 42.37 µs |

A change of pages costs about 6 µs a command (the alternating command against the mean
of the two alone; 5.2 to 6.4 µs over the ten runs); against `first` alone the
difference is 42 µs, most of which is `second`'s larger items.
