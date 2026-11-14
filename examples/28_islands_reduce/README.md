# 28 islands reduce

An all-reduce across islands in one call. A tensor is held as a copy on every island
(`copy=True`); each island's copy is its partial. After one call every island's copy
holds the sum of all of them, added in the tensor's format inside the call. The same
script and kernels run on two cards and on one.

The call from a kernel, on every rank of the command (`kernels/reader.cpp`):

```cpp
km::all_reduce(
    reader, command, command.a0, command.a1, command.a2, command.a3,
    static_cast<km::Sum>(command.a4), fold, kFoldPages * kFoldBytes
);
```

`a0` is the tensor's reference word, `a1` and `a2` the first page and how many, `a3` a
scratch tensor (a copy of twice the pages) and `a4` what is added: `km::Sum::Int32`
(modulo 2^32), `Float32`, or `Bfloat16` (each pair added in float32 and rounded to
nearest even). `fold` is a buffer of L1 the call reads and adds pages in. Every rank of
the command makes the call and shares the adds of its island; it returns once its
island's copy holds the sum. The caller says nothing about links or neighbours: inside
the call the islands reduce their shares along the line (each island ends with the
sum of its own share), then gather them as example 27 does.

The call from the host (`reduce.py`):

```python
run.all_reduce(tensor, scratch)
```

It takes what to add from the tensor's format and sends one REDUCE command to every
rank, which the runtime's service overlay serves by the same `km::all_reduce` (through
the run's transfer buffer).

On one island (`--cards 0`) both calls return at once: the partial is the sum.

## Running it

`reduce.py` writes each island's partial (integers in [-8, 8], so every sum is exact
in each format), calls, and checks every island's copy against the numpy sum, for
`--rounds` rounds, the partials written again before each; the median time of a call
from submit to the event on the host. Then the same with `run.all_reduce`.

```
python reduce.py                      # cards 0 and 1, float32
python reduce.py --dtype bfloat16     # or int32
python reduce.py --cards 0            # one card: one island
```

Blackhole p150a cards, AICLK 1350 MHz, the median of 10 rounds (5 on the split card);
on the card: the command from submit to done on its slowest island; on the host: from
the call to its return:

| Cards | Format | Pages | From a kernel, on the card | From the host, on the card | On the host | Every copy |
|---|---|---|---|---|---|---|
| 0 and 1 | float32 | 256 (1 MiB) | 222.1 µs | 220.8 µs | 491.8 µs | exact |
| 0 and 1 | float32 | 2 | 134.6 µs | 134.5 µs | 400.1 µs | exact |
| 0 and 1 | bfloat16 | 256 (512 KiB) | | | 548.7 µs | exact |
| 0 split in 3 (`--islands 3`) | float32 | 256 | 978.2 µs | 978.6 µs | 1339.7 µs | exact |

The bfloat16 row is from a run before the card's time was recorded.

## Files

| File | What it is |
|---|---|
| `kernels/reduce.h` | the handler, its words, the fold buffer |
| `kernels/reader.cpp` | the call |
| `kernels/writer.cpp`, `kernels/compute.cpp` | nothing to do |
| `reduce.py` | the partials, both calls, the check; one JSON line |
