# 27 islands gather

An all-gather across islands in one call. A tensor is held as a copy on every island
(`copy=True`); island c has written only its share, the pages
`partition(pages, c, islands)`. After one call every island's copy holds every share.
The same script and kernels run on two cards and on one.

The call from a kernel, on every rank of the command (`kernels/reader.cpp`):

```cpp
km::all_gather(reader, command, command.a0, command.a1, command.a2);
```

`a0` is the tensor's reference word, `a1` and `a2` the first page and how many. Every
rank of the command makes the call; it returns once its island's copy is whole. The
caller says nothing about links or neighbours: the first of each island's ranks posts
the island's share to the islands beside it on the line and passes on each share that
arrives, the others wait for it.

The call from the host, with no kernel of the program's own (`gather.py`):

```python
run.all_gather(tensor)
```

It sends one GATHER command to every rank, which the runtime's service overlay serves
by the same `km::all_gather`, and returns its events.

On one island (`--cards 0`) both calls return at once: the copy is the whole tensor.

## Running it

`gather.py` writes each island's share into that island's copy (zeros elsewhere),
runs GATHER once and checks every island's copy against the whole, then times
`--rounds` calls (the median, from submit to the event on the host); then the same
with `run.all_gather`.

```
python gather.py               # cards 0 and 1
python gather.py --cards 0     # one card: one island
```

Blackhole p150a cards, AICLK 1350 MHz, 256 float32 tiles (1 MiB), each card writing
half of it, the median of 10 rounds:

| Cards | From a kernel | From the host | Every copy |
|---|---|---|---|
| 0 and 1 | 351.7 µs | 342.8 µs | exact |

## Files

| File | What it is |
|---|---|
| `kernels/gather.h` | the handler and its words |
| `kernels/reader.cpp` | the call |
| `kernels/writer.cpp`, `kernels/compute.cpp` | nothing to do |
| `gather.py` | the shares, both calls, the check; one JSON line |
