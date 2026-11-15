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

Blackhole p150a cards, AICLK 1350 MHz, float32 tiles, each island writing its share,
the median of 10 rounds (5 on the split card); on the card: the command from submit to
done on its slowest island; on the host: from the call to its return:

| Cards | Pages | From a kernel, on the card | From the host, on the card | On the host | Every copy |
|---|---|---|---|---|---|
| 0 and 1 | 256 (1 MiB) | 62.9 µs | 62.8 µs | 326.6 µs | exact |
| 0 and 1 | 2 | 53.7 µs | 53.7 µs | 322.9 µs | exact |
| 0 split in 3 (`--islands 3`) | 256 | 158.7 µs | 2454.4 µs | 2824.3 µs | exact from a kernel; from the host no page moved: its command was taken as the dispatcher's inspect (fixed, not yet run again) |
| 0 (one island) | 256 | 2.2 µs | no command | 0.8 µs | exact |

Most of a call is fixed: 254 more pages add 9.2 µs. On the host each call costs about
270 µs more than on the card (the host's submit and wait). On one island the call
returns at once: from a kernel the time is the command's own, and `run.all_gather`
sends nothing.

These rows are from before two changes: the leader now releases its island's other
ranks with one multicast (it took 4.4 µs as an atomic a rank, now 0.9), and the wait
stores every island's command by one native call (the second island's store, about
5 µs, is gone). Measured since by the account experiment
(`experiments/islands_collectives/account`, two cards, from a kernel, every share
exact): 2 pages 12.7 µs (the command 2.3, the entries and the meeting 1.4, the lines
out 2.0, the other island's share 5.3, the confirm 0.1, the release 0.9); 272 pages
(557 KB each way) 22.6 µs, of which the share's line 16.2 (51 GB/s past its first
page: the link cores' reads in flight). `docs/reference/islands.md` has the account.

## Output

Two cards, then one (`python gather.py`, `python gather.py --cards 0`), the last line
of each run:

```
{"cards": [0, 1], "pages": 256, "bytes": 1048576, "islands": 2, "kernel_exact": true, "kernel_us": 330.95, "kernel_card_us": 62.87, "host_exact": true, "host_us": 326.59, "host_card_us": 62.76, "kernel_errors": ["none", "none"], "ok": true, "wall_s": 10.6}
{"cards": [0], "pages": 256, "bytes": 1048576, "islands": 1, "kernel_exact": true, "kernel_us": 381.62, "kernel_card_us": 2.18, "host_exact": true, "host_us": 0.83, "host_card_us": null, "kernel_errors": ["none"], "ok": true, "wall_s": 16.2}
```

## Files

| File | What it is |
|---|---|
| `kernels/gather.h` | the handler and its words |
| `kernels/reader.cpp` | the call |
| `kernels/writer.cpp`, `kernels/compute.cpp` | nothing to do |
| `gather.py` | the shares, both calls, the check; one JSON line |
