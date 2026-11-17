# 14 message handlers

A message names its handler. The program has three dynamic handlers, and a thread of the
program is one message that goes through them as it moves over the grid:

| Handler | Where | What it does |
|---|---|---|
| `lookup` | the key's owner, rank `key mod count` | looks the key up; a rank that does not hold it forwards to the next rank, still as `lookup`; the rank that holds it forwards to the value's owner as `value` |
| `value` | the value's owner, rank `value mod count` | adds the value to the thread's sum; while steps are left, forwards to the next key's owner as `lookup`, else to the thread's origin as `report` |
| `report` | the rank that seeded the thread | counts the thread's sum and probes, and consumes it |

A message is `{thread, origin, key, value, steps left, sum, probes}` (words 1 to 7 of
16); word 0 is the runtime's and holds the handler. The table is a 32-bit mixer, so the
cores load nothing: a rank holds a key three times in four, and the rank before the
key's owner (the last one asked) always holds it. When an owner is the core itself,
the forward goes to the core itself, and the message comes back to it as the next handler.

The `RUN` command is `lookup`: the seeds, sent without a handler, have it. The host
replays every thread in numpy and checks each core's counts per handler, its forwards to
itself, its reported sums (64 bits) and probes; a report at the wrong core or a message
with another handler is counted as wrong. The [dynamic handlers](../../docs/reference/dynamic.md)
page has the calls.

```python
program = km.Program(
    "message_handlers",
    kernels={"reader": ..., "compute": ..., "writer": ...},
    handlers=HERE / "kernels" / "handlers.h",
).dynamic("lookup", "value", "report", max_live=1 << 14, queue_messages=1 << 14)
handle = run.submit("lookup", (0, count), [threads, steps], exclusive=True)
```

## Running it

```
cd examples/14_message_handlers
python message_handlers.py                                   # 1000 threads, 10 steps
python message_handlers.py --threads 10000 --steps 43 --repeat 10
python message_handlers.py --count 6 --threads 400 --steps 100
```

`--count` 0 (the default) is every core; the group's last two cores are its counter
tiles. Each core seeds its share of the threads with its quota, `max_live` over the
run's ranks (138 with the defaults, every core): more threads per rank than that are
refused seeds, and the run is not exact.

## Measured

Card 0, 2026-11-14, 64-byte messages, queues of 16384 messages, `max_live` 16384. Times
are the `RUN` command's, from its submit by the dispatcher to its completion after the
quiet end; a message is one handling (a lookup, a value or a report).

| Case | Result |
|---|---|
| 120 cores, 10000 threads of 43 steps: 1,013,360 messages (573,360 lookup, 430,000 value, 10,000 report; 7,292 to their own core) | exact in 10 of 10 runs; median 10.80 ms (10.68 to 10.90), 93.9 M messages a second |
| 120 cores, 1000 threads of 10 steps: 24,280 messages | exact; 1.30 ms, 18.7 M messages a second |
| 6 cores (4 ranks), 400 threads of 100 steps: 93,626 messages, 19,993 to their own core | exact in 3 of 3; 6.24 ms, 15.0 M messages a second (card 1) |

The [forward example](../05_forward/README.md#measured), with one handler, moves 10^6
messages in 11.9 ms on the same card.
