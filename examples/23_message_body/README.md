# 23 message body

A message carries a body. A thread of the program is one message whose 64-byte line
holds its state and whose body, 4 KiB here, holds its data; the thread moves from rank to
rank, and every core it arrives at changes every word of the body and forwards it, body
and all, to the next rank:

| Step | Where | What it does |
|---|---|---|
| seed | rank t, for thread t | writes the thread's first body (`chain_seed`) into its L1 area and sends the message with it to rank t + 1 |
| hop | every rank the thread arrives at | changes each of the body's 1024 words with the hop's key (`chain_word`), the line's state word too, and forwards the message with its body to the next rank |
| end | the rank of the last hop | copies the body into its L1 area and consumes the message |

The three compute threads change a third of the body each. Words 0 and 1 of the line are
the runtime's (its mark and handler, the body's bytes); the program's fields are words 2
to 4. The host replays every thread in numpy and checks each rank's arrivals, which
thread ended where, its state word and every word of its final body. The
[dynamic handlers](../../docs/reference/dynamic.md#a-message-with-a-body) page has the
calls.

```c++
const km::Body body = packet.body(index);  // the body in L1: address, bytes
... change body.words()[0 .. body.bytes / 4) ...
messages.forward((rank + 1) % count, message, body);  // returns once the body has left
```

```python
program = km.Program(
    "message_body",
    kernels={"reader": ..., "compute": ..., "writer": ...},
    handlers={"chain": 1},
).dynamic("chain", max_live=128, queue_messages=128, body_bytes=4096)
handle = run.submit("chain", (0, count), [threads, hops, 4096, 1], exclusive=True)
```

## Running it

```
cd examples/23_message_body
python message_body.py                       # the grid, 1 thread, 1200 hops, 0 and 4 KiB
python message_body.py --count 14 --hops 1200 --keep --repeat 3
python message_body.py --count 4 --hops 10 --threads 2 --bodies 0 4096 --keep
```

`--count` 0 (the default) is every core; the group's last two cores are its counter
tiles. `--bodies` lists the cases: 0 is the same chain with the 64-byte line alone (a
program without bodies); `--keep` adds each body carried unchanged, which separates
moving the body from changing it. Threads are at most the ranks, so each ends on its own
rank.

## Measured

Card 1, 2026-11-14, one thread (so the hops follow one another), queues of 128 messages,
`max_live` 128, host load 5.7 to 6.6. A hop is the `RUN` command's time, from its submit
by the dispatcher to its completion after the quiet end, over the hops; three runs each,
all exact (every word of the final body checked).

| Chain | 64-byte message | 4 KiB body, carried | 4 KiB body, changed at every hop |
|---|---|---|---|
| 12 cores (14 with the counter tiles), 1200 hops | 1.974 to 1.978 µs a hop | 2.394 to 2.397 µs | 5.452 to 5.454 µs |
| the grid, 118 ranks (120 cores), 1180 hops | 2.081 to 2.099 µs a hop | 2.532 to 2.541 µs | 5.564 to 5.599 µs |

Moving a 4 KiB body costs a hop about 0.42 µs more on 12 cores and 0.45 µs more on the
grid than the line alone. Changing all 1024 words on the three compute threads costs
about 3.06 µs (a third of the body each).
