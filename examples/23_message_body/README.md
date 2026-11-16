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

## Over two cards

`islands.py` runs the same kernels on two cards (`Session(cards=[0, 1])`): one `RUN`
over every message rank of both islands, numbered on from island 0 to island 1, so a
thread crosses the link twice a lap. A body crosses in the packet of its message,
right after it; the far link core writes the body into the rank's body page, then the
message, as a local send does. A body over a link is at most 4,224 bytes (a 64-byte
message and its body in one packet of 4,352 bytes); the load refuses a larger one.

```
python islands.py --threads 8                # one link; 0 and 4 KiB
python islands.py --threads 8 --links 4 --hold-links --message-links
```

The numbers below are of the old path, now `--hold-links` (`Dynamic(hold_links=True)`);
by default a hop across is a delivery by the link service: 2.378 µs a hop without a
body, 7.570 with 4 KiB (cards 0 and 1, 2026-11-16).

Cards 0 and 1 (Blackhole p150a, one link), 2026-11-15, 8 threads of 1200 hops over 234
message ranks, 40 crossings each way: both cases exact (every word of every final body
checked); 3.001 µs a hop with the 64-byte message alone, 6.109 µs with a 4 KiB body
changed at every hop (one service). Four services (`--links 4 --message-links`, 232 message
ranks): exact, 2.990 and 6.056 µs a hop; the crossing ranks' messages took links 1 and 3, 40 each way.

On a split card (`--cards 1 --islands 2`, `--graph 0-1`) `--direct` loads the handler as a
part with the pairs r to r + 1 declared and sends each hop by `messages.direct()`; a pair
across islands goes by the link service's L1 write. Card 1, 2026-11-16, 1199 of 1199 hops
direct, exact: split in two 3.556 µs a hop without a body, 6.891 with 4 KiB; a line of
three (`--kernel-bytes 180224`) 3.900 and 7.406.

With `--parts` the handler is an overlay part under Komorebi's own kernels
(`kernels/parts/chain.cpp`, `km_message`): 12 cores, 1200 hops, the body changed, exact
in one run, 6.39 µs a hop against 5.45 compiled in.

`--parts --direct` declares the pairs r to r + 1 (`direct=(1,)`) and the part sends each
hop by `messages.direct()`, through the queue when the slot is busy. On the grid (118
ranks, card 1, 2026-11-16), exact, 1199 of 1199 hops direct: 3.05 µs a hop without a
body, 6.30 with 4 KiB changed at every hop, against 3.13 and 8.25 with `--parts` alone.

## An overlay sent by name

`--parts --send` gives the program two handlers as parts: `start`, the `RUN` command's,
which every core loads at the start and which seeds the thread on rank 0, and `chain`,
which `start` names in its send (`messages.send(0, message, kHandlerChain, body)`). No
core holds `chain` until the thread first arrives there; the reader then loads it from
DRAM before compute calls it. Rank 0 records the wall clock each time the thread comes
round; the first lap holds the 11 other ranks' loads.

```
python message_body.py --count 14 --hops 1200 --bodies 4096 --parts --send
```

Card 1, 2026-11-14, one run, host load 3.7: exact; every rank loaded two handlers
(`start`, then `chain` once), the counter tiles none. Laps of 12 hops: the first 131,822
cycles, the later ones 104,620 to 106,168. At the clock the run gives (1,345 MHz) a later
arrival is 8,759 cycles, 6.51 µs; a first arrival, with the load of `chain`'s three
parts (1,408 + 1,216 + 4,608 bytes), 11,188 cycles, 8.32 µs.

Moving a 4 KiB body costs a hop about 0.42 µs more on 12 cores and 0.45 µs more on the
grid than the line alone. Changing all 1024 words on the three compute threads costs
about 3.06 µs (a third of the body each).

## A cast of a body

`cast_body.py` casts a body to a group instead: the first core of each broadcast of a
rectangle reads a 4 KiB body from DRAM into its broadcast slot and casts it, and every
core of the broadcast, the sender too, gets it in its buffer
([broadcast](../../docs/reference/broadcast.md)). No new call is needed: the broadcast's
buffer has one 4 KiB page a block (`km.Buffer(0, "uint32", 4096, window)`). Each core's
unpack thread folds every word of every body into its L1 area (`cast_fold`), and the
host checks each core's fold against numpy; `--no-fold` only frees each body, for the
cost of the cast alone.

```
python cast_body.py --rect 0,0,12,1 --along row             # 12 cores: two broadcasts of six
python cast_body.py --rect 0,0,12,10 --along column         # the grid: twelve of ten
python cast_body.py --rect 0,0,12,10 --along column --no-fold
```

Card 1, 2026-11-14, 4 jobs of 1000 casts of 8 distinct bodies, window 2, host load 3.0
to 5.7; a cast is the job's time over its casts:

| Group | Folded, every word checked | Freed unread |
|---|---|---|
| a row of 12 cores (two broadcasts of six) | 0 of 12 cores wrong, 12,288,000 words; 10.42 µs a cast | 1.33 µs a cast |
| the grid, 120 cores (twelve broadcasts of ten) | 0 of 120 cores wrong, 122,880,000 words; 10.43 µs a cast | 1.39 µs a cast |

A cast of 4 KiB to a row or a column costs about 1.3 to 1.4 µs; the fold of 1024 words on
one compute thread (about 9 µs) bounds the folded runs.

