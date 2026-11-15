# 14 message handlers, written as a class

Experimental. The program of [../message_handlers.py](../message_handlers.py) as a
class deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md#dynamic-handlers-and-messages)):
its three dynamic handlers, `lookup`, `value` and `report`, are methods, each a loop
over the messages that name it; the build makes them one message loop of the compute
kernel, a case for each handler (`build/messagehandlers-<hash>/` beside the script). The
reader and writer are the runtime's.

```bash
python handlers_class.py --beside --threads 10000 --steps 43 --repeat 3
```

- `lookup(threads, steps)` is the RUN command's: it zeroes the core's totals, seeds
  its share of the threads (`self.send(owner, seed)`: a seed has lookup's handler) and
  looks keys up; a rank that does not hold the key forwards it to the next rank, still
  as lookup, the rank that holds it to the value's owner with
  `to=MessageHandlers.value`.
- `value` adds the value to the thread's sum and forwards to the next key's owner
  (`to=MessageHandlers.lookup`) or to the thread's origin (`to=MessageHandlers.report`).
- `report` counts the thread's sum (64 bits over two words) and probes, and consumes it.
- Each core's counts are `totals: km.L1[km.u32, 16]`, a field of words: the class is
  made with `totals=`, an L1 tensor of a row a core; the unpack CPU writes the words
  (`self.totals[LOOKUPS] += 1`), and the host reads the tensor after the run and checks
  every core against the numpy replay, as the hand program checks its results words.
  The class writes no run word (the hand's `kResRun`): each run zeroes the totals.

The table is the hand's 32-bit mixer written in the class's lines (each word wraps at
2^32, as the C++ does).

## Numbers

A Blackhole p150a, card 1, `python handlers_class.py --beside --threads 10000 --steps 43
--repeat 3`: 120 cores (2 counter tiles), 64-byte messages, queues of 16,384 messages,
`max_live` 16,384, host load 1.34, a load a run. Time is the RUN command's, from its
submit by the dispatcher to its completion after the quiet end. Kernel text from the
mock build.

| Program | Result | RUN time, 3 runs | Messages a second (median) | Kernel text |
|---|---|---|---|---|
| hand | exact in 3 of 3: 1,013,360 messages (573,360 lookup, 430,000 value, 10,000 report; 7,292 to their own core) | 11,133.7 / 11,184.1 / 11,149.0 us | 90.9 M | 43,568 B |
| class | exact in 3 of 3, the same counts | 10,986.6 / 10,972.7 / 11,075.0 us | 92.2 M | 42,640 B |

The class's median run is 1.5 % shorter than the hand's; the difference is not
accounted.
