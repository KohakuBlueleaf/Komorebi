# 20 message overlays

A message's handler loads where the message arrives. This is the program of
[14_message_handlers](../14_message_handlers/README.md), its three handlers `lookup`,
`value` and `report`, with each handler's body an overlay: a compute part that a core
loads into a slot when a message for it arrives and the core does not hold it. The
program has no kernels of its own; Komorebi's reader, compute and writer run the message
loop, and the program gives one source per handler:

| Source | Handler | What it does |
|---|---|---|
| `kernels/lookup.cpp` | `lookup` | seeds the run (the `RUN` command's handler); at a key's owner, looks the key up, and forwards to the next rank (still `lookup`) or to the value's owner as `value` |
| `kernels/value.cpp` | `value` | adds the value to the thread's sum; forwards to the next key's owner as `lookup`, or to the thread's origin as `report` |
| `kernels/report.cpp` | `report` | counts the thread's sum and probes, and consumes it |

Each source defines one function, called for a packet's messages of its handler:

```c++
void km_message(km::Messages& messages, km::Packet& packet) {
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];
        ...
        messages.forward(message[kMsgValue] % count, message, kHandlerValue);
    }
    UNPACK((results[kResValues] += packet.count));  // the state: in L1, one core adds
}
```

A part keeps nothing between calls. What example 14 kept in its compute kernel's locals
(each core's counts and sums) is here 16 words of an L1 tensor per core, its address the
program's compute argument 0 (`messages.arg(0)`); the unpack core adds to it. The checks
are example 14's: the host replays every thread in numpy and compares each core's counts
per handler, its forwards to itself, its reported sums and probes.

```python
program = km.Program(
    "message_overlays",
    kernels=None,  # the runtime's: each handler's body is a part
    args=lambda core: ([], [], [results.address]),  # the handlers' state in L1
    handlers=KERNELS / "handlers.h",
).dynamic("lookup", "value", "report", parts=KERNELS, **options(args))
program.slots(count=args.slots)
```

With three slots, a core loads each handler once, when its first message arrives. With
two or one, a core that meets a handler it does not hold loads it in place of the one it
used least recently; `--loads` counts the loads of every core after the run (the overlay
area's epoch, three steps a handler). `--stray` adds one seed for handler 9, which the
program does not have: the run stops where it arrives, with the reason `handler` and the
message's word 0, and the script checks exactly that. See
[Dynamic handlers](../../docs/reference/dynamic.md#handlers-as-overlays).

## Running it

```
cd examples/20_message_overlays
python message_overlays.py                                   # 1000 threads, 10 steps
python message_overlays.py --threads 10000 --steps 43 --repeat 10 --loads
python message_overlays.py --threads 10000 --steps 43 --slots 2 --loads
python message_overlays.py --count 3 --threads 138 --steps 300 --slots 1 --loads
python message_overlays.py --count 6 --deadline 1 --stray
```

## Measured

Card 0, 2026-11-14, AICLK 1,350 MHz before and after every run, host load 4.8 to 10.3,
64-byte messages, queues of 16,384 messages, `max_live` 16,384. Times are the `RUN`
command's, from its submit by the dispatcher to its completion after the quiet end; a
message is one handling (a lookup, a value or a report). Every run exact against the
replay. The first row is example 14 in the same job; one core is a group of 3 cores (one
rank and two counter tiles).

| Case | Example 14 (bodies compiled in) | As parts, 3 slots | 2 slots | 1 slot |
|---|---|---|---|---|
| 120 cores, 10000 threads of 43 steps: 1,013,360 messages (7,292 to their own core), median of 10 (1 slot: of 3) | 10.83 ms, 93.6 M messages a second | 10.76 ms, 94.2 M a second, 354 loads | 10.77 ms, 94.1 M a second, 13,230 loads | 11.40 ms, 88.9 M a second, 97,394 loads |
| 1 core, 138 threads of 300 steps: 82,938 messages, median of 3 | 21.92 ms, 357 cycles a message | 21.92 ms, 357 cycles, 3 loads | 21.92 ms, 3 loads | 22.21 ms, 362 cycles, 601 loads |

```
python ../14_message_handlers/message_handlers.py --threads 10000 --steps 43 --repeat 10
python message_overlays.py --threads 10000 --steps 43 --repeat 10 --loads [--slots 2]
python message_overlays.py --threads 10000 --steps 43 --repeat 3 --loads --slots 1
python message_overlays.py --count 3 --threads 138 --steps 300 --repeat 3 --loads [--slots 2|1]
python ../14_message_handlers/message_handlers.py --count 3 --threads 138 --steps 300 --repeat 3
```

- With a slot for each handler, the handlers as parts cost nothing measurable: a core
  loads each once, and a held handler costs one look at the slots' records per packet.
- With two slots on the grid, a core loads 112 times in the run (one load per 77 messages
  it handles) and the run takes the same time: it is bound by the queues in DRAM. On one
  core with two slots the core loaded three times in the run, no more: no load replaced
  a handler that a later packet needed.
- With one slot, a load costs the run about 0.5 µs on one core (601 loads, +0.29 ms) and
  0.8 µs a core on the grid.
- A packet's handlers that the core holds go first. Without that order, the same runs
  loaded 20,249 times (two slots, grid) and 5,137 times (one slot, one core, 28.70 ms).
- `--stray`: core 0 gives up with `where` `handler`, `seen` 0x90001 (handler 9, mark 1),
  and the dispatcher ends the run at its deadline.
