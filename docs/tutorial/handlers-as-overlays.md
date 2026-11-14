---
title: "Handlers as overlays"
part: 4
summary: "each handler's code is loaded by the core where its messages land, when they first land there"
---

# Handlers as overlays

A dynamic handler's code can stay out of the kernels: each handler is a source of its own,
and a core loads it when a message for that handler lands there. The program of
[[message-handlers]] is written here that way, and prints the same lines. This lesson
relies on [[message-handlers]].

## Why load a handler where its messages land

In [[message-handlers]] the compute kernel held every handler's code, and so did every
core. A program with many handlers, or with large ones, would not fit one kernel, and most
cores need only some of the handlers at a time. Here the code of a handler is an
**overlay**: Komorebi builds it on the host and keeps it in DRAM; a core reads it into a
**slot** of its L1 when a message for that handler arrives and the core does not hold it,
then calls it. A core that holds it calls it at once.

## The general form

In Python, the handlers get a folder of sources, and the program no kernels:

```python
program = km.Program(
    "NAME",
    kernels=None,  # Komorebi's reader, compute and writer
    args=lambda core: ([], [], [STATE.address]),  # what the handlers keep, in L1
    handlers=KERNELS / "handlers.h",
).dynamic("FIRST", "SECOND", ..., parts=KERNELS, message_bytes=64, ...)
```

Each source, `KERNELS / "FIRST.cpp"` and so on, defines one function:

```cpp
void km_message(km::Messages& messages, km::Packet& packet) {
    if (messages.seeding()) {  // only in the RUN command's handler, once
        ... messages.send(DESTINATION, message) ...
        return;
    }
    for (uint32_t index = 0; index < packet.count; index++) {
        ... packet[index][WORD] ..., then forward, send or consume
    }
}
```

- `parts=` names the folder; each handler's source is `<name>.cpp`. Komorebi then runs its
  own compute kernel: it takes each packet, and for each handler that has messages in it,
  loads that handler's code where needed and calls its `km_message` with those messages.
- `packet` holds only messages of this handler. There is no `next()`, `done()` or
  `switch` in a handler: Komorebi does those.
- The `RUN` command's handler is called once more, with an empty packet, while
  `messages.seeding()` is true: that is where the first messages are sent.
- A handler keeps nothing between calls. What it counts goes to L1 that the program owns:
  here a tensor with a row per core, its address the program's compute argument 0
  (`messages.arg(0)`). The code runs on the three compute cores at once, so one of them
  writes: `UNPACK((...))`.

## One line at a time

```{ .python }
--8<-- "tutorial/49_handlers_as_overlays/handlers_as_overlays.py:program"
```

No kernels: `parts=KERNELS` says that `climb.cpp` and `land.cpp` in that folder are the
handlers' code. The notes tensor has a row of 4 words in every core's L1; its address goes
to every core as compute argument 0.

```{ .cpp }
--8<-- "tutorial/49_handlers_as_overlays/kernels/climb.cpp:climb"
```

`climb` is the `RUN` command's handler, so it seeds: rank 0 sends the 6 messages, with
`a0` and `a1` of the `RUN` command. In a packet, each message adds the rank to its value
and climbs on, or is forwarded to this core as `land`. The unpack core adds the number of
messages to the core's climbs.

```{ .cpp }
--8<-- "tutorial/49_handlers_as_overlays/kernels/land.cpp:land"
```

`land` adds up the values of its messages, consumes them, and the unpack core adds both
counts to the notes.

## The whole program

=== "handlers_as_overlays.py"

    ``` { .python }
    --8<-- "tutorial/49_handlers_as_overlays/handlers_as_overlays.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/49_handlers_as_overlays/kernels/handlers.h"
    ```

=== "kernels/climb.cpp"

    ``` { .cpp }
    --8<-- "tutorial/49_handlers_as_overlays/kernels/climb.cpp"
    ```

=== "kernels/land.cpp"

    ``` { .cpp }
    --8<-- "tutorial/49_handlers_as_overlays/kernels/land.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/49_handlers_as_overlays/output.txt"
```

The lines are those of [[message-handlers]]: the same messages took the same paths. Each
of the 6 cores loaded `climb` before the seeds, and `land` when its first landing message
arrived.

## Common mistakes

- Keeping a count in a variable outside `km_message`, or in a `static`. It is zeroed on
  every call; the count belongs in the program's L1.
- Writing the state from every compute core. All three run the handler; without
  `UNPACK((...))` the count is added three times.
- Calling `messages.next()` or `messages.done(packet)` in a handler. Komorebi takes and
  frees the packets.
- A message for a handler the program does not have. The core stops the run and says
  where (`handler`, with the message's word 0 in `run.state()`).

Next: [[full-buffers]] is about a core with too many messages to send.
