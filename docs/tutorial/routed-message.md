---
title: "A routed message"
part: 4
summary: "send: a message to a core that the data chooses while the program runs"
---

# A routed message

A routed message is a small block of numbers that one core sends to another core, where the
sender picks the receiver while the program runs, from the data. This lesson relies on
[[why-cores-talk]], [[core-group]] and [[partitions]].

## Why routed messages

A port, a reduce or a broadcast has its senders and receivers fixed by the command before
anything runs. Many programs cannot say that. A ray of light goes to the core that holds the
part of the scene it hit next; a key to be sorted goes to the core in charge of its range.
Which core that is depends on the data.

For this, Komorebi has **dynamic handlers**: a handler whose cores send **messages**, each of
64 bytes (16 numbers), to the core they choose with a single call. Nothing about who
receives is decided in advance. The run ends by itself when every message has ended.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "seeding: rank 0 sends 12 messages;\nmessage v goes to rank (v x v) mod 6", "gem": "amber"}, {"t": "each message waits in the queue of its\ndestination core"}, {"t": "each core: next() gives a packet of\nmessages; it adds the values and\nconsumes them", "gem": "amber"}, {"t": "every message is consumed: the run\nends by itself"}, {"t": "Python reads each core's count and sum\nfrom its L1"}]}
</script>
<figcaption>The life of the program of this lesson.</figcaption>
</figure>

## The general form

In Python:

```python
program = km.Program(
    NAME,
    kernels={"reader": READER, "compute": COMPUTE, "writer": WRITER},
    handlers=HEADER,
).dynamic(HANDLER, message_bytes=64, max_live=LIVE, queue_messages=LIVE)
run.submit(HANDLER, group=(FIRST, COUNT), args=[...], exclusive=True)
```

In the compute kernel, inside the function you give to `run`:

```cpp
km::Messages messages(compute, command);
if (messages.seeding()) {
    messages.send(DESTINATION, message);
    messages.seeded();
}
while (km::Packet packet = messages.next()) {
    for (uint32_t index = 0; index < packet.count; index++) {
        ... packet[index][WORD] ...
        messages.consume();
    }
    messages.done(packet);
}
```

- `READER` and `WRITER` are Komorebi's own: `messages_reader.cpp` and `messages_writer.cpp`, in the
  runtime's kernel folder. Only the compute kernel is yours.
- `.dynamic(HANDLER, ...)` marks the handler as dynamic. `message_bytes` is the size of a
  message. `max_live` is how many messages may exist at once in the whole group;
  `queue_messages` is how many each core's inbox in DRAM holds, and may not be smaller.
- The command starts the **run**. It stays in flight until the run has ended.
- The last 2 cores of the group are **counter tiles**: they keep the inboxes' counters and
  take no messages. A group of 8 has 6 cores that take messages, ranks 0 to 5.
- `message` is an array of 16 numbers. Word 0 is Komorebi's; the others are yours.
- `messages.seeding()` is true once, at the start, and is the place for the first sends.
  `messages.seeded()` ends it.
- `messages.send(DESTINATION, message)` sends a copy of `message` to the core of rank
  `DESTINATION` of the group. It returns false when this core has used up its quota
  ([[full-buffers]]).
- `messages.next()` waits for the next **packet**: up to 32 messages that arrived. It returns an
  empty packet when the run has ended.
- `packet[index]` is the address in L1 of message number `index`; `packet[index][WORD]` is
  one of its numbers.
- `messages.consume()` says that a message ends here. `messages.done(packet)` frees the packet.
- `messages.rank()` is this core's rank, and `messages.count()` the cores that take messages.

## One line at a time

```{ .python }
--8<-- "tutorial/28_routed_message/routed_message.py:program"
```

The program takes its reader and writer from the runtime and its compute kernel from this
folder, and declares the handler dynamic with room for 4096 messages.

```{ .cpp }
--8<-- "tutorial/28_routed_message/kernels/compute.cpp:run"
```

The first part is seeding: rank 0 sends one message for each value from 0 to 11. The value
goes in word 1, and the destination is `value * value % messages.count()`: the square of the
value, modulo the 6 cores. The squares of 0 to 11 modulo 6 can only be 0, 1, 3 and 4, so cores
2 and 5 will receive nothing. The destination comes from the data: change the values and
the destinations change, with no change to anything else.

The `while` loop is the same on every core, including rank 0. It takes packets, counts each
message, adds its value, and consumes it. At the end, the compute kernel stores
the count and the sum in the core's L1, where Python reads them ([[why-cores-talk]]).

```{ .python }
--8<-- "tutorial/28_routed_message/routed_message.py:route"
```

One command, which stays in flight until every message has been consumed. `wait` returns
then. The command's arguments are the number of messages and the address of the L1
tensor.

## The whole program

=== "routed_message.py"

    ``` { .python }
    --8<-- "tutorial/28_routed_message/routed_message.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/28_routed_message/kernels/handlers.h"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/28_routed_message/kernels/compute.cpp"
    ```

This folder holds only the compute kernel: `reader.cpp` and `writer.cpp` are the runtime's.

```{ .text .km-output }
--8<-- "tutorial/28_routed_message/output.txt"
```

Rank 0 receives the values 0 and 6, rank 1 values 1, 5, 7 and 11 (24 in all), rank 3
values 3 and 9, and rank 4 values 2, 4, 8 and 10. The program checks every row against
the same calculation done in numpy.

## Common mistakes

- A destination outside the group, such as rank 8 here. Komorebi stops the run with the
  reason `bad dest`. The ranks of the counter tiles take no message either.
- Sending without consuming. Every message has to end with a `consume` somewhere, or the
  run never ends and the card stops the command at its deadline
  ([[when-it-goes-wrong]]).
- Sending more messages than the quota ([[full-buffers]]).
- Forgetting `exclusive=True`, or sending a second dynamic command while the first is in
  flight. One dynamic run at a time.

Next: [[forward-and-consume]] is about a message that goes on to several cores.
