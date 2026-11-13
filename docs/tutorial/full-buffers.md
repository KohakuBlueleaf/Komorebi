---
title: "Full buffers"
part: 4
summary: "what stops a sender when the receiver is busy, and the quota of messages in a dynamic run"
---

# Full buffers

A sender must not overrun a receiver that is busy, and Komorebi does it differently for each
kind of message: in a dynamic run it gives each core a **quota**, a number of sends that it
may have alive, and `send` returns false when the quota is used up. This lesson relies on
[[routed-message]] and [[forward-and-consume]].

## Why a limit

A receiver takes messages at its own speed. If senders may send as many as they like, the
messages pile up and something overflows: a buffer in L1 with no more room, or a queue in
DRAM that wraps round over messages not yet read. A program that loses messages is wrong in
ways that depend on timing, which is the worst kind. So every way of sending has a rule
that makes overflow impossible:

| Kind | What stops a sender | What the sender sees |
|---|---|---|
| port ([[port]]) | each sender has a slot of its own in the receiver's arrival buffer, and `exclusive=True` keeps two commands' tiles apart | nothing: it can always write |
| broadcast ([[broadcast]]) | a window of blocks in flight: a slot is rewritten only when every receiver has finished with it | `reserve` waits |
| routed message ([[routed-message]]) | a quota of messages alive for each core | `send` returns false |

In a dynamic run every core has a **quota** to start with: `max_live` divided by the number
of cores that take messages. `send` takes one unit. `consume` gives one unit back to the core
where the message ends. A `forward` changes nothing. With `max_live` messages at most alive
in the whole run, and an inbox of at least `max_live` messages for each core, no inbox can
overflow, even when every message is addressed to one core. The price is that quota stays
with the messages: a core that only sends runs dry, and a core that only consumes collects
more than it started with.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "r0", "x": 0, "y": 0, "w": 150, "h": 56, "t": "rank 0\nquota 10: sends 10,\nthe 11th says no", "gem": "amber", "small": true}, {"id": "q", "x": 220, "y": 0, "w": 160, "h": 56, "t": "inboxes in DRAM\n(never more than\nmax_live in all)", "small": true}, {"id": "rn", "x": 450, "y": 0, "w": 150, "h": 56, "t": "ranks 0 to 5\nconsume: quota\ncomes back", "gem": "amber", "small": true}], "arrows": [{"from": "r0", "to": "q", "kind": "data"}, {"from": "q", "to": "rn", "kind": "data"}]}
</script>
<figcaption>Rank 0 starts with 10 units of quota. Each send takes one; each consume, on the receiving cores, gives one back to them.</figcaption>
</figure>

## The general form

```cpp
if (!dyn.send(DESTINATION, message)) {
    ...  // no quota left: do something else
}
const uint32_t left = dyn.quota();
```

```python
program.dynamic(HANDLER, message_bytes=64, max_live=LIVE, queue_messages=QUEUE)
```

- `dyn.send` returns `true` when the message was sent and `false` when this core has no
  quota. When it returns false, nothing was sent.
- `dyn.quota()` is the sends this core has left.
- `LIVE` is `max_live`, the most messages alive at once on the card. Each core starts with
  `LIVE / 118`: 120 cores less 2 counter tiles. `LIVE` must be at least 118.
- `QUEUE` is `queue_messages`, the room for messages in each core's inbox, a power of two. It
  may not be smaller than `LIVE`: Komorebi refuses such a program before it opens the card.

## One line at a time

```{ .python }
--8<-- "tutorial/29_full_buffers/full_buffers.py:program"
```

`max_live` is 1180: 10 units of quota for each of the 118 cores. The inboxes hold 2048.

```{ .cpp }
--8<-- "tutorial/29_full_buffers/kernels/compute.cpp:run"
```

Rank 0 tries twice as many sends as its quota, counting the ones that were accepted: the
first 10 are, the other 10 return false. The messages are spread round the 6 cores. Every
core consumes what it gets and reads its quota at the end.

## The whole program

=== "full_buffers.py"

    ``` { .python }
    --8<-- "tutorial/29_full_buffers/full_buffers.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/29_full_buffers/kernels/handlers.h"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/29_full_buffers/kernels/compute.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/29_full_buffers/output.txt"
```

Rank 0 had 10 units, sent 10 messages and was refused the other 10 sends. Its 10 messages
went to ranks 0 to 5 (ranks 0 to 3 got two each, ranks 4 and 5 one), and each message
that ended at a core gave that core a unit back. The core's quota at the end is its 10, less
what it sent, plus what it received: 2 for rank 0, 12 for ranks 1 to 3, 11 for ranks 4 and 5.
The script checks each of these.

## Common mistakes

- Ignoring the result of `send`. A refused send sends nothing; a program that does not look
  has lost that message.
- A seed that sends more messages than the quota. The seeding sends count too.
- A `max_live` that is less than the number of cores: "no quota" is the error, before the
  card is opened.
- Moving the quota between cores. Komorebi does not do it: a program in which some cores
  only send and others only consume runs the senders dry.

Next: [[streams-and-graphs]] describes a whole chain of steps once, instead of core by core.
