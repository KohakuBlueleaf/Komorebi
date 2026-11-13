---
title: "Forward and consume"
part: 4
summary: "a message that visits several cores, one after the other, until it ends"
---

# Forward and consume

A core that receives a routed message can send it on to another core, which is a forward, or
end it, which is a consume; a message lives until some core consumes it. This lesson
relies on [[routed-message]].

## Why forward

In [[routed-message]] a message went from the core that made it to one other core and ended
there. Often the work on a message is not done in one place: a ray bounces from one part
of a scene to the next, a record is passed to the next stage. The core that holds the next
part is known only after this part has been worked on. So the receiving core looks at the
message, changes it if it likes, and forwards it to the core the new data names. The
message keeps going until some core decides it is done, and consumes it.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "seeded: rank 0 sends message 0 (value\n1, 4 hops) to rank 0", "gem": "amber"}, {"t": "visit core 0: value 6, 4 hops left:\nforward to rank 0"}, {"t": "visit core 0: value 31, 3 hops left:\nforward to rank 1"}, {"t": "visit core 1: value 59, 2 hops left:\nforward to rank 5"}, {"t": "visit core 5: value 5, 1 hop left:\nforward to rank 5"}, {"t": "visit core 5: value 26, no hops left:\nconsume", "gem": "sage"}]}
</script>
<figcaption>One message of this lesson, 0, with 4 hops: five visits, on cores 0, 0, 1, 5 and 5.</figcaption>
</figure>

## The general form

Inside the `while` loop of [[routed-message]], for each message of the packet:

```cpp
message[WORD] = packet[index][WORD];
...
if (DONE) {
    messages.consume();
} else {
    messages.forward(DESTINATION, message);
}
```

- `message` is a local array of numbers. The message that arrived is read from
  `packet[index]`; the one that goes on is the local array, so the core can change it first.
- Every message that arrives ends in exactly one of two ways: `consume()` or `forward(...)`.
  A core that does neither leaves the message in limbo and the run does not end.
- `messages.forward(DESTINATION, message)` sends `message` on to the core of rank `DESTINATION`.
  A forward to the core itself works the same way.
- `DONE` is the program's rule for when a message ends. Here it is a count of hops left,
  carried in the message.
- A forward does not change the core's quota, and a `consume` gives one unit of quota
  back to the core where the message ends ([[full-buffers]]).
- The run ends when every message that was sent has been consumed. Nobody counts them in
  advance: each core reports how many it has sent and consumed, and the dispatcher ends the
  run when, twice in a row, the two totals are equal and every core is idle.

## One line at a time

```{ .cpp }
--8<-- "tutorial/28_forward_and_consume/kernels/compute.cpp:run"
```

Rank 0 seeds 6 messages, message `id` to rank `id % 6`, each with 4 hops and the value
`id + 1`. At each visit the core makes a new value, `(value * 5 + 1) % 97`, and counts the
visit. If the message has no hops left, it ends here: the core counts it and consumes it.
Otherwise the hops left drop by one, and the message is forwarded to the rank that the new
value names, `value % 6`. At the end, each core stores its visit count and its count of
messages that ended there in L1.

```{ .python }
--8<-- "tutorial/28_forward_and_consume/forward_and_consume.py:run"
```

The command carries the number of messages, the hops and the address of the tensor.

```{ .python }
--8<-- "tutorial/28_forward_and_consume/forward_and_consume.py:replay"
```

The check: the same rules in a few lines of Python, followed message by message. The cores
and the replay must agree on every count.

## The whole program

=== "forward_and_consume.py"

    ``` { .python }
    --8<-- "tutorial/28_forward_and_consume/forward_and_consume.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/28_forward_and_consume/kernels/handlers.h"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/28_forward_and_consume/kernels/compute.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/28_forward_and_consume/output.txt"
```

6 messages with 4 hops are 5 visits each: 30 visits, spread over the cores by the data.
Core 3 and core 4 are visited but no message ends there; at core 2 two messages end. The
last line is the check against the replay.

## Common mistakes

- Neither forwarding nor consuming a message. The message is lost from the run's point of
  view, the totals never match, and the card stops the command at its deadline
  ([[when-it-goes-wrong]]).
- A rule that never ends a message, such as a hop count that is not decreased. The run
  does not end, and the card stops the command at its deadline.
- Forwarding the address `packet[index]` itself instead of your own array. Copy the words
  you want into `message` first, as the program does.

Next: [[full-buffers]] is about a core with too many messages to send.
