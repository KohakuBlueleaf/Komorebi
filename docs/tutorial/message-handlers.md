---
title: "Message handlers"
part: 4
summary: "a message names the handler that takes it where it lands, so a message can change what it does as it moves, or without moving"
---

# Message handlers

A program can have several dynamic handlers, and each message names the one that handles
it where it lands. A forward can keep the handler or give the message a new one, and a
forward to the core itself changes the handler without moving the message. This lesson
relies on [[forward-and-consume]].

## Why a handler in the message

In [[forward-and-consume]] every visit did the same thing. Often a message goes through
stages: it looks something up at one core, uses the answer at another, and reports back
where it started. Each stage is its own piece of code. With one handler, the program
would carry a stage number in the message and switch on it. With several handlers, the
message carries its handler, Komorebi keeps it beside the message, and the compute kernel
asks `messages.handler()` which stage this is.

## The general form

In Python, the program names all its dynamic handlers in one call:

```python
program.dynamic(
    "FIRST", "SECOND", ..., message_bytes=64, max_live=LIVE, queue_messages=QUEUE
)

run.submit("FIRST", group=(0, GROUP), args=[...], exclusive=True)
```

In the compute kernel, inside the `while` loop of [[routed-message]]:

```cpp
message[WORD] = packet[index][WORD];
switch (messages.handler()) {
    case kHandlerFirst:
        messages.forward(DESTINATION, message);  // the same handler
        messages.forward(DESTINATION, message, kHandlerSecond);  // another handler
        ...
}
```

- `.dynamic(...)` takes the names of up to 4 handlers. All of them share one message size
  and one set of queues.
- The `RUN` command names the handler that the run's first messages have: a message from
  `send` during `seeding()` has the `RUN` command's handler.
- `packet[index]` makes message `index` the message being handled; `messages.handler()`
  is its handler.
- `forward(DESTINATION, message)` and `send(DESTINATION, message)` keep the handler of the
  message being handled. With a third argument, the message has that handler where it
  lands.
- `forward(messages.rank(), message, kHandlerSecond)` goes to the core itself: the message
  comes back to this core in a later packet, as the other handler.
- A program with one dynamic handler is written as in [[routed-message]]; nothing changes.

## One line at a time

```{ .python }
--8<-- "tutorial/44_message_handlers/message_handlers.py:program"
```

The program has two dynamic handlers, `climb` and `land`, numbered in `handlers.h`.

```{ .cpp }
--8<-- "tutorial/44_message_handlers/kernels/compute.cpp:run"
```

Rank 0 seeds 6 messages, message `id` to rank `id % 6`, with 3 hops and the value
`id + 1`; the `RUN` command is `climb`, so they climb. A climbing message adds the rank to
its value and goes on to the next rank, keeping its handler. When its hops are used up,
it is forwarded to the same core as `land`. A landing message is counted and consumed.

```{ .python }
--8<-- "tutorial/44_message_handlers/message_handlers.py:replay"
```

The check: the same rules in Python, message by message.

## The whole program

=== "message_handlers.py"

    ``` { .python }
    --8<-- "tutorial/44_message_handlers/message_handlers.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/44_message_handlers/kernels/handlers.h"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/44_message_handlers/kernels/compute.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/44_message_handlers/output.txt"
```

Each message climbs 4 times, on four ranks in a row, and lands on the fourth. Message 0
starts at rank 0 with the value 1 and lands at rank 3 with 1 + 0 + 1 + 2 + 3 = 7.

## Common mistakes

- Switching on `km::handler(command)` inside the loop. That is the `RUN` command's
  handler, the same for the whole run; the message's own is `messages.handler()`.
- Reading `messages.handler()` before `packet[index]`. It is still the handler of the
  previous message.
- A handler the program did not declare as dynamic. Komorebi carries any number, but the
  `switch` has no case for it.

Next: [[full-buffers]] is about a core with too many messages to send.
