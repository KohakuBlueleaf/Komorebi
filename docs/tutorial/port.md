---
title: "A port"
part: 4
summary: "one core sends a tile into the L1 of one other core, and the receiver waits for it at a port"
---

# A port

A port is a counter on the receiving core: the sending core writes a tile into the receiver's
L1 and then adds 1 to the port, and the receiver waits until the port says the tile is
there. This lesson relies on [[why-cores-talk]], [[core-group]] and [[circular-buffers]].

## Why a port

The sender can write straight into another core's L1 over the NoC, but the receiver has to
know when the write has finished, or it would use half a tile. The sender therefore does two
things in order: it writes the tile, and, when the write is done, it adds 1 to a counter in
the receiver's L1. The receiver waits for the counter, and only then uses the tile. The
counter is the **port**; every core has eight of them, numbered 0 to 7.

All three kernels of the two cores take part: on the sender `reader.cpp` fetches the tile and
`writer.cpp` sends it; on the receiver `reader.cpp` waits at the port and `writer.cpp`
uses the tile.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "dram", "x": 0, "y": 0, "w": 190, "h": 40, "t": "DRAM: the source tensor"}, {"id": "out", "x": 0, "y": 80, "w": 170, "h": 52, "t": "core 1\noutgoing buffer: 1 tile", "gem": "amber", "small": true}, {"id": "arr", "x": 280, "y": 80, "w": 170, "h": 52, "t": "core 0\narrival buffer + port", "gem": "amber", "small": true}, {"id": "py", "x": 280, "y": 172, "w": 170, "h": 40, "t": "Python: the result", "gem": "sapphire"}], "arrows": [{"from": "dram", "to": "out", "kind": "data", "t": "reader.cpp", "dx": 8, "dy": 2}, {"from": "out", "to": "arr", "kind": "data", "t": "writer.cpp: emit", "dy": -10, "dx": -60}, {"from": "arr", "to": "py", "kind": "reply", "t": "writer.cpp: result", "dx": 8, "dy": 2}], "narrow": {"boxes": [{"id": "dram", "x": 0, "y": 0, "w": 190, "h": 40, "t": "DRAM: the source tensor"}, {"id": "out", "x": 0, "y": 90, "w": 190, "h": 52, "t": "core 1\noutgoing buffer: 1 tile", "gem": "amber", "small": true}, {"id": "arr", "x": 0, "y": 192, "w": 190, "h": 52, "t": "core 0\narrival buffer + port", "gem": "amber", "small": true}, {"id": "py", "x": 0, "y": 294, "w": 190, "h": 40, "t": "Python: the result", "gem": "sapphire"}], "arrows": [{"from": "dram", "to": "out", "kind": "data", "t": "reader.cpp", "dx": 8, "dy": 2}, {"from": "out", "to": "arr", "kind": "data", "t": "writer.cpp: emit", "dy": 2, "dx": 8}, {"from": "arr", "to": "py", "kind": "reply", "t": "writer.cpp: result", "dx": 8, "dy": 2}]}}
</script>
<figcaption>A tile goes from DRAM to the outgoing buffer of core 1, over the NoC to the arrival buffer of core 0, and a number of it goes to Python.</figcaption>
</figure>

## The general form

In the kernels, with `km::Arrival arrival = {BUFFER, PAGES, 1};` made before `run`, and
`#include "komorebi/device/movement/km_port.hpp"`:

Sender's `writer.cpp`:

```cpp
km::emit(writer, command, arrival, OUTGOING, 1, RECEIVER, PORT, SLOT);
```

Receiver's `reader.cpp`:

```cpp
km::arrive(reader, command, arrival, PORT, 1);
```

Receiver, to use the tile, after `reader.cpp`:

```cpp
writer.wait_front(BUFFER, PAGES, km::kBehindPort);
... get_read_ptr(BUFFER) ...
cb_pop_front(BUFFER, PAGES);
```

In Python:

```python
run.submit(KIND, group=(FIRST, COUNT), args=[...], exclusive=True)
```

- `BUFFER` is the number of the **arrival buffer**, a buffer of `PAGES` tiles that every core
  has. It is declared in `km.Buffer` like any other. A tile sent to a core lands in one
  of its pages.
- `OUTGOING` is the buffer in which the sender has the tile to send.
- `RECEIVER` is the number of the receiving core, as the core numbers of lesson 3.
- `PORT` is which of the 8 counters the sender increments and the receiver waits for: 0 to 7.
- `SLOT` is which page of the arrival buffer the tile lands in: 0 to `PAGES - 1`. Two tiles
  sent to the same core by different senders use different slots.
- `emit` copies the tile, waits until the write is done, increments the port, waits until
  that is done, and pops the outgoing tile.
- `arrive` waits until the port counts `1`, clears it, and tells the other kernels that the
  arrival buffer is full.
- `exclusive=True` makes the dispatcher hold back a command whose group overlaps one still
  running, so that a tile of one command can never land in the slots of the next.

## One line at a time

```{ .cpp }
--8<-- "tutorial/25_port/kernels/reader.cpp:run"
```

Both cores run this `reader.cpp`. The one with rank 1 is the sender: it reads the tile into the
outgoing buffer, as in [[reading-from-dram]]. The one with rank 0 waits at port 0 for 1
arrival.

```{ .cpp }
--8<-- "tutorial/25_port/kernels/writer.cpp:run"
```

The sender's `writer.cpp` calls `emit`: the tile goes to `receiver`, the first core of the
group, port 0, slot 0. The receiver's `writer.cpp` waits until the arrival buffer is full, reads
number `a2` of the tile, sets it as the result ([[the-leader]]), and frees the buffer. The
receiver is rank 0 on purpose: it is the leader, so its result reaches Python.

```{ .python }
--8<-- "tutorial/25_port/port.py:send"
```

Two commands to the group of cores 0 and 1. The first arguments are the source tensor, which
tile core 1 reads, and which number of it core 0 reports.

## The whole program

=== "port.py"

    ``` { .python }
    --8<-- "tutorial/25_port/port.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/25_port/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/25_port/kernels/reader.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/25_port/kernels/writer.cpp"
    ```

The compute file is the empty one.

```{ .text .km-output }
--8<-- "tutorial/25_port/output.txt"
```

The numbers are 1000 plus their place in the whole tensor, so tile 1 starts at 2024 and
tile 3 number 5 is 4077. Core 0 never read the source tensor; the tile came from core 1.

## Common mistakes

- Leaving out `exclusive=True`. The documentation of Komorebi requires it for every
  command that uses ports.
- A sender that sends to one port while the receiver waits at another. The receiver never
  sees its arrival, and the card stops the command at its deadline
  ([[when-it-goes-wrong]]).
- An arrival buffer with fewer slots than the ones used. The call gives up and the command
  ends the same way.
- A receiver that is not the leader and sets the result: Python receives the leader's
  ([[the-leader]]).

Next: [[reduce]] sends from many cores to one.
