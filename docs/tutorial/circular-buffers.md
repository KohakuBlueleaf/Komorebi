---
title: "Circular buffers"
part: 3
summary: "a buffer of several pages: a queue in L1 that lets the reader run ahead of the writer"
---

# Circular buffers

A circular buffer is a queue of pages in a core's L1: one side puts tiles in at the end,
the other takes them out at the front, and when either side reaches the last page it goes
on at the first. This lesson relies on [[three-kernels]].

## Why more than one page

In the last lesson the buffer held one tile. The reader put a tile in and the writer took
it out, and the reader could not start the next read until the writer had finished with
the tile in the only page. The two CPUs took turns when they could have worked together.

With room for two tiles the reader fills the second page while the writer still writes the
first. The buffer is the place where the two meet, and the more pages it has, the further
the reader may run ahead before it has to wait.

The name comes from how it is built: the pages lie one after the other in L1, and the
buffer remembers where the next tile goes and where the oldest tile is. When either reaches
the last page, it wraps to the first, as if the pages were in a ring. Your code never
sees this: `get_write_ptr` and `get_read_ptr` give the right address.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "s0", "x": 0, "y": 0, "w": 100, "h": 44, "t": "page 0\na tile", "small": true, "gem": "amber"}, {"id": "s1", "x": 110, "y": 0, "w": 100, "h": 44, "t": "page 1\na tile", "small": true, "gem": "amber"}, {"id": "s2", "x": 220, "y": 0, "w": 100, "h": 44, "t": "page 2\nempty", "small": true}, {"id": "s3", "x": 330, "y": 0, "w": 100, "h": 44, "t": "page 3\nempty", "small": true}, {"id": "w", "x": 0, "y": 84, "w": 150, "h": 44, "t": "writer: wait_front\nreads here", "small": true}, {"id": "r", "x": 220, "y": 84, "w": 150, "h": 44, "t": "reader: reserve\nwrites here", "small": true}], "arrows": [{"from": "w", "to": "s0", "kind": "data"}, {"from": "r", "to": "s2", "kind": "data"}]}
</script>
<figcaption>A buffer of four pages with two tiles in it. The writer takes the oldest, page 0; the reader puts the next at page 2.</figcaption>
</figure>

## The general form

```python
km.Buffer(index=INDEX, format=DTYPE, page_bytes=BYTES, pages=PAGES)
```

```cpp
reader.reserve(INDEX, 1);   // wait for a free page
...                         // fill get_write_ptr(INDEX)
cb_push_back(INDEX, 1);     // the page is full: the writer may take it
```

```cpp
writer.wait_front(INDEX, 1);  // wait for a full page
...                           // use get_read_ptr(INDEX)
cb_pop_front(INDEX, 1);       // the page is free: the reader may fill it
```

- `PAGES` is the number of pages: how many tiles may be waiting in the buffer at once.
- `reserve(INDEX, 1)` returns when at least one page is free. With every page full, it
  waits.
- `cb_push_back` hands the page just filled to the other side; `cb_pop_front` hands back
  the oldest full page.
- Pages come out in the order they went in.
- One buffer has one side that fills it and one that empties it. A program with more
  stages has more buffers, each with its own `INDEX`; the next lesson has three.

## One line at a time

```{ .python }
--8<-- "tutorial/19_circular_buffers/circular_buffers.py:buffer"
```

This is the only line that changes from the last lesson: `pages=pages`. The program
loads three times, with 1, 2 and 8 pages in the buffer, and each time copies 64 tiles in
one command. The kernels are those of [[three-kernels]], unchanged.

## The whole program

=== "circular_buffers.py"

    ``` { .python }
    --8<-- "tutorial/19_circular_buffers/circular_buffers.py"
    ```

The kernel files and the header are those of lesson 18.

```{ .text .km-output }
--8<-- "tutorial/19_circular_buffers/output.txt"
```

With one page the copy took 55 microseconds. With two pages it took 33, and with eight it
took 33: room for a second tile made the loop faster by a third, and more room did not
help, since room was no longer what held it back. The script checks after every run that
all 64 tiles arrived.

## Common mistakes

- Thinking that more pages are always faster. Two were enough here. Pages take L1,
  1.4 MB in all, and a program needs its L1 for other things.
- Forgetting `cb_pop_front`. The buffer fills up and the reader waits for ever, as in
  [[reading-from-dram]]: more pages only make that happen a command or two later.
- Letting the reader and the writer disagree on how many tiles pass. Each tile pushed is
  one that has to be popped, or the buffer is left with tiles in it for the next command.

Next: [[computing-on-a-tile]] puts the compute kernel between the reader and the writer.
