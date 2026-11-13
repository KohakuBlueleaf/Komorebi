---
title: Buffers and data
summary: Tensors in DRAM, the L1 memory of a core, circular buffers, and moving pages over the NoC with the reader and the writer.
---

# 6. Buffers and data

Commands carry ten words. Real data lives in DRAM, and a kernel gets it into L1, the
memory next to the core, by asking the NoC to copy it. This page covers the three
things a kernel needs for that: a **tensor** in DRAM that the host creates, a **circular
buffer** in L1 that the reader and the writer share, and the calls that start a **NoC
read** and a **NoC write**.

<figure class="km-grid">
<script type="application/json">
{"steps": [
 {"label": "The host puts a tensor in DRAM with session.tensor. Its data is split into pages and spread over the 8 DRAM banks, page p in bank p mod 8. The program's circular buffer is made in the L1 of every core.",
  "lit": ["host", "dram:0"],
  "arrows": [{"from": "host", "to": "dram:0:4", "kind": "data"}]},
 {"label": "The reader kernel of core 0 starts a NoC read of page t: a copy from a DRAM bank into a free page of the circular buffer, in L1. It waits until the read has landed, then pushes the page.",
  "lit": ["core:0"],
  "arrows": [{"from": "dram:0:3", "to": "core:0", "kind": "data"}]},
 {"label": "The writer kernel of the same core sees the pushed page, starts a NoC write of it to the output tensor in DRAM and, when the write has left, pops the page so that the reader can reuse it.",
  "lit": ["core:0"],
  "arrows": [{"from": "core:0", "to": "dram:9:1", "kind": "data"}]},
 {"label": "After the last page the core reports. The host reads the output tensor back and compares it with the input.",
  "lit": ["host", "core:0", "dram:9"],
  "arrows": [{"from": "dram:9:3", "to": "host", "kind": "data"}]}
]}
</script>
</figure>

## The idea

**DRAM** holds tensors. A tensor is stored as **pages** of equal size, spread over the
card's eight DRAM banks. How a tensor is cut into pages is its **layout**:

- `"row"`: one page is one row (the last dimension). A tensor of 8 rows of 256 words
  has 8 pages of 1,024 bytes.
- `"tile"`: one page is one 32 x 32 block, a **tile**, the unit the compute cores work
  on (chapter 9). A tile of `bfloat16` numbers (a 16-bit floating-point format) is
  32 x 32 x 2 = 2,048 bytes. The tiles are ordered row by row.

**L1** is the core's own memory, about 1.4 MB for your data. It is not a cache: nothing is
copied into it unless a kernel asks the NoC to. Part of it is divided into **circular
buffers**: queues of pages, each with a number (0 to 28 are yours). A circular buffer is
how two kernels of one core hand pages to each other:

```text
 DRAM --NoC read--> [ circular buffer in L1: page page page page ] --NoC write--> DRAM
        reader kernel   reserve, read, push           wait, write, pop   writer kernel
```

The producer **reserves** a free page, fills it, and **pushes** it. The consumer **waits**
for a pushed page, uses it, and **pops** it, which makes it free again. With four pages
in the buffer the reader can be up to four pages ahead of the writer.

| Call | Where | What it does |
|---|---|---|
| `session.tensor(data, dtype, layout)` | host | Puts a numpy array in DRAM as pages; `dtype` is `"bfloat16"`, `"float32"`, `"uint32"` and others. Returns a `Tensor` |
| `session.empty(shape, dtype, layout)` | host | The same, without writing data |
| `tensor.address`, `.pages` | host | Where it is, in every bank; how many pages |
| `tensor.accessor_args()` | host | The compile-time arguments that let a kernel address the tensor's pages; goes in `compile_args` |
| `tensor.read()` | host | The whole tensor as a numpy array. Call it after `run.stop()` |
| `km.Buffer(index, format, page_bytes, pages)` | host | Declares a circular buffer on every core |
| `TensorAccessor(args, address)` | kernel | Turns "page t" into a bank and an address |
| `noc_async_read_page(t, acc, l1_addr)` | kernel | Starts a copy of page t of the tensor to `l1_addr` |
| `noc_async_read_barrier()` | kernel | Waits until all reads started so far have landed |
| `noc_async_write_page(t, acc, l1_addr)` | kernel | Starts a copy from `l1_addr` to page t of the tensor |
| `noc_async_writes_flushed()` | kernel | Waits until the writes started so far have left L1 |
| `rd.reserve(cb, n)` / `get_write_ptr(cb)` / `cb_push_back(cb, n)` | reader | Wait for n free pages; the L1 address of the first; make them visible to the consumer |
| `wr.wait_front(cb, n)` / `get_read_ptr(cb)` / `cb_pop_front(cb, n)` | writer | Wait for n pushed pages; the L1 address of the first; free them |

`rd.reserve` and `wr.wait_front` are Komorebi's versions of tt-metal's `cb_reserve_back`
and `cb_wait_front`: they give up after a time limit instead of waiting for ever, and
return `false` when they do (chapter 21). A body that gets `false` returns at once.

## Example 1: copy tiles from one tensor to another

The reader reads tile t of the input into the circular buffer. The writer writes it to
tile t of the output. The compute kernel is empty. First the reader's body:

```cpp title="kernels/reader.cpp" callouts="TensorAccessor(in_args|rd.reserve|noc_async_read_page|noc_async_read_barrier|cb_push_back"
--8<-- "tutorial/06_copy_tiles/kernels/reader.cpp:body"
```

1.  The command's word `a0` is the input tensor's address; `in_args` is the layout
    information that the host put in `compile_args`. Together they let the accessor find
    page t.
2.  Wait for one free page in buffer `kCb`. `false` means the core is stopping.
3.  Start the read: page `t` of the tensor goes to the free page's L1 address
    (`get_write_ptr`). This only starts the copy.
4.  Wait until the copy has landed. Without this the page would be pushed before it is
    full.
5.  Make the page visible to the writer.

The writer's body mirrors it:

```cpp title="kernels/writer.cpp" callouts="wr.wait_front|noc_async_write_page|noc_async_writes_flushed|cb_pop_front"
--8<-- "tutorial/06_copy_tiles/kernels/writer.cpp:body"
```

1.  Wait for the reader's push. On `false` the writer returns `false`: it reports
    nothing, because the core is stopping anyway.
2.  Start the write of the page at `get_read_ptr` to page `t` of the output tensor.
3.  Wait until the data has left L1. Only then may the page be freed.
4.  Free the page. The reader's `reserve` can now return.

The handler's number, the buffer's index and the tile size are shared in
`kernels/handlers.h`:

```cpp title="kernels/handlers.h"
--8<-- "tutorial/06_copy_tiles/kernels/handlers.h"
```

The host makes the tensors, tells the program about them, and checks the copy:

``` { .python title="copy_tiles.py" callouts="session.tensor|session.empty|km.Buffer|compile_args|args = |dst.read()" }
--8<-- "tutorial/06_copy_tiles/copy_tiles.py:host"
```

1.  Puts the array in DRAM. `bf16_values(bf16_bits(x))` first rounds the numbers to
    what `bfloat16` can hold, so that the host knows exactly what the card should copy.
2.  The output tensor, allocated but not written.
3.  One circular buffer on every core: index 0, four pages of 2,048 bytes.
4.  The tensors' layouts become compile-time arguments of the reader and of the writer,
    the `in_args` and `out_args` of the kernels.
5.  The command's words: the two addresses and the number of tiles.
6.  Reads the output tensor, a numpy array of `float32` holding the `bfloat16` values.
    Reading is done after `run.stop()`.

## Example 2: use a page in L1 as ordinary memory

L1 is memory a core can read with an ordinary load. This handler adds up all words of a
`uint32` tensor: the reader is as above (rows instead of tiles), and the writer adds up
each page where it lies in L1 and returns the total as its result word.

```cpp title="kernels/writer.cpp" callouts="wr.wait_front|invalidate_l1_cache|get_read_ptr|sum += row|wr.result"
--8<-- "tutorial/06_sum_in_l1/kernels/writer.cpp:body"
```

1.  Wait for a page from the reader.
2.  The NoC wrote the page behind the core's back; the core keeps a small cache of L1
    that it does not know is stale. This call drops it. Without it the loop below
    can read old words.
3.  `get_read_ptr` is an L1 address. Cast it to a pointer and it is memory.
4.  An ordinary loop over 256 words.
5.  The total goes back as the result word.

``` { .python title="sum_rows.py" }
--8<-- "tutorial/06_sum_in_l1/sum_rows.py:host"
```

## Run it

```bash
cd tutorial/06_copy_tiles
TT_LOGGER_LEVEL=error python copy_tiles.py
cd ../06_sum_in_l1
TT_LOGGER_LEVEL=error python sum_rows.py
```

```{ .text .km-output }
6 tiles copied, identical: True
sum of 2048 words: card 743169458, numpy 743169458
```

The numbers come from a fixed random seed, so they are the same on every card.

## Notes

- A circular buffer holds whole pages and its size is a multiple of the page. Every
  reserve, push, wait and pop on it is a number of pages, and the same number as the
  other side uses. Komorebi's wrappers check this and the core gives up with the code `bad_item` if it is wrong.
- Reserve and push by the same amount, wait and pop by the same amount. A push of 2
  after a reserve of 1 corrupts the buffer.
- NoC calls are asynchronous: `noc_async_read_page` returns before the data is in L1.
  Nothing may use the data before the barrier.
- Buffers 29, 30 and 31 belong to the runtime. Use 0 to 28.
- Do not read a tensor with `tensor.read()` while the program runs; read after
  `run.stop()`. `session.read` reads a tensor while a run is loaded; see the host API.
- Reference: [host API: tensors](../reference/host.md#tensors), [runtime: the three kernels](../reference/runtime.md#the-three-kernels).
