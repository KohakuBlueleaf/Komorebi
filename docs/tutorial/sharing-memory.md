---
title: "Sharing memory"
part: 5
summary: "regions: Python gives the thread an array, the thread writes another, and Python reads it after the run"
---

# Sharing memory

A region is a named block of memory that Python and a thread both reach: Python fills it
before the thread starts and reads it after the thread has ended, and the thread uses it as
an ordinary array. This lesson relies on [[hello-thread]] and [[tiles-and-tensors]].

## Why regions

Words are not enough for most work. A thread's argument words are 64 numbers, and its
result is 48. The data of a real program is a thousand numbers or a million. A region is the
place for them. It lives in the DRAM of the card, in the memory bank of the thread's
cluster, and it is the meeting place: Python writes into it when the program is loaded, the
thread reads and writes it while it runs, and Python reads it when the run is over.

Python never reads a region while the thread runs. That is a rule of the library, and it
decides how a program is shaped: what has to travel while the program runs travels as
result words ([[thread-submits]]); the bulk of the data travels in regions, before and
after.

## The general form

In Python:

```python
program.thread(..., regions={NAME: CONTENT, ...})

run.stop()
data = run.thread.read(NAME)
```

In the C file:

```c
uint64_t bytes;
TYPE* pointer = (TYPE*)kmx_region(ID, &bytes);
kmx_invalidate(ID);
...
kmx_flush(ID);
```

- `NAME` is the region's name. It is only for Python.
- `CONTENT` is either a number of bytes, for a region that starts empty, or a numpy array,
  for a region that starts with the array's bytes. Its size is the array's size, in bytes.
- `ID` is the region's number in the thread: 0 for the first name in `regions`, 1 for the
  second, in the order of the dictionary.
- `kmx_region(ID, &bytes)` returns the region as memory, and stores its size in `bytes`.
  For a number it does not know, it returns 0, a null pointer. Check for it.
- `TYPE` is what you read the memory as: here `uint32_t`, since the numpy array holds
  32-bit numbers.
- `kmx_invalidate(ID)` comes before the thread reads what someone else wrote, and
  `kmx_flush(ID)` after the thread has written what someone else will read. The thread
  reads memory through a cache, and these two calls keep the cache and the DRAM in step.
  Tensix cores write to DRAM past that cache, so for them the rule matters while the thread
  runs.
- `run.thread.read(NAME)` returns the region as a numpy array of bytes (`uint8`); `.view`
  turns it into the type you stored. It works only after `run.stop()`: before it, it raises
  `RuntimeError: regions are read after run.stop()`.

## One line at a time

```{ .python }
--8<-- "tutorial/34_sharing_memory/sharing_memory.py:regions"
```

Two regions. `numbers` starts as the array of 0 to 1023, as 32-bit numbers, 4096 bytes.
`tripled` starts empty, with room for 4096 bytes. The thread's region 0 is `numbers`, its
region 1 is `tripled`.

```{ .c }
--8<-- "tutorial/34_sharing_memory/thread/triple.c:regions"
```

The two regions as arrays. `numbers_bytes` is the size of the first: 4096.

```{ .c }
--8<-- "tutorial/34_sharing_memory/thread/triple.c:work"
```

The thread drops the cache's copy of `numbers`, so that it reads what Python wrote. Then
it writes three times each number into `tripled`, and flushes `tripled`, so that what it
wrote reaches the DRAM.

```{ .python }
--8<-- "tutorial/34_sharing_memory/sharing_memory.py:read"
```

After the thread's result and `run.stop()`, Python reads `tripled` and views its bytes as
32-bit numbers again.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "Python: regions={numbers: array,\ntripled: 4096}; load puts them in DRAM"},
  {"t": "thread: kmx_region(0) and kmx_region(1)\ngive it pointers to the two regions", "gem": "sapphire"},
  {"t": "thread: tripled[i] = 3 * numbers[i]\nthen kmx_flush", "gem": "sapphire"},
  {"t": "the thread ends, run.stop() stops the run"},
  {"t": "Python: run.thread.read(\"tripled\")"}]}
</script>
<figcaption>Where the data of the program is, and who touches it when.</figcaption>
</figure>

## The whole program

=== "sharing_memory.py"

    ``` { .python }
    --8<-- "tutorial/34_sharing_memory/sharing_memory.py"
    ```

=== "thread/triple.c"

    ``` { .c }
    --8<-- "tutorial/34_sharing_memory/thread/triple.c"
    ```

The kernels are the empty ones. Run it from the folder `tutorial/34_sharing_memory`:

```bash
python sharing_memory.py
```

```{ .text .km-output }
--8<-- "tutorial/34_sharing_memory/output.txt"
```

The thread went through the 1024 numbers that Python wrote, and Python read back what the
thread wrote. The program checks all 1024 of them against numpy, not only the five at each
end.

One thing the run does not show. With the `kmx_flush` left out, Python reads the same
numbers: the library hands every region back when the thread ends, and Python reads only
after that. The two calls matter when the Tensix cores use a region while the thread runs,
and the Tensix side of that is not in this tutorial yet.

## Common mistakes

- Reading too early. `run.thread.read(...)` before `run.stop()` raises
  `RuntimeError: regions are read after run.stop()`.
- A region id that does not exist. `kmx_region(5, &bytes)` returns 0, and a thread that
  uses that pointer breaks. Test it against 0.
- No region at all, or more than 16 ([[hello-thread]]).
- Treating `tripled` as numbers in Python without `.view(np.uint32)`. `read` returns bytes,
  four of them for each 32-bit number.

Next: [[twelve-cores]] runs the same thread on three clusters, and the lesson uses a
region of each.
