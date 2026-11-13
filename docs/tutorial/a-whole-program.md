---
title: "A whole program"
part: 6
summary: "the sum of 48 tiles, from numpy to a check in numpy: every part of a Komorebi program, in order, in one program"
---

# A whole program

A whole program has the same few steps whatever it computes, and each step is something an
earlier lesson taught. This lesson is one program, the sum of 48 tiles, read from start to
end with the lesson of every part. It relies on all of Parts 1 to 4.

## Why one more program

Every program so far showed one idea, and used the others only as far as it had to. A real
program is the ideas side by side, and the question becomes where each one goes. The answer is
the same each time. The data is made in Python. The cores' work is written once, in three
kernels. The program is loaded once. Commands carry the numbers that change. The result is
read after the cores have stopped, and checked.

## The plan

| Step | In this program | Lesson |
|---|---|---|
| Make the data | 48 tiles of small integers in numpy | [[tiles-and-tensors]] |
| Put it on the card | `session.tensor(...)` for the source, and one for each result | [[reading-from-dram]] |
| Describe the cores' work | three kernels, a handler `sum_tiles`, buffers | [[handlers]], [[three-kernels]] |
| Share the tiles | each core takes its part of the 48 | [[partitions]] |
| Move tiles through L1 | the reader pushes, the compute kernel pops | [[circular-buffers]] |
| Compute | each core adds its tiles into one | [[computing-on-a-tile]] |
| Combine the cores | the sums of the cores are added into one | [[reduce]] |
| Write the answer | the writer of the leader writes the total | [[writing-to-dram]], [[the-leader]] |
| Send the command | `submit` with the arguments, then `wait` | [[arguments]], [[commands]] |
| Check | read the result after `stop` and compare with numpy | [[a-result]] |

## One part at a time

```{ .python }
--8<-- "tutorial/39_a_whole_program/a_whole_program.py:tensors"
```

Python makes 48 tiles of integers from -3 to 3, and puts them on the card as one tensor of
`bfloat16` tiles. It makes two tensors for the answer, one for each of the two commands,
before the program is loaded: a tensor cannot be written while the kernels run
([[a-whole-map]]).

```{ .python }
--8<-- "tutorial/39_a_whole_program/a_whole_program.py:program"
```

The program: the folder of the kernels, the header with the handler, three buffers, and the
arithmetic setting of [[computing-on-a-tile]]. The first buffer holds the core's own tiles on their
way from the reader to the compute kernel, two at a time. The second receives the sums that
other cores send ([[port]]). The third holds the output.

```{ .cpp }
--8<-- "tutorial/39_a_whole_program/kernels/reader.cpp:run"
```

The reader. Its core takes its share of the 48 tiles with `km::partition`, reads them one by
one into the first buffer, and then takes its part in the reduction.

```{ .cpp }
--8<-- "tutorial/39_a_whole_program/kernels/compute.cpp:run"
```

The compute kernel takes the same share, and adds its tiles into one: the first tile goes
into register 0, and each later tile into register 1 and is added to 0. Then `km::reduce`
adds the sums that the other cores send. The result goes to the output buffer.

```{ .cpp }
--8<-- "tutorial/39_a_whole_program/kernels/writer.cpp:run"
```

The writer. A core that is not the leader sends its sum to the core it reports to; the leader
writes the total into the destination tensor.

```{ .python }
--8<-- "tutorial/39_a_whole_program/a_whole_program.py:sum"
```

The same command twice, for 1 core and for 8. The arguments are the source and destination
addresses, the first tile of each, 0 for the unused form, and the number of tiles. When both
commands are done, the program stops the cores and checks each destination against numpy.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "numpy: 48 tiles of integers\nsession.tensor: they are on the card"},
  {"t": "session.load: three kernels start\non all 120 cores", "gem": "amber"},
  {"t": "submit sum_tiles to 8 cores:\neach reads and adds its 6 tiles", "gem": "amber"},
  {"t": "the 8 sums are reduced into one tile\nat core 0, and written to DRAM", "gem": "amber"},
  {"t": "run.stop, then destination.read()\nthe check against numpy"}]}
</script>
<figcaption>The life of the program, from numpy to numpy.</figcaption>
</figure>

## The whole program

=== "a_whole_program.py"

    ``` { .python }
    --8<-- "tutorial/39_a_whole_program/a_whole_program.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/39_a_whole_program/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/39_a_whole_program/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/39_a_whole_program/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/39_a_whole_program/kernels/writer.cpp"
    ```

Run it from the folder `tutorial/39_a_whole_program`:

```bash
python a_whole_program.py
```

```{ .text .km-output }
--8<-- "tutorial/39_a_whole_program/output.txt"
```

Both sums equal numpy's. Eight cores took 6.9 microseconds against 25.7 for one: 3.7 times
as fast with 8 times the cores, since the eight sums still have to be combined.

## Common mistakes

- Writing a tensor after `load`. The write waits for the cores to be idle, and they never are:
  after 60 seconds it raises `RuntimeError` with `Timeout (60000 ms) waiting for physical
  cores to finish`. Make every tensor first.
- Reading a tensor before `stop`. The same wait, the same error, after the same 60 seconds.
  Read it after `stop`, as every program of this tutorial does.

Next: [[your-own-kernel]] starts from this program and changes it.
