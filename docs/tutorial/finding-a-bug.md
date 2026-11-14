---
title: "Finding a bug"
part: 6
summary: "reading the card's report of a command that did not finish: which cores, which kernel, waiting for what"
---

# Finding a bug

When a command does not finish, the card's report says which kernel of which core was
waiting, and for what; with that line you can find the missing step in a kernel. This
lesson relies on [[when-it-goes-wrong]], [[circular-buffers]] and [[three-kernels]].

## Why a report

A kernel that waits for something that never comes does not fail with a message of its
own: it waits. Looking at the code for the mistake is slow, because a copy is a loop of
five calls in two kernels, and the mistake is one missing call. The card helps. When the
deadline passes ([[when-it-goes-wrong]]), it first reads where each kernel of the group is
waiting, then tells the cores to leave the command. That report ends in the message of
`CommandTimeout`.

## The general form

The message has this shape:

```text
command N was given up, the run goes on; at the deadline: slot S, sequence Q,
    group first F count C; cores: core X reader waits in WHERE writer waits in WHERE
```

- `command N` is the handle of the command that did not finish, `group first F count C` the
  cores it was for ([[hello-core]]). `slot S` and `sequence Q` are the card's own numbers for
  it, and you do not need them.
- `was given up, the run goes on`: the cores left the command, and the run takes more
  commands. `error.given_up` is True.
- `cores: core X` names each core of the group as the card found it at the deadline.
- `reader waits in` and `writer waits in` say, for each of the core's two data-moving
  kernels, where it was waiting. If a kernel is not named, it was not waiting.
- `WHERE` is the step the kernel was in. Three matter here:
    - `next`: waiting for its next command. The kernel had finished this command.
    - `data`: waiting on a circular buffer: `writer.cpp` for a tile to be there
      (`wait_front`), `reader.cpp` for room to put one (`reserve`).
    - `cdone`: `writer.cpp`, at the end of its work, waiting for the other kernels' work to end.

Other steps exist, and the message names them in the same way. A core busy in a loop of
its own cannot leave the command; the run then ends with the message of
[[when-it-goes-wrong]], `timed out; not reported`, which names the same steps.

## Two bugs, one at a time

The kernels of this lesson are the copy of [[circular-buffers]], with two switches. Argument
`a5` of the command turns a bug on. In `reader.cpp`:

```{ .cpp }
--8<-- "tutorial/39_finding_a_bug/kernels/reader.cpp:run"
```

With `a5` equal to 1 `reader.cpp` never calls `cb_push_back`: it reads each tile into the
buffer, and never says that the tile is there. In `writer.cpp`, `a5` equal to 2 skips
`cb_pop_front`: `writer.cpp` copies each tile out, and never says that the page is free again.

```{ .python }
--8<-- "tutorial/39_finding_a_bug/finding_a_bug.py:wait"
```

The program runs the copy three times, with bug 1, bug 2 and no bug. The deadline of each is
one second ([[when-it-goes-wrong]]). When the card gives up, the program prints the message,
stops the run and loads the kernels again for the next bug.

Read what the card said about bug 1:

```text
cores: core 0 reader waits in next writer waits in data
```

`reader.cpp` was in `next`: it had run its whole loop and was waiting for another command. The
writer was in `data`: it was waiting for a tile to be there. So the tile was read but never
announced. The place to look is where `reader.cpp` hands a tile on: the `cb_push_back`.

Bug 2:

```text
cores: core 0 reader waits in data writer waits in cdone
```

Now `reader.cpp` was the one in `data`: it was waiting for room to put a tile. `writer.cpp` was
in `cdone`: it had written all its tiles and was waiting for `reader.cpp` to end. `reader.cpp`
waited for room that `writer.cpp` should have made, so look at where `writer.cpp` is done with
a tile: the `cb_pop_front`.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "the message names a kernel and a step:\nreader in next, writer in data"},
  {"t": "which kernel is waiting, and for what?\nwriter.cpp waits for a tile"},
  {"t": "who should have made it?\nreader.cpp, with cb_push_back"},
  {"t": "look there: the call is missing.\nfix it and run again", "gem": "sage"}]}
</script>
<figcaption>From the message to the line of code.</figcaption>
</figure>

## The whole program

=== "finding_a_bug.py"

    ``` { .python }
    --8<-- "tutorial/39_finding_a_bug/finding_a_bug.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/39_finding_a_bug/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/39_finding_a_bug/kernels/reader.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/39_finding_a_bug/kernels/writer.cpp"
    ```

The compute file is the empty one. Run it from the folder `tutorial/39_finding_a_bug`:

```bash
python finding_a_bug.py
```

```{ .text .km-output }
--8<-- "tutorial/39_finding_a_bug/output.txt"
```

The two bugs were found from the messages alone, and the run without a bug copied all eight
tiles and the program's check passed.

## Common mistakes

- Reading the timeout as the cause. It says only that the card ended the command at the
  deadline. The cause is where the kernel was: the `WHERE`.
- Looking only at the kernel that is in `data`. In bug 1 it was `writer.cpp` that was in `data`,
  and the mistake was in `reader.cpp`. The one that waits is the victim; the one that did not
  send what it should is the cause.
- Taking a given-up command for a lost run. The run goes on, and the next command on
  the same cores runs as usual; this program stops it only to load the kernels with the
  next bug. After a `CommandTimeout` whose `given_up` is False the run is over
  ([[when-it-goes-wrong]]): load again.
- A deadline too short for honest work. The card then reports a kernel that was working.
  Check the time of the command with [[measuring]] before you suspect a bug.

Next: [[a-whole-program]] puts the parts of the tutorial together in one program.
