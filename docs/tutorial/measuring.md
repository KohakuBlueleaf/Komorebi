---
title: "Measuring"
part: 6
summary: "the card's clock and the host's clock, why they differ, and how to read many runs of one command"
---

# Measuring

A time of a command means something only when you know whose clock measured it and how
many runs it comes from. This lesson relies on [[commands]], [[many-commands]] and
[[circular-buffers]].

## Why two clocks

Every number in this tutorial so far that ended in "microseconds" was the card's. The card
starts a clock when a command begins and stops it when the cores have reported, and the event
carries the result: `event.microseconds`. Python has a clock of its own, and it sees more: the
trip from Python to the card, the trip back, and the work of `submit` and `wait`. The two
numbers answer two questions. The card's says how long the cores worked. The host's says how
long your program waited. A kernel that you tune is judged by the first; a program that you
tune is judged by the second.

One run of a command is also a poor measure. The first command after a load can be slow, and
any run can be disturbed by something else. A program that measures sends the command many
times, and reads the **middle** value, the median: half the runs were faster, half slower.
The smallest and the largest show how much the runs differ.

## The general form

```python
import statistics
import time

started = time.perf_counter()
handle = run.submit(...)
event = run.wait(handle, timeout=SECONDS)
host_microseconds = (time.perf_counter() - started) * 1e6
card_microseconds = event.microseconds

middle = statistics.median(LIST)
```

- `time.perf_counter()` is Python's clock for timing. It returns seconds, as a decimal
  number; only the difference of two readings means something.
- `(... - started) * 1e6` turns seconds into microseconds: `1e6` is 1,000,000.
- `event.microseconds` is the card's clock for the same command.
- `LIST` is the list of times of all the runs. `statistics.median` returns the middle one.
  `min` and `max` give the smallest and the largest.

## One line at a time

```{ .python }
--8<-- "tutorial/38_measuring/measuring.py:loop"
```

Twenty copies of the 64 tiles of [[circular-buffers]], one command after the other. For each
run the program reads Python's clock before `submit` and after `wait`, and the card's clock
from the event, and keeps both in lists.

```{ .python }
--8<-- "tutorial/38_measuring/measuring.py:report"
```

For each clock, the first run, the middle, the smallest and the largest.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "host clock starts\ntime.perf_counter()"},
  {"t": "run.submit: the command travels\nto the card"},
  {"t": "card clock starts: the cores copy 64 tiles\ncard clock stops: the cores have reported", "gem": "amber"},
  {"t": "the event travels back\nrun.wait returns"},
  {"t": "host clock stops\ntime.perf_counter()"}]}
</script>
<figcaption>The card's clock covers only the middle step; the host's covers all five.</figcaption>
</figure>

## The whole program

=== "measuring.py"

    ``` { .python }
    --8<-- "tutorial/38_measuring/measuring.py"
    ```

The kernels are those of [[circular-buffers]], with 8 pages in the buffer. Run it from the
folder `tutorial/38_measuring`:

```bash
python measuring.py
```

```{ .text .km-output }
--8<-- "tutorial/38_measuring/output.txt"
```

Read the two lines.

- On the card's clock the 20 runs agree to within a microsecond: 32.1 to 32.8. The first is
  no different from the others. The copy is steady work, and the card's clock sees nothing
  else.
- On the host's clock the middle run takes about 161 microseconds, five times the card's
  32. The 130 or so microseconds more are the trips and the calls of Python. They are
  there for every command, however short it is.
- The first run on the host's clock took 595 microseconds, nearly four times the middle.
  Run it again and the first is again the slowest, by about as much. A program that times
  one run, and happens to time the first, reports a number that is not the cost of a
  command.

When commands do not depend on each other, [[many-commands]] hides the trips: the card
works while Python sends the next one.

## Common mistakes

- Timing one run. Time many, and give the middle, with the smallest and largest.
- Comparing a card time with a host time. Say which clock a number is from.
- Timing a program that includes `load`. Loading takes about a second ([[commands]]), and
  it swamps the commands that follow. Start the clock after `load`.
- Taking the first run as typical. On the host's clock it was nearly four times the middle.

Next: [[finding-a-bug]] is about the other thing a command can do: not finish.
