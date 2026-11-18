---
title: Measure a command
summary: How to time a command on the card and on the host, and read the numbers of many runs.
tags:
  - guide
  - measuring
---

# Measure a command

Use this to find out how long a command takes. The lesson behind it is
[Measuring](../tutorial/measuring.md).

## Steps

1. Load the program once, and start timing after `load`: loading takes about a second.
2. Send the command many times, 20 or more, one after the other, each with its own `wait`.
3. For each run, keep two numbers: `event.microseconds`, the card's clock, and the
   difference of two `time.perf_counter()` readings around `submit` and `wait`, the host's.
4. Report the middle value (`statistics.median`) with the smallest and the largest. Do not
   report the first run: on the host's clock it was nearly four times the middle.
5. Say which clock the number is from. The card's clock is the time the cores worked; the
   host's is the time your program waited, and it includes the trips to the card.

To hide the trips, send many commands before the first `wait`
([Many commands](../tutorial/many-commands.md)). Then `event.microseconds` is not one
command's time: a program queues up to `depth` commands (4 by default), and each queued
command's event counts the ones before it (five commands of 3.3 ms each read 3.3, 6.6,
9.8, 13.1 and 13.1 ms). Time one command with its own `wait`, as in step 2.
