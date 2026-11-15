# Dispatcher watch: core 0's loop stalls, the run goes on

Whether the dispatcher comes back by itself when its loop on X280 core 0 stops
making passes: core 2 of the same cluster watches the loop, reports the stall to the
host, then raises core 0's machine software interrupt, whose handler re-enters the
loop with the run's state kept. No host action between the stall and the commands
after it.

## Result

One run on a Blackhole p150a (card 0), X280 at 1350 MHz, the map/collect example on
120 cores, a dispatcher image built with `KMD_STALL_DEBUG`:

| What | Measured |
|---|---|
| The stall command | answered (PONG), then core 0 makes no pass (its bound 3 s) |
| Core 2 saw the stall | after 100.009 ms without a pass (bound `KMD_WATCH_STALL_TICKS`, 100 ms) |
| Event to the host | `KMD_EVENT_STALL` "stalled", at host command 6, ticks 5,000,453 |
| Interrupt to back in the loop | 1.2 microseconds (58 ticks of the 50 MHz timer) |
| Event to the host | `KMD_EVENT_STALL` "back", core 0's interrupted pc in `req` |
| Stall to back | 100.01 ms, the watch's bound plus the re-entry |
| MAP commands after (on cores 14 and 15, and on all 120) | 4 of 4 done and exact; 4 of 4 before |
| The run's end | a clean stop: state DONE, the watch off, 1 stall, 1 return |

The time from a stall to the loop running again is the watch's bound: the re-entry is
a few instructions and the loop starts again where its state says it is.

## How it works

- Core 0's loop counts its passes in its state (cached and coherent in the cluster);
  its long waits (a core reset's wait for the firmware) count beats. Core 2 reads the
  two every millisecond while core 0's loop runs, so the watch costs the loop nothing.
- No change for `KMD_WATCH_STALL_TICKS` (100 ms) is a stall. The longest wait the loop
  makes on purpose is a failed core reset's (1 s, which beats); the next longest are a
  few milliseconds. Core 2 then pushes `KMD_EVENT_STALL` "stalled" into the host's event
  ring, sets core 0's software interrupt (the cluster's CLINT, `msip` of hart 0), and
  waits up to `KMD_WATCH_BACK_TICKS` (50 ms).
- Core 0 enables only that interrupt, and only while its loop runs. Its trap entry
  (`trap.h`) sees cause 3, takes run()'s stack pointer and the state that run() saved
  before the loop, and calls `kmd_resumed`: the interrupt cleared, a command line taken
  before the stall freed, the reset windows cleared, "back" pushed, the loop entered
  again. The stalled call chain is dropped; the frame of run() with the rings' lines
  is kept.
- If core 0 does not come back, core 2 pushes "lost" and stops watching; the host
  stops the run with that reason (`KernelFailed`), so it never waits without one.

What it does not cover: a core 0 that stopped executing (an access that never
completes takes no interrupt), and a stall in the middle of a command that changed
part of the state: that command is not taken again only if it was consumed.

## Running it

On card 0 with tt-metal's environment set:

    python experiments/recovery/dispatcher_watch/stall.py

It builds the `KMD_STALL_DEBUG` image (the stall command, `KMD_HANDLER_STALL`, exists
only there), runs MAPs, the stall, MAPs, and prints one JSON line: the stall events,
the core block's watch words, the times, whether every MAP is exact. Exit status 0
when the events are "stalled" then "back", the MAPs exact and the return before the
stall's own bound.
