# Commands to an X280 thread

A command whose group names a work cluster goes to that cluster's thread: the dispatcher
hands it to the thread, the thread library calls the program's

```c
uint32_t kmx_program(const km_command_t* command);  /* switch on kmx_handler(command) */
```

and the dispatcher sends the command's event with the returned word as its result, as
for a command to a core group. The host, the stream engine or another cluster's thread
submits it like any command. This page is the prototype's protocol, what was checked on
a card and what it costs.

## Results

One Blackhole p150a, slow dispatch, AICLK 1350 MHz (three samples during the run), the
X280 at the session's 1000 MHz, host load 4.6.

| Path | Command to a thread (`KM_TC_ECHO`) | NOP to one grid core |
|---|---|---|
| Thread on cluster 1 to the target, one at a time, median (p99) | 4.74 µs (4.96) | 3.60 µs (3.76) |
| The same, 8 in flight, per command | 2.02 µs | 1.69 µs |
| The dispatcher's own time, store to answer and submit to acknowledgement (median) | 2.56 µs | 1.64 µs |
| Host, one at a time, median | 127.2 µs | 127.6 µs |
| Host, 8 in flight, per command | 22.3 µs | |

| Check | Outcome |
|---|---|
| 10,000 commands from the host, every 16th one whose function submits a grid command and waits for it | every result word right |
| 10,000 from the thread on cluster 1, 1 and 8 in flight | every result word right |
| 1,000 waves of 8, thread and grid commands mixed | every word right, 0 rejected |
| A function that spins 0.2 s against a deadline of 0.02 s | its timeout event after 20.05 ms, auxiliary 2 (the cluster); 100 grid commands meanwhile right; 100 thread commands after it returned right; dispatcher error none |
| A thread with `kmx_main` only, as before | examples 07 (every path, clusters 1 to 3), 08 and a rasterizer frame on the thread exact |

The host's times are its own round trip through the PCIe window; they do not tell
the two kinds apart. A thread's command costs the dispatcher about 0.9 µs more than a
NOP to one core: one 64-byte line stored through the window it already holds onto the
thread's local memory, and the answer taken from the thread's ring. The thread's
side, from the line to its answer, is the rest.

**Cost for the commands that do not use it** (`experiments/dispatcher/measure.py
bench`, the image of the main branch against this one, alternated twice):

| | main | with thread commands |
|---|---|---|
| 63 commands stored before the start, per command | 1.2225, 1.2247 µs | 1.2247, 1.2231 µs |
| 100,000 commands fed by the host, per command | 1.1950, 1.1921 µs | 1.1858, 1.1858 µs |
| An idle pass of the loop | 68.02 ns | 68.3 ns |
| The kernel image | 28,088 bytes | 30,224 bytes |

The first build cost 0.06 µs a command: the loop had grown past the point where the
compiler inlined `slots_sweep`, and an outlined call spills to the uncached stack. It is
`KMD_HOT` now; no other change was needed.

## The protocol

**Naming a cluster.** The group word: `first` = `KM_GROUP_CLUSTER_BASE` (0xFF00) + the
cluster, 1 to 3, count 1 (`KM_GROUP_CLUSTER(c)` in `km_abi.h`). A first core of 0xFF01
is no core (at most 128 cores, `first` is 16 bits) and today such a group is rejected
with reason 3 (`group`), so no program used it. No header bit is used: bits 16 to 31 of
the header carry the slot and the stream tag, and bits 24 to 31 are kept for the
program field. The DONE event's auxiliary word names the cluster in its low 8 bits.

**Dispatcher to thread.** Eight command lines (`KMD_WORK_IN_DEPTH`) in the thread's
local memory after its event ring (`KMD_WORK_IN`), line n in slot n mod 8. Core 0
writes a line as four 16-byte stores through the window it already holds for the events.
A line keeps the command's words, with its sequence word set to the line's number and
the landing words set so a torn line is never taken (`kmx_ring_command` checks both).

**Thread to dispatcher.** The answer is a command on the thread's own ring, handler 243
(`KMD_HANDLER_ANSWER`): a0 the line's number, a1 the word. The dispatcher takes it
before the ring's room check, sends no event for it, and sends the line's command's
DONE event: with `KM_FLAG_RESULT` the word is the result, else the event carries the
time, as for a core group.

**Lines kept until the thread serves.** The host starts the threads after the
dispatcher is ready and writes each thread's block, its ring area included, when that
thread starts; a line stored before that is written over. So the dispatcher keeps
every line and stores it only once the thread said it serves: the library answers line
0 when it enters its serve loop, and the dispatcher then stores the waiting lines in
order. A command sent to a cluster whose thread has not started waits for it, within
its deadline.

**The library.** `kmx_main` and `kmx_program` are both weak. After `kmx_main` returns,
if there is one, core 0 serves: it takes each line, calls `kmx_program` and answers
with the word, checking the stop line, the runtime's stop and the deadline every 256
spins. A thread with `kmx_main` and no `kmx_program` runs exactly as before. The
function may submit commands and wait for their events (`KM_TC_SUBMIT` does).

**Full, deadline, stop.** With all 8 lines of a thread unanswered, a further command
from the stream engine or another thread is rejected busy (reason 11) and sent again
after the kernel's wake; one from the host is rejected with reason 14
(`KMD_REJECT_THREAD`). Each line has its source's command deadline: when it passes, the
source gets the timeout event and the dispatcher goes on (it never waits for a thread);
the line stays taken until the thread answers, and that late answer sends no event. On
the host such a timeout does not fail the run, unlike a core group's. A stop or an EXIT
ends the thread as before; lines still unanswered then count among the commands the
kernel abandoned at its end.

**Not yet a framework feature.** The host submits by hand (`run._queue_command` with
group `(KM_GROUP_CLUSTER_BASE + c, 1)`); there is no `run.submit` target for a cluster,
no graph or stream node for one, no reference page and no lesson. A thread still lives
at most its deadline (60 s at most), so a command to a thread is only served while that
thread runs. Events of commands the thread itself submitted are not drained while it
waits in its serve loop between lines.

## Running it

`commands.py` loads the program (the grid's kernels, `x280/program.c` with
`kmx_program` only on `--server`, `x280/client.c` with `kmx_main` on `--client`) and
prints one JSON line; `ok` is false, and it exits non-zero, when a check fails.

```sh
python commands.py run --count 10000 --mix 1000 --latency 1000 --server 2 --client 1
python commands.py deadline --server 2
```

- `run`: the client's commands to the server and to grid core `--core`, one at a time
  (each timed) and 8 in flight, every word checked; then the host's checked commands,
  the mixed waves and the timed ones, alternating thread and grid.
- `deadline`: a load with a 0.02 s command deadline; `KM_TC_SPIN` keeps the function
  0.2 s, then grid commands, then thread commands again.

The tests that need no card: `tests/core/test_thread.py` (the line layout on both
sides), `tests/core/test_run.py` (a late command to a thread is its event and the run
goes on) and `tests/core/kmx_ring_check.c` (taking a line, a torn line, the answer).
