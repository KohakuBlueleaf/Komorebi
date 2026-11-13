---
title: "A thread that submits"
part: 5
summary: "the thread sends a command to the Tensix cores, waits for its event, and uses the result word"
---

# A thread that submits

A thread sends commands to the Tensix cores with `kmx_submit` and waits for them with
`kmx_wait`, the same two steps that `run.submit` and `run.wait` are in Python. This lesson
relies on [[hello-thread]], [[commands]], [[a-result]] and [[arguments]].

## Why a thread submits

Until now Python decided what the cores do next: it sent a command, waited, and sent the
next one. Every decision then goes through the host and the cable to the card. A thread
sits on the card. It can read a result and send the next command at once, and Python is
only told at the end. [[streams-and-graphs]] sent commands one after another to save that
trip; a thread is the other way to save it, and it can decide with real C code.

## The general form

```c
km_command_t command;
command.sequence = command.land1 = command.land2 = command.land3 = 0;
command.hdr = HANDLER | KM_FLAG_RESULT << 8;
command.group = FIRST | COUNT << 16;
command.a0 = A0;
...
command.a9 = A9;

uint32_t handle;
int sent = kmx_submit(&command, &handle);

kmx_event event;
int waited = kmx_wait(handle, kmx_time() + TICKS, &event);
```

- `km_command_t` is the command, the very structure that a kernel receives as `command`
  ([[hello-core]]). The thread fills it field by field. A single assignment such as
  `command = (km_command_t){0}` may become a call of a library function that the thread
  does not have, and then it does not link.
- `sequence`, `land1`, `land2` and `land3` are fields the library fills in. Set them to 0.
- `hdr` holds the handler number ([[handlers]]) and, shifted left by 8 bits, the flags.
  `KM_FLAG_RESULT` is the flag that asks for a result word ([[a-result]]).
- `group` holds the first core and the count: `FIRST | COUNT << 16`, the `group=(FIRST,
  COUNT)` of Python.
- `a0` to `a9` are the arguments ([[arguments]]). Set every one, also the ones you do not
  use.
- `kmx_submit(&command, &handle)` stores the command for the cores. It returns `KMX_OK`
  and the handle, or another code, for instance `KMX_FULL` when no slot is free.
- `kmx_wait(handle, DEADLINE, &event)` waits for the event of that handle. `DEADLINE` is a
  time on the thread's clock, which ticks 50 million times a second: now,
  `kmx_time()`, plus the `TICKS` you will wait. `KMX_TICKS_PER_SECOND` is one second. It
  returns `KMX_OK` and the event, or `KMX_LATE` when the time passed first.
- `event.kind` says how the command ended: `KM_EVENT_DONE` is the normal end. With
  `KM_FLAG_RESULT` set, `event.ticks` is the result word that the leader of the group
  sent; without it, it is the time the command took.

Only core 0 of the cluster submits and waits.

## One line at a time

The kernel is the writer of [[arguments]], with a product in place of the sum:

```{ .cpp }
--8<-- "tutorial/32_thread_submits/kernels/writer.cpp:run"
```

The command, built from the two numbers that Python gave the thread:

```{ .c }
--8<-- "tutorial/32_thread_submits/thread/submit.c:command"
```

Handler 0, the flag for a result, and the group of the one core 0. The ten arguments are
all set.

```{ .c }
--8<-- "tutorial/32_thread_submits/thread/submit.c:send"
```

The thread submits and waits at most one second. Either step may fail, and the thread
then ends with a code of its own, 1 or 2, which Python can read.

```{ .c }
--8<-- "tutorial/32_thread_submits/thread/submit.c:report"
```

The answer is the result word, and the thread returns it to Python as its own result.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "thread: kmx_submit\nthe command for core 0", "gem": "sapphire"},
  {"t": "core 0: the writer computes 6 * 7\nand sets the result word", "gem": "amber"},
  {"t": "core 0 reports: done"},
  {"t": "thread: kmx_wait returns the event\nevent.ticks is 42", "gem": "sapphire"},
  {"t": "thread: kmx_result\nPython reads 42 from thread_result"}]}
</script>
<figcaption>The command goes from the thread to a core and the answer comes back, with no host in between.</figcaption>
</figure>

## The whole program

=== "thread_submits.py"

    ``` { .python }
    --8<-- "tutorial/32_thread_submits/thread_submits.py"
    ```

=== "thread/submit.c"

    ``` { .c }
    --8<-- "tutorial/32_thread_submits/thread/submit.c"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/32_thread_submits/kernels/writer.cpp"
    ```

The reader and compute files are the empty ones. Python sends no command: it only starts
the program and reads the thread's result.

```{ .text .km-output }
--8<-- "tutorial/32_thread_submits/output.txt"
```

The 42 was made by a Tensix core, passed to the thread in the event, and passed on to
Python in the thread's result.

## Common mistakes

- Forgetting `KM_FLAG_RESULT`. The event is then still `KM_EVENT_DONE`, but `event.ticks`
  is a time, not the number the writer set.
- Leaving an argument unset. The command has ten, and an `a5` that was never written holds
  whatever the stack held. Set all ten.
- Not checking `kmx_submit` and `kmx_wait`. Both return a code, and a thread that ignores
  it goes on with an event that is not there.
- Submitting from cores 1 to 3 of the cluster ([[four-cores]]). Only core 0 may; the other
  three end the run with a library error.

Next: [[four-cores]] uses the cluster's other three cores.
