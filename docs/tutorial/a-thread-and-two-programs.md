---
title: "A thread and two programs"
part: 6
summary: "an X280 thread in a run of two programs sends a command to each and adds their answers"
---

# A thread and two programs

In [[two-programs]] the host sent commands to two programs held by the same cores. An X280
thread can do the same: each command names its program, and the thread gets every
program's number and handlers by name. This lesson relies on [[two-programs]] and
[[commands-to-a-thread]].

## Why a thread

The host's commands go one at a time over PCIe, and each answer comes back the same way.
A thread on the card can send a command to one program, look at the answer, and send the
next command to another program at once. It is how a program can choose, while it runs,
which of its programs does the next piece of work and on how many cores
(`examples/19_thread_programs` does that every round).

## The general form

In Python, any one of the run's programs declares the thread:

```python
first = km.Program("first", ...)
second = km.Program("second", ...)
second.thread(cluster=1, sources=[...], regions={...}, handlers={"name": 1})
run = session.load([first, second])
```

In the thread's C, a command for handler `HANDLER` of program `PROGRAM` on cores `FIRST`
to `FIRST + COUNT - 1`:

```c
km_command_t command;
kmx_command(&command, kProgramPROGRAM, kProgramPROGRAMHandlerHANDLER, FIRST, COUNT);
command.a0 = ...;
kmx_submit(&command, &handle);
```

- The thread's C has two names for each program of the run: `kProgramAdd` is the number
  of program `add`, and `kProgramAddHandlerApply` is the number of its handler `apply`.
  Names in Python are written in camel case in C, as for the thread's own handlers
  (`kHandlerBoth` for `both`).
- `kmx_command(&command, program, handler, first, count)` fills the command: the program
  and the handler in its header, the cores in its group, every argument 0. Set the
  arguments after it.
- Only one program of a run may declare threads. If two do, `session.load` stops with
  `ProgramError: ... belong to one program of a run`.

## One line at a time

```{ .python }
--8<-- "tutorial/48_a_thread_and_two_programs/a_thread_and_two_programs.py:programs"
```

The two programs of [[two-programs]], unchanged, and a thread declared by the second.

```{ .c }
--8<-- "tutorial/48_a_thread_and_two_programs/thread/both.c:apply"
```

One command to one core of one program, with `KM_FLAG_RESULT` so that the answer comes
back in the event, then the wait for it.

```{ .c }
--8<-- "tutorial/48_a_thread_and_two_programs/thread/both.c:program"
```

The thread's handler `both` asks `add` on core 0 and `multiply` on core 1, and answers
with the sum.

```{ .python }
--8<-- "tutorial/48_a_thread_and_two_programs/a_thread_and_two_programs.py:run"
```

Python loads both programs on every core and sends one command, to the thread.

## The whole program

=== "a_thread_and_two_programs.py"

    ``` { .python }
    --8<-- "tutorial/48_a_thread_and_two_programs/a_thread_and_two_programs.py"
    ```

=== "thread/both.c"

    ``` { .c }
    --8<-- "tutorial/48_a_thread_and_two_programs/thread/both.c"
    ```

The programs' sources are those of `tutorial/46_two_programs`. Run it from the folder
`tutorial/48_a_thread_and_two_programs`:

```bash
python a_thread_and_two_programs.py
```

```{ .text .km-output }
--8<-- "tutorial/48_a_thread_and_two_programs/output.txt"
```

55 is 6 + 7 from `add` and 6 x 7 from `multiply`.

## Common mistakes

- Declaring a thread in both programs. Declare it in one; it can command them all.
- A plain handler number in `kmx_submit`. The header's program field is then 0, the
  run's first program; use `kmx_command` with `kProgram...`.
- A name with a dash in a program, such as `km.Program("my-add", ...)`. It has no C name;
  the build of the thread stops with a `ValueError`.

Next: [[every-primitive]] lists every call that this tutorial used, with its lesson.
