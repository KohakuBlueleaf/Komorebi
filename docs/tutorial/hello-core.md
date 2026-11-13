---
title: "Hello, core"
part: 1
summary: "the smallest program, taken apart line by line: one kernel file per role, one Python file, one command"
---

# Hello, core

A Komorebi program is a folder with three C++ files, one for each of three roles, and a
Python file that loads them onto the cores and sends the cores a command. This lesson
takes the smallest such program apart, line by line. It relies on [[what-komorebi-is]],
[[setup]] and [[the-card]].

## Why there are three files

Every Tensix core has five CPUs ([[the-card]]). Komorebi runs one small C++ function on
each kind of them:

- the **reader** runs on the CPU that moves data in,
- the **writer** runs on the CPU that moves data out,
- the **compute** function runs on the three that do arithmetic.

Such a function is called a **kernel**. It is the code that runs on a core, as opposed to
the Python that runs on your computer. Each role is a file of its own: `reader.cpp`,
`compute.cpp`, `writer.cpp`, all in one folder. Komorebi always loads all three onto the
cores, so a program always has all three files, even when a role has nothing to do. In
this lesson none has: the program does nothing on the core except to be asked.

## The general form of a kernel file

Each of the three files has the same shape. Words in capitals are the parts you choose.

```cpp
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::ROLE name;
    name.run([&](const km_command_t& command) {
        BODY
    });
}
```

- `#include "komorebi/device/km_worker.hpp"` brings in everything a kernel needs from
  Komorebi.
- `void kernel_main()` is the function the card calls. The name is fixed.
- `km::ROLE name;` makes the object that stands for the role: `km::Reader`, `km::Compute`
  or `km::Writer`, the one that belongs to the file. `name` is yours; the tutorial uses the
  role's own word: `reader`, `compute`, `writer`.
- `name.run(...)` is a loop that you do not write. It waits until a command arrives for
  this core, calls the function you gave it once, and waits again. It goes on until the
  program is stopped.
- `[&](const km_command_t& command) { BODY }` is that function: a C++ lambda that takes
  the command that arrived. `BODY` is the work. The command is the only parameter, and
  you do not use it yet.

In the writer, `BODY` must be `return true;`. That `true` is the writer saying "I am
done with this command, report it". The reader and the compute function return nothing.

This is the whole writer of the program in this lesson:

```{ .cpp }
--8<-- "tutorial/04_hello_core/kernels/writer.cpp:run"
```

The function has one line. A command arrives, the function returns `true`, the core
reports.

## The general form of the Python file

```python
program = km.Program(NAME, kernels=FOLDER)

with km.Session() as session:
    run = session.load(program)
    handle = run.submit(KIND, group=(FIRST, COUNT))
    event = run.wait(handle, timeout=SECONDS)
    run.stop()
```

- `NAME` is a name for the program, a string. It labels messages.
- `FOLDER` is the folder with the three kernel files. The programs of this tutorial write
  it `HERE / "kernels"`, where `HERE` is the folder that holds the Python file itself, so
  that the program runs from any directory.
- `KIND` is a number that says what kind of work the command is. It is always `0` in this
  lesson; [[handlers]] gives it a meaning.
- `FIRST` and `COUNT` say which cores the command is for: `COUNT` cores, starting at core
  number `FIRST`. Together they are the command's **group**.
- `SECONDS` is how long `wait` may wait before it gives up with an error.

## One line at a time

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:program"
```

A `Program` is a description of what the cores will run: a name and the folder with the
three files. Nothing is compiled and the card is not touched yet.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:session"
```

A `Session` is the open card. `with` opens it at the start of the block and closes it at
the end, also when an error ends the block early. Everything that follows is inside the
block.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:load"
```

`load` compiles the three files, copies the kernels onto all 120 cores, and starts them.
The kernels now run and wait for commands. What `load` returns, `run`, is the program
loaded and running: the handle on it for the rest of the program.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:submit"
```

`submit` sends one **command** to the cores. A command is a message from Python to cores
that says: run your kernels once, now. `submit` does not wait for the cores; it returns at
once with a **handle**, a ticket that names this command, so that you can ask about it
later.

<figure class="km-diagram">
<script type="application/json">
{"kind": "anatomy",
 "parts": [
  {"t": "handle = ", "n": "the ticket for\nthis command"},
  {"t": "run.submit(", "n": "send a command"},
  {"t": "0", "n": "the kind of work:\n0 for now"},
  {"t": ", "},
  {"t": "group=(0, 1)", "n": "which cores: start at\ncore 0, take 1"},
  {"t": ")"}]}
</script>
<figcaption>The parts of the line that sends a command.</figcaption>
</figure>

Only core 0 runs the command. Every core sees every command, but a core that is not in
the group lets it pass.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:wait"
```

`wait` blocks until the command is finished, then returns an **event**: the card's report
about that command. If the command takes longer than `timeout` seconds, `wait` raises an
error instead.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:stop"
```

`stop` tells the kernels on every core to finish, and waits until they have. After
`stop` no command can be sent.

```{ .python }
--8<-- "tutorial/04_hello_core/hello.py:print"
```

`event.kind` says how the command ended. `done` means that every core of the group
reported that it finished.

## What happens, in order

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "session.load(program)\nthe kernels start on all 120 cores and\nwait", "gem": "amber"}, {"t": "run.submit(0, group=(0, 1))\nthe command goes to the cores"}, {"t": "core 0: reader, compute and writer\neach run their function once", "gem": "amber"}, {"t": "core 0 reports: done"}, {"t": "run.wait(handle) returns the event"}]}
</script>
<figcaption>The life of the one command of this program.</figcaption>
</figure>

## The whole program

=== "hello.py"

    ``` { .python }
    --8<-- "tutorial/04_hello_core/hello.py"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/04_hello_core/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ``` { .cpp }
    --8<-- "tutorial/04_hello_core/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/04_hello_core/kernels/writer.cpp"
    ```

Run it from the folder `tutorial/04_hello_core`:

```bash
python hello.py
```

```{ .text .km-output }
--8<-- "tutorial/04_hello_core/output.txt"
```

It prints one line. The program did nothing on the core, and that is the point: the
command arrived, three empty functions ran, the core reported, and Python heard it.
Everything the later lessons do is a way to make those three functions do something.

## Common mistakes

- A missing file. If the folder lacks one of the three, `load` stops with
  `FileNotFoundError: a: no kernel source [.../compute.cpp, .../writer.cpp]`, naming
  the missing files.
- A group outside the card. `group=(0, 500)` stops at `submit` with
  `ValueError: group (0, 500) is not inside 120 cores`. The cores are numbered 0 to 119.
- A writer that returns `false`, or does not return `true`. The core never reports, and
  after about a second the card gives up on the command and `wait` raises
  `CommandTimeout`. [[when-it-goes-wrong]] is about this.

Next: [[commands]] looks at what a command is, and at what it costs.
