---
title: "Handlers"
part: 1
summary: "two pieces of work in one kernel, chosen by the command"
---

# Handlers

A handler is one piece of work inside a kernel, and the command says which one to run by
its number. This lesson relies on [[hello-core]], [[a-result]] and [[arguments]].

## Why handlers

A program rarely does one thing. The kernels are loaded once ([[commands]]), so the one
kernel has to contain every piece of work the program may ask for, and each command has to
say which one it wants. That is what the number you gave to `submit` first, the `0` of
lesson 4, was for. A **handler** is a piece of work with a number and a name.

All three kernels of a core receive every command. Each looks at the handler and does the
part that belongs to it. A kernel that has nothing for a handler ignores it, as the empty
reader and compute kernels have so far ignored everything.

## The general form

The numbers and names live in one C++ header, which the kernels include and Python reads:

```cpp
constexpr uint32_t kHandlerNAME = NUMBER;
```

In the kernel, inside the function you give to `run`:

```cpp
if (km::handler(command) == kHandlerNAME) {
    WORK
}
```

In Python:

```python
program = km.Program("...", kernels=FOLDER, handlers=HEADER)
handle = run.submit("name", group=(0, 1))
```

- `NAME` is a word that starts with a capital letter, such as `Add` or `BigStep`. `NUMBER`
  is its number, from 1 to 239. The prefix `kHandler` is required.
- `km::handler(command)` is the number of the handler that this command asked for.
- `WORK` is what the kernel does for that handler.
- `HEADER` is the path of the header file. `Program` reads it, and from then on `submit`
  accepts the handler's name in place of its number. The name is the C++ name without the
  prefix, in lower case with underscores between the words: `kHandlerBigStep` is
  `"big_step"`, `kHandlerAdd` is `"add"`.
- The number `0` is the one handler you do not write. A command with handler 0 runs the
  kernels, and no `if` matches it.

## One line at a time

```{ .cpp }
--8<-- "tutorial/08_handlers/kernels/handlers.h"
```

Two handlers, `Add` with number 1 and `Multiply` with number 2. Both kernels and Python
agree on these numbers because they all read this one file.

```{ .cpp }
--8<-- "tutorial/08_handlers/kernels/writer.cpp:run"
```

The writer reads the two arguments, then checks the handler. A command for `Add` makes it
the sum, a command for `Multiply` the product. The two `if`s do not overlap, so at most
one of them runs.

```{ .python }
--8<-- "tutorial/08_handlers/handlers.py:program"
```

`handlers=` is the one new argument of `Program`: the header.

```{ .python }
--8<-- "tutorial/08_handlers/handlers.py:submit"
```

`"add"` is the handler's name in place of the number. The same line with `"multiply"`
asks for the other piece of work.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "a command arrives at the writer\nwith a handler number", "gem": "amber"},
  {"t": "is it kHandlerAdd (1)?\nyes: the result is left + right"},
  {"t": "is it kHandlerMultiply (2)?\nyes: the result is left * right"},
  {"t": "return true: report the command"}]}
</script>
<figcaption>What the writer of this program does with each command.</figcaption>
</figure>

## The whole program

=== "handlers.py"

    ``` { .python }
    --8<-- "tutorial/08_handlers/handlers.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/08_handlers/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/08_handlers/kernels/writer.cpp"
    ```

The reader and compute files are the empty ones; the header need not be included in them,
since they do not look at the handler.

```{ .text .km-output }
--8<-- "tutorial/08_handlers/output.txt"
```

Two commands with the same arguments, 6 and 7, and different handlers, and two different
answers from the same loaded kernel.

## Common mistakes

- A name that is not in the header. `run.submit("nothing", ...)` stops with
  `KeyError: "handlers has no handler 'nothing'"`. The same happens when you forgot
  `handlers=` in `Program`: Python then knows no names, only numbers.
- A handler number the kernel has no `if` for. `run.submit(77, ...)` is accepted, runs
  the kernels, and the result is 0. No message tells you that nothing matched.
- A number outside 1 to 239. `run.submit(240, ...)` stops with
  `ValueError: handler 240: not NOP and not in 1..KM_HANDLER_USER_MAX`. The numbers above
  239 belong to Komorebi.

Next: [[many-commands]] sends many commands at once.
