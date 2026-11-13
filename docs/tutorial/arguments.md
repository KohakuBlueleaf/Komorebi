---
title: "Arguments"
part: 1
summary: "numbers from Python into the kernel"
---

# Arguments

A command can carry up to ten 32-bit numbers, its arguments, and the kernel reads them as
`command.a0` to `command.a9`. This lesson relies on [[hello-core]] and [[a-result]].

## Why arguments

A kernel that does the same thing every time is not much use. The numbers it works on
have to come from somewhere, and the command is the message that arrives at the right
moment. Python puts the numbers in the command; the kernel takes them out.

## The general form

In Python, in the call of `submit`:

```python
handle = run.submit(0, group=(0, 1), args=[FIRST, SECOND, ...])
```

In a kernel, inside the function you give to `run`:

```cpp
const uint32_t name = command.a0;
const uint32_t other = command.a1;
```

- `args=[...]` is a list of up to ten whole numbers, from 0 to 4,294,967,295 (32 bits,
  not negative).
- `command.a0` is the first number of the list, `command.a1` the second, and so on to
  `command.a9`. They are `uint32_t`.
- `name` is yours. Read each argument once, at the top of the function, into a variable
  that says what the number is: `left`, `right`, `count`. After that line the rest of the
  function never writes `a0` again. The tutorial does this in every kernel, because
  `a3` means nothing to a reader and `first_page` does.
- An argument that you did not send reads as 0.

## One line at a time

```{ .python }
--8<-- "tutorial/07_arguments/arguments.py:submit"
```

`args=[left, right]` puts the two numbers in the command. The program sends the sums of
three pairs, one command each, and `left` and `right` are the loop's variables.

<figure class="km-diagram">
<script type="application/json">
{"kind": "anatomy",
 "parts": [
  {"t": "args=["},
  {"t": "left", "n": "arrives as\ncommand.a0"},
  {"t": ", "},
  {"t": "right", "n": "arrives as\ncommand.a1"},
  {"t": "]"}]}
</script>
<figcaption>The first number of the list is argument 0, the second argument 1.</figcaption>
</figure>

```{ .cpp }
--8<-- "tutorial/07_arguments/kernels/writer.cpp:run"
```

The kernel reads the two numbers into `left` and `right`, adds them, and sets the sum as
the result ([[a-result]]). The addition runs on the core, in C++.

## The whole program

=== "arguments.py"

    ``` { .python }
    --8<-- "tutorial/07_arguments/arguments.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/07_arguments/kernels/writer.cpp"
    ```

This is the program of [[what-komorebi-is]], the one that had not been explained. Its
reader and compute kernels are the empty ones.

```{ .text .km-output }
--8<-- "tutorial/07_arguments/output.txt"
```

## Common mistakes

- Eleven numbers, or more. `submit` stops with `ValueError: 11 arguments; a command has 10`.
- A negative number, or one above 4,294,967,295. `submit` stops with
  `ValueError: arguments are 32-bit unsigned words`.
- A number with a fraction. `args=[1.5, 0]` is not refused: the 1.5 arrives as 1, with no
  message. Pass whole numbers.
- A sum that does not fit in 32 bits. The kernel's addition wraps around: with
  `args=[4294967295, 1]` the result is 0.

Next: [[handlers]] puts two different pieces of work into one kernel.
