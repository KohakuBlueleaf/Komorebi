---
title: "Two programs"
part: 6
summary: "several programs in one run, on the same cores or on groups of their own, and a group that switches"
---

# Two programs

Until now a run held one program, and its kernels were built before the run started. A
program can instead give one function to each kernel, `km_program`. Komorebi then builds the
program as an [overlay](overlays.md), and its own kernels run on every core. For each
command they load the program the command names and call its function. One run can hold
several programs. This lesson runs two, `add` and `multiply`, first on the same cores, then
each on its own group. It relies on [[handlers]], [[a-result]] and [[overlays]].

## The runtime's form

A kernel source in the form you have used so far defines `kernel_main`, which loops over the
commands itself. A source in the **runtime's form** defines `km_program` instead. The runtime
calls it once for each command, with the kernel's object and the command:

```{ .cpp }
--8<-- "tutorial/46_two_programs/add/writer.cpp:program"
```

The reader and the compute kernel of `add` have nothing to do, so their `km_program` is
empty. Komorebi tells the forms apart by the sources: a source that defines `km_program` is in
the runtime's form. All the kernels of one program use the same form.

## Two programs, one handler each

```{ .python }
--8<-- "tutorial/46_two_programs/two_programs.py:programs"
```

Both programs number their handlers from 1, and both have a handler 1. A command names its
program as well as its handler: `add.handler("apply")` is handler 1 of `add`, and
`multiply.handler("apply")` is handler 1 of `multiply`. `run.submit` takes either.

## Every core holds both

```{ .python }
--8<-- "tutorial/46_two_programs/two_programs.py:combined"
```

`session.load([add, multiply])` loads both on every core. The commands alternate between
the programs, and each core loads the program that a command names. Programs that share
cores also share one L1 layout: a buffer index that both declare must be declared the same
way.

## A group each

```{ .python }
--8<-- "tutorial/46_two_programs/two_programs.py:groups"
```

`session.load({add: first, multiply: second})` gives each program its own half of
`session.ranks`, the cores a program runs on (120 with the X280 dispatcher). The host
refuses a command to cores that do not hold its program before sending it.

## A group changes program

```{ .python }
--8<-- "tutorial/46_two_programs/two_programs.py:switch"
```

`run.load(multiply, group=part)` is a command like any other: cores 0 to 29 take it
after their earlier commands, and from then on they hold `multiply`. Its code was built
and placed in DRAM with the run, so nothing moves from the host. The cores' buffers are
set up again for `multiply`, and whatever `add` kept in them is gone. Commands of `add`
to those cores are now refused, and cores 30 to 59 still hold `add`.

## Run it

Run it from the folder `tutorial/46_two_programs`:

```bash
python two_programs.py
```

```{ .text .km-output }
--8<-- "tutorial/46_two_programs/output.txt"
```

## Common mistakes

- Defining `kernel_main` in one kernel and `km_program` in another. The build refuses the
  mix: every kernel of a program is in one form.
- Submitting a bare number such as `1` to a run of two programs. A number means a handler of
  the first program; use `program.handler(...)` for the others.
- Keeping state in a global of a `km_program` source between commands. Another program can
  replace it in its slot, so its globals start again when it is loaded.

The example [`16_two_programs`](../../examples/16_two_programs/README.md) runs two programs
that read and write tiles, checks every result and measures the price of the runtime's form.
The [overlay reference](../reference/overlays.md#a-program-as-an-overlay) has every rule.

Next: [[a-thread-and-two-programs]] lets an X280 thread command both programs.
