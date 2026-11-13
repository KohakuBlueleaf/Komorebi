---
title: "Overlays"
part: 6
summary: "work that is not in the kernels, loaded into a core while they run"
---

# Overlays

An **overlay** is a piece of kernel code that is not built into the kernels. Komorebi builds it
on the host and keeps it in DRAM. A core **loads** it into a **slot** of its L1 while the
kernels run, then **calls** it like a function. This lesson runs two overlays, `add` and
`multiply`, through one slot of the writer, in turn. It relies on [[handlers]] and
[[a-result]].

## Why overlays

In [[handlers]] the writer held both pieces of work, and each command named the one it
wanted. Every handler a program might need is built into its kernels before the run starts,
and all of them take room in every core. An overlay is built into the kernels only as a
number. The code arrives when a command needs it, and it can replace code that is no longer
needed.

## The program

```{ .python }
--8<-- "tutorial/43_overlays/overlays.py:program"
```

`program.overlay(name, folder)` declares an overlay. The folder holds one source file for
each kernel that has a part of it: here only `writer.cpp`. Its number comes from the handlers
header, the way a handler's does: `add` is `kOverlayAdd`, which is 1. `program.slots(count=1)`
gives each kernel that has parts one slot, as large as its largest part. When the program
loads, Komorebi builds every part, checks it, and prints how much of the core's L1 the slots
take.

The part is one function with a fixed name, `km_overlay`. It takes the kernel object and one
argument of your own type:

```{ .cpp }
--8<-- "tutorial/43_overlays/kernels/add/writer.cpp:part"
```

## Loading and calling

The writer names the overlay the command asks for, loads it and calls it:

```{ .cpp }
--8<-- "tutorial/43_overlays/kernels/writer.cpp:run"
```

The reader does the loading. It reads the parts of an overlay from DRAM into their slots, the
writer's part and the compute cores' parts as well as its own. So the reader calls `load()`
for the same overlay in the same command, and the writer's `load()` waits for it:

```{ .cpp }
--8<-- "tutorial/43_overlays/kernels/reader.cpp:run"
```

`load()` does nothing when a slot already holds the overlay. With one slot, each load of the
other overlay replaces it.

```{ .python }
--8<-- "tutorial/43_overlays/overlays.py:submit"
```

## The whole program

=== "overlays.py"

    ``` { .python }
    --8<-- "tutorial/43_overlays/overlays.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/43_overlays/kernels/writer.cpp"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/43_overlays/kernels/reader.cpp"
    ```

=== "kernels/multiply/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/43_overlays/kernels/multiply/writer.cpp"
    ```

Run it from the folder `tutorial/43_overlays`:

```bash
python overlays.py
```

```{ .text .km-output }
--8<-- "tutorial/43_overlays/output.txt"
```

The third command loaded `add` again, because `multiply` had replaced it. With
`program.slots(count=2)` both would stay, and the third load would do nothing.

## Common mistakes

- Calling `load()` on the writer or on compute and not on the reader. The writer waits for
  the reader, then gives up when its time limit passes, and the command fails.
- Calling `call()` after a `load()` that returned false. The call does nothing and returns 0;
  the command reports why the load gave up.
- Calling a function of the kernel's own source from a part. The part is built on its own and
  cannot reach the kernel's code, so the build refuses it. Header-only calls such as
  `writer.wait_front` are compiled into the part and work.

The example [`12_overlays`](../../examples/12_overlays/README.md) loads an overlay with
parts on the reader, the writer and the three compute cores on every core, and times each
load and call. The [overlay reference](../reference/overlays.md) has every rule.

Next: [[two-programs]] runs two programs in one run, each loaded as an overlay.
