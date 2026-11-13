---
title: Debug a command that does not finish
summary: How to read the card's report of a command that missed its deadline, and find the missing call in a kernel.
tags:
  - guide
  - debugging
---

# Debug a command that does not finish

Use this when a program stops with `km.CommandTimeout`, or when `wait` raises
`km.WaitTimeout` and you are not sure why. The lessons behind it are
[When it goes wrong](../tutorial/when-it-goes-wrong.md) and
[Finding a bug](../tutorial/finding-a-bug.md).

## Steps

1. Tell the two errors apart. `WaitTimeout` means that Python stopped waiting; the command
   may be slow and not broken. Wait again with a longer `timeout`, or time the command with
   [Measure a command](measure-a-command.md). `CommandTimeout` means that the card stopped
   the cores at the command's deadline, and that the run is over.
2. Read the message of `CommandTimeout` in full. It names the command, the group, and
   each core of the group that had not reported, with the step that its reader and its
   writer were in.
3. Read each step:
    - `next`: the kernel was waiting for its next command, so it had finished this one.
    - `data`: the kernel was waiting on a circular buffer. A writer waits for a tile to be
      there, a reader for room to put one.
    - `cdone`: the writer had finished and was waiting for the other kernels of the core.
4. Name the victim and the cause. The kernel that waits is the victim. The cause is the
   kernel that did not do what the waiting one needs: a writer that waits in `data` needs
   the reader's `cb_push_back`; a reader that waits in `data` needs the writer's
   `cb_pop_front`.
5. Look at that call in the kernel, fix it, and load again. After a `CommandTimeout` the run
   cannot be used.
6. If the kernel is right and the command is only slow, raise the deadline of
   `session.load(program, deadline=...)` to the longest honest command of the program.

## What to check first

- The writer of a handler returns `true` when it is done ([Hello, core](../tutorial/hello-core.md)).
- Every `cb_push_back` has a `cb_pop_front` of the same number of tiles
  ([Circular buffers](../tutorial/circular-buffers.md)).
- Every tensor is made before `load`; a tensor written or read while the kernels run
  waits 60 seconds and then raises a `RuntimeError`
  ([A whole program](../tutorial/a-whole-program.md)).
