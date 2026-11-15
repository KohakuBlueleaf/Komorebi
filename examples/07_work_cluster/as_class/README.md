# 07 work cluster, written as a class

`STEP` and `SUM` of [../kernels](../kernels) as two handlers of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md#x280-threads),
experimental), driven by an X280 thread on cluster 1. Made, the class is compiled into
an ordinary program's files under `build/workcluster-<hash>/` beside the script, with
`workcluster_thread.h`: each handler's number and the place of each of its command
words. The thread ([thread/driver.c](thread/driver.c)) runs [../decide.c](../decide.c)
as the hand program's thread does and places each command's words by that header
before `kmx_submit`; it waits for each with `kmx_wait`.

```bash
python work_class.py              # the class's thread
python work_class.py --beside     # the hand program's thread first
```

- `step(state, round, first)`: each core of the group reads the word of its page
  (`state.pages(page, 1).read()`, page = `first` + its rank), adds `round` and the
  page, and writes it back (`state.pages(page, 1).write(word)`).
- `sum(state, cores, first)`: the leader reads the `cores` words from page `first` and
  returns their total, the command's result.

The class's commands carry the group's first core as a word (`first`); the hand
kernels read it from the core's own number. The checks are the hand program's: the
code, the rounds, the sum, the trail and every round's group against
[../work_cluster.py](../work_cluster.py)'s numpy replay, and every core's state word.

Measured on one Blackhole card (`--beside`, the defaults: 119 cores, 30 a STEP, limit
1,000,000): both the same as numpy: 204 rounds, sum 1,004,917, every state word. The
loop on the thread: 10.55 ms against the hand program's 10.75 ms (408 commands, two
submitted and waited for in turn each round). Kernel text 21,200 B against 21,424 B.
