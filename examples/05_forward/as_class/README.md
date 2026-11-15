# 05 forward, written as a class

The dynamic handler of [../forward.py](../forward.py) as one method of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md#dynamic-handlers-and-messages),
experimental). Its loop `for message in self.arrive():` makes it dynamic: made, the
class is compiled into a compute kernel alone under `build/forward-<hash>/` beside the script,
and the reader and writer are the runtime's message movers. A message is `Visit`, a
class deriving `km.Message` whose int annotations are its words (`id`, `forwards`,
`state`, `sum`: words 1 to 4, as the hand program's). The program is made dynamic as
the hand program is, with `.dynamic("forward", **options)`.

```bash
python forward_class.py --messages 10000 --forwards 100              # the class
python forward_class.py --beside --messages 10000 --forwards 100 --repeat 3
```

`--beside` runs the hand program before each run of the class, in the same session.
The flags and the report are the hand program's; the class has no stall or probe
test (`--stall`, `--probe`) and no engine run (`--engine`). Each core's totals go to
the same L1 area as the hand program's, by a `km.cxx` body at the end of the handler
that takes them; its address is a command word (`results`).

Measured on one Blackhole card (`--beside --messages 10000 --forwards 100 --repeat 3`):
both exact in 3 of 3 runs (10,000 messages consumed, 10^6 forwards, no core gave up).
The `RUN` command, min / median / max: 12,131 / 12,162 / 12,194 us against the hand
program's 12,289 / 12,297 / 12,406 us. Kernel text 41,600 B against 43,040 B: the
movers' are the same runtime kernels; the class's compute loop has no stall flag and
no test of the probe's null messages.
