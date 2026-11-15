# 11 movement, written as a class

`READ` and `COPY` of [../movement.py](../movement.py) as two methods of a class deriving
`km.Program` ([Programs written as classes](../../../docs/reference/classes.md), experimental).
Made, the class is compiled into the program's files under `build/movement/` beside the
script. The copies, checked word by word, and the rates are the hand program's.

```bash
python movement_class.py --beside --cores 8 --pattern AB AAAB
```

`--beside` runs the hand program first in the same session. The class's tensors are
bfloat16 (the same bits as the hand program's pages): its buffers take the format of
the tensor a stream reads. Its block is a size of the build (`plan={"BLOCK": 16}`,
`--block`), not a command word.

Measured on one Blackhole card (the line above): every copy exact for both programs;
READ GB/s, the hand program's and the class's: A 234.52 and 234.51, AB 185.61 and
185.15, AAAB 310.44 and 308.74. Kernel text 38,288 B against 36,720 B.
