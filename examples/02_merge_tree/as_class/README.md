# 02 merge tree, written as a class

`MERGE` of [../merge.py](../merge.py) as one method of a class deriving `km.Program`
([Programs written as classes](../../../docs/reference/classes.md), experimental): each core
adds its part of the tiles, `km.reduce` sums the parts by owner or along a tree, and the
leader writes the sum. Made, the class is compiled into the program's files under
`build/mergetree/` beside the script. The jobs and the checks are the hand program's.

```bash
python merge_class.py --beside                   # 20 groups of 8, owner and tree in turn
```

`--beside` runs the hand program first in the same session and from the same seed.
The class sums parts of one tile (`--part 1`) and stalls no rank (`--stall-rank`
is refused), and it asks for no result word: the hand program's checksum of the sum
is not in it.

Measured on one Blackhole card (`--beside`): both ok, 40 jobs, every exact job equal
to numpy and each owner and tree pair equal bit for bit, the random ones at relative
RMS 8.6e-08; kernel text 43,360 B against the hand program's 36,176 B (the hand
program's reader and writer read and write pages directly; the class moves streams).
A job, exclusive: host 42.0 us against 35.4 us (the hand program's asks for a result
word).
