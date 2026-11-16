# Attention of tile functions

The attention project's ATTEND as a class whose arithmetic is tile functions and class
lines only, with no inline C++. It runs beside the hand program on the same Q, K and V.
The movement is [attention_class](../attention_class/README.md)'s: the job table, Q
held in a read stream, and K^T and V cast down each column by a broadcast. Each core
takes one job (the plans of t1 and causal64 give at most one a core; the script refuses
a plan that gives more, because the softmax states and O's rows are filled once a
handler).

## The lines and what each lowers to

| Line | Lowers to | Class line ([classes](../../../docs/reference/classes.md)) |
|---|---|---|
| `self.state.fill(...)`, `self.o.fill(...)` | R zero states and R rows of DV zero fp32 tiles read from `init` into L1 once, and each field's token pushed | a fill; rows of a field |
| `qs = km.Stream(q.tiles[...], block=QB)` | the job's Q rows read by mover A, waited for once, freed after the job | a held stream |
| `kv = down.cast(k.tiles[...].T, v.tiles[...])` | K^T and V of the block cast down the column; compute waits for it at the cast and frees it at the end of the block's lines | a broadcast; a field by a broadcast's block |
| `s += score(qs[r], kv[0][:, c * SUB : c * SUB + SUB])` | DT products of Q's row r by rows of K^T's window of SUB keys, transposed (HiFi2) | a held stream's row by a broadcast's block |
| `self.p.write(softmax_step(s, self.state[r]))` | state r copied into DST after the scores; the step's vector program updates m, l and the rescale and makes P = 2^(x - m); P packed to bf16 into the field p and the state packed back | a step function on DST sums; a product whose left side is a field |
| `acc = rescale(self.o[r], self.state[r])` | O's row r and state r copied into DST in place; the row times the rescale | rows of a field; sums started from a field |
| `acc += weigh(self.p, kv[1][c * SUB : c * SUB + SUB, :])` | SUB products of P's tiles by V's rows of the same keys (HiFi2) | a field by a broadcast's block |
| `self.o[r].write(acc)` | O's row r packed back in place, its token pushed, then the state | rows of a field |
| `last = norm(self.o[r], self.state[r])`, `o.pages(...).write(last)` | the row divided by l and written to O by mover B | sums started from a field |

The step functions:

```python
@km.tile_function
def softmax_step(row, state):
    x = row * (SCALE * LOG2E)
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    l_new = alpha * l_old + km.reduce_sum(p, exact=True)
    return p, (m_new, l_new, alpha)


@km.tile_function
def rescale(row, state):
    return row * state[2].col(), (state[0], state[1], state[2])
```

A step takes SUB = 2 key tiles, where the hand takes 3: t1's block of 4 keys divides
into 3s only with a mask on the last step.

## Results

Card 1, a Blackhole p150a, 120 cores (aiclk 800 MHz at the start). The plan has R 13,
KB 12, window 2 and L 10. Q, K and V are the project run's (`sdpa.inputs`, seed 1).
Each program ran 20 commands back to back; the table gives the median.

| t1 (1 x 1 x 128 x 64) | hand | tile class |
|---|---|---|
| O against float64: relative RMS, largest error | 5.50e-3, 6.41e-3 | 6.58e-3, 6.41e-3 |
| O against the hand's: relative RMS, largest error | | 4.24e-3, 3.91e-3 |
| time a command | 13.24 us | 63.32 us (62.60 to 64.92) |
| kernel text (mover A, mover B, unpack, math, pack) | 23,596, 8,648, 14,108, 7,796, 9,092 | 24,712, 10,120, 17,664, 15,428, 12,756 |

[attention_class](../attention_class/README.md), the same movement with the hand's
compute as inline C++, took 13.42 us on t1.

**Where the 50 us went (an estimate from the build, not measured).** The fills are the
only lines the tile class adds to the movement of attention_class. On mover A of every
core they read 39 fp32 pages of 4,096 B (159,744 B), so 120 cores read 19.2 MB of DRAM.
The hand reads Q, K, V (48 KB) and the job tables (480 KB). At 400 to 500 GB/s of DRAM
reads, 19.2 MB takes 38 to 48 us. Compute does the same arithmetic as the hand: one
row a core and two steps of two key tiles on t1, about the hand's three-tile step. No
probe has yet split the time between mover A's fills and compute's steps. The fix
would be a field of rows zeroed by compute rather than read from DRAM; that line is
not built.

```bash
python attention.py card --shape t1    # the hand, then the class: O against float64, times
python attention.py build --shape t1   # a mock device: the class's kernel text
```
