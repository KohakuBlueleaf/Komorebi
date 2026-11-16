# Attention of tile functions

The attention project's ATTEND as a class whose arithmetic is tile functions and class
lines only, with no inline C++. It runs beside the hand program on the same Q, K and V.
The movement is [attention_class](../attention_class/README.md)'s: the job table, Q
held in a read stream, and K^T and V cast down each column by a broadcast. Each core
takes one job (the plans of t1 and causal64 give at most one a core; the script refuses
a plan that gives more, because the softmax states and O's rows are zeroed once a
handler).

## The lines and what each lowers to

| Line | Lowers to | Class line ([classes](../../../docs/reference/classes.md)) |
|---|---|---|
| `self.state.zero()`, `self.o.zero()` | R zero states and R rows of DV zero fp32 tiles packed into L1 on the core, once (no DRAM read) | a field zeroed on the core; rows of a field |
| `self.mask.fill(...)` (causal) | the three mask rows read from `init` once | a fill |
| `s = start(self.mask[i], self.state[r])` (causal) | the mask row for the step's first key (0, the diagonal tile's triangle, -inf past it) and state r copied into DST; the scores then add into it, and the softmax step runs on the state already there | sums started from a field; a step on sums that carry their state |
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
| time a command | 13.30 us | 16.36 us (16.18 to 17.68) |
| kernel text (mover A, mover B, unpack, math, pack) | 23,596, 8,648, 14,108, 7,796, 9,092 | 24,668, 10,120, 17,664, 15,796, 15,496 |

[attention_class](../attention_class/README.md), the same movement with the hand's
compute as inline C++, took 13.42 us on t1.

**Where the time goes on t1 (measured by the run's trace).** The trace stamps, on each
core, when mover A took the command and when mover B was done with it. Times are
microseconds after the first core took the command, the median of the 20 commands:

| | t1 |
|---|---|
| the last core takes the command | 0.20 us |
| mover B done, the 116 cores without rows (no O pages to write) | 2.80 us |
| mover B done, the 4 cores with a row (O's two pages written after compute packed them) | 14.77 us |
| the command, host to host | 16.36 us |

The command ends with the four cores that hold a row. On them mover B is done once
compute has run the row's two steps after the block of K and V arrived down the
column, and the row's O has been divided and written. The trace has no stamp between
the block's arrival and compute's last pack, so it does not split those 14.77 us
between the broadcast and compute. attention_class moves the same blocks the same way
in 13.42 us a command. The rest, about 3 us, is therefore compute's tile steps as
lowered here (my reading, not a measurement). Each step of two keys is two DST
cycles. The first makes the scores, runs the softmax step and packs P and the state.
The second copies O's row, rescales it, adds P by V and packs O and the state. Each
cycle waits for the state's token from the cycle before it, so pack and unpack do not
overlap the steps of one row.

The first build read the states and O's rows from a tensor of zeros
([results/t1_fills.json](results/t1_fills.json)): 39 fp32 pages a core on mover A,
19.2 MB of DRAM for 120 cores. It took 63.32 us a command (62.60 to 64.92) with the same
O. `self.f.zero()`, which makes them on the core instead, took 47 us off.

| causal64 (1 x 8 x 1,024 x 64, causal), the build that read zeros | hand | tile class |
|---|---|---|
| O against float64: relative RMS, largest error | 4.74e-3, 8.96e-3 | 6.09e-3, 1.56e-2 |
| O against the hand's: relative RMS, largest error | | 4.59e-3, 1.56e-2 |
| time a command | 57.59 us | 206.97 us (206.56 to 208.48) |

causal64 ([results/causal64_fills.json](results/causal64_fills.json)) was measured with
the states and O's rows still read from DRAM: 45 fp32 pages a core with the masks,
22.1 MB. It has not been measured with them zeroed on the core. Each core holds 4 rows,
10 rows apart, and each row takes the keys up to its own diagonal tile, so a core runs
about 34 steps of two keys where t1 runs two.

```bash
python attention.py card --shape t1        # the hand, then the class: O, times, trace
python attention.py card --shape causal64
python attention.py build --shape t1       # a mock device: the class's kernel text
```
