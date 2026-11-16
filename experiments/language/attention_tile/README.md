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
Each program ran 20 commands back to back; the tables give the median. The hand and
the tile class ran in the same job. [attention_class](../attention_class/README.md)
(the same movement, the hand's compute as inline C++) is from its own README, card 0.

| t1 (1 x 1 x 128 x 64) | hand | attention_class | attention_tile |
|---|---|---|---|
| O against float64: relative RMS, largest error | 5.50e-3, 6.41e-3 | the hand's words exactly | 6.58e-3, 6.41e-3 |
| O against the hand's: relative RMS, largest error | | 0 of 8,192 words differ | 4.24e-3, 3.91e-3 |
| time a command | 13.30 us | 13.42 us | 16.36 us (16.18 to 17.68) |
| kernel text (mover A, mover B, unpack, math, pack) | 23,596, 8,648, 14,108, 7,796, 9,092 | 25,836, 10,236, 16,376, 7,340, 8,288 | 24,668, 10,120, 17,664, 15,796, 15,496 |

| causal64 (1 x 8 x 1,024 x 64, causal) | hand | attention_class | attention_tile |
|---|---|---|---|
| O against float64: relative RMS, largest error | 4.74e-3, 8.96e-3 | the hand's words exactly | 6.09e-3, 1.56e-2 |
| O against the hand's: relative RMS, largest error | | 0 of 524,288 words differ | 4.59e-3, 1.56e-2 |
| time a command | 57.62 us | 57.75 us | 160.90 us (160.50 to 162.94) |
| kernel text (mover A, mover B, unpack, math, pack) | 23,596, 8,648, 14,388, 8,436, 9,340 | 25,836, 10,236, 14,928, 7,988, 8,592 | 24,564, 10,120, 17,452, 16,896, 16,068 |

**Where the time goes (measured by the run's trace).** The trace stamps, on each core,
when mover A took the command and when mover B was done with it. Microseconds after
the first core took the command, the median of the 20 commands:

| | t1 | causal64 |
|---|---|---|
| the last core takes the command | 0.20 us | 0.19 us |
| mover B done, the cores without rows (no O pages to write) | 2.80 us (116 cores) | 2.89 us (40 cores) |
| mover B done, the cores with rows (O written after compute packed it) | 14.77 us (4 cores) | 159.25 us (80 cores) |
| the command, host to host | 16.36 us | 160.90 us |

Each command ends with the cores that hold rows: mover B is done once compute has run
the rows' steps as the K and V blocks arrive down the column and has divided and
written O. The trace has no stamp between a block's arrival and compute's last pack,
so it does not split that time between the broadcast and compute. Since
attention_class moves the same blocks the same way in the hand's time, the rest is
compute's tile steps (a reading, not a measurement). On causal64 the slowest core runs
34 steps of two keys (4 rows, 10 rows apart, each to its own diagonal tile), about
4.7 us a step. The same rows in steps of three keys would be 24.

**What the tile class still cannot match, and why.** A step of two keys is two DST
cycles: the scores, the softmax step, then P and the state packed; then O's row copied,
rescaled, P by V added, then O and the state packed. Each cycle waits for the state's
token from the cycle before it. Against the hand's compute (flash.hpp), in a user's
words:

- The step's own cost. This is t1's 3 us: two steps, each slower than the hand's. Each
  call loads its tile function's constants again because the class has no kernel-wide
  programmable constants. The softmax is a general vector program, not flash's
  macros. Math and pack take turns on the token, where the hand gives pack vector work
  while math runs the products. None of the three is measured apart from the others.
- More steps on causal64. A step is two keys, not the hand's three: t1's four keys and
  a row's last keys divide into threes only with a mask on a partial step, which a
  step cannot take. So causal64's slowest core runs 34 steps where 24 would do.
- The rescale skip. Every step rescales O's row, even when the row's maximum did not
  move. This adds to causal64's long rows and does not bear on t1's two steps.

Together these are causal64's remaining 103 us over the hand: about 1.4 times the
steps, each about twice the hand's cost.

The first build read the states and O's rows from a tensor of zeros
([results/t1_fills.json](results/t1_fills.json),
[results/causal64_fills.json](results/causal64_fills.json)): 39 fp32 pages a core on
mover A (45 with causal64's masks), 19.2 and 22.1 MB of DRAM. It took 63.32 us on t1 and
206.97 us on causal64 with the same O. `self.f.zero()`, which makes them on the core,
took 47 and 46 us off.

```bash
python attention.py card --shape t1        # the hand, then the class: O, times, trace
python attention.py card --shape causal64
python attention.py build --shape t1       # a mock device: the class's kernel text
```
