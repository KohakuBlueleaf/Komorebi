# 04 uop stream

The deciding is done by a micro-op stream: 1,000 `MAP` jobs, each on a group whose size
the stream draws at random, run three ways that must give the same tiles.

```python
s = ks.Stream(seed=seed)
j, x, count, first, room, base, page = s.regs(7)
top = s.label()
s.rand(x)
s.mod(count, x, n_cores)
s.addi(count, count, 1)  # 1 to every core
if form == "submit":  # the stream picks the first core too
    s.sub(room, n_cores, count)
    s.addi(room, room, 1)
    s.shr(first, x, 8)
    s.mod(first, first, room)
    s.submit(MAP, (first, count), args=[inp, T, out, base, page])
    s.wait(0)  # one command in flight
else:  # the dispatcher's queue picks it
    s.enqueue(MAP, count, args=[inp, T, out, base, page])
s.add(base, base, count)  # the next job's output tiles
s.addi(page, page, 61)  # the input window moves (modulo W)
...
s.addi(j, j, 1)
s.jlt(j, n_jobs, top)
s.wait(0)
s.halt(j)
```

| Way | Who decides | How the commands go |
|---|---|---|
| `hostfed` | the host | `run.run_hostfed(stream)`: the package's stream interpreter runs the SUBMIT stream on the host and sends its commands through the host ring, each waited for as the stream says |
| `submit` | the X280 (core 1 of cluster 0) | `run.upload(stream)`, `run.start`, `run.wait_engine`: the SUBMIT stream submits each command and waits for it |
| `enqueue` | the X280 and the dispatcher's job queue | the same calls with the ENQUEUE stream; the queue gives each job the lowest free group of its size, so several jobs are in flight |
| `errors` | - | three streams that end in an error event, then 20 jobs of the SUBMIT stream on the same run |

`MAP` (`kernels/`): rank m of a job adds its partition of the job's T input tiles (from
tile a4) into one tile and stores it at output page a3 + m. Groups differ between the ways
(only `submit` and `hostfed` pick the same ones), but every job writes the same tiles. The
inputs are integers in [-2, 2], so every sum is exact in bf16, and the host compares every
output tile of every way bit for bit with numpy. It also checks each core's command count
against the simulated groups (`hostfed`, `submit`) or the total of core commands
(`enqueue`), and gives microseconds per job, from the start to the stream's end. Each run
is traced: the groups the grid saw must be the simulator's in order for `submit` (one
command in flight), and the trace gives the most commands in flight at once.

The `errors` way uploads a stream whose first micro-op has a bad opcode (the kernel's
check ends it with reason `opcode`), one that loops past its step limit (`steps`) and one
that enqueues a job larger than the grid (`never_fits`). Each ends in its error event; the
run goes on, and 20 jobs of the SUBMIT stream after them are exact, every core's count is
right, and the stop is clean.

## Running it

    python uop_stream.py --card 0                   # every way
    python uop_stream.py --card 0 --ways hostfed    # one way

Options: `--jobs` (1,000), `--tiles` T (64), `--seed` (the stream's xorshift seed),
`--depth` (ring slots, 16), `--deadline` (per command, seconds), `--no-trace`. It prints
one JSON line with `ok`, true only when every check above holds, and exits non-zero
without it. Reading the 59,456 output tiles (116 MB) back takes 65 to 80 s per way, in
chunks of 2 MB with a line per chunk (one every 1 to 2 s).

On a p150a (card 0, 120 cores, default arguments, X280 at the session's 1000 MHz;
the 1750 MHz opt-in's figures in the last column): 1,000 jobs, 59,456 tiles, 0 wrong
and every core's count right in every way; the SUBMIT and ENQUEUE streams give the same
jobs, and the grid saw the simulator's groups in order.

| Way | Microseconds per job | Most commands in flight | Upload | 1750 MHz |
|---|---|---|---|---|
| `hostfed` | 163 | 1 | - | 161 |
| `submit` | 13.4 | 1 | 576 bytes | 11.5 |
| `enqueue` | 9.3 | 2 | 512 bytes | 8.4 (3 in flight) |

Host-fed, every job is a round trip to the host; on the X280 it is a round trip to the
dispatcher on the same cluster. The three error streams ended with `opcode`,
`steps` and `never_fits`. The dispatcher's image with both engines is 27,664
bytes.
