# 04 uop stream

The deciding is done by a micro-op stream: 1,000 `MAP` jobs, each on a group whose size
the stream draws at random, run three ways that must give the same tiles.

```python
stream = micro_ops.Stream(seed=seed)
job, random_word, count, first, room, base, page = stream.registers(7)
top = stream.label()
stream.rand(random_word)
stream.mod(count, random_word, core_count)
stream.addi(count, count, 1)  # 1 to every core
map_args = [addresses["in"], tiles, addresses["out"], base, page]
if form == "submit":  # the stream picks the first core too
    stream.sub(room, core_count, count)
    stream.addi(room, room, 1)
    stream.shr(first, random_word, 8)
    stream.mod(first, first, room)
    stream.submit(addresses["map"], (first, count), args=map_args)
    stream.wait(0)  # one command in flight
else:  # the dispatcher's queue picks it
    stream.enqueue(addresses["map"], count, args=map_args)
stream.add(base, base, count)  # the next job's output tiles
stream.addi(page, page, STEP)  # the input window moves (modulo the window)
...
stream.addi(job, job, 1)
stream.jlt(job, job_count, top)
stream.wait(0)
stream.halt(job)
```

| Way | Who decides | How the commands go |
|---|---|---|
| `hostfed` | the host | `run.run_hostfed(stream)`: the package's stream interpreter runs the SUBMIT stream on the host and sends its commands through the host ring, each waited for as the stream says |
| `submit` | the dispatcher's engine: the X280's core 1 of cluster 0, or TRISC 0 of a Tensix dispatcher's core | `run.upload(stream)`, `run.start`, `run.wait_engine`: the SUBMIT stream submits each command and waits for it |
| `enqueue` | the engine and the dispatcher's job queue | the same calls with the ENQUEUE stream; the queue gives each job the lowest free group of its size, so several jobs are in flight |
| `errors` | - | three streams that end in an error event, then 20 jobs of the SUBMIT stream on the same run |

`MAP` (`kernels/`): each rank of a job adds its partition of the job's T input tiles
(`tiles`, a1, from tile `first_tile`, a4) into one tile and stores it at output page
`first_page` (a3) plus its rank. Groups differ between the ways
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

Under the Tensix dispatcher (`KMR_DISPATCHER=tensix`) the engine runs on TRISC 0 of the
dispatcher's core. Card 1 (p150a), `--ways submit,enqueue`, 1,000 jobs, 2026-11-19,
1-minute load 23 to 26, each dispatcher in the same job, every tile and core count right:

| Way | X280, 120 cores | Tensix dispatcher, 119 ranks |
|---|---|---|
| `submit` | 21.76 us a job, 1 in flight | 22.02 us, 1 in flight |
| `enqueue` | 18.68 us, 4 in flight | 19.46 us, 3 in flight |

On a Wormhole chip (79 ranks, 2026-11-19) the same ways give 21.3 and 17.0 us a job. The
engine's code on TRISC 0 is 16,960 bytes on Wormhole and 16,608 on Blackhole.
