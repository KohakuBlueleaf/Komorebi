# A table of jobs, as a class

Question: can a class walk a table of jobs that a page of words holds, on the movers and
on compute alike, with a constant built once at the compute kernel's start and a write
stream its compute body fills? Each line is a line of the class compiler
([classes](../../../docs/reference/classes.md)); nothing here names a buffer, a command
word or a generated type.

`table_jobs.py` makes the class `TableJobs`. Each core's page of the table holds a count
(word 0) and then four words a job, from word 4. A job names its source pages (first,
count), its target's first page and whether the constant tile is added:

```python
class Job(km.Entry):
    source: int
    count: int
    target: int
    add: int


class TableJobs(km.Program, defines={"TABLE_BIAS": "0.5f"}):
    bias: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)

    @km.handler
    def walk(self, tables: km.Tensor, source: km.Tensor, target: km.Tensor):
        km.cxx(
            "...fill_tile(0, TABLE_BIAS), pack into bias...",
            takes=[self.bias],
            at="start",
        )
        km.cxx("noc_async_write_barrier();", on="mover_b", at="end")
        for job in tables.pages(self.rank, 1).entries(Job, first=4, size=4, count=0):
            out = km.Stream(target.pages(job.target, job.count), write=True)
            for tile in km.Stream(source.pages(job.source, job.count)):
                km.cxx(
                    "...copy, add bias if job_add, pack into out_buffer...",
                    takes=[tile, job, out, self.bias],
                )
```

| Line | What it lowers to |
|---|---|
| `class Job(km.Entry)` | the words' indices, `kJobSource` to `kJobAdd` (0 to 3), in every kernel |
| `defines={"TABLE_BIAS": "0.5f"}` | a define of every kernel of the class |
| `km.own(push=None)` and a body `at="start"` | a field compute writes in place, once, after the start-up and before the commands |
| a body `on="mover_b", at="end"` | mover B's lines after its run |
| `.entries(Job, first=4, size=4, count=0)` | the page read by mover A and pushed to compute, by mover B into its own; the loop over the count (word 0, refused past the page's 255 entries) on every CPU; `job_add` and the others read from the entry's words |
| a write stream inside the loop | each job's target pages, packed by the body into `out_buffer` |

A second handler, `gather`, runs on a rectangle of 2 x 2 cores (the class is made
with `rect=(0, 0, 2, 2)`) and takes the ranges of a rectangle of tiles:

```python
down = km.Broadcast(self, along="column")  # a core outside the rectangle skips
r0 = self.y * COLUMNS + self.x
copied = km.Stream(
    qo.tiles[r0 : min(r0 + R * stride, r0 + rows * stride) : stride, 0:DT],
    write=True,
    block=QB,
)
for tile in km.Stream(
    q.tiles[r0 : min(r0 + R * stride, r0 + rows * stride) : stride, 0:DT], block=QB
):
    copied.push(tile)
for chunk in range(chunks):
    k0 = chunk * KB
    if down.sender():
        kv = down.cast(
            k.tiles[k0 : min(k0 + KB, keys), 0:DT].T,
            v.tiles[k0 : min(k0 + KB, keys), 0:DV],
        )
    else:
        kv = down.arrive()
    seen = km.Stream(
        slots.pages((r0 * chunks + chunk) * SLOT, SLOT), write=True, block=SLOT
    )
    km.cxx(
        "...copy the SLOT_PAGES pages of kv_buffer from kv into seen_buffer...",
        takes=[kv, seen],
    )
```

| Line | What it lowers to |
|---|---|
| `q.tiles[r0:min(r0 + R * stride, r0 + rows * stride):stride, 0:DT]`, `block=QB` | Q's `rows` rows (at most R = 4), `stride` rows apart: a `km::Walk` of a row pitch of `stride x DT` pages, one block of R x DT tiles reserved whole and partly read; the write stream of the same rectangle writes the rows read |
| `down.cast(k.tiles[...].T, v.tiles[...])` | the sender reads K's chunk a page at a time, column h of it to slot `h x KB + key`, then V's rows after K's most (slot page `KB x DT`), then casts the block of `KB x DT + KB x DV` = 12 pages; a short last chunk leaves the rest in place |
| `km.cxx(..., takes=[kv, seen])` | compute's three wait for the block, the body copies it page by page into the write stream's block, the block is freed after |

```bash
python table_jobs.py --cores 8 --jobs 3    # walk on 8 cores, 3 jobs a core; gather on 2 x 2; both checked against numpy
```

## Results

Card 0, a Blackhole p150a. The jobs are random (seed 1): 1 to 5 source pages each,
anywhere in 64 pages of bfloat16 integers -4 to 4, half of them adding 0.5.

| | walk alone | walk and gather |
|---|---|---|
| walk: cores, jobs | 8, 24 (3 a core) | the same |
| walk: target against numpy | 0 words differ | 0 words differ |
| walk: time a command | 11.16 us | 11.62 us |
| gather: Q's rows (3 of 4, 4 rows apart) copied | | 0 words differ |
| gather: slot pages (7 keys, 3 chunks, the last of 1 key; 112 filled) | | 0 words differ |
| gather: time a command | | 16.16 us |
| Kernel text, B (mover A, mover B, unpack, math, pack) | 14,840, 9,968, 6,004, 2,324, 3,128 | 17,348, 22,336, 9,572, 3,440, 6,208 |

A second class, `BandCasts` (TableJobs with it did not fit the kernels' text: 75,968
B of 70,656), runs `banded` on 2 x 4 cores: a job loop around a chunk loop of the
job's chunks (job j casts j + 2), its broadcast on mover A in bands of 2 cores:

```python
down = km.Broadcast(self, along="column", length=BAND, mover="A", unread=unread)
for job in range(jobs):
    for chunk in range(job + 2):
        ...  # the cast and the copy, as gather's
```

| Line | What it lowers to |
|---|---|
| `km.Broadcast(..., length=BAND, mover="A")` | 4 broadcasts, one a band of 2 cores of a column, on NoC 0: the senders are rows 0 and 2 |
| the cast inside two loops | the blocks counted by `km_blocks_down` on mover A and compute (5 for 2 jobs), the finish and compute's padding (1 block to the window of 2) after the job loop |
| `unread=unread` | the same command with the word 1: each core's mover pushes each block's slot as it is, then pads |

```bash
python table_jobs.py --part bands    # BandCasts: banded, then again unread
```

| | BandCasts |
|---|---|
| banded: the bands' pages (8 cores, 5 blocks; 416 filled) | 0 words differ |
| banded: time a command | 13.98 us |
| unread: time a command | 9.84 us |
| Kernel text, B (mover A, mover B, unpack, math, pack) | 22,048, 8,820, 4,884, 1,536, 3,340 |

gather places each core by `self.y * COLUMNS + self.x`: a rectangle's group is the
contiguous core indices from its first core to its last, so `self.rank` of its second
row is the grid's width on, and the cores between are in the group but not the
rectangle (they skip the handler at the broadcast's line). The first run on the card
placed the cores by `self.rank`, and every slot page of the second row was read from
pages no core wrote.

The ray tracer's class reads its directory by the same line
([raytrace_class](../raytrace_class/README.md)).
