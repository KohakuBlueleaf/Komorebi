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

```bash
python table_jobs.py --cores 8 --jobs 3    # the class on 8 cores, 3 jobs a core, checked against numpy
```

## Results

Card 0, a Blackhole p150a. The jobs are random (seed 1): 1 to 5 source pages each,
anywhere in 64 pages of bfloat16 integers -4 to 4, half of them adding 0.5.

| | Class |
|---|---|
| Cores, jobs | 8, 24 (3 a core) |
| Target against numpy | 0 words differ |
| Time a command | 11.16 us |
| Kernel text, B (mover A, mover B, unpack, math, pack) | 14,840, 9,968, 6,004, 2,324, 3,128 |

The ray tracer's class reads its directory by the same line
([raytrace_class](../raytrace_class/README.md)).
