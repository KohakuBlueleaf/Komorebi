# Attention, as a class: what a class cannot say

Question: can the attention project's program
([projects/attention](../../projects/attention/README.md)) be written as a class
([Programs written as classes](../../docs/reference/classes.md)), as the ray tracer's
trace pass was ([raytrace_class](../raytrace_class/README.md)): every arithmetic body
the hand program's, inline and unchanged (`ktt_flash`'s compute with `sfpu.hpp`),
every move a class line? And where it cannot, which line is missing?

It cannot be written yet. The compute body needs buffers that only it uses, and a
class cannot make one: a body that takes a field which no line fills is refused at
its line ("self.state is not filled above in this handler ..., and no other handler
keeps it filled"), and an inline body cannot declare a buffer. Without them the
compute body would have to change, so the pass stops at this table: no class file and
no card run. Each row is one thing the hand program does and no line says, with the
general line that would say it, its size in the compiler (an estimate, in lines of
`src/komorebi/dsl/`), and whether the ray tracer's and the rasterizer's hand programs
need the same. The rows are ordered by how many of the three places need them.

## What a class can say of it today

- The two handlers and their words: `attend(q, k, v, o, tables, mode, passes,
  first_job, job_count)` and `prof(profile, zero)`, the command words a0 to a8 and a0
  to a1 in the hand's order.
- The plan's sizes as module constants (head tiles, value tiles, rows R, block keys
  KB, the window) and the exponential's constants (scale bits, threshold bits).
- The passes, `for _ in range(max(passes, 1))`, and the jobs' end (all jobs when
  job_count is 0, else `min(first_job + job_count, jobs)`), once the job count is an
  int of the handler (row 3).
- The job table's page as a field of words filled each command,
  `self.table.fill(km.Stream(tables.pages(self.rank, 1)))`. Only bodies could read
  it (row 3).
- The compute options HiFi2 and fp32 DST (`compute={...}`); the float32 buffers'
  unpack to DST comes with row 1.
- The submit with `KM_FLAG_EXCLUSIVE` (`Run.submit(..., exclusive=True)`).

## The table

| # | What the hand program does | The general line that would say it | Size (estimate) | Ray tracer | Rasterizer |
|---|---|---|---|---|---|
| 1 | Compute's own buffers: P (3 bf16 pages, a FIFO from pack to unpack), the row state (2 R float32 tiles written in place, `pack_tile<true>` at a slot), O and the row sums (R x (Dv + 1) float32 tiles written in place with L1 accumulation), two token buffers (pack to unpack and math), the ones tile and the causal mask; each reserved once or by the body, never moved | a field that only bodies use, `p: km.L1[km.bf16, 3, 32, 32] = km.own()`: a buffer of its format and pages, given to bodies by name (`p_buffer`), never filled, waited for or freed by a line; a float32 one unpacked to DST as float32 (`unpack_to_dest_mode`, as a float32 stream's) | 50 to 70 | yes: the treelet path's work pages, signal, packet states (float32 to DST), lists and slots | yes: the pack CPU's work space, the writer's empty tile, the histograms, the list slots |
| 2 | The profile page of five CPUs' words: the reader's (Q waits and reads, the sender's slot waits, K / V reads, casts, arrivals, finish), unpack's, math's and pack's (each at its own submit), the writer's; stored by `prof` and zeroed | a mover-to-mover hand-over and a body on mover B over a write stream's page before the movers write it (the row the ray tracer's class already names) | more than 40 | yes (its profile page: the walk's, the reader's and the writer's words) | yes (the directory page's end: the reader's and the writer's counts) |
| 3 | The job table: a 4,096 B page each command, its word 0 the job count, then 8 words a job (first Q, K, V, O pages, rows, the first row, key tiles, the row step); the reader pushes it to compute, the writer reads its own copy, and each CPU walks the jobs, its streams and broadcasts inside that loop | a page's words as a table of entries, `for job in tables.pages(self.rank, 1).entries(first=8, size=8, count=0)`, each `job[i]` an int of every CPU (mover A's read pushed to compute, mover B's own), the streams and broadcasts inside the loop kept in step on the three kernels | 150 to 250 | no (two words of its directory: `.read()[0:2]`) | yes: the directory's item ids (up to 997 a page) |
| 4 | Q's rows of a job: `rows` rows of DT tiles, `stride` tile rows apart (a causal call's rows L apart), into one block of R x DT pages reserved whole (a short job fills part of it) | a rectangle with a row step, `q.tiles[r0:r1:s, 0:DT]` (a `km::Walk` with a row pitch of s x T), and a stream of one block a job whose block may be partly read | 40 | no | no |
| 5 | A broadcast's block from two tensors: K's tiles head-tile-major `[DT][KB]` (page `(k0 + key) x DT + h` to slot `h x KB + key`), then V's `[KB][Dv]`, read by the sender into the slot it reserved, then cast | a cast of several page ranges into one block in order, and a transposed rectangle, `b.cast(k.tiles[k0:k1, 0:DT].T, v.tiles[k0:k1, 0:DV])` | 60 | no | no |
| 6 | Broadcasts in bands of L rows down each grid column, on the reader (NoC 0), the sender each band's top; L from the plan (2, 5 or 10) | a broadcast's length and mover, `km.Broadcast(self, along="column", length=L, mover="A")`; today a column's broadcast is the rectangle's whole column on mover B (NoC 1), the hand's `--noc 1` form, and the same as the hand's only at L 10 | 40 | no | no |
| 7 | A broadcast's blocks counted over the whole command (passes x jobs x the job's blocks), the receivers' padding to the window after the last (`km::broadcast_end` with the count) | the broadcast's blocks counted over every loop around its cast; today block by block of the one loop around it | 30 | no | no |
| 8 | Compute takes a block from the broadcast and its bodies index K's and V's pages inside it (the score's `head_tile x KB + chunk_start`, P V's `first value page + key x Dv + column`), then hands it back (`km::broadcast_done`) | a body that takes a broadcast's received block whole: its buffer and the block's first page, freed after the body; today `for tile in v:` gives each page | 30 | no | no |
| 9 | Compute's start and end, outside the command loop: `compute_kernel_hw_startup<SrcOrder::Reverse>(Q, KV, P)`, `flash::init(scale)`, `macro_init`, `addr_init`, the ones and causal tiles built and packed once, the row state and O reserved once; `restore_constants` after the loop | bodies at the kernel's start and end, `km.cxx(..., at="start")` and `at="end"`, and the start-up's operand buffers and source order on the class | 40 | no (its start-up is the class's) | no |
| 10 | Defines the unchanged bodies test: `FLASH_CAUSAL`, `FLASH_APPROX`; `BROADCAST_ON_WRITER` moves the broadcast to the writer | the class's defines, `class A(km.Program, defines={...})`, or made from the plan | 10 | no | no |
| 11 | The cut-down modes: compute only pushes the broadcast buffer's blocks unread and pads to the window; broadcast only has compute pop the blocks unused; no row written | a broadcast's block pushed without a read or a cast (the `if mode` around it is a line today) | 20 | no | no |
| 12 | Compute packs each finished row's Dv tiles into the output buffer (`pack_tile`, a timed reserve of the row, then push), the writer writes them at O's page `first + row x stride x Dv` | a write stream that a pack body fills by `pack_tile` (its buffer given, the reserve and push the line's), and a write stream of a rectangle with a row step (row 4's) | 25 | no (its hits are written by address) | no (its pages are written by address) |

Rows 1 and 3 to 12 are the attention pass's skeleton: with them every move of the
hand program is a line and the bodies are the hand's own, the profile page (row 2)
aside, as in the ray tracer's class. Row 1 alone lets the pass
build with every move still an inline body. That proves only that the bodies run
unchanged; it says nothing of the moves, so it was not built here. Rows 1 and 2 are
needed by all three places, row 3 by two; the rest are attention's own.

## The proof that was not run

With the lines of rows 1 and 3 to 12, the proof would be the README's `t1` (1 x 1 x 128
x 64, L 10, its RMS 5.50e-3 recorded) or `causal64` (58.1 us a call recorded), the
class's O beside the hand program's, word for word. The recorded runs keep only the
sampled rows' errors, not O, so the hand program would run once beside the class in
the same job.
