# Attention, as a class: what a class cannot say

Question: can the attention project's program
([projects/attention](../../../projects/attention/README.md)) be written as a class
([Programs written as classes](../../../docs/reference/classes.md)), as the ray tracer's
trace pass was ([raytrace_class](../raytrace_class/README.md)): every arithmetic body
the hand program's, inline and unchanged (`ktt_flash`'s compute with `sfpu.hpp`),
every move a class line? And where it cannot, which line is missing?

It cannot be written yet. Rows 1 to 3 below are now built; the rows from 4 on are
not. Before row 1 the compute body's own buffers could not be made: a body that takes a field which no line fills is refused at
its line ("self.state is not filled above in this handler ..., and no other handler
keeps it filled"), and an inline body cannot declare a buffer. Without them the
compute body would have to change, so the pass stops at this table: no class file and
no card run. Each row is one thing the hand program does and no line says, with the
general line that would say it, its size in the compiler (an estimate, in lines of
`src/komorebi/dsl/`), and where the ray tracer's, the rasterizer's and the decode
project's hand programs need the same (paths under each project). The rows are ordered
by how many of the four places need them.

## What a class can say of it today

- The two handlers and their words: `attend(q, k, v, o, tables, mode, passes,
  first_job, job_count)` and `prof(profile, zero)`, the command words a0 to a8 and a0
  to a1 in the hand's order.
- The plan's sizes as module constants (head tiles, value tiles, rows R, block keys
  KB, the window) and the exponential's constants (scale bits, threshold bits).
- The passes, `for _ in range(max(passes, 1))`, and the jobs' end (all jobs when
  job_count is 0, else `min(first_job + job_count, jobs)`), once the job count is an
  int of the handler (row 5).
- The job table's page as a field of words filled each command,
  `self.table.fill(km.Stream(tables.pages(self.rank, 1)))`. Only bodies could read
  it (row 5).
- The compute options HiFi2 and fp32 DST (`compute={...}`); the float32 buffers'
  unpack to DST is row 2.
- The submit with `KM_FLAG_EXCLUSIVE` (`Run.submit(..., exclusive=True)`).

## The table

| # | What the hand program does | The general line that would say it | Size (estimate) | Ray tracer | Rasterizer | Decode |
|---|---|---|---|---|---|---|
| 1 | Compute's own buffers, of three kinds: a FIFO between its CPUs (P, 3 bf16 pages from pack to unpack; two token buffers from pack to unpack and math), slots written in place (the row state, 2 R float32 tiles at `pack_tile<true>` slots; O and the row sums, R x (Dv + 1) float32 tiles with L1 accumulation), constants built once (the ones tile, the causal mask); none moved by a mover | BUILT: `p: km.L1[km.bf16, 3, 32, 32] = km.own()`, a buffer of its format and pages given to bodies by name (`takes=[self.p]` binds `p`, its index), never filled, waited for or freed by a line; one pushing CPU, `km.own(push="pack")` or `push=None` for a buffer written in place ([own_buffer](../own_buffer/README.md)) | 50 to 70 | yes: `kernels/compute.cpp`'s work pages, signal, packet states, lists and slots (the treelet path) | yes: `kernels/setup_compute.cpp`'s work space, `kernels/writer.cpp`'s empty tile, the histograms and list slots | yes: `kernels/attend/compute.cpp`'s scores, reference, P, rescale and state buffers |
| 2 | Unpack to DST as float32 keyed by those buffers (O and the row state, `unpack_to_dest_mode`); the class refuses the option and sets it from its float32 streams alone | BUILT with row 1: a float32 own field is unpacked to DST as float32, as a float32 stream is | 10 (with row 1) | yes: `card.py`, the packet states | no | yes: `card.py`, the arrivals, the sums and the state buffers |
| 3 | The profile page of five CPUs' words: the reader's (Q waits and reads, the sender's slot waits, K / V reads, casts, arrivals, finish), unpack's, math's and pack's (each at its own submit), the writer's; stored by `prof` and zeroed | BUILT: pack's body fills the page, `km.cxx(..., takes=[report, rays, self.handoff], on="mover_b")` adds the movers' words before the movers write it; mover A's words by an own field it pushes (`km.own(push="mover_a")`); the loop's name on a mover after the loop gives `rays_tiles`, `rays_cycles` ([raytrace_class](../raytrace_class/README.md)). Attention's page needs unpack's and math's words too: a word of one compute CPU into pack's page is still not a line | more than 40 | yes: the trace pass's profile page (the walk's, the reader's and the writer's words) | yes: `kernels/reader.cpp`'s and `kernels/writer.cpp`'s counts at the directory page's end | no |
| 4 | Defines the unchanged bodies test: `FLASH_CAUSAL`, `FLASH_APPROX`; `BROADCAST_ON_WRITER` moves the broadcast to the writer | the class's defines, `class A(km.Program, defines={...})`, or made from the plan | 10 | no | no | yes: `card.py`'s `DC_KEYS_MOST`, `DC_EPSILON` and the emitted bodies' defines |
| 5 | The job table: a 4,096 B page each command, its word 0 the job count, then 8 words a job (first Q, K, V, O pages, rows, the first row, key tiles, the row step); the reader pushes it to compute, the writer reads its own copy, and each CPU walks the jobs, its streams and broadcasts inside that loop | a page's words as a table of entries, `for job in tables.pages(self.rank, 1).entries(first=8, size=8, count=0)`, each `job[i]` an int of every CPU (mover A's read pushed to compute, mover B's own), the streams and broadcasts inside the loop kept in step on the three kernels | 150 to 250 | no (two words of its directory: `.read()[0:2]`) | yes: `kernels/reader.cpp`'s directory, its item ids (up to 997 a page) | no (its words are the command's) |
| 6 | Compute packs each finished row's Dv tiles into the output buffer (`pack_tile`, a timed reserve of the row, then push), the writer writes them at O's page `first + row x stride x Dv` | a write stream that a pack body fills by `pack_tile` (its buffer given, the reserve and push the line's), and a write stream of a rectangle with a row step (row 7's) | 25 | no (its hits are written by address) | no (its pages are written by address) | yes: `kernels/attend/compute.cpp` packs a head's tiles into the split's output buffer |
| 7 | Q's rows of a job: `rows` rows of DT tiles, `stride` tile rows apart (a causal call's rows L apart), into one block of R x DT pages reserved whole (a short job fills part of it) | a rectangle with a row step, `q.tiles[r0:r1:s, 0:DT]` (a `km::Walk` with a row pitch of s x T), and a stream of one block a job whose block may be partly read | 40 | no | no | no |
| 8 | A broadcast's block from two tensors: K's tiles head-tile-major `[DT][KB]` (page `(k0 + key) x DT + h` to slot `h x KB + key`), then V's `[KB][Dv]`, read by the sender into the slot it reserved, then cast | a cast of several page ranges into one block in order, and a transposed rectangle, `b.cast(k.tiles[k0:k1, 0:DT].T, v.tiles[k0:k1, 0:DV])` | 60 | no | no | no |
| 9 | Broadcasts in bands of L rows down each grid column, on a chosen NoC: the reader's NoC 0 by default, the writer's NoC 1 with `BROADCAST_ON_WRITER`; the sender each band's top; L from the plan (2, 5 or 10) | a broadcast's length and mover, `km.Broadcast(self, along="column", length=L, mover="A")`; today a column's broadcast is the rectangle's whole column on mover B (NoC 1), the hand's `--noc 1` form, and the same as the hand's only at L 10 | 40 | no | no | no (its heads meet by ports) |
| 10 | A broadcast's blocks counted over the whole command (passes x jobs x the job's blocks), the receivers' padding to the window after the last (`km::broadcast_end` with the count) | the broadcast's blocks counted over every loop around its cast; today block by block of the one loop around it | 30 | no | no | no |
| 11 | Compute takes a block from the broadcast and its bodies index K's and V's pages inside it (the score's `head_tile x KB + chunk_start`, P V's `first value page + key x Dv + column`), then hands it back (`km::broadcast_done`) | a body that takes a broadcast's received block whole: its buffer and the block's first page, freed after the body; today `for tile in v:` gives each page | 30 | no | no | no |
| 12 | Compute's start and end, outside the command loop: `compute_kernel_hw_startup<SrcOrder::Reverse>(Q, KV, P)`, `flash::init(scale)`, `macro_init`, `addr_init`, the ones and causal tiles built and packed once, the row state and O reserved once; `restore_constants` after the loop | bodies at the kernel's start and end, `km.cxx(..., at="start")` and `at="end"`, and the start-up's operand buffers and source order on the class | 40 | no (its start-up is the class's) | no | no |
| 13 | The cut-down modes: compute only pushes the broadcast buffer's blocks unread and pads to the window; broadcast only has compute pop the blocks unused; no row written | a broadcast's block pushed without a read or a cast (the `if mode` around it is a line today) | 20 | no | no | no |

Rows 1, 2 and 4 to 13 are the attention pass's skeleton: with them every move of the
hand program is a line and the bodies are the hand's own, the profile page (row 3)
aside, as in the ray tracer's class. Row 1 alone lets the pass build with every move
still an inline body. That proves only that the bodies run unchanged; it says nothing
of the moves, so it was not built here. Row 1 is needed by all four places; rows 2
and 3 by three; rows 4 to 6 by two; the rest are attention's own.

## The proof that was not run

With the lines of rows 1, 2 and 4 to 13, the proof would be the project README's `t1`
(1 x 1 x 128 x 64, L 10, its RMS 5.50e-3 recorded) or `causal64` (58.1 us a call
recorded), the class's O beside the hand program's, word for word. The recorded runs
keep only the sampled rows' errors, not O, so the hand program would run once beside
the class in the same job.
