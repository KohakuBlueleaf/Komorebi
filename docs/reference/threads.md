---
title: X280 threads reference
summary: Every kmx call with its return codes and the hart it runs on, the thread's memory layout and result block, the command ring's line, and the host side (Program.thread, ThreadRun, ThreadResult, regions).
tags:
  - reference
  - x280
  - threads
---

# X280 threads reference

Status: implemented on one Blackhole p150a, work clusters 1, 2 and 3, each with its
command ring to the dispatch kernel (its sources 2, 3 and 4). The library's ring calls
are also checked on the CPU against a fake kernel.

The guide is [X280 threads](../threads.md): what a thread is for, how regions change
hands, the rules. This page lists the interface. The sources are
`src/komorebi_native/x280/kmx/` (`kmx.h`, `kmx_layout.h`, `kmx_ring.h`, `kmx.c`) and
`src/komorebi/thread.py`.

## C: `#include "kmx.h"`

Plain C, no libc. The library's start-up is the image's `main`; the thread supplies
`kmx_main`.

| Call | Hart | Does | Returns |
|---|---|---|---|
| `int kmx_main(int hart)` | 0 | the thread's entry, written by the program | its code, unless `kmx_result` gave one |
| `const kmx_args* kmx_args_get(void)` | any | the argument block: `nwords`, `words[]`, `nregions`, `region[]`, `deadline`, `flags` | never 0 |
| `void* kmx_region(int id, uint64_t* bytes)` | any | region `id` as memory, `*bytes` its size | 0 for an unknown id |
| `void kmx_take(int id)` | any | drops the cluster's cached lines of the region: before reading what the grid or the host wrote | - |
| `void kmx_give(int id)` | any | writes the region's cached lines back to DRAM: after writing what the grid or the host will read | - |
| `void kmx_take_at(int id, uint64_t offset, uint64_t bytes)`, `kmx_give_at` | any | the same for the lines holding `bytes` of the region from `offset` | - |
| `int kmx_spawn(int hart, void (*fn)(int, void*), void* arg)` | 0 | starts `fn(hart, arg)` on hart 1, 2 or 3 and returns | `KMX_OK`; `KMX_FULL` while that hart runs its last spawn; `KMX_BAD` for another hart |
| `int kmx_join(int hart)` | 0 | waits until `hart` returned from its last spawn (at once if none) | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`); `KMX_BAD` |
| `int kmx_par(void (*fn)(int, void*), void* arg)` | 0, not nested | spawns `fn` on harts 1 to 3, runs `fn(0, arg)`, joins them | `KMX_OK`; `KMX_FULL` if a spawned hart is busy (nothing started); `KMX_LATE` or `KMX_STOPPED`; `KMX_STOPPED` with error `par` from another hart or inside a `kmx_par` |
| `uint32_t kmx_clusters(void)` | any | the clusters running this image in the program, bit `c` for cluster `c` | a mask |
| `int kmx_lead(void)` | any | the lowest of them: the one that calls `kmx_par_all` | 1, 2 or 3 |
| `int kmx_par_all(void (*fn)(int, int, uint64_t), uint64_t arg)` | the lead's 0, not nested | runs `fn(cluster, hart, arg)` on every hart of `kmx_clusters()`; the first call waits until every other cluster serves | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED`; error `par` elsewhere |
| `int kmx_serve(void)` | 0 | runs the lead's `kmx_par_all` calls until the lead's `kmx_main` returns | `KMX_OK`; `KMX_NONE` on the lead; `KMX_LATE` / `KMX_STOPPED` |
| `int kmx_push(int cluster, int id, uint64_t offset, const void* src, uint64_t bytes)` | any | stores `bytes` from `src` at `offset` of region `id` of `cluster`'s thread; returns once the last line's store is acknowledged; on hart 0 outside `kmx_par`, over the four harts | `KMX_OK`; `KMX_BAD` (a cluster not in `kmx_clusters()`, an unknown region, out of range, not 64-byte multiples, a window, or before the lead's first call on another cluster); the lead's first push to another cluster waits as `kmx_par_all` does |
| `int kmx_issue(const km_record_t* r, uint32_t* ticket)` | 0 | stores record `r` as the next command of the ring | `KMX_OK` and the ticket (1, 2, ...); `KMX_FULL` when no slot is free; `KMX_STOPPED` without a ring |
| `int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev)` | 0 | waits for the ticket's event until `kmx_time()` passes `deadline`; other tickets' events are kept for `kmx_poll`, wake events are passed over | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`); `KMX_FULL` when 32 events are kept |
| `int kmx_poll(kmx_event* ev)` | 0 | one event, a kept one first; never waits | `KMX_OK` or `KMX_NONE` |
| `int kmx_cluster(void)` | any | the work cluster the thread runs on | 1, 2 or 3 |
| `uint64_t kmx_time(void)` | any | the 50 MHz timer (`KMX_TICKS_PER_SECOND`) | ticks |
| `int kmx_stop_requested(void)` | any | true once the host asked for a stop or the thread's deadline passed | 0 or 1 |
| `void kmx_progress(uint32_t where)` | any | stamps `where` and its time into the result block | - |
| `void kmx_result(uint32_t code, const void* words, uint32_t n)` | 0 | the thread's code and up to 48 words | error `result` if `n > 48` (the first 48 kept) |

Ring calls, `kmx_spawn`, `kmx_join`, `kmx_par`, `kmx_par_all` and `kmx_serve` from
harts 1 to 3 record the library error `par` and return `KMX_STOPPED`. A record for `kmx_issue` is a `km_record_t` (`komorebi/abi/km_abi.h`):
`hdr = seg | flags << 8`, `team = first | count << 16`, arguments `a0` to `a9`; the
library writes `seq` and the landing words. Flags: `KM_F_EXCL` (issued only when no
team in flight overlaps it), `KM_F_RESULT` (the team leader's `wr.result(value)`
comes back in the event's `ticks`), `KM_F_QUEUE` (the kernel chooses `first`). A
record that cannot be issued now is refused at once (`KM_EV_REFUSED`, aux 11) and the
kernel then sends one `KMX_EV_WAKE` (ticket 0) when a slot frees or the queue issues a
job: the record may be issued again after it. The kernel's other refusals are its README's ("Refused commands"), and STOP, EXIT, RUN and
CANCEL from a thread are refused with aux 1.

`kmx_event`: `ticket`, `kind` (`KM_EV_DONE`, `KM_EV_TIMEOUT`, `KM_EV_REFUSED`, `KMX_EV_WAKE` from `kmx_poll` only),
`aux` (the first core of a done record or the slot of a timeout in its low 8 bits, the
low 16 bits of the record's sequence number above them; the reason of a refusal), `ticks` (issue to completion
in timer ticks, or the result word of a `KM_F_RESULT` record).

Return codes (`kmx_layout.h`): `KMX_OK` 0, `KMX_FULL` 1, `KMX_STOPPED` 2,
`KMX_LATE` 3, `KMX_NONE` 4, `KMX_BAD` 5.

## The image

| Rule | Value |
|---|---|
| size | at most 126,976 bytes, globals included |
| globals | a non-zero initialiser and `__attribute__((section(".data")))`: the image has no `.bss` |
| stack | 32 KiB per hart |
| libraries | none: a struct assignment or a large zero initialiser may become a `memset` call that does not link |
| start-up | floating point and vector on; the L2 prefetchers set to the documented values unless `prefetch="keep"`; argument block, regions and bank checked; then `kmx_main` on hart 0, spawns served on harts 1 to 3; after it, every region given back |
| build | `Thread.sources` with `Thread.includes` and `Thread.defines`, compiled with the dispatch kernel's toolchain into the build cache |

## Memory

The thread's block in the cluster's local memory (LIM), bytes from its base:

| Offset | Bytes | What | Written by |
|---|---|---|---|
| `0x000` | `0x240` | `kmx_args`: run value, magic `kmx1`, version 3 (`KMX_VERSION`), deadline (ticks), region and word counts, check line address and value, flags (`KMX_FLAG_KEEP_PREFETCH` 1), the cluster (`cluster`, at `0x38`); region table at `0x040` (16 x `{offset, bytes}`); words at `0x140` (64 x `uint32`) | host, before the start |
| `0x240` | 64 | stop line: word 0 = `0x53544F50` asks for a stop | host |
| `0x280` | 256 | result block | thread |
| `0x380` | 4 x 64 | spawn lines, hart `h`'s at `+ 64 h`: 64-bit words `gen`, `fn`, `arg` (hart 0), `done` (hart `h`); hart 0's word 0 ends the service | harts |
| `0x480` | 64 | `kmx_ring_desc`, zero for no ring | host, before the start |
| `0x4C0` | 64 | `kmx_peers`: a value for this load (`nonce`), the lead, the mask of clusters running the image, the offset `all` of the `kmx_par_all` lines, then each cluster's tile and its bank's DRAM tile (`x \| y << 16`, indexed by cluster); zero: no peers | host, before the start |
| `all` | 4 x 64 | after the block the host writes (and the ring's events): the call line the lead writes into each other cluster (32-bit words `nonce`, `gen`, `fn`, `arg`, `exit`), and in the lead's, cluster `c`'s done line at `+ 64 c` (`nonce`, the last `gen` done; 0: it serves) | the lead; the others |

Result block: `state` (1 boot, 2 running, 3 done, 4 failed), `error` (the first
library error: 0 none, 1 args, 2 bank, 3 late, 4 stopped, 5 par, 6 result), `code`,
`n`, `where`, `t_where` (ticks from the start to the last stamp), `ticks` (start to
the end of `kmx_main`), then 48 result words at byte 64.

Windows on a work cluster: the ring's 16 (`KMX_RING_WINDOW`), cluster `c`'s LIM
through `16 + c` (`KMX_PEER_WINDOW + c - 1`) while the thread runs, hart `h`'s push
through `20 + h` (`KMX_PUSH_WINDOW + h`) during a push.

Regions are whole 4 KiB pages of the work cluster's DRAM bank (on the p150a bank 6 for
cluster 1, bank 7 for clusters 2 and 3); page 0 of a bank's pages is the check line,
then each thread's own regions, then the shared ones. The thread sees region `id` at
the cached alias of the bank plus its offset; a Tensix kernel reaches it with
`get_noc_addr_from_bank_id<true>(bank, address)`. A thread reads only its own bank:
data for a thread on the other bank is pushed there by its producer.

`kmx_ring_desc` (one 64-byte line): `cmd` (`cmd_depth` 64-byte slots, 64-byte
aligned), `consumed` (the kernel's count of commands taken), `events` (`event_depth`
16-byte events), `read` (a 64-byte line, word 0 the events the thread took),
`cmd_depth`, `event_depth` (powers of two, at most 1,024), `flags`
(`KMX_RING_CACHED` 1: each hand-over is a sweep; `KMX_RING_REMOTE` 2: `cmd` and `read`
are NoC addresses in the tile `reserved[0]` names, `x | y << 16`, inside one 2 MiB
window that the start-up opens in slot `KMX_RING_WINDOW` and closes after `kmx_main`),
`reserved[0]`. Other addresses are as the thread's cluster sees them. A line that
names a ring but is out of range ends the start-up with error `args`. The session's
ring: 16 command lines and the read line in the dispatch kernel's LIM after its layout
(`KMD_LIM_BYTES`, cluster `c`'s ring `c - 1` rings further), 64 events after the kmx
block in the thread's LIM
(`KMD_WORK_*` from `KMX_LIM_BYTES`). The protocol is the host ring's (the dispatch kernel's README,
"Protocol with the host").

## Python

`Program.thread(cluster=1, clusters=None, **fields)` declares the thread (one per
cluster) and returns a `komorebi.thread.Thread`; with `clusters`, a list of one per
cluster, the same fields on each:

| Field | Meaning |
|---|---|
| `sources` | C files, linked with the kmx library |
| `regions` | `{name: size in bytes or an array of initial contents}`; ids in dict order |
| `shared` | names of the program's shared regions the thread uses; ids after its own; with `regions`, 1 to 16 in all |
| `words` | up to 64 unsigned 32-bit words |
| `cluster` | the work cluster, 1 to 3 (bank 6 for 1, bank 7 for 2 and 3) |
| `host_fallback` | `fn(regions, words, run) -> (code, words)`: the thread on the host, on `uint32` copies of the regions, issuing records through `run` (the session's `Run`) where the thread uses its ring |
| `deadline` | seconds for the whole thread, at most 60 (default 10) |
| `includes`, `defines` | extra include directories and `-D` definitions |
| `prefetch` | `"documented"` (default) or `"keep"` |

`Program.region(name, value)` declares a shared region (a size in bytes or an array of
initial contents). Every thread naming it must be on one bank, and every shared region
must be named by a thread; `session.load` refuses the program otherwise.

`session.load(program, thread="card" | "host")` writes the regions before the grid
starts, then starts every image (or readies the fallbacks) once the dispatch kernel is
ready. `run.threads` holds a `ThreadRun` per cluster; `run.thread` is the lowest
cluster's:

| Member | Meaning |
|---|---|
| `run.thread_result(timeout)` | waits for the thread to return (asks it to stop at the timeout), stops it, returns a `ThreadResult`; with `thread="host"` runs the fallback; raises `ThreadFailed` when the thread is `stuck` |
| `run.thread_results(timeout)` | the same for every thread, `{cluster: ThreadResult}`; on the first that does not return in time or ends with a library error, stops the others and raises `ThreadFailed` (`cluster`, `results`; its message names a `stuck` thread) |
| `run.thread.read(name)` | the region's bytes (`uint8`), only after `run.stop()` |
| `run.thread.regions.address(name)` | `(bank, bank address)` for a Tensix kernel |
| `run.thread.ring` | the `RingPlacement` the host wrote into the ring line (`None` with `thread="host"`) |
| `run.stop()` | stops every thread still running, then the run |

`ThreadResult`: `ran` (false when it was stopped at the timeout), `state`, `error`
(names, as in the result block), `code`, `words`, `where`, `where_us`, `us`, `stuck`
(it did not return within 1 s of the stop word either: a hart outside the library or
halted by a trap) and `detail` (then the runtime's account of a stop that failed).

`RingPlacement(cmd, consumed, events, read, cmd_depth, event_depth, cached=False,
remote=None)`, `work_ring(kernel_lim, kernel_xy, work_lim, work_xy, cluster=1)` (the
session's placement of one cluster's ring and the kernel's end of it),
`kernel_work_args(ends)` (the kernel's `work_*` words for every ring) and
`args_block(thread, base, check_value, run_value=0, ring=None, table=None,
peers=None)` build the LIM block; `Peers(nonce, lead, tiles, banks)` is the peers line
(tiles and banks: `{cluster: x | y << 16}`) and `peer_groups({cluster: image})` the
clusters that run one image. They need no device; the session fills `Peers` from the
clusters' tiles and their banks' DRAM tiles for every thread on the card.

## Measured

On work cluster 1 of a p150a at 1,750 MHz: the bank read at about 1.3 GB/s and written
at 1.3 to 1.8 GB/s with four harts; a hand-over sweep at 5 to 8 GB/s of region
(`experiments/x280_threads/`). At the session's 1,000 MHz on each of clusters 1 to 3:
about 1.2 GB/s read and written per cluster, alone or together (three at once 3.5 GB/s);
a push into the other bank 2.3 GB/s, a pull 0.024 GB/s; take 4.6 GB/s, give 4.3 GB/s
(`experiments/work_clusters/`).
An empty `kmx_par` call: 0.71 µs; `kmx_par_all` over three clusters, the map, heavy
map and prefix sum on 1, 4, 8 and 12 harts: [X280 threads](../threads.md#twelve-harts)
and `examples/08_parallel_thread/` (`--twelve`).
