---
title: X280 threads reference
summary: Every kmx call with its return codes and the hart it runs on, the thread's memory layout and result block, the command ring's line, and the host side (Program.thread, ThreadRun, ThreadResult, regions).
tags:
  - reference
  - x280
  - threads
---

# X280 threads reference

Status: implemented on one Blackhole p150a, work cluster 1, with the command ring to
the dispatch kernel (its source 2). The library's ring calls are also checked on the
CPU against a fake kernel.

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
| `int kmx_par(void (*fn)(int, void*), void* arg)` | 0, not nested | runs `fn(h, arg)` on harts 0 to 3 and waits for all | `KMX_OK`; `KMX_LATE` or `KMX_STOPPED` (library error `late` / `stopped`); `KMX_STOPPED` with error `par` from another hart or inside a `kmx_par` |
| `int kmx_issue(const km_record_t* r, uint32_t* ticket)` | 0 | stores record `r` as the next command of the ring | `KMX_OK` and the ticket (1, 2, ...); `KMX_FULL` when no slot is free; `KMX_STOPPED` without a ring |
| `int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev)` | 0 | waits for the ticket's event until `kmx_time()` passes `deadline`; other tickets' events are kept for `kmx_poll`, wake events are passed over | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`); `KMX_FULL` when 32 events are kept |
| `int kmx_poll(kmx_event* ev)` | 0 | one event, a kept one first; never waits | `KMX_OK` or `KMX_NONE` |
| `uint64_t kmx_time(void)` | any | the 50 MHz timer (`KMX_TICKS_PER_SECOND`) | ticks |
| `int kmx_stop_requested(void)` | any | true once the host asked for a stop or the thread's deadline passed | 0 or 1 |
| `void kmx_progress(uint32_t where)` | any | stamps `where` and its time into the result block | - |
| `void kmx_result(uint32_t code, const void* words, uint32_t n)` | 0 | the thread's code and up to 48 words | error `result` if `n > 48` (the first 48 kept) |

Ring calls from harts 1 to 3 record the library error `par` and return
`KMX_STOPPED`. A record for `kmx_issue` is a `km_record_t` (`komorebi/abi/km_abi.h`):
`hdr = seg | flags << 8`, `team = first | count << 16`, arguments `a0` to `a9`; the
library writes `seq` and the landing words. Flags: `KM_F_EXCL` (issued only when no
team in flight overlaps it), `KM_F_RESULT` (the team leader's `wr.result(value)`
comes back in the event's `ticks`), `KM_F_QUEUE` (the kernel chooses `first`). A
record that cannot be issued now is refused at once (`KM_EV_REFUSED`, aux 11) and the
kernel then sends one `KMX_EV_WAKE` (ticket 0) when a slot frees or the queue issues a
job: the record may be issued again after it. The kernel's other refusals are its README's ("Refused commands"), and STOP, EXIT, RUN and
CANCEL from a thread are refused with aux 1.

`kmx_event`: `ticket`, `kind` (`KM_EV_DONE`, `KM_EV_TIMEOUT`, `KM_EV_REFUSED`, `KMX_EV_WAKE` from `kmx_poll` only),
`aux` (the slot of a timeout, the reason of a refusal), `ticks` (issue to completion
in timer ticks, or the result word of a `KM_F_RESULT` record).

Return codes (`kmx_layout.h`): `KMX_OK` 0, `KMX_FULL` 1, `KMX_STOPPED` 2,
`KMX_LATE` 3, `KMX_NONE` 4.

## The image

| Rule | Value |
|---|---|
| size | at most 126,976 bytes, globals included |
| globals | a non-zero initialiser and `__attribute__((section(".data")))`: the image has no `.bss` |
| stack | 32 KiB per hart |
| libraries | none: a struct assignment or a large zero initialiser may become a `memset` call that does not link |
| start-up | floating point and vector on; the L2 prefetchers set to the documented values unless `prefetch="keep"`; argument block, regions and bank checked; then `kmx_main` on hart 0, `kmx_par` service on harts 1 to 3 |
| build | `Thread.sources` with `Thread.includes` and `Thread.defines`, compiled with the dispatch kernel's toolchain into the build cache |

## Memory

The thread's block in the cluster's local memory (LIM), bytes from its base:

| Offset | Bytes | What | Written by |
|---|---|---|---|
| `0x000` | `0x240` | `kmx_args`: run value, magic `kmx1`, version 1, deadline (ticks), region and word counts, check line address and value, flags (`KMX_FLAG_KEEP_PREFETCH` 1); region table at `0x040` (16 x `{offset, bytes}`); words at `0x140` (64 x `uint32`) | host, before the start |
| `0x240` | 64 | stop line: word 0 = `0x53544F50` asks for a stop | host |
| `0x280` | 256 | result block | thread |
| `0x380` | 64 | `kmx_par` mailbox | hart 0 |
| `0x3C0` | 32 | `kmx_par` done words, one per hart | harts |
| `0x400` | 64 | `kmx_ring_desc`, zero for no ring | host, before the start |

Result block: `state` (1 boot, 2 running, 3 done, 4 failed), `error` (the first
library error: 0 none, 1 args, 2 bank, 3 late, 4 stopped, 5 par, 6 result), `code`,
`n`, `where`, `t_where` (ticks from the start to the last stamp), `ticks` (start to
the end of `kmx_main`), then 48 result words at byte 64.

Regions are whole 4 KiB pages of the work cluster's DRAM bank (bank 6 for cluster 1
on the p150a); page 0 of the thread's pages is the check line. The thread sees region
`id` at the cached alias of the bank plus its offset; a Tensix kernel reaches it with
`get_noc_addr_from_bank_id<true>(bank, address)`.

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
(`KMD_LIM_BYTES`), 64 events after the kmx block in the thread's LIM
(`KMD_WORK_*` from `KMX_LIM_BYTES`). The protocol is the host ring's (the dispatch kernel's README,
"Protocol with the host").

## Python

`Program.thread(cluster=1, **fields)` declares the thread (one per program) and
returns a `komorebi.thread.Thread`:

| Field | Meaning |
|---|---|
| `sources` | C files, linked with the kmx library |
| `regions` | `{name: size in bytes or an array of initial contents}`, 1 to 16; ids in dict order |
| `words` | up to 64 unsigned 32-bit words |
| `cluster` | the work cluster, 1 to 3; only cluster 1 (bank 6) was run on a card |
| `host_fallback` | `fn(regions, words, run) -> (code, words)`: the thread on the host, on `uint32` copies of the regions, issuing records through `run` (the session's `Run`) where the thread uses its ring |
| `deadline` | seconds for the whole thread, at most 60 (default 10) |
| `includes`, `defines` | extra include directories and `-D` definitions |
| `prefetch` | `"documented"` (default) or `"keep"` |

`session.load(program, thread="card" | "host")` writes the regions before the grid
starts, then starts the image (or readies the fallback) once the dispatch kernel is
ready. `run.thread` is a `ThreadRun`:

| Member | Meaning |
|---|---|
| `run.thread_result(timeout)` | waits for the thread to return (asks it to stop at the timeout), stops it, returns a `ThreadResult`; with `thread="host"` runs the fallback |
| `run.thread.read(name)` | the region's bytes (`uint8`), only after `run.stop()` |
| `run.thread.regions.address(name)` | `(bank, bank address)` for a Tensix kernel |
| `run.thread.ring` | the `RingPlacement` the host wrote into the ring line (`None` with `thread="host"`) |
| `run.stop()` | stops a thread still running, then the run |

`ThreadResult`: `ran` (false when it was stopped at the timeout), `state`, `error`
(names, as in the result block), `code`, `words`, `where`, `where_us`, `us`.

`RingPlacement(cmd, consumed, events, read, cmd_depth, event_depth, cached=False,
remote=None)`, `work_ring(kernel_lim, kernel_xy, work_lim, work_xy)` (the session's
placement and the kernel's `work_*` words) and `args_block(thread, base, check_value,
run_value=0, ring=None)` build the LIM block; they need no device.

## Measured

On work cluster 1 of a p150a: the bank read at about 1.3 GB/s and written at 1.3 to
1.8 GB/s with four harts; a hand-over sweep at 5 to 8 GB/s of region.
`experiments/x280_threads/` has the commands and numbers.
