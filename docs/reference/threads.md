---
title: X280 threads reference
summary: Every kmx call with its return codes and the core it runs on, the thread's memory layout and result block, the command ring's line, and the host side (Program.thread, ThreadRun, ThreadResult, regions).
tags:
  - reference
  - x280
  - threads
---

# X280 threads reference

Status: implemented on one Blackhole p150a, work clusters 1, 2 and 3, each with its
command ring to the dispatcher (its sources 2, 3 and 4). The library's ring calls
are also checked on the CPU against a fake kernel.

The guide is [X280 threads](../threads.md): what a thread is for, how regions change
hands, the rules. This page lists the interface. The sources are
`src/komorebi_native/x280/kmx/` (`kmx.h`, `kmx_layout.h`, `kmx_ring.h`, `kmx.c`) and
`src/komorebi/program/thread.py`.

## C: `#include "kmx.h"`

Plain C, no libc. The library's start-up is the image's `main`; the thread supplies
`kmx_main`, `kmx_program` or both ([commands to a thread](#commands-to-a-thread)).

| Call | Core | Does | Returns |
|---|---|---|---|
| `int kmx_main(int core)` | 0 | the thread's entry, written by the program; optional with `kmx_program` | its code, unless `kmx_result` gave one |
| `uint32_t kmx_program(const km_command_t* command)` | 0 | after `kmx_main` (if any), called for each command to this cluster until the run stops; written by the program | the command's result word |
| `uint32_t kmx_handler(const km_command_t* command)` | any | the command's handler, for `kmx_program`'s switch | 0 to 255 |
| `const kmx_args* kmx_args_get(void)` | any | the argument block: `nwords`, `words[]`, `nregions`, `region[]`, `deadline`, `flags` | never 0 |
| `void* kmx_region(int id, uint64_t* bytes)` | any | region `id` as memory, `*bytes` its size | 0 for an unknown id |
| `void kmx_invalidate(int id)` | any | drops the cluster's cached lines of the region: before reading what the grid or the host wrote | - |
| `void kmx_flush(int id)` | any | writes the region's cached lines back to DRAM: after writing what the grid or the host will read | - |
| `void kmx_invalidate_at(int id, uint64_t offset, uint64_t bytes)`, `kmx_flush_at` | any | the same for the lines holding `bytes` of the region from `offset` | - |
| `void kmx_evict(void)` | any | empties the cluster's cache: reads 128 KiB of the cached Zero Device, one line for each line of the cache's one way, so dirty lines are written back and every other line dropped; 21 to 26 µs on one core, against 10.4 ns a line for a sweep. `kmx_fetch` and `kmx_store` of 160 KiB or more use it only when the thread's start found that shape (WayEnable 0 and way 0 in core 0's way mask, both read once, never written); otherwise they sweep | - |
| `int kmx_spawn(int core, void (*function)(int core, void* arg), void* arg)` | 0 | starts `function(core, arg)` on core 1, 2 or 3 and returns | `KMX_OK`; `KMX_FULL` while that core runs its last spawn; `KMX_BAD` for another core |
| `int kmx_join(int core)` | 0 | waits until `core` returned from its last spawn (at once if none) | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`); `KMX_BAD` |
| `int kmx_parallel(void (*function)(int core, void* arg), void* arg)` | 0, not nested | spawns `function` on cores 1 to 3, runs `function(0, arg)`, joins them | `KMX_OK`; `KMX_FULL` if a spawned core is busy (nothing started); `KMX_LATE` or `KMX_STOPPED`; `KMX_STOPPED` with error `parallel` from another core or inside a `kmx_parallel` |
| `uint32_t kmx_clusters(void)` | any | the clusters running this image in the program, bit `c` for cluster `c` | a mask |
| `int kmx_lead(void)` | any | the lowest of them: the one that calls `kmx_parallel_all` | 1, 2 or 3 |
| `int kmx_parallel_all(void (*function)(int cluster, int core, uint64_t arg), uint64_t arg)` | the lead's 0, not nested | runs `function(cluster, core, arg)` on every core of `kmx_clusters()`; the first call waits until every other cluster serves | `KMX_OK`; `KMX_FULL` while a core of `kmx_spawn` still runs (nothing started on any cluster); `KMX_LATE` / `KMX_STOPPED`; error `parallel` elsewhere |
| `int kmx_serve(void)` | 0 | runs the lead's `kmx_parallel_all` calls until the lead's `kmx_main` returns | `KMX_OK`; `KMX_NONE` on the lead; `KMX_LATE` / `KMX_STOPPED` |
| `int kmx_push(int cluster, int id, uint64_t offset, const void* source, uint64_t bytes)` | any | stores `bytes` from `source` at `offset` of region `id` of `cluster`'s thread; returns once the last line's store is acknowledged; on core 0 outside `kmx_parallel`, over the four cores | `KMX_OK`; `KMX_BAD` (a cluster not in `kmx_clusters()`, an unknown region, out of range, not 64-byte multiples, a window, or before the lead's first call on another cluster); the lead's first push to another cluster waits as `kmx_parallel_all` does |
| `int kmx_port_cores(void)` | any | entries of the port tables: the grid's cores (0: the program declared no ports) | a count |
| `int kmx_port_owner(int core)` | any | the X280 core of this cluster that serves `core` | 0 to 3, or -1 (another cluster's, or none) |
| `int kmx_port_tell(int core, uint32_t first, uint32_t second, uint32_t third)` | the one serving `core` | stores a tell into the core's `KM_CONTROL_X280` line; the core sees the newest | `KMX_OK`; `KMX_BAD` for a core this X280 core does not serve |
| `int kmx_port_asked(int core, kmx_message* message)` | the same | `core`'s newest ask, if not answered yet; never waits | `KMX_OK` and `message`; `KMX_NONE`; `KMX_BAD` |
| `int kmx_port_answer(int core, uint32_t sequence, uint32_t first, uint32_t second, uint32_t third)` | the same | answers ask `sequence`; never waits | `KMX_OK`; `KMX_BAD` (also for `sequence` 0) |
| `int kmx_port_told(int core, kmx_message* message)` | the same | `core`'s newest tell, if newer than the last one looked at (latest wins) | `KMX_OK` and `message`; `KMX_NONE`; `KMX_BAD` |
| `int kmx_port_run(const km_command_t* command, void (*function)(int cluster, int x280_core, uint64_t arg), uint64_t arg, kmx_event* event)` | the lead's 0, not nested | submits `command`, runs `function(cluster, x280_core, arg)` on every X280 core as `kmx_parallel_all` does, ends the serve at the command's event | `KMX_OK` and `event` once every `function` returned; `KMX_FULL`: nothing submitted (a spawned core still runs, or 32 events of other commands kept), or, when the 32 filled during the run, the command still runs and `event->handle` alone is set: its handle, for `kmx_wait` once `kmx_poll` took the others; `KMX_LATE` / `KMX_STOPPED`; `KMX_BAD` without ports; error `parallel` elsewhere |
| `int kmx_port_open(void)` | any, inside `kmx_port_run`'s `function` | whether the command still runs; the lead's X280 core 0 looks for its event every fourth call | 1, or 0 once the event came, a stop or the deadline (and outside a run) |
| `int kmx_pull_run(const km_command_t* command, const kmx_pull* settings, kmx_event* event)` | the lead's 0, not nested | submits `command` and serves its pulls on every X280 core until its event: units 0 to `settings->units - 1` in ranges of `settings->per`, by `settings->mode` (`KMX_PULL_COUNT`, `_LOG`, `_REPLAY`; log region `settings->log`), `settings->pick` (the range) or `settings->pick_word` (the range and a word the core gets with it, [the pull's word](#the-pulls-word)) | as `kmx_port_run`; `KMX_BAD` also for a bad mode, both `pick` and `pick_word` set, `per` 0, a log region under `KMX_PULL_LOG_BYTES(1)`, or a log that ran out of room (the command ran) |
| `int kmx_ranks(void)` | any | X280 cores 0 to `kmx_ranks() - 1` are ranks of the program's dynamic run ([X280 cores as ranks](#x280-cores-as-ranks)) | 0 to 4 (0: none on this cluster) |
| `int kmx_messages_run(void (*function)(int x280_core, kmx_messages* messages, void* arg), void* arg, uint64_t deadline)` | 0, not nested | waits by `deadline` for the next dynamic run's `RUN`, then runs `function(x280_core, messages, arg)` on each X280 rank, cores 1 to `kmx_ranks() - 1` spawned, and joins them; afterwards it takes and consumes what a `function` left | `KMX_OK` once all returned; `KMX_LATE` / `KMX_STOPPED`; `KMX_BAD` without ranks or inside `kmx_parallel`, or at once after a rank of the load gave up for room (each rank's `gave` `KMX_GAVE_BROKEN`) |
| `const volatile uint32_t* kmx_messages_next(kmx_messages* messages)` | the rank's | the next message of the rank's queue, `message_bytes / 4` words (word 0 the runtime's, `messages->handler` its handler), valid until the next call; waits while the queue is empty | the message; 0 once the run ended or the rank gave up |
| `int kmx_messages_forward(kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler)` | the rank's | the message taken last goes on at core rank `rank` (below `messages->count`; over islands, of any island), handled there by `handler`; words 1 on of `message` are sent | `KMX_OK`; `KMX_BAD` once it gave up (no such core rank, no room in the core's inbox, or another island's rank's ring place unconfirmed, within the wait) |
| `int kmx_messages_send(...)`, same arguments | the rank's | a new message, as forward, taking one of the quota | as forward; `KMX_FULL` with no quota left (nothing sent) |
| `void kmx_messages_consume(kmx_messages* messages)` | the rank's | the message taken last ends here: one send back to the quota | - |
| `void kmx_command(km_command_t* command, uint32_t program, uint32_t handler, uint32_t first, uint32_t count)` | any | fills `command`: `hdr = handler \| program << 24`, `group = first \| count << 16`, every other word 0 ([several programs](#a-thread-in-a-run-of-several-programs)) | - |
| `int kmx_block(km_command_t* command, int id, uint64_t offset, uint32_t bytes)` | any | makes `bytes` (1 to 1,024) of region `id` from `offset` (a multiple of 64) the command's block: `a7` and `a8` ([the block of a command](#the-block-of-a-command)) | `KMX_OK`; `KMX_BAD` outside the region or for a bad size or offset |
| `int kmx_load(uint32_t program, uint32_t first, uint32_t count, uint64_t deadline, kmx_event* event)` | 0 | in a run loaded with a dict, makes cores `[first, first + count)` hold `program` (its field, `kProgram<Name>`): submits the load, waits for its event and, while the dispatcher rejects it busy, waits for the wake and sends it again, all by `deadline` ([a thread gives a group another program](#a-thread-gives-a-group-another-program)) | `KMX_OK` and `event` (may be 0) once the load is done; `KMX_BAD` when the dispatcher refused it or its timeout came (`event` says which) or the run has no record for `program`, or 32 events of other handles are kept (the stash full); `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`) |
| `int kmx_fetch(int id, uint64_t offset, const kmx_dram* from, uint64_t bytes, uint32_t first, uint32_t count, uint64_t deadline, kmx_event* event)` | 0 | cores `[first, first + count)` copy `bytes` from `from` (a range of any DRAM bank, or a tensor's pages) into region `id` at `offset`: the range's lines dropped, then one command (reserved handler 245) sent again after a busy rejection and waited for by `deadline`, then the range's lines dropped again (a line the L2 prefetcher brought in during the copy is stale); the thread's first read of a line takes the new bytes ([another bank](#reading-and-writing-another-bank)) | `KMX_OK` and `event` (may be 0); `KMX_BAD` for a bound (nothing sent), a refusal or the command's timeout (`event` says which), or 32 events of other handles kept (the stash full); `KMX_LATE` / `KMX_STOPPED` |
| `int kmx_fetch_clean(int id, uint64_t offset, const kmx_dram* from, uint64_t bytes, uint32_t first, uint32_t count, uint64_t deadline, kmx_event* event)` | 0 | `kmx_fetch` without dropping the range's lines, for a range none of whose lines is in the cluster's cache (not read or written since the load, its last flush or the last fetch into it): the call costs the copy | as `kmx_fetch` |
| `int kmx_store(int id, uint64_t offset, const kmx_dram* to, uint64_t bytes, uint32_t first, uint32_t count, uint64_t deadline, kmx_event* event)` | 0 | the other way: the region's lines written back, then the cores copy them to `to` | as `kmx_fetch` |
| `int kmx_fetch_submit(int id, uint64_t offset, const kmx_dram* from, uint64_t bytes, uint32_t first, uint32_t count, uint32_t* handle)`, `kmx_store_submit` | 0 | the same copy, submitted and not waited for: `kmx_wait` (or `kmx_poll`) takes its event, and a fetch's DONE taken there drops the range's lines again | `KMX_OK` and the handle; `KMX_FULL` when no slot is free, or for a fetch when a fetch submitted 16 commands before it still has its event untaken; `KMX_BAD` for a bound |
| `int kmx_submit(const km_command_t* command, uint32_t* handle)` | 0 | stores `command` as the next command of the ring | `KMX_OK` and the handle (1, 2, ...); `KMX_FULL` when no slot is free; `KMX_STOPPED` without a ring |
| `int kmx_wait(uint32_t handle, uint64_t deadline, kmx_event* event)` | 0 | waits for the handle's event until `kmx_time()` passes `deadline`; other handles' events are kept for `kmx_poll`, wake events are passed over | `KMX_OK`; `KMX_LATE` / `KMX_STOPPED` (library error `late` / `stopped`); `KMX_FULL` when 32 events are kept |
| `int kmx_poll(kmx_event* event)` | 0 | one event, a kept one first; never waits | `KMX_OK` or `KMX_NONE` |
| `int kmx_cluster(void)` | any | the work cluster the thread runs on | 1, 2 or 3 |
| `int kmx_island(void)`, `int kmx_islands(void)` | any | the island the thread's cores are on, and the run's islands ([islands](runtime.md#islands)) | 0 and 1 on one island |
| `int kmx_island_of(uint32_t number)` | any | the island of thread `number` (`KMX_THREAD(island, cluster)` = 3 x island + cluster, the group `KM_GROUP_CLUSTER(number)`) ([commands across islands](#commands-across-islands)) | the island; -1 for no thread of the run |
| `uint64_t kmx_time(void)` | any | the 50 MHz timer (`KMX_TICKS_PER_SECOND`) | ticks |
| `int kmx_stop_requested(void)` | any | true once the host asked for a stop or the thread's deadline passed (never once it serves commands) | 0 or 1 |
| `void kmx_progress(uint32_t where)` | any | writes `where` and its time into the result block as a progress marker | - |
| `void kmx_result(uint32_t code, const void* words, uint32_t word_count)` | 0 | the thread's code and up to 48 words | error `result` if `word_count > 48` (the first 48 kept) |

Ring calls, `kmx_spawn`, `kmx_join`, `kmx_parallel`, `kmx_parallel_all`, `kmx_port_run` and
`kmx_serve` from X280 cores 1 to 3 command the library error `parallel` and return `KMX_STOPPED`.
`kmx_message`: `sequence`, `w[3]`. A command for `kmx_submit` is a `km_command_t` (`komorebi/abi/km_abi.h`):
`hdr = handler | flags << 8 | program << 24` (the program field, 0 for the run's first
program), `group = first | count << 16`, arguments `a0` to `a9`; the
library writes `sequence` and the landing words. Flags: `KM_FLAG_EXCLUSIVE` (submitted only when no
group in flight overlaps it), `KM_FLAG_RESULT` (the group leader's `writer.result(value)`
is returned in the event's `ticks`), `KM_FLAG_QUEUE` (the kernel chooses `first`). A
command that cannot be submitted now is rejected at once (`KM_EVENT_REJECTED`, auxiliary 11) and the
kernel then sends one `KMX_EVENT_WAKE` (handle 0) when a slot frees or the queue submits a
job: the command may be submitted again after it. The kernel's other rejections are its README's ("Rejected commands"), and STOP, EXIT, RUN and
CANCEL from a thread are rejected with auxiliary 1.

`kmx_event`: `handle`, `kind` (`KM_EVENT_DONE`, `KM_EVENT_TIMEOUT`, `KM_EVENT_REJECTED`, `KMX_EVENT_WAKE` from `kmx_poll` only),
`auxiliary` (the first core of a done command or the slot of a timeout in its low 8 bits, the
low 16 bits of the command's sequence number above them; the reason of a rejection), `ticks` (submit to completion
in timer ticks, or the result word of a `KM_FLAG_RESULT` command).

Return codes (`kmx_layout.h`): `KMX_OK` 0, `KMX_FULL` 1, `KMX_STOPPED` 2,
`KMX_LATE` 3, `KMX_NONE` 4, `KMX_BAD` 5.

### The block of a command

A command from a thread carries a block as one from the host does
([runtime](runtime.md#the-block)): `kmx_block` puts the region's bank address in `a7`
and `bytes | (bank + 1) << 16` in `a8`, the bank being the cluster's (6 for cluster 1,
7 for clusters 2 and 3), where the region's one copy lies. Every core of the group then
reads it from that bank with `reader.block(command)` or `writer.block(command)`. The
thread writes the bytes and flushes them (`kmx_flush_at`) before `kmx_submit`, and
leaves them until the command's event; arguments `a0` to `a6` stay its own.

```c
kmx_command(&command, 0, kHandlerDot, 0, 12);
command.a0 = x0;
kmx_flush_at(WEIGHTS, 0, 64 * 4);
if (kmx_block(&command, WEIGHTS, 0, 64 * 4) == KMX_OK) {
    kmx_submit(&command, &handle);
}
```

### The pull's word

`kmx_pull`'s `pick_word(core, index, arg)` gives core's `index`-th range and a 32-bit
word for it, `KMX_PULL_WORD(range, word)`; the core gets the word with the range from
`port.next(first, word)` ([X280 ports](ports.md#x280-ports)). It runs on the X280 core
that serves the core, so it may read what the core told it (`kmx_port_told`) and
decide from that, as `examples/18_thread_split` picks an overlay. The word is 0 with
the counter, a log, a replay, `pick`, and when no range is left. A pull without a word
costs what it did: `examples/10_x280_ports --pull` 0.66 to 0.70 µs a pull both ways.

```c
static uint64_t pick_word(int core, uint32_t index, uint64_t ranges) {
    const uint32_t range = index < ranges ? index : KMX_PULL_NONE;
    return KMX_PULL_WORD(range, choose(core)); /* choose: from kmx_port_told */
}
const kmx_pull pull = {units, per, KMX_PULL_COUNT, 0, 0, ranges, pick_word};
```

### X280 cores as ranks

A thread can be ranks of the program's dynamic run
([X280 ranks](dynamic.md#x280-ranks)): with `x280_ranks=n` on the program's `dynamic`,
X280 cores 0 to n - 1 of the thread's cluster are ranks `count` to `count + n - 1`
after the group's `count` core ranks. `kmx_main` calls `kmx_messages_run` once for each
dynamic `RUN` it takes part in; each rank's function takes its messages with
`kmx_messages_next` until it returns 0, and ends each with a forward to a core rank or a
consume. `kmx_messages` holds the run (`run`, the `RUN`'s sequence), the rank (`rank`),
the core ranks (`count`, `first` the group's first core), the taken message's
`handler`, the `quota`, `sent`, `consumed` and `gave` (`KMX_GAVE_ROOM` 1, `_RANK` 2,
`_STOPPED` 3, `_LINE` 4, `_BROKEN` 5: a run after a `ROOM` give-up of the load, refused;
`_FAR` 6: a send to another island's rank whose ring place stayed unconfirmed, not
exercised on a card; 0: it did not give up).

```c
static void rank(int x280_core, kmx_messages* messages, void* arg) {
    uint32_t message[16];
    const volatile uint32_t* in;
    while ((in = kmx_messages_next(messages)) != 0) {
        for (uint32_t w = 1; w < 16; w++) message[w] = in[w];
        if (done(message)) { kmx_messages_consume(messages); continue; }
        kmx_messages_forward(messages, message[3] % messages->count, message, kHandlerGrid);
    }
}
int kmx_main(int core) {
    return kmx_messages_run(rank, 0, kmx_time() + 20 * KMX_TICKS_PER_SECOND);
}
```

The rank's queue is in the cluster's bank and the library reads it itself; the thread's
regions and words are as without ranks, and `x280_queues` is a region after the
thread's own (the ids of shared regions move one on).

## Commands to a thread

A command whose group names a work cluster goes to that cluster's thread instead of to
cores: `first` = `KM_GROUP_CLUSTER_BASE` (0xFF00) + the cluster, count 1
(`KM_GROUP_CLUSTER(c)` in C, `cluster=c` in Python). The host, a graph, a stream and
another cluster's thread submit it as any command. The thread answers it in C:

```c
#include "kmx.h"

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerFold: /* from Thread(handlers={"fold": 1, ...}) */
            return fold(command->a0);
        case kHandlerServed:
            return served;
    }
    return 0;
}
```

**The serve loop.** After `kmx_main` returns (at once without one), core 0 serves: it
tells the dispatcher it serves, then takes each command in turn, calls `kmx_program` and
answers with its word, until the run stops. The command's event is then sent to its
source with the word as its result (`KM_FLAG_RESULT`) or the time, as for a core group;
`auxiliary` holds the cluster in its low 8 bits and the sequence number above them. A
command sent before the thread serves is kept by the dispatcher until it does, within
its deadline.

**The deadline.** A thread's `deadline` bounds `kmx_main`. Once the thread serves, the
library lifts it on every core: the thread lives until the run stops (the host's stop
flag or the runtime's stop), and each command has its own deadline, the dispatcher's
(`session.load(deadline=)`, or a stream's or graph's). When a command's deadline passes,
its source gets its `KM_EVENT_TIMEOUT` and the dispatcher goes on; the command's line
stays taken until the thread answers, and that late answer sends no event.

**The limits.** A thread holds 8 unanswered commands (`KMD_WORK_IN_DEPTH`). With all 8
taken, a further command from the stream engine, the graph engine or another thread is
rejected busy (reason 11) and sent again after the dispatcher's wake; one from the host
is rejected with reason 14, `thread` (`KMD_REJECT_THREAD`), and the run goes on: the 8
are answered and later commands are taken (measured with one 0.2 s command and 7 behind
it). A command to a thread is
never exclusive or queued (reason 2, `flags`), and a cluster without a thread rejects it
with reason 3 (`group`).

**From a thread.** `kmx_submit` with `group = KM_GROUP_CLUSTER(c)` commands another
cluster's thread: 4.74 µs one at a time against 3.60 µs for a NOP to one grid core, 2.02
µs a command with 8 in flight (`experiments/x280/thread_commands`). `kmx_program` may submit
commands to core groups and wait for them inside the call (`kmx_wait`); while it waits
the thread answers no other command, so the dispatcher keeps up to 8 more as above.
At depth: a command to the thread's own cluster, or to a thread that is waiting for
this one, is answered only after the call returns, so its wait ends at that command's
deadline with its timeout event (or at `kmx_wait`'s own deadline with `KMX_LATE`), never
in a deadlock. `kmx_parallel` inside `kmx_program` works as in `kmx_main` (cores 1 to
3 serve spawns until the run stops).

**Events nobody waited for.** The event of a command `kmx_program` submitted and did not
wait for is taken from the ring by the serve loop between commands (and while an answer
waits for room) and kept for a later `kmx_poll` or `kmx_wait`: the newest 32
(`KMX_STASH`), older ones dropped. A program that needs every event takes them inside
the call that submitted the commands (`kmx_wait` each, or `kmx_poll` until `KMX_NONE`).
Measured: 200 commands never waited for, over 20 calls, each with a command from the host
between them; every one answered, and `kmx_poll` then took 32
(`tests/hardware/thread/commands/open_points_case.py`). Without this a ring full of unread
events stopped the thread's commands and its answer queued behind them.

### Commands across islands

In a run over several islands ([islands](runtime.md#islands)) a command's group is in
flat ranks and a thread's number is flat: `KMX_THREAD(island, cluster)`, 3 x island +
the cluster, so island 0's numbers are the ones of one island and island 1's work
cluster 1 is 4. `kmx_island_of(number)` names the island of a thread. A thread submits
to any of them, and to any flat group, with `kmx_submit` as on one island:

```c
kmx_command(&question, 0, kHandlerFold, KM_GROUP_CLUSTER(KMX_THREAD(1, 1)), 1);
kmx_command(&map, 0, kHandlerMap, 0, 238); /* both islands' ranks (119 each) */
```

The dispatcher splits the group by the island table. It runs its own part and sends
each other island's part to that island's dispatcher as a command packet: the 64-B
command line unchanged and a 64-B crossing line (the origin island, the origin ticket,
the time left, a sequence). Any group of flat ranks may be commanded, over any number of
islands: the islands a group touches are a range (each island's ranks follow the one
before), and the parts leave one after another as the dispatcher's queue toward the link
has room. The dispatcher stages a packet in the link area of the island's link core 0
(the core after its ranks) and gives that core a `PAGES` line (`KM_PAGES_LINE`, group
`KM_GROUP_LINK_BASE`); its CPU that holds link 0's connection, opened at the load,
writes the packet's bytes and then its count line into the far dispatcher's inbox,
forwarded by the link cores of every island between by their next link, as a page ask
is. That is the link's write of bytes to a place on another island: when the link
service carries a write to any NoC address, the line becomes that call, the same packet,
one path. Each island writes only its own part of a far inbox (whole lines, its count
word last; nothing atomic crosses), and every 16-byte row of a packet that holds words
carries the packet's number: the link core sends it and the far dispatcher takes it only
when every row shows it ([hazards](hazards.md#a-line-read-across-as-it-lands)). The same CPU serves page commands in turn when link 0
is the session's only link ([islands](islands.md#pages-across-a-link)). The other
dispatcher runs the packet as its own command and sends DONE back the same way (ticket,
status, completion words). The ticket's event comes once every part is back: `ticks` is
the whole command's, a result (`KM_FLAG_RESULT`) is the part's that holds the group's
first rank. A part not back by the command's deadline gives the command up:
`KM_EVENT_TIMEOUT` with `KM_EVENT_ISLAND` (0x80) + the lowest island whose part is not
back in the low 8 bits of `auxiliary` (this island's when only its own part is out), and
the run goes on. For a command from the host the dispatcher
counts that TIMEOUT once among its give-ups, so the host's run goes on past it as past
any command given up; a part run for the other island counts nothing there. A line that
would cross may not be EXCLUSIVE (reason 2, `flags`); the host's own split of a group
over islands marks its lines `KM_FLAG_PART` and does not cross ([islands](islands.md#the-run)).

Every table of the crossing is sized by the session, none by a constant of the cards:
each dispatcher's link area (its island entries and an inbox part for each other
island) is laid out by the host (`komorebi.islands.link_area`) and put at the end of the upload
region, which is that much shorter: 2,176 bytes with two islands, 65,984 with 32. Its
own tables (the entries copied, its counts, a bit an island for each command in flight:
192 bytes with two islands, 3,200 with 128) are in its state's other pages in its DRAM
bank, cached, where the X280 reads them faster than the uncached LIM. An inbox part holds 16 packets
while the parts take at most 64 KiB (up to 33 islands), fewer beyond, at least 4; a
dispatcher keeps half its part's packets in commands across in flight, at most 8. A
dispatcher reads four other islands' parts a pass, in turn. A give-up names an island in
one byte (`KM_EVENT_ISLAND` + island), so a session of more than 128 islands is refused
by name. With one island none of this runs and the upload region keeps its size.

A stream's `SUBMIT` and a graph's node take a group over the same flat ranks: the engine
checks it against the session's ranks (from the link line) instead of its island's, and
the dispatcher sends each other island's part as it does a thread's command. A queued
job (`ENQUEUE`, `KM_FLAG_QUEUE`) stays on the island whose dispatcher places it. A
host-fed stream or graph through one island's run (`run.island(i).run_hostfed`) checks
its groups against the session's ranks too. Example 06's `islands.py` (two cards,
2026-11-16): its graph from island 0 alone, `RIGHT` on island 1's 118 ranks and `STEP`
and `NORM` on all 236, host-fed and by island 0's engine alike: 4 and 6 rounds as the
replay, every state word right.

Measured (`examples/25_islands_threads --links 4`, two cards, 2026-11-14): a NOP
submitted to island 0's ring, 2,000 pipelined, 2.98 µs to island 0's 118 ranks and
3.94 µs to island 1's; example 01's MAP from island 0's thread, one at a time, 6.05 µs
on island 0's ranks and 13.11 µs on all 236, every tile exact; a FOLD with its result
from island 0's thread to island 1's, one at a time, 10.61 µs; a HANG on ranks 100 to
139 gave the command up naming island 1, and a MAP on all ranks after it was exact.
(With the link core a rank of its own, opened on its first packet: 3.08, 4.36, 6.00,
15.41 on 240 ranks and 10.69 µs.)

### Fetch and store across islands

`kmx_fetch` and `kmx_store` take a tensor another island holds as they take one of
their own: `from` or `to` names the tensor (`KM_COPY_INTERLEAVED`, its bank address,
its page bytes) and the range, never the island. The copy's cores find the tensor in
the table of island tensors every core holds (32 entries at most; the host writes it
again when a tensor is made while the run lives) and move the pages another island holds through their island's link core 0: a
fetch asks for them a chunk at a time and writes them into the region, a store puts
them from the chunk straight into the holder's DRAM
([islands](islands.md#pages-across-a-link)). The cache is handled as on one island: a
fetch drops or evicts the range before the copy, a store writes the region back before
it. A page that crosses is at most one fabric packet (4,352 bytes); a larger one gives
the copy's command up. On one island the calls are what they
were.

Measured (`experiments/x280/bank_fetch/islands.py`, two cards, four links, 2026-11-14, 16
cores, 4 KiB pages, best of 3, every word checked): from island 0's work cluster 1,
`kmx_fetch` of 1 MiB and 64 MiB 51 µs and 1.30 ms from island 0's tensor, 204 µs and
11.0 ms (6.10 GB/s) from island 1's; `kmx_store` 47 µs and 1.30 ms into island 0's,
197 µs and 10.7 ms into island 1's. With every page CPU serving (2026-11-15, one run):
228 µs and 12.4 ms from island 1's, 213 µs and 11.7 ms into it, four pages a line.
With a line a whole run straight between the two DRAMs (2026-11-15, one run): 107 µs and
1.125 ms (59.7 GB/s) from island 1's, 102 µs and 1.118 ms into it.

Not done: a `kmx_push` larger than one packet does not cross (one command line and its
crossing line are a packet); runs of more than two islands (one link, no routing); a
crossing's local part that times out from a thread is given up as on one island.

### A thread in a run of several programs

In a run loaded with several programs in the runtime's form (`session.load([a, b])`,
[host](host.md#several-programs-in-one-run)), one of them declares the threads, and the
thread commands any of them: a command's header names its program in bits 24 to 31. The
C gets each program's field and handlers as defines, `kProgram<Name>` and
`kProgram<Name>Handler<Handler>` (a name in camel case, as `kHandler<Name>` for the
thread's own handlers); a run of one program adds none, so its thread's defines are what
they were. A define that the thread's own defines or handlers also give is refused at
the build (`ValueError`).

```c
km_command_t command; /* a run of programs spin and scale */
kmx_command(&command, kProgramScale, kProgramScaleHandlerScale, first, count);
command.a1 = tiles;
uint32_t handle;
kmx_submit(&command, &handle);
```

A command to a group with a core that does not hold the program (with
`{program: group}`) is refused by the dispatcher: `kmx_wait` returns its event,
`KM_EVENT_REJECTED` with reason 13, and the run goes on; in the combined form every core
holds every program.
[19_thread_programs](../../examples/19_thread_programs/README.md) splits the grid between
two programs every round from the thread.

**What the host sees.** `run.submit(handler, cluster=c, args=..., result=True)` returns
a handle; `run.wait` and `run.event` give its `Event` as for a core group. Over several
islands `c` is the thread's flat number (`KMX_THREAD(island, cluster)`), and the call
returns a ticket ([the run](islands.md#the-run)). A rejection
raises `CommandRejected` naming the reason (`thread` for a full thread). A timeout does
not end the run: `run.wait` returns the event with kind `"timeout"` and the run goes
on. A command the thread submits to a core group is given up at its deadline as the
host's is ([giving up one command](runtime.md#giving-up-one-command)): `kmx_wait`
returns `KMX_OK` with `KM_EVENT_TIMEOUT` (`auxiliary` the slot and sequence, `ticks` from
the submit to the give-up), the run goes on and the next command on those cores runs
as usual. A host-fed graph or stream ends with its deadline error, the run going on.

### A thread gives a group another program

In a run loaded with a dict (`session.load({a: group, b: group})`,
[host](host.md#a-group-switches-program)), the thread moves cores from one program to
another as the host's `run.load` does, with one call:

```c
kmx_event event;
if (kmx_load(kProgramWiden, k, old_k - k, kmx_time() + KMX_TICKS_PER_SECOND, &event)
    != KMX_OK) {
    return KMX_BAD; /* event: why */
}
/* cores k to old_k - 1 hold widen: commands of widen to them are taken */
```

- **The records.** A load names the program's record in DRAM, which exists only once
  the session has built the programs. The session writes the records into the thread's
  ring line before the thread starts (`records`, `record_step`, `programs`: program
  field `p`'s record at `records + p * record_step` for `p` below `programs`); the
  library reads them, the program names only the field. A run whose groups cannot
  switch (loaded with one program or a list, or groups whose runtime kernels differ)
  has `programs` 0, and `kmx_load` returns `KMX_BAD` without sending anything.
- **A busy dispatcher.** A load is exclusive: it goes only when no job is queued and no
  command in flight shares a core with it. A thread's command that cannot go now is
  rejected at once with reason 11 and a wake owed (the thread never holds the
  dispatcher's core 0); `kmx_load` then waits for that wake and sends the load again,
  so a load to cores with a command in flight returns after that command. The events
  of other commands that arrive meanwhile are kept for `kmx_wait` and `kmx_poll`.
- **What the program sees.** The load takes effect for later commands: when `kmx_load`
  returns `KMX_OK` the dispatcher holds the cores as the new program's, a command of
  the old program to them is refused (reason 13) and one of the new program is taken.
  The cores take the load after their earlier commands.
- **What the host sees.** Its own record of which program each core holds is stale
  once a thread loads, so in a run loaded with a dict that has a thread on the card
  `run.check_programs` starts `False`: the host sends what it is asked to and the
  dispatcher refuses a command for a program its cores do not hold (`CommandRejected`,
  reason 13). `run.thread.records` holds what the thread was given.
- **Not possible.** A load from a stream or a graph (rejected, reason 1), from a cast or
  a pull; a load in the combined form; a load of a dynamic program.

#### A class added to the live run

A class added with `run.addon(program)` ([host](host.md#a-class-added-to-a-live-run))
takes the next program field, and its record goes into the row the load reserved for
it: the load reserves a record row for every field an added class may take (each field
whose overlay number stays above the programs' own overlays, at most 61; one record is
576 B in each bank). The add then raises `programs` in the live thread's ring line, so
`kmx_load` reaches it as it reaches a loaded program; a field past the reserved rows
is refused by name. The thread's image was built before the class existed, so it has
no `kProgram...` or handler constant for it: the host gives the thread the field, the
handler number and the call's words in a command,

```python
field, handler, words = run.thread_words(Project.rows, (x, w, y))  # nothing sent
run.submit("call", cluster=1, args=[field | handler << 8, first | count << 16, *words])
```

and the thread uses them in place of the constants:

```c
kmx_load(job->a0 & 0xFF, first, count, kmx_time() + KMX_TICKS_PER_SECOND, &event);
kmx_command(&command, job->a0 & 0xFF, (job->a0 >> 8) & 0xFF, first, count);
```

The dispatcher is unchanged: it takes any field below 64. On the card (the gate on
cores 0 to 59, the added class on 60 to 119, 16 calls each waited for): the thread's
`kmx_load` of the added class took 10.5 µs, a call 10.4 to 11.3 µs from submit to its
event, against 232 to 319 µs for the host's `run.submit` and `run.wait` of the same
call (9.1 to 9.5 µs of it on the card); every output exact, the gate after it too.

An added class with per-core args needs nothing more from the thread: its load command
copies each core's row into L1 before any of the core's five CPUs takes the next
command, and `kmx_load` returns only when that command is done. A thread's command to
a group with a core that does not hold the class is refused by the dispatcher, as for
any program: `kmx_wait` returns `KM_EVENT_REJECTED` with reason 13, and no core takes
it.

[21_thread_load](../../examples/21_thread_load/README.md): after one command from the
host, the thread moves the boundary between two programs' groups with a load every
round and commands both; 40 rounds on 120 cores in 1,825 µs against 16,950 µs with the
host issuing every load and command, a load 6.2 µs from `kmx_load`'s call to its
return against 134.9 µs for the host's `run.load` and wait, 10,345 thread-issued loads
with every result exact.

## Reading and writing another bank

A thread's own regions are in its cluster's bank; any other DRAM, another bank or a
tensor interleaved over all of them, it reads and writes through Tensix cores. The X280
core's own loads over the NoC move 0.024 GB/s; four Tensix cores copy 64 MiB from one
bank to another in 1.06 ms.

```c
const kmx_dram from = {address, 7, 0, 0}; /* a range of bank 7 */
kmx_event event;
if (kmx_fetch(0, 0, &from, bytes, 0, 4, kmx_time() + KMX_TICKS_PER_SECOND, &event)
    != KMX_OK) {
    return 3; /* event: a refusal or the command's timeout */
}
/* region 0 holds the bytes: read it */
```

- **The other side.** `kmx_dram` is `{address, bank, page, first}`. A range of one bank:
  `bank` 0 to 7 and its bank address; the library cuts it into pages of the largest
  power of two up to 16 KiB that divides `bytes` (`page` and `first` 0). A tensor:
  `bank` `KM_COPY_INTERLEAVED`, its address, its aligned page bytes (at most 16 KiB) and
  the first page, read and written as the host channel's transfers read them; `bytes` is
  then whole pages. Python gives a tensor's address as `tensor.address`; a range of one
  bank of a tensor with pages of P bytes is its pages b, b + 8, ..., one after another
  from `tensor.address` in bank b.
- **The command.** Reserved handler 245 (`KM_HANDLER_COPY`, words in
  `komorebi/abi/km_copy.h`) to cores `[first, first + count)`. It is served by the
  runtime's loop of every program, compiled in as the host transfers are, so any
  loaded cores take it whatever program they hold, and the dispatcher accepts it from
  the host and from a work cluster's thread (as the load), never from a stream or a
  graph. Each core copies its share of the pages (`km::partition`) through two 16 KiB
  chunks of the run's transfer buffer, the read of one chunk beside the write of the
  other; every wait is bounded by the run's wait limit.
- **The bounds.** The library checks the region (offset and bytes in 64-byte multiples,
  inside it), the page and the run's buffer before it sends anything (`KMX_BAD`); each
  core checks every bound again before it issues a read (`km_copy_fits`: pages in whole
  64-byte lines, both chunks inside L1, both sides inside 4 GiB of their banks, banks
  the card has, two ranges of one bank apart). A core that finds one false gives up
  with where `copy` and gave `bad_transfer`, having moved nothing, and the run stops as
  for any give-up. Two tensors, or a tensor and a range, are not compared for overlap.
- **The cache.** A fetch drops the range's lines before the copy (a dirty line written
  back afterwards would land over the new bytes, a stale one would be read in their
  place); nothing on the cluster touches the range while the cores copy, so nothing
  follows. A store writes the lines back first. A sweep is one line at a time through
  the cluster's cache controller, 6.1 GB/s of range on one core or four, so from
  160 KiB on a fetch or a store calls `kmx_evict` instead, about 20 µs whatever the
  range (the cache has one 128 KiB way): a blocking fetch of 64 MiB takes 1.08 ms
  against the copy's 1.06 ms, and 11.9 ms with a sweep; a store of 64 MiB within its
  own bank 2.30 ms, and 13.3 ms with a sweep. A range the thread has not touched since
  the load, its last flush or the last fetch into it needs neither: `kmx_fetch_clean`
  takes 1.07 ms for the 64 MiB. So fetch what the thread will read,
  once, into a range it has not read since; the submit form lets the thread work on
  something else while the cores copy.
- **What the cores do meanwhile.** The copy is a command: the cores take it in their
  turn behind earlier commands and run nothing else while they copy.

[bank_fetch](../../experiments/x280/bank_fetch/README.md) has the rates;
[08_parallel_thread](../../examples/08_parallel_thread/README.md) `--fetch` runs its
prefix sum on an input fetched from bank 7, which cluster 1 cannot read.

## The image

| Rule | Value |
|---|---|
| size | at most 126,976 bytes, globals included |
| globals | a non-zero initialiser and `__attribute__((section(".data")))`: the image has no `.bss` |
| stack | 32 KiB per core |
| libraries | none: a struct assignment or a large zero initialiser may become a `memset` call that does not link |
| start-up | floating point and vector on; the L2 prefetchers set to the documented values unless `prefetch="keep"`; argument block, regions and bank checked; then `kmx_main` on core 0, spawns served on cores 1 to 3; after it, every region given back |
| build | `Thread.sources` with `Thread.includes` and `Thread.defines`, compiled with the dispatcher's toolchain into the build cache |

## Memory

The thread's block in the cluster's local memory (LIM), bytes from its base:

| Offset | Bytes | What | Written by |
|---|---|---|---|
| `0x000` | `0x240` | `kmx_args`: run value, magic `kmx1`, version 5 (`KMX_VERSION`), deadline (ticks), region and word counts, check line address and value, flags (`KMX_FLAG_KEEP_PREFETCH` 1), the cluster (`cluster`, at `0x38`); region table at `0x040` (16 x `{offset, bytes}`); words at `0x140` (64 x `uint32`) | host, before the start |
| `0x240` | 64 | stop line: word 0 = `0x53544F50` asks for a stop | host |
| `0x280` | 256 | result block | thread |
| `0x380` | 4 x 64 | spawn lines, one per core, core `core`'s at `+ 64 * core`: 64-bit words `gen`, `fn`, `arg` (core 0), `done` (that core); core 0's word 0 ends the service | cores |
| `0x480` | 64 | `kmx_ring_descriptor`, zero for no ring | host, before the start |
| `0x4C0` | 64 | `kmx_peers`: a value for this load (`nonce`), the lead, the mask of clusters running the image, the offset `all` of the `kmx_parallel_all` lines, then each cluster's tile and its bank's DRAM tile (`x \| y << 16`, indexed by cluster); zero: no peers | host, before the start |
| `0x500` | 64 | `kmx_ports`: `at` (LIM offset of the ports area, 0: no ports), `control` (the cores' control block address), `cores` (table entries), `flags` (`KMX_PORTS_NOC1` 1), `state` (`kmx_port_run`'s: `n << 1` while run `n` serves, `\| 1` once it ended), `next`, `lost`, `pull` (units, per, mode and log, 1 when `pick` is a `pick_word`), `pick`, `arg` (`kmx_pull_run`'s, written by the lead) | host, before the start; `state` the lead |
| `0x540` | 64 | `kmx_ranks_line` (0: no X280 ranks): `at` (LIM offset of the ranks area), `ranks`, `cores`, `queue` (bank address of rank 0's queue), `cap`, `message_bytes`, `quota`, `counters`, `idle` (ticks), `inbox` (the cores' L1 inbox), `inbox_messages`, `lines` and `lines_tile` (the dispatcher's status lines), `wait` (ticks) | host, before the start |
| `all` | 4 x 64 | after the block the host writes (and the ring's events): the call line the lead writes into each other cluster (32-bit words `nonce`, `gen`, `fn`, `arg`, `exit`), and in the lead's, cluster `cluster`'s done line at `+ 64 * cluster` (`nonce`, the last `gen` done; 0: it serves) | the lead; the others |

The ports area at `at` (after the `kmx_parallel_all` lines, written by the host before any
thread starts, `KMX_PORTS_BYTES` = 8,832): each core's tile at `+ 4 i` (`x | y << 16`),
its serving X280 core at `0x200 + i` (`cluster << 2 | x280_core`, 0: none), its mailbox
line at `0x280 + 64 i` (ask at `+ 0x00`, tell at `+ 0x10`, written by the core).

Result block: `state` (1 boot, 2 running, 3 done, 4 failed), `error` (the first
library error: 0 none, 1 args, 2 bank, 3 late, 4 stopped, 5 parallel, 6 result), `code`,
`n`, `where`, `t_where` (ticks from the start to the last progress marker), `ticks` (start to
the end of `kmx_main`), then 48 result words at byte 64.

Windows on a work cluster: the ring's 16 (`KMX_RING_WINDOW`), cluster `c`'s LIM
through `16 + c` (`KMX_PEER_WINDOW + c - 1`) while the thread runs, core `h`'s push
through `20 + h` (`KMX_PUSH_WINDOW + h`) during a push, core `i`'s control block through
`24 + i` (`KMX_PORT_WINDOW + i`) for the run when this cluster serves it.
With X280 ranks, the same windows onto the cores' L1 for their inboxes while
`kmx_messages_run` runs, and X280 rank `r`'s status line through `152 + r`
(`KMX_RANKS_WINDOW + r`). The ranks area at `at` (`KMX_RANKS_BYTES` = 3,072): each
core's tile at `+ 4 i`, its credit line at `0x200 + 16 i` (word 0 the head of its inbox,
written by the core), its next inbox position at `0xA00 + 4 i` (an atomic add of the
X280 ranks).

Regions are whole 4 KiB pages of the work cluster's DRAM bank (on the p150a bank 6 for
cluster 1, bank 7 for clusters 2 and 3); page 0 of a bank's pages is the check line,
then each thread's own regions, then the shared ones. The thread sees region `id` at
the cached alias of the bank plus its offset; a Tensix kernel reaches it with
`get_noc_addr_from_bank_id<true>(bank, address)`. A thread reads only its own bank
itself: data on another bank is pushed there by its producer, or fetched with
`kmx_fetch` ([another bank](#reading-and-writing-another-bank)).

`kmx_ring_descriptor` (one 64-byte line): `cmd` (`command_depth` 64-byte slots, 64-byte
aligned), `consumed` (the kernel's count of commands taken), `events` (`event_depth`
16-byte events), `read` (a 64-byte line, word 0 the events the thread took),
`command_depth`, `event_depth` (powers of two, at most 1,024), `flags`
(`KMX_RING_CACHED` 1: each flush or invalidate is a sweep; `KMX_RING_REMOTE` 2: `cmd` and `read`
are NoC addresses in the tile `reserved[0]` names, `x | y << 16`, inside one 2 MiB
window that the start-up opens in slot `KMX_RING_WINDOW` and closes after `kmx_main`),
`reserved[0]`, `buffer` (the cores' L1 for `kmx_fetch` and `kmx_store`: the run's
transfer buffer; 0: none), then `records`, `record_step` and `programs` for
`kmx_load` (program field `p`'s record at `records + p * record_step`, `p` below
`programs`; 0: none). Other addresses are as the thread's cluster sees them. A line that
names a ring but is out of range ends the start-up with error `args`. The session's
ring: 16 command lines and the read line in the dispatcher's LIM after its layout and
its upload region (`KMD_LIM_BYTES` + 512 KiB, `komorebi.program.thread.rings_at()`; cluster
`c`'s ring `c - 1` rings further), 64 events after the kmx
block in the thread's LIM
(`KMD_WORK_*` from `KMX_LIM_BYTES`). The protocol is the host ring's (the dispatcher's README,
"Protocol with the host").

## Python

`Program.thread(cluster=1, clusters=None, **kwargs)` declares the thread (one per
cluster) and returns a `komorebi.program.thread.Thread`; with `clusters`, a list of one per
cluster, the same fields on each:

| Field | Meaning |
|---|---|
| `sources` | C files, linked with the kmx library |
| `regions` | `{name: size in bytes or an array of initial contents}`; ids in dict order |
| `shared` | names of the program's shared regions the thread uses; ids after its own; with `regions`, 1 to 16 in all |
| `words` | up to 64 unsigned 32-bit words |
| `cluster` | the work cluster, 1 to 3 (bank 6 for 1, bank 7 for 2 and 3) |
| `host_fallback` | `function(regions, words, run) -> (code, words)`: the thread on the host, on `uint32` copies of the regions, submitting commands through `run` (the session's `Run`) where the thread uses its ring |
| `deadline` | seconds for `kmx_main` (a thread without `kmx_program`: the whole thread), at most 60 (default 10) |
| `handlers` | `kmx_program`'s handlers: `{name: number}` or a header path read as `Program`'s (`constexpr uint32_t kHandlerName = n;`), numbers 1 to 239; the C sees `kHandlerName` defined as `n` |
| `includes`, `defines` | extra include directories and `-D` definitions |
| `prefetch` | `"documented"` (default) or `"keep"` |

A program written as a class declares its thread in its body, with `km.region(bytes)`
for each region and `km.x280_thread(*sources, cluster=1, ...)` (the fields above but
`regions`, `words`, `shared` and `handlers`), and gives `thread_words=` at the make.
The thread is then made and built at the make, with the class's generated header
`<name>_thread.h`: a function per handler that fills its command, the regions' ids,
the class's constants, and the words the load gives the thread after its own (each
tensor of the make, its address). `program.thread(...)` on a made class does the same
for a thread declared after the make ([classes, X280 threads](classes.md#x280-threads)).

`Program.region(name, value)` declares a shared region (a size in bytes or an array of
initial contents). Every thread naming it must be on one bank, and every shared region
must be named by a thread; `session.load` rejects the program otherwise.

`Program.ports(serve=None, noc=0)` declares X280 ports ([X280 ports](../threads.md#x280-ports)):
`serve` `{core: (cluster, x280_core)}` (`x280_core` 0 to 3), or `None` for every core of the grid dealt over the
threads' X280 cores in grid order, cluster by cluster (`komorebi.program.thread.port_plan`); `noc`
the NoC of the X280 cores' stores into the cores. Once per program (`ValueError` again, for
`noc` other than 0 or 1, or an empty `serve`). `session.load` rejects ports with
`thread="host"` or without a thread, a core outside the grid, a cluster without a
thread and an X280 core outside 0 to 3. At the load it writes each served core's
`KM_X280_TO` line (its X280 core's tile and mailbox address) into the control blocks, and
every cluster's ports area before any thread starts.

`session.load(program, thread="card" | "host")` writes the regions before the grid
starts, then starts every image (or readies the fallbacks) once the dispatcher is
ready. `run.threads` holds a `ThreadRun` per cluster; `run.thread` is the lowest
cluster's:

| Member | Meaning |
|---|---|
| `run.thread_result(timeout)` | waits for the thread to return (asks it to stop at the timeout), stops it, returns a `ThreadResult`; with `thread="host"` runs the fallback; raises `ThreadFailed` as `thread_results` does when the thread did not return in time, failed or is `stuck` (its `results[cluster]` is the `ThreadResult`) |
| `run.thread_results(timeout)` | the same for every thread, `{cluster: ThreadResult}`; on the first that does not return in time or ends with a library error, stops the others and raises `ThreadFailed` (`cluster`, `results`; its message names a `stuck` thread) |
| `run.thread.read(name)` | the region's bytes (`uint8`), only after `run.stop()` |
| `run.thread.regions.address(name)` | `(bank, bank address)` for a Tensix kernel |
| `run.thread.write(name, data, offset=0, barrier=True)` | `data`'s bytes (uint8, or any array as its bytes) into the region from byte `offset` on, also while the run is loaded: a list the thread reads, written as it runs. With `barrier` they are in DRAM on return; without it, the next write with a barrier carries them. A range outside the region raises `ValueError`; a thread run with `thread="host"` raises `RuntimeError` |
| `run.thread.ring` | the `RingPlacement` the host wrote into the ring line (`None` with `thread="host"`) |
| `run.thread.records` | `(records, record_step, programs)` written into the ring line for `kmx_load` (`None` in a run whose groups cannot switch, and with `thread="host"`) |
| `run.stop()` | stops every thread still running, then the run |

`ThreadResult`: `ran` (false when it was stopped at the timeout), `state`, `error`
(names, as in the result block), `code`, `words`, `where`, `where_microseconds`, `microseconds`, `stuck`
(it did not return within 2.5 s of the stop flag either: a core outside the library or
halted by a trap) and `detail` (then the runtime's error message for a stop that failed).

`parse_result(raw)` is the `ThreadResult` of a result block's 256 bytes; `ring_bytes()`
the bytes of one work ring's command lines and read line in the dispatcher's local
memory, `ring_area_bytes(ring)` those of a thread's own memory after the kmx block that
a ring (a `RingPlacement`) uses.

`RingPlacement(cmd, consumed, events, read, command_depth, event_depth, cached=False,
remote=None)`, `work_ring(kernel_lim, kernel_coordinates, work_lim, work_coordinates, cluster=1)` (the
session's placement of one cluster's ring and the kernel's end of it),
`kernel_work_args(ends)` (the kernel's `work_*` words for every ring) and
`args_block(thread, base, check_value, run_value=0, ring=None, table=None,
peers=None, ports=None, records=None)` build the LIM block (`records` from
`load_records(switch)`, the run's records for `kmx_load`); `Peers(nonce, lead, tiles, banks)` is the peers line
(tiles and banks: `{cluster: x | y << 16}`) and `peer_groups({cluster: image})` the
clusters that run one image. They need no device; the session fills `Peers` from the
clusters' tiles and their banks' DRAM tiles for every thread on the card.
`BANK_OF_CLUSTER` is the DRAM bank each work cluster reaches through its memory port
(`{1: 6, 2: 7, 3: 7}` on a p150a, checked at the start-up), where its thread's regions
live; `TIMER_HZ` the X280 timer's ticks a second (50 MHz, `kmx_time()`).

## Measured

On work cluster 1 of a p150a at 1,750 MHz: the bank read at about 1.3 GB/s and written
at 1.3 to 1.8 GB/s with four cores; a flush or invalidate sweep at 5 to 8 GB/s of region
(`experiments/x280/x280_threads/`). At 1,000 MHz on each of clusters 1 to 3:
about 1.2 GB/s read and written per cluster, alone or together (three at once 3.5 GB/s);
a push into the other bank 2.3 GB/s, a pull 0.024 GB/s; invalidate 4.6 GB/s, flush 4.3 GB/s
(`experiments/x280/work_clusters/`).
An empty `kmx_parallel` call: 0.71 µs; `kmx_parallel_all` over three clusters, the map, heavy
map and prefix sum on 1, 4, 8 and 12 cores: [X280 threads](../threads.md#twelve-cores)
and `examples/08_parallel_thread/` (`--twelve`).
