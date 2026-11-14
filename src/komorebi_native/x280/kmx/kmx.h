/* kmx: what an X280 thread on a work cluster is written against. The library's
 * start-up enters kmx_main on core 0; cores 1 to 3 wait for kmx_spawn. Plain C. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"
#include "komorebi/abi/km_copy.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define KMX_TICKS_PER_SECOND 50000000ull /* kmx_time() counts the 50 MHz timer */

/* The user's entry, on core 0. Its return value is the result code unless the thread
 * called kmx_result. Optional when the thread has kmx_program. */
int kmx_main(int core);

/* Optional: after kmx_main, if any, core 0 calls it for each command to this cluster
 * until a stop (no deadline then); its word is the result of the command's event. */
uint32_t kmx_program(const km_command_t* command);

/* A command's handler, for kmx_program's switch. */
static inline uint32_t kmx_handler(const km_command_t* command) {
    return command->hdr & 0xFFu;
}

/* The argument block: region table and the program's words. */
const kmx_args* kmx_args_get(void);

/* The work cluster this thread runs on, 1 to 3 (the same image may run on several). */
int kmx_cluster(void);

/* The island this thread's cores are on, and the run's islands (0 and 1 on one). */
int kmx_island(void);
int kmx_islands(void);

/* Thread numbers are flat over islands: island i's cluster c is KMX_THREAD(i, c); a
 * command's group (KM_GROUP_CLUSTER_BASE + number, 1) names it, and crosses if far. */
#define KMX_THREAD(island, cluster) (3u * (uint32_t)(island) + (uint32_t)(cluster))

/* The island of thread `number` (1 to 3 * kmx_islands()), or -1 for none. */
int kmx_island_of(uint32_t number);

/* Region `id` of the DRAM bank, as memory, and its size; 0 for an unknown id. */
void* kmx_region(int id, uint64_t* bytes);

/* Cache flush / invalidate with the grid: invalidate before reading what the grid
 * wrote, flush after writing what it will read; core 0 spreads it over the four. */
void kmx_invalidate(int id);
void kmx_flush(int id);

/* The same for `bytes` of region `id` from `offset` (whole lines around them). */
void kmx_invalidate_at(int id, uint64_t offset, uint64_t bytes);
void kmx_flush_at(int id, uint64_t offset, uint64_t bytes);

/* Any core: one cache way of the cached Zero Device read, every line of the cluster's
 * cache replaced: dirty lines written back, every region's lines dropped. */
void kmx_evict(void);

/* Fills `command`: `handler` of `program` (kProgramName in a run of several programs,
 * else 0) on cores [first, first + count), arguments 0. */
void kmx_command(
    km_command_t* command,
    uint32_t program,
    uint32_t handler,
    uint32_t first,
    uint32_t count
);

/* Makes `bytes` (to KM_BLOCK_BYTES) of region `id` at `offset` (64-aligned) the block
 * of `command` (a7, a8): KMX_OK, or KMX_BAD outside it. Flush before submitting. */
int kmx_block(km_command_t* command, int id, uint64_t offset, uint32_t bytes);

/* Core 0: cores [first, first + count) hold `program` once KMX_OK (sent again after a
 * busy rejection, by `deadline`); KMX_BAD: refused (`event`, may be 0) or no record. */
int kmx_load(
    uint32_t program,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
);

/* The command ring to the dispatcher (core 0; KMX_STOPPED without a ring).
 * submit: KMX_OK and the handle, or KMX_FULL when no slot is free. */
int kmx_submit(const km_command_t* command, uint32_t* handle);

/* The event of `handle`, by kmx_time() == `deadline` at the latest (KMX_LATE), or a
 * stop; events of other handles are kept for kmx_poll, up to KMX_STASH (KMX_FULL). */
int kmx_wait(uint32_t handle, uint64_t deadline, kmx_event* event);

/* One event if any (a kept one first): KMX_OK, or KMX_NONE. Never waits. */
int kmx_poll(kmx_event* event);

/* Core 0: starts function(core, arg) on core 1, 2 or 3 and returns. KMX_OK; KMX_FULL
 * while that core has not finished its last spawn; KMX_BAD for another core. */
int kmx_spawn(int core, void (*function)(int core, void* arg), void* arg);

/* Core 0: waits until `core` finished its last spawn (at once if none). KMX_OK, or
 * KMX_LATE / KMX_STOPPED (library error late / stopped); KMX_BAD for another core. */
int kmx_join(int core);

/* Core 0, not nested: spawns function on cores 1 to 3, runs function(0, arg), joins the
 * three: KMX_OK, KMX_FULL if one of them is still busy, or KMX_LATE / KMX_STOPPED. */
int kmx_parallel(void (*function)(int core, void* arg), void* arg);

/* The clusters running this image in the program (bit c for cluster c), and the one
 * that leads them: it calls kmx_parallel_all, the others kmx_serve. */
uint32_t kmx_clusters(void);
int kmx_lead(void);

/* The lead's core 0, not nested: runs function(cluster, core, arg) on every core of
 * every cluster of kmx_clusters(), waits. KMX_OK, or KMX_LATE / KMX_STOPPED. */
int kmx_parallel_all(
    void (*function)(int cluster, int core, uint64_t arg), uint64_t arg
);

/* Core 0 of a non-lead cluster: runs the lead's kmx_parallel_all calls until the lead's
 * kmx_main returns (KMX_OK), or KMX_LATE / KMX_STOPPED; KMX_NONE on the lead. */
int kmx_serve(void);

/* Any core: stores `bytes` from `source` at `offset` of `cluster`'s region `id`
 * (64-byte multiples, else KMX_BAD); the receiver calls kmx_invalidate_at first. */
int kmx_push(int cluster, int id, uint64_t offset, const void* source, uint64_t bytes);

/* Pages in DRAM a copy names: a range of one bank (`bank` 0 to 7, `page` 0 and `first`
 * 0 for the library's choice) or a tensor's pages (`bank` KM_COPY_INTERLEAVED). */
typedef struct {
    uint32_t address; /* a range's bank address; a tensor's address */
    uint32_t bank;
    uint32_t page;  /* a tensor's aligned page bytes */
    uint32_t first; /* the first page */
} kmx_dram;

/* Core 0: cores [first, first + count) copy `bytes` from `from` to region `id` at
 * `offset` (64-byte multiples) by `deadline`; the range's lines dropped before it. */
int kmx_fetch(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
);

/* The same with no sweep: the caller says no line of the range is in the cluster's
 * cache (not read or written since the load, its last flush or the last fetch). */
int kmx_fetch_clean(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
);

/* Core 0: the region's lines written back, then copied out to `to`. KMX_OK; KMX_BAD (a
 * bound, a refusal, a timeout: `event`, may be 0); KMX_LATE / KMX_STOPPED. */
int kmx_store(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
);

/* The same, submitted and not waited for: KMX_OK and the handle for kmx_wait (after a
 * fetch's event, kmx_invalidate_at of the range); KMX_FULL; KMX_BAD for a bound. */
int kmx_fetch_submit(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
);
int kmx_store_submit(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
);

/* X280 ports: a sequence number and three words with the grid's cores, fixed to X280
 * cores at load. A call names a core the calling X280 core serves (else KMX_BAD). */
typedef struct {
    uint32_t sequence, w[3];
} kmx_message;

/* Entries of the port tables, the grid's cores (0: the program declared no ports). */
int kmx_port_cores(void);

/* The X280 core of this cluster that serves `core`, or -1 (another cluster's, or none).
 */
int kmx_port_owner(int core);

/* Tells `core` three words; the core sees the newest (latest wins). Never waits. */
int kmx_port_tell(int core, uint32_t first, uint32_t second, uint32_t third);

/* `core`'s newest ask if it is not answered yet: KMX_OK and `message`, else KMX_NONE.
 */
int kmx_port_asked(int core, kmx_message* message);

/* Answers `core`'s ask `sequence` with three words. Never waits. */
int kmx_port_answer(
    int core, uint32_t sequence, uint32_t first, uint32_t second, uint32_t third
);

/* `core`'s newest tell if it is newer than the last one looked at: KMX_OK and
 * `message`, else KMX_NONE (tells in between are lost: latest wins). */
int kmx_port_told(int core, kmx_message* message);

/* X280 core 0 of the lead: submits command, runs function as kmx_parallel_all does
 * until its event: KMX_OK and `event` once all returned; FULL, LATE, STOPPED, BAD. */
int kmx_port_run(
    const km_command_t* command,
    void (*function)(int cluster, int x280_core, uint64_t arg),
    uint64_t arg,
    kmx_event* event
);

/* Inside kmx_port_run's function: 1 while its command runs, 0 once its event came (or a
 * stop, or the deadline). function then returns and stores into no core any more. */
int kmx_port_open(void);

/* The pull: kmx_pull_run hands units 0 to units - 1 out in ranges of `per` (range j:
 * units j * per on) to the cores that ask (km::X280Port::next), each range once. */
#define KMX_PULL_COUNT 0u  /* a counter per cluster: cluster i of n hands i, i + n.. */
#define KMX_PULL_LOG 1u    /* the same, and each core's ranges logged in order */
#define KMX_PULL_REPLAY 2u /* each core gets the ranges the log holds for it */
#define KMX_PULL_NONE 0xFFFFFFFFu

/* `log`: the log's region id (each cluster its own copy, KMX_PULL_LOG_BYTES). `pick`
 * or `pick_word` (one at most): core's index-th range instead of the counter. */
typedef struct {
    uint32_t units, per, mode, log;
    uint32_t (*pick)(int core, uint32_t index, uint64_t arg); /* the range or NONE */
    uint64_t arg;
    uint64_t (*pick_word)(int core, uint32_t index, uint64_t arg); /* KMX_PULL_WORD */
} kmx_pull;

/* pick_word's answer: the range (or KMX_PULL_NONE) and the core's word with it. */
#define KMX_PULL_WORD(range, word) \
    ((uint64_t)(uint32_t)(range) | (uint64_t)(uint32_t)(word) << 32)

/* A log region: a count per core, then `per_core` range numbers per core. */
#define KMX_PULL_LOG_BYTES(per_core) (4u * KMX_PORTS_CORES * (1u + (per_core)))

/* X280 core 0 of the lead: submits `command`, serves its pulls on every X280 core until
 * its event, as kmx_port_run; KMX_BAD also when it ran but a log ran out of room. */
int kmx_pull_run(
    const km_command_t* command, const kmx_pull* settings, kmx_event* event
);

/* X280 ranks of a dynamic run: X280 core h < kmx_ranks() is rank count + h after the
 * run's `count` core ranks. A rank takes the messages of its queue, sends to cores. */
typedef struct {
    uint32_t run;            /* the RUN command's sequence */
    uint32_t rank;           /* this X280 core's rank */
    uint32_t count;          /* the core ranks: the group without its counter tiles */
    uint32_t first;          /* the group's first core */
    uint32_t handler;        /* of the message taken last; before it the RUN's */
    uint32_t quota;          /* sends left */
    uint32_t sent, consumed; /* this run's messages, as its status line has them */
    uint32_t gave;           /* KMX_GAVE_*: why this rank stopped (0: it did not) */
    /* the library's: the queue's head, the messages of the take not yet handed out,
     * the next of them, whether the line says idle, the last time there was work */
    uint32_t head, left, at, idle;
    uint64_t last;
} kmx_messages;

/* An X280 rank's give-up, word 3 of its status line (word 2 its queue's head). */
#define KMX_GAVE_ROOM 1u    /* a core's inbox had no room within the wait */
#define KMX_GAVE_RANK 2u    /* forward or send named no core rank */
#define KMX_GAVE_STOPPED 3u /* a stop, or the thread's deadline */
#define KMX_GAVE_LINE 4u    /* the status line's window could not be opened */

/* X280 ranks of this cluster (0: the program declares none). */
int kmx_ranks(void);

/* Core 0, not nested: waits for the next dynamic run (by `deadline`), then runs
 * function(x280_core, messages, arg) on each X280 rank; KMX_OK once all returned. */
int kmx_messages_run(
    void (*function)(int x280_core, kmx_messages* messages, void* arg),
    void* arg,
    uint64_t deadline
);

/* The next message of this rank's queue (message_bytes / 4 words, word 0 the
 * runtime's), valid until the next call; 0 once the run ended or this rank gave up. */
const volatile uint32_t* kmx_messages_next(kmx_messages* messages);

/* The message goes on at core rank `rank`, handled by `handler` there: KMX_OK, or
 * KMX_BAD (gave up: no such core rank, or no room in its inbox within the wait). */
int kmx_messages_forward(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
);

/* A new message, as forward: KMX_FULL with no quota left (nothing sent). */
int kmx_messages_send(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
);

/* The message taken last ends here: one send returns to the quota. */
void kmx_messages_consume(kmx_messages* messages);

/* Timer ticks since an arbitrary point (KMX_TICKS_PER_SECOND per second). */
uint64_t kmx_time(void);

/* Nonzero once the host or the runtime asked the thread to stop, or the run's
 * deadline passed (never while serving commands): the thread should then return. */
int kmx_stop_requested(void);

/* A progress marker the host reads with the result, also after a timeout. */
void kmx_progress(uint32_t where);

/* The thread's result: a code and up to KMX_RESULT_WORDS_MAX words. The thread then
 * returns from kmx_main. */
void kmx_result(uint32_t code, const void* words, uint32_t word_count);

#if defined(__cplusplus)
}
#endif
