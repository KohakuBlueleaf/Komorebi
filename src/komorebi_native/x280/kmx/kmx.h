/* kmx: what an X280 thread on a work cluster is written against. The library's
 * start-up enters kmx_main on core 0; cores 1 to 3 wait for kmx_spawn. Plain C. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define KMX_TICKS_PER_SECOND 50000000ull /* kmx_time() counts the 50 MHz timer */

/* The user's entry, on core 0. Its return value is the result code unless the thread
 * called kmx_result. */
int kmx_main(int core);

/* The argument block: region table and the program's words. */
const kmx_args* kmx_args_get(void);

/* The work cluster this thread runs on, 1 to 3 (the same image may run on several). */
int kmx_cluster(void);

/* Region `id` of the DRAM bank, as memory, and its size; 0 for an unknown id. */
void* kmx_region(int id, uint64_t* bytes);

/* Cache flush / invalidate with the grid: invalidate before reading what the grid
 * wrote, flush after writing what it will read; core 0 spreads it over the four. */
void kmx_invalidate(int id);
void kmx_flush(int id);

/* The same for `bytes` of region `id` from `offset` (whole lines around them). */
void kmx_invalidate_at(int id, uint64_t offset, uint64_t bytes);
void kmx_flush_at(int id, uint64_t offset, uint64_t bytes);

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

/* `log`: the log's region id (each cluster its own copy, KMX_PULL_LOG_BYTES). `pick`,
 * when set, gives core's index-th range (or KMX_PULL_NONE) instead of the counter. */
typedef struct {
    uint32_t units, per, mode, log;
    uint32_t (*pick)(int core, uint32_t index, uint64_t arg);
    uint64_t arg;
} kmx_pull;

/* A log region: a count per core, then `per_core` range numbers per core. */
#define KMX_PULL_LOG_BYTES(per_core) (4u * KMX_PORTS_CORES * (1u + (per_core)))

/* X280 core 0 of the lead: submits `command`, serves its pulls on every X280 core until
 * its event, as kmx_port_run; KMX_BAD also when it ran but a log ran out of room. */
int kmx_pull_run(
    const km_command_t* command, const kmx_pull* settings, kmx_event* event
);

/* Timer ticks since an arbitrary point (KMX_TICKS_PER_SECOND per second). */
uint64_t kmx_time(void);

/* Nonzero once the host or the runtime asked the thread to stop, or the run's
 * deadline passed: the thread should then return. */
int kmx_stop_requested(void);

/* A progress marker the host reads with the result, also after a timeout. */
void kmx_progress(uint32_t where);

/* The thread's result: a code and up to KMX_RESULT_WORDS_MAX words. The thread then
 * returns from kmx_main. */
void kmx_result(uint32_t code, const void* words, uint32_t word_count);

#if defined(__cplusplus)
}
#endif
