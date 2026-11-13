/* kmx: what an X280 thread on a work cluster is written against. The library's
 * start-up enters kmx_main on hart 0; harts 1 to 3 wait for kmx_spawn. Plain C. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define KMX_TICKS_PER_SECOND 50000000ull /* kmx_time() counts the 50 MHz timer */

/* The user's entry, on hart 0. Its return value is the result code unless the thread
 * called kmx_result. */
int kmx_main(int hart);

/* The argument block: region table and the program's words. */
const kmx_args* kmx_args_get(void);

/* The work cluster this thread runs on, 1 to 3 (the same image may run on several). */
int kmx_cluster(void);

/* Region `id` of the DRAM bank, as memory, and its size; 0 for an unknown id. */
void* kmx_region(int id, uint64_t* bytes);

/* Hand-over with the grid: take before reading what the grid wrote, give after
 * writing what it will read; on hart 0 the sweep is spread over the four harts. */
void kmx_take(int id);
void kmx_give(int id);

/* The same for `bytes` of region `id` from `offset` (whole lines around them). */
void kmx_take_at(int id, uint64_t offset, uint64_t bytes);
void kmx_give_at(int id, uint64_t offset, uint64_t bytes);

/* The command ring to the dispatch kernel (hart 0; KMX_STOPPED without a ring).
 * issue: KMX_OK and the ticket, or KMX_FULL when no slot is free. */
int kmx_issue(const km_record_t* r, uint32_t* ticket);

/* The event of `ticket`, by kmx_time() == `deadline` at the latest (KMX_LATE), or a
 * stop; events of other tickets are kept for kmx_poll, up to KMX_STASH (KMX_FULL). */
int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev);

/* One event if any (a kept one first): KMX_OK, or KMX_NONE. Never waits. */
int kmx_poll(kmx_event* ev);

/* Hart 0: starts fn(hart, arg) on hart 1, 2 or 3 and returns. KMX_OK; KMX_FULL while
 * that hart has not finished its last spawn; KMX_BAD for another hart. */
int kmx_spawn(int hart, void (*fn)(int hart, void* arg), void* arg);

/* Hart 0: waits until `hart` finished its last spawn (at once if none). KMX_OK, or
 * KMX_LATE / KMX_STOPPED (library error late / stopped); KMX_BAD for another hart. */
int kmx_join(int hart);

/* Hart 0, not nested: spawns fn on harts 1 to 3, runs fn(0, arg), joins the three:
 * KMX_OK, KMX_FULL if one of them is still busy, or KMX_LATE / KMX_STOPPED. */
int kmx_par(void (*fn)(int hart, void* arg), void* arg);

/* The clusters running this image in the program (bit c for cluster c), and the one
 * that leads them: it calls kmx_par_all, the others kmx_serve. */
uint32_t kmx_clusters(void);
int kmx_lead(void);

/* The lead's hart 0, not nested: runs fn(cluster, hart, arg) on every hart of every
 * cluster of kmx_clusters() and waits for all. KMX_OK, or KMX_LATE / KMX_STOPPED. */
int kmx_par_all(void (*fn)(int cluster, int hart, uint64_t arg), uint64_t arg);

/* Hart 0 of a cluster that does not lead: runs the lead's kmx_par_all calls until the
 * lead's kmx_main returns (KMX_OK), or KMX_LATE / KMX_STOPPED; KMX_NONE on the lead. */
int kmx_serve(void);

/* Any hart: stores `bytes` from `src` at `offset` of `cluster`'s region `id` (64-byte
 * multiples; else KMX_BAD); the receiver takes them (kmx_take_at) before reading. */
int kmx_push(int cluster, int id, uint64_t offset, const void* src, uint64_t bytes);

/* Timer ticks since an arbitrary point (KMX_TICKS_PER_SECOND per second). */
uint64_t kmx_time(void);

/* Nonzero once the host or the runtime asked the thread to stop, or the run's
 * deadline passed: the thread should then return. */
int kmx_stop_requested(void);

/* A stamp the host reads with the result, also after a timeout. */
void kmx_progress(uint32_t where);

/* The thread's result: a code and up to KMX_RESULT_WORDS_MAX words. The thread then
 * returns from kmx_main. */
void kmx_result(uint32_t code, const void* words, uint32_t n);

#if defined(__cplusplus)
}
#endif
