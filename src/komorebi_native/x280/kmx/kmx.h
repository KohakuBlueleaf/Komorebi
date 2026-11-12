/* kmx: what an X280 thread on a work cluster is written against. The library's
 * start-up enters kmx_main on hart 0; harts 1 to 3 wait for kmx_par. Plain C. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Return codes of the calls that can fail. */
#define KMX_OK 0
#define KMX_FULL 1    /* the command ring has no free slot */
#define KMX_STOPPED 2 /* a stop was asked for, or the call is not available yet */
#define KMX_LATE 3    /* the deadline passed */

#define KMX_TICKS_PER_SECOND 50000000ull /* kmx_time() counts the 50 MHz timer */

/* An event of a command the thread issued: its ticket, kind, aux byte and ticks. */
typedef struct {
    uint32_t ticket, kind, aux, ticks;
} kmx_event;

/* The user's entry, on hart 0. Its return value is the result code unless the thread
 * called kmx_result. */
int kmx_main(int hart);

/* The argument block: region table and the program's words. */
const kmx_args* kmx_args_get(void);

/* Region `id` of the DRAM bank, as memory, and its size; 0 for an unknown id. */
void* kmx_region(int id, uint64_t* bytes);

/* Hand-over with the grid: take before reading what the grid wrote (drops the cached
 * lines), give after writing what the grid will read (writes them back). */
void kmx_take(int id);
void kmx_give(int id);

/* The command ring to the dispatch kernel. Not available yet: they return
 * KMX_STOPPED. */
int kmx_issue(const km_record_t* r, uint32_t* ticket);
int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev);
int kmx_poll(kmx_event* ev);

/* Hart 0 only: runs fn(hart, arg) on all four harts and returns when all are done:
 * KMX_OK, or KMX_LATE / KMX_STOPPED if the deadline passed or a stop came meanwhile. */
int kmx_par(void (*fn)(int hart, void* arg), void* arg);

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
