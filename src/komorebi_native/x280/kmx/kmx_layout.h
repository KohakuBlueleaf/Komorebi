/* The layout an X280 thread shares with the host: its block in the work cluster's
 * local memory (LIM), the argument block, the region table and the result block. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define KMX_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KMX_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

#define KMX_MAGIC 0x31786D6Bu /* "kmx1" */
#define KMX_VERSION 1u
#define KMX_REGIONS_MAX 16u
#define KMX_WORDS_MAX 64u              /* the program's own argument words */
#define KMX_RESULT_WORDS_MAX 48u       /* words a thread returns */
#define KMX_DEADLINE_MAX 3000000000ull /* 60 s of the 50 MHz timer */

/* LIM layout, in bytes from the cluster's available LIM base. The host writes the
 * argument block and the stop line as whole 64-byte lines; the rest is the thread's. */
#define KMX_LIM_ARGS 0x000u   /* kmx_args (host, before the start) */
#define KMX_LIM_STOP 0x240u   /* line from the host: word 0 = KMX_STOP_WORD */
#define KMX_LIM_RESULT 0x280u /* kmx_result_block (the thread) */
#define KMX_LIM_PAR 0x380u    /* hart 0's kmx_par mailbox */
#define KMX_LIM_DONE 0x3C0u   /* hart h's last finished kmx_par at + 8 * h */
#define KMX_LIM_BYTES 0x400u
#define KMX_STOP_WORD 0x53544F50u /* "STOP" */

/* Argument block offsets, in bytes. */
#define KMX_ARGS_REGIONS 0x040u
#define KMX_ARGS_WORDS 0x140u
#define KMX_ARGS_BYTES 0x240u

/* One region of the cluster's DRAM bank: its bank address (a multiple of 64) and
 * size. The thread sees it at X280_LOCAL_DDR_TILE_BASE + offset. */
typedef struct {
    uint64_t offset;
    uint64_t bytes;
} kmx_region_t;

typedef struct {
    uint64_t run_value;   /* this run's value: the block is stale otherwise */
    uint32_t magic;       /* KMX_MAGIC */
    uint32_t version;     /* KMX_VERSION */
    uint64_t deadline;    /* timer ticks the whole thread may take, 1 to the maximum */
    uint32_t nregions;    /* entries of region[] in use */
    uint32_t nwords;      /* entries of words[] in use */
    uint64_t check;       /* bank address of the check line (the bank check) */
    uint64_t check_value; /* what the host wrote at the start of that line */
    uint64_t reserved[2];
    kmx_region_t region[KMX_REGIONS_MAX];
    uint32_t words[KMX_WORDS_MAX];
} kmx_args;

/* States, in the result block. */
#define KMX_STATE_BOOT 1u    /* start-up, before kmx_main */
#define KMX_STATE_RUNNING 2u /* in kmx_main */
#define KMX_STATE_DONE 3u    /* kmx_main returned, no library error */
#define KMX_STATE_FAILED 4u  /* a library error: see error */

/* Library errors, in the result block (the thread's own code is separate). */
#define KMX_ERR_NONE 0u
#define KMX_ERR_ARGS 1u    /* the argument block is not this run's, or out of range */
#define KMX_ERR_BANK 2u    /* the check line does not hold check_value: wrong bank */
#define KMX_ERR_LATE 3u    /* a wait reached the run's deadline */
#define KMX_ERR_STOPPED 4u /* the host or the runtime asked for a stop */
#define KMX_ERR_PAR 5u     /* kmx_par called from a hart other than 0 */
#define KMX_ERR_RESULT 6u  /* kmx_result with more than KMX_RESULT_WORDS_MAX words */

typedef struct {
    uint32_t state;   /* KMX_STATE_* */
    uint32_t error;   /* KMX_ERR_* (the first one) */
    uint32_t code;    /* the thread's code: kmx_result's, else kmx_main's return */
    uint32_t n;       /* result words */
    uint32_t where;   /* the last kmx_progress stamp */
    uint32_t t_where; /* timer ticks from the start to that stamp */
    uint32_t ticks;   /* timer ticks from the start to the end of kmx_main */
    uint32_t reserved[9];
    uint32_t words[KMX_RESULT_WORDS_MAX];
} kmx_result_block;

KMX_STATIC_ASSERT(offsetof(kmx_args, deadline) == 16, "kmx_args.deadline at 16");
KMX_STATIC_ASSERT(offsetof(kmx_args, check) == 32, "kmx_args.check at 32");
KMX_STATIC_ASSERT(offsetof(kmx_args, check_value) == 40, "check_value at 40");
KMX_STATIC_ASSERT(offsetof(kmx_args, region) == KMX_ARGS_REGIONS, "region table");
KMX_STATIC_ASSERT(offsetof(kmx_args, words) == KMX_ARGS_WORDS, "argument words");
KMX_STATIC_ASSERT(sizeof(kmx_args) == KMX_ARGS_BYTES, "kmx_args size");
KMX_STATIC_ASSERT(KMX_LIM_ARGS + KMX_ARGS_BYTES <= KMX_LIM_STOP, "LIM: args");
KMX_STATIC_ASSERT(sizeof(kmx_result_block) == 256, "kmx_result_block size");
KMX_STATIC_ASSERT(KMX_LIM_RESULT + 256 <= KMX_LIM_PAR, "LIM: result");
KMX_STATIC_ASSERT(KMX_LIM_DONE + 32 <= KMX_LIM_BYTES, "LIM: kmx_par");
