/* An X280 core as a rank of a dynamic run: what the grid's kernels and the X280 thread
 * share (README.md). Plain C. */
#pragma once

#include <stdint.h>

/* Handlers. RUN is the program's dynamic handler: a0 round trips per rank core, a1 the
 * messages a rank core keeps in flight, a2 XR_MODE_*, a3 the stats page. */
#define XR_HANDLER_RUN 1u
#define XR_HANDLER_FINISH 2u /* to the stand-ins: their X280 cores leave */
#define XR_MODE_RANK 1u /* messages through the X280 core's queue and the inboxes */
#define XR_MODE_ASK 2u  /* the same round trips as port asks */

/* Tells: the X280 core to its cores {HELLO, its queue's bank address, the bank's tile};
 * a stand-in to its X280 core {START, run, mode}, {END, run}, {FINISH}. */
#define XR_TELL_HELLO 0x48454C4Fu /* "HELO" */
#define XR_TELL_START 1u
#define XR_TELL_END 2u
#define XR_TELL_FINISH 3u

/* An X280 core's area in its region: its queue (XR_QUEUE messages of 64 bytes), then a
 * credit line per core (word 0: the core's inbox head). */
#define XR_QUEUE 1024u
#define XR_CREDITS (XR_QUEUE * 64u)
#define XR_AREA 0x20000u
#define XR_REGION_BYTES (4u * XR_AREA)
/* messages of a core's inbox in its L1, written by its X280 core */
#define XR_INBOX 64u

/* A message: 16 words, word 0 its mark (position + 1). */
#define XR_W_MARK 0u
#define XR_W_FROM 1u  /* the sender: a core, or 0x100 | cluster << 2 | X280 core */
#define XR_W_INDEX 2u /* the round trip's number on its core */
#define XR_W_VALUE 3u
#define XR_W_CHECK 4u
#define XR_VALUE(core, k) ((((core) << 20) ^ (k)) * 2654435761u)
#define XR_REPLY(value, k) ((((value) ^ 0x5BD1E995u) * 0x9E3779B1u) + (k))
#define XR_CHECK(from, k, value) \
    (((from) * 0x85EBCA6Bu) ^ ((k) * 0xC2B2AE35u) ^ (value) ^ 0x27D4EB2Fu)
#define XR_ANSWER(k, core) (((k) * 0x9E3779B1u) ^ ((core) << 8) ^ 0x5A5A0000u)

/* A core's work area in L1: the tail line of a stand-in (its X280 core's queue), the
 * reservation's returned slot, the credit line, then XR_INBOX staged messages. */
#define XR_WORK_TAIL 0u
#define XR_WORK_RETURNED 16u
#define XR_WORK_CREDIT 64u
#define XR_WORK_STAGE 128u
#define XR_WORK_BYTES (XR_WORK_STAGE + 64u * XR_INBOX)

/* A core's stats page (32-bit words), one per RUN's a3. */
#define XR_PAGE_BYTES 1024u
#define XR_PAGES 4u
#define XR_ST_COUNT 0u /* timed round trips */
#define XR_ST_BAD 1u   /* replies or answers that differ */
#define XR_ST_MIN 2u   /* cycles */
#define XR_ST_MAX 3u
#define XR_ST_SUM_LO 4u
#define XR_ST_SUM_HI 5u
#define XR_ST_SENT 6u /* the run's counters as the status line had them */
#define XR_ST_CONSUMED 7u
#define XR_ST_GAVE 8u        /* nonzero: where a wait gave up (XR_GAVE_*) */
#define XR_ST_SEND_CYCLES 9u /* in the sends: reserve, write, commit */
#define XR_ST_SENDS 10u      /* reservations */
#define XR_ST_FIRST_LO 11u   /* wall clock of the first send and the last reply */
#define XR_ST_FIRST_HI 12u
#define XR_ST_LAST_LO 13u
#define XR_ST_LAST_HI 14u
#define XR_ST_EXTRA 15u   /* messages after the last round trip */
#define XR_ST_CREDITS 16u /* credit lines written */
#define XR_ST_RUN 17u     /* the run's sequence */
#define XR_ST_DONE 18u    /* round trips done */
#define XR_ST_IDLES 19u   /* idle lines written */
#define XR_ST_HIST 32u
#define XR_HIST_BINS 192u
#define XR_GAVE_HELLO 1u   /* no HELLO from the X280 core */
#define XR_GAVE_ATOMIC 2u  /* a reservation did not answer */
#define XR_GAVE_WRITE 3u   /* a write did not flush */
#define XR_GAVE_REPLY 4u   /* no reply within the wait limit */
#define XR_GAVE_ASK 5u     /* no answer to an ask */
#define XR_GAVE_STOPPED 6u /* the stop flag */
#define XR_GAVE_LINE 7u    /* a status line did not flush */

/* Thread words. */
#define XR_WORD_INBOX 0u /* the inboxes' L1 address (the same on every core) */
#define XR_WORD_IDLE 1u  /* ticks without a message before the line says idle */
/* + c - 1: cluster c's ring commands minus the status lines */
#define XR_WORD_BACK 2u
#define XR_WORD_BANK 5u    /* + c - 1: the tile of cluster c's bank */
#define XR_WORD_STANDIN 8u /* + 4 (c - 1) + h: X280 core h of cluster c's stand-in */
#define XR_WORDS 20u
#define XR_NONE 0xFFFFFFFFu

/* Result words of a cluster, per X280 core h at XR_RES_CORE + XR_RES_STRIDE h. */
#define XR_RES_CORE 0u
#define XR_RES_CONSUMED 0u
#define XR_RES_SENT 1u
#define XR_RES_BAD 2u
#define XR_RES_GAVE 3u
#define XR_RES_ASKS 4u
#define XR_RES_TAKE 5u  /* cycles / 16 taking messages and storing replies */
#define XR_RES_POLLS 6u /* looks at the queue that found nothing */
#define XR_RES_RUNS 7u
#define XR_RES_STORE 8u /* cycles / 16 of those storing replies, fences included */
#define XR_RES_TAKES 9u /* looks that found messages */
#define XR_RES_STRIDE 10u
#define XR_RES_WORDS 40u
#define XR_X280_GAVE_REGION 1u
#define XR_X280_GAVE_WINDOW 2u
#define XR_X280_GAVE_CREDIT 3u
/* a message from a core this X280 core does not serve */
#define XR_X280_GAVE_FOREIGN 4u
