/* Komorebi ABI: the definitions shared by the Tensix kernels, the X280 dispatch kernel
   and the host. Plain C, usable from C and C++. */
#ifndef KOMOREBI_ABI_KM_ABI_H
#define KOMOREBI_ABI_KM_ABI_H

#include <stdint.h>

#define KM_RING_SLOTS 16u /* D: mailbox slots, 64 bytes each */
#define KM_CORES_MAX 128u
#define KM_SEG_NOP 0u
#define KM_SEG_EXIT 255u /* every core leaves its kernels */
#define KM_SEG_PING 254u /* host -> dispatch kernel only: answer with an event */
#define KM_SEG_STOP 253u /* host -> dispatch kernel only: stop the grid and finish */
/* host -> dispatch kernel only: a0 = microseconds for later records */
#define KM_SEG_DEADLINE 252u
#define KM_SEG_USER_MAX 239u /* segments 1..239 belong to the program */
/* Flags: bit 0 is reserved and zero. Every record reports completion (revision 2). */
#define KM_TEAM_ALL(n) ((uint32_t)(n) << 16) /* first 0, count n */

/* One record: four 16-byte chunks. Word 0 of each chunk carries the sequence number;
   a record has landed when all four agree (KM_LANDED). */
#define KM_LANDED(r) \
    ((r)->land1 == (r)->seq && (r)->land2 == (r)->seq && (r)->land3 == (r)->seq)

typedef struct {
    uint32_t seq;  /* chunk 0: sequence number, starts at 1, increases */
    uint32_t hdr;  /* seg | flags << 8 | slot << 16 */
    uint32_t team; /* first | count << 16: cores [first, first + count) in core order */
    uint32_t a0;
    uint32_t land1, a1, a2, a3; /* chunk 1 */
    uint32_t land2, a4, a5, a6; /* chunk 2 */
    uint32_t land3, a7, a8, a9; /* chunk 3 */
} km_record_t;                  /* sizeof == 64; ten argument words */

/* Control block in L1: the same address on every core. Offsets in bytes. */
#define KM_CTL_RING 0x000u /* KM_RING_SLOTS records */
#define KM_CTL_RSVD 0x400u /* reserved, 16 bytes; not written in this version */
#define KM_CTL_STOP 0x410u /* nonzero: every wait gives up */
#define KM_CTL_DONE 0x440u /* KM_RING_SLOTS counters, 16 bytes apart: members done */
#define KM_CTL_BOOT 0x540u /* start-up check-in area (column groups, as today) */
/* per core: last sequence consumed per slot, gave-up code, where code */
#define KM_CTL_STATE 0x640u
/* the record being run, for the compute and writer kernels */
#define KM_CTL_JOB 0x700u
#define KM_CTL_OUT 0x740u   /* scratch for outgoing 4-byte NoC writes */
#define KM_CTL_CORES 0x780u /* core index -> NoC x | y << 16, KM_CORES_MAX words */
#define KM_CTL_BYTES 0x980u

/* Words the Tensix grid pushes into the X280's local memory. */
#define KM_ACK_STRIDE 16u /* ack word of slot j at ack_base + j * KM_ACK_STRIDE */
#define KM_ACK(seq) ((uint32_t)(seq)) /* the whole sequence number (revision 2) */
#define KM_READY 0xFFFFFFFFu          /* READY words are not in the ack area */

/* Host -> dispatch kernel: one command is one 64-byte record in the host ring (revision
   2). The host puts the command's number (from 1) in `seq` and in the three landing
   words; the kernel replaces them with its own sequence, fills the slot, and stores the
   record to the grid. Segments 252..254 are commands to the kernel itself and are not
   forwarded. */

/* Dispatch kernel -> host: one 16-byte event per finished command, in a ring in host
   memory. */
#define KM_EV_DONE 0u
#define KM_EV_TIMEOUT 1u /* aux = slot; the cores are named by their state words */
#define KM_EV_PONG 2u

typedef struct {
    uint32_t index;  /* event number, starts at 1: the ring slot is index % depth */
    uint32_t req;    /* the host's command number */
    uint32_t status; /* KM_EV_* | aux << 8 */
    uint32_t ticks;  /* issue to completion, in the X280's timer ticks */
} km_event_t;        /* sizeof == 16: one chunk, lands whole */

#endif /* KOMOREBI_ABI_KM_ABI_H */
