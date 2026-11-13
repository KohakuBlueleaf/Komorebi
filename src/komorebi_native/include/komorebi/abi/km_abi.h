/* Komorebi ABI: the definitions shared by the Tensix kernels, the X280 dispatcher
   and the host. Plain C, usable from C and C++. */
#ifndef KOMOREBI_ABI_KM_ABI_H
#define KOMOREBI_ABI_KM_ABI_H

#include <stdint.h>

#if defined(__cplusplus)
#define KM_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KM_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

#define KM_RING_SLOTS 16u /* D: mailbox slots, 64 bytes each */
#define KM_CORES_MAX 128u
#define KM_HANDLER_NOP 0u
#define KM_HANDLER_EXIT 255u /* every core leaves its kernels */
#define KM_HANDLER_PING 254u /* host -> dispatcher only: answer with an event */
#define KM_HANDLER_STOP 253u /* host -> dispatcher only: stop the grid and finish */
/* host -> dispatcher only: a0 = microseconds for later commands */
#define KM_HANDLER_DEADLINE 252u
#define KM_HANDLER_USER_MAX 239u /* handlers 1..239 belong to the program */
/* host -> dispatcher only: a0 = offset of a stream upload, a1 = its bytes */
#define KM_HANDLER_RUN 251u
/* host -> dispatcher only: a0 = request id of the stream to cancel */
#define KM_HANDLER_CANCEL 250u
/* Flags: bit 0 is reserved and zero. Every command reports completion. */
/* dispatcher: submitted only when no in-flight group overlaps the command's group */
#define KM_F_EXCL 2u
/* group leader: acks with {ack, result} as one 8-byte write; the kernel keeps it */
#define KM_F_RESULT 4u
/* dispatcher: a queued job; `first` is ignored, the group allocator chooses it */
#define KM_F_QUEUE 8u
#define KM_GROUP_ALL(n) ((uint32_t)(n) << 16) /* first 0, count n */

/* One command: four 16-byte chunks. Word 0 of each chunk carries the sequence number;
   a command has landed when all four agree (KM_LANDED). */
#define KM_LANDED(r) \
    ((r)->land1 == (r)->seq && (r)->land2 == (r)->seq && (r)->land3 == (r)->seq)

typedef struct {
    uint32_t seq; /* chunk 0: sequence number, starts at 1, increases */
    uint32_t hdr; /* handler | flags << 8 | slot << 16 */
    uint32_t
        group; /* first | count << 16: cores [first, first + count) in core order */
    uint32_t a0;
    uint32_t land1, a1, a2, a3; /* chunk 1 */
    uint32_t land2, a4, a5, a6; /* chunk 2 */
    uint32_t land3, a7, a8, a9; /* chunk 3 */
} km_command_t;                 /* sizeof == 64; ten argument words */

KM_STATIC_ASSERT(sizeof(km_command_t) == 64, "a command is one 64-byte line");

/* Control block in L1: the same address on every core. Offsets in bytes. */
#define KM_CTL_RING 0x000u /* KM_RING_SLOTS commands */
#define KM_CTL_RSVD 0x400u /* reserved, 16 bytes; not written in this version */
#define KM_CTL_STOP 0x410u /* nonzero: every wait gives up */
#define KM_CTL_DONE 0x440u /* KM_RING_SLOTS counters, 16 bytes apart: cores done */
#define KM_CTL_BOOT 0x540u /* start-up check-in area (column groups, as today) */
/* per core: last sequence consumed per slot, gave-up code, where code */
#define KM_CTL_STATE 0x640u
/* the command being run, for the compute and writer kernels */
#define KM_CTL_JOB 0x700u
#define KM_CTL_OUT 0x740u   /* scratch for outgoing 4-byte NoC writes */
#define KM_CTL_CORES 0x780u /* core index -> NoC x | y << 16, KM_CORES_MAX words */
/* KM_PORTS port slots of 16 bytes: count (bumped by senders), then the count a wait
   wants (nonzero only while waiting, or after the wait gave up) */
#define KM_CTL_PORT 0x980u
#define KM_CTL_BROADCAST 0xA00u /* KM_BROADCASTS areas, KM_BROADCAST_BYTES each */

/* Ports: the receiving side of a value sent to one core. */
#define KM_PORTS 8u
#define KM_PORT_STRIDE 16u
#define KM_PORT_COUNT 0u /* word offsets inside a port slot */
#define KM_PORT_WANT 1u
/* a port or emit give-up: the gave-up word's top byte names the port */
#define KM_GAVE_PORT_SHIFT 24u

KM_STATIC_ASSERT(KM_CTL_PORT == KM_CTL_CORES + 4u * KM_CORES_MAX, "ports after cores");
KM_STATIC_ASSERT(
    KM_CTL_PORT + KM_PORTS * KM_PORT_STRIDE == KM_CTL_BROADCAST, "ports end"
);
KM_STATIC_ASSERT(KM_CTL_PORT % 16u == 0 && KM_PORT_STRIDE == 16u, "16-byte slots");

/* Broadcast: one area per broadcast, offsets inside it (16-byte slots). */
#define KM_BROADCASTS 2u
#define KM_BROADCAST_RANKS 16u   /* DONE-BY words of ranks 1 to 15; rank 0 sends */
#define KM_BROADCAST_VALID 0x00u /* seq << 16 | blocks valid, from the sender */
#define KM_BROADCAST_OUT 0x10u   /* source slot of this core's VALID or DONE write */
#define KM_BROADCAST_GAVE 0x20u  /* first give-up: where | area << 8, block, seq */
#define KM_BROADCAST_STATS 0x30u /* blocks, wait cycles low and high words */
#define KM_BROADCAST_DONE 0x40u  /* rank r's DONE-BY word: KM_BROADCAST_DONE + 16 r */
#define KM_BROADCAST_BYTES 0x140u
#define KM_BROADCAST_GAVE_WORDS 3u   /* words of the KM_BROADCAST_GAVE slot in use */
#define KM_BROADCAST_STATS_WORDS 3u  /* words of the KM_BROADCAST_STATS slot in use */
#define KM_BROADCAST_DONE_STRIDE 16u /* bytes between two ranks' DONE-BY words */

KM_STATIC_ASSERT(
    KM_CTL_BROADCAST == KM_CTL_CORES + 4u * KM_CORES_MAX + 0x80u, "ports room"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_DONE + KM_BROADCAST_DONE_STRIDE * KM_BROADCAST_RANKS ==
        KM_BROADCAST_BYTES,
    "broadcast area"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_GAVE_WORDS * 4u <= KM_BROADCAST_STATS - KM_BROADCAST_GAVE, "gave slot"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_STATS_WORDS * 4u <= KM_BROADCAST_DONE - KM_BROADCAST_STATS,
    "stats slot"
);
KM_STATIC_ASSERT(
    KM_CTL_BROADCAST % 16u == 0 && KM_BROADCAST_BYTES % 16u == 0, "16-byte slots"
);

#define KM_CTL_DYN 0xC80u   /* routed messages: queue counters, idle words, status */
#define KM_CTL_TAILS 0xD40u /* routed messages: a counter tile's tail lines */
#define KM_CTL_X280                                                   \
    0x1580u /* X280 port: the lines shared with the serving X280 core \
             */
#define KM_CTL_BYTES 0x15C0u

/* Routed messages: offsets inside KM_CTL_DYN, each word in its own 16-byte slot. */
#define KM_DYN_IN_TAIL 0x00u /* reader: its tail as last read from its counter tile */
#define KM_DYN_IN_HEAD 0x10u /* reader */
#define KM_DYN_SENT 0x20u    /* writer: messages sent to any queue, new ones included */
#define KM_DYN_CONSUMED 0x30u /* compute: messages of the packets it finished */
#define KM_DYN_QUOTA 0x40u    /* compute: sends left */
#define KM_DYN_FLAGS 0x50u /* idle words of reader, compute, writer, 16 bytes apart */
#define KM_DYN_OUT 0x80u   /* source slot of the status line */
#define KM_DYN_GAVE 0x90u  /* first give-up: where, position, run, word seen */
#define KM_DYN_RET 0xA0u   /* writer: the reservation's returned position */
#define KM_DYN_BYTES 0xC0u
/* Counter tiles: the last K cores of a dynamic run's group hold the cores' tails and
 * never submit an atomic; offsets inside KM_CTL_TAILS. */
#define KM_TAILS_READY 0x00u  /* counter tile: {run, 1, cores it serves, 0} */
#define KM_TAILS_READER 0x10u /* reader: a counter's READY line read here */
#define KM_TAILS_WRITER 0x20u /* writer: the same */
#define KM_TAILS_LINE 0x30u   /* + 16 c: group core c's tail, the atomic adds' target */
#define KM_TAILS_BYTES (KM_TAILS_LINE + 16u * KM_CORES_MAX)
#define KM_COUNTERS_MAX 4u
/* the submitter ends a dynamic run with this command to its group; a0 = run's seq */
#define KM_HANDLER_DYN_END 249u

/* Status line of a core in a dynamic run, 16 bytes at the submitter, one store. */
#define KM_DYN_LINE_RUN 0u /* word offsets: the run's seq */
#define KM_DYN_LINE_STATE 1u
#define KM_DYN_LINE_SENT 2u
#define KM_DYN_LINE_CONSUMED 3u
#define KM_DYN_BUSY 1u /* line states */
#define KM_DYN_IDLE 2u
#define KM_DYN_GAVE_UP 3u

KM_STATIC_ASSERT(
    KM_CTL_DYN == KM_CTL_CORES + 4u * KM_CORES_MAX + 0x300u, "after broadcasts"
);
KM_STATIC_ASSERT(
    KM_CTL_BROADCAST + KM_BROADCASTS * KM_BROADCAST_BYTES == KM_CTL_DYN, "broadcast end"
);
KM_STATIC_ASSERT(KM_CTL_DYN + KM_DYN_BYTES == KM_CTL_TAILS, "dynamic area ends");
KM_STATIC_ASSERT(
    KM_CTL_TAILS + KM_TAILS_BYTES <= KM_CTL_BYTES && KM_CTL_BYTES % 64u == 0, "tails"
);
KM_STATIC_ASSERT(KM_CTL_DYN % 16u == 0 && KM_DYN_RET + 16u <= KM_DYN_BYTES, "slots");
KM_STATIC_ASSERT(KM_DYN_IN_TAIL % 16u == 0 && KM_DYN_OUT % 16u == 0, "16-byte slots");
KM_STATIC_ASSERT(KM_HANDLER_DYN_END > KM_HANDLER_USER_MAX, "not a program handler");

/* X280 port: 16-byte lines {seq, w0, w1, w2}, each with one writer, offsets inside
   KM_CTL_X280. A line lands whole; seq names it (a tell's latest wins). */
#define KM_X280_TELL                                                 \
    0x00u /* the X280 core's tell: seq counts its tells to this core \
           */
#define KM_X280_ANSWER 0x10u /* the X280 core's answer: seq repeats the ask's */
#define KM_X280_OUT 0x20u    /* source of this core's stores to its X280 core */
#define KM_X280_TO                                                    \
    0x30u /* the host's, at load: X280 core tile x | y << 16, mailbox \
           */
#define KM_X280_BYTES 0x40u
#define KM_X280_TO_TILE 0u /* word offsets of the KM_X280_TO line */
#define KM_X280_TO_BOX 1u  /* LIM address of this core's mailbox line; 0: not served */
/* A core's mailbox line in its X280 core's LIM (64 bytes, the core writes it). */
#define KM_BOX_ASK 0x00u  /* the core's ask: seq counts its asks */
#define KM_BOX_TELL 0x10u /* the core's tell: seq counts its tells */
#define KM_BOX_BYTES 0x40u
/* The pull: an ask {KM_PULL_ASK} answered {first unit, count (0: none left), pulls}. */
#define KM_PULL_ASK 0x50554C4Cu /* "PULL" */

KM_STATIC_ASSERT(KM_CTL_TAILS + KM_TAILS_BYTES <= KM_CTL_X280, "tails end");
KM_STATIC_ASSERT(
    KM_CTL_X280 % 64u == 0 && KM_CTL_X280 + KM_X280_BYTES == KM_CTL_BYTES,
    "X280 core lines"
);

/* Words the Tensix grid pushes into the X280's local memory. */
#define KM_ACK_STRIDE 16u /* ack word of slot j at ack_base + j * KM_ACK_STRIDE */
#define KM_ACK(seq) ((uint32_t)(seq)) /* the whole sequence number */
#define KM_READY 0xFFFFFFFFu          /* READY words are not in the ack area */

/* Host -> dispatcher: one command is one command, its number in `seq` and the
   landing words. Handlers 252..254 are commands to the kernel and are not forwarded. */

/* Dispatcher -> host: one 16-byte event per finished command, in a ring in host
   memory. */
#define KM_EV_DONE 0u    /* aux = first core of the running group | seq low 16 << 8 */
#define KM_EV_TIMEOUT 1u /* aux = slot | seq low 16 << 8; cores: their state words */
#define KM_EV_PONG 2u
#define KM_EV_REJECTED 3u /* consumed, not run; aux = reason (KMD_REJECT_*) */

typedef struct {
    uint32_t index;  /* event number, starts at 1: the ring slot is index % depth */
    uint32_t req;    /* the host's command number */
    uint32_t status; /* KM_EV_* | aux << 8 */
    uint32_t ticks;  /* submit to completion, in the X280's timer ticks */
} km_event_t;        /* sizeof == 16: one chunk, lands whole */

KM_STATIC_ASSERT(sizeof(km_event_t) == 16, "an event is one 16-byte chunk");

/* A DONE or TIMEOUT event's aux: the first core or the slot in its low 8 bits, the low
   16 bits of the command's sequence number above them. */
#define KM_EV_AUX_LOW 0xFFu
#define KM_EV_SEQ_SHIFT 8u
#define KM_EV_SEQ_MASK 0xFFFFu
KM_STATIC_ASSERT(KM_CORES_MAX <= KM_EV_AUX_LOW + 1u, "a first core fits in 8 bits");

#endif /* KOMOREBI_ABI_KM_ABI_H */
