/* The dispatch kernel's interface to the host, in plain C: the argument block, the
 * layouts of the cluster's local memory (LIM) and the host buffer, the result block. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
#define KMD_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KMD_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

/* LIM layout, in bytes from the argument block (64-byte aligned, written by the host).
 * The host zeroes the whole layout before every run. */
#define KMD_LIM_ARGS 0x000u /* kmd_args_t */
#define KMD_LIM_MAILBOX \
    0x100u /* hello from core 0: word 0 = 1, word 1 = control block */
#define KMD_LIM_RESULT 0x140u /* result block, KMD_R_WORDS words */
#define KMD_LIM_READ 0x1C0u   /* line from the host: word 0 = events it has read */
#define KMD_LIM_READY 0x200u  /* KM_READY of column leader c at + c * 16 */
#define KMD_LIM_HARTS 0x300u  /* hart block, KMD_H_WORDS words */
#define KMD_LIM_ACKS 0x400u   /* ack word of slot j at + j * KM_ACK_STRIDE */
#define KMD_LIM_HRING 0x800u  /* host command ring: host_depth km_record_t */
#define KMD_LIM_DYN 0x1800u   /* status line of core c in a dynamic run at + c * 16 */
#define KMD_LIM_BYTES 0x2000u
#define KMD_LIM_UPLOAD                                            \
    KMD_LIM_BYTES /* upload region: args upload_bytes, not zeroed \
                   */
#define KMD_READY_MAX 16u
/* A hello's control block lies in [KMD_CTL_FLOOR, KMD_L1_BYTES - KM_CTL_BYTES]. */
#define KMD_CTL_FLOOR 0x10000u /* below: the Tensix firmware and its mailboxes */
#define KMD_L1_BYTES 0x180000u /* Blackhole Tensix L1, 1536 KiB */
#define KMD_HOST_DEPTH_MAX 64u

/* Host buffer (pinned host memory the kernel writes through a posted window): four
 * state words, then the event ring of event_depth km_event_t. */
#define KMD_HOST_STATE 0u    /* word: KMD_STATE_* */
#define KMD_HOST_CONSUMED 1u /* word: commands taken from the host ring */
#define KMD_HOST_EVENTS                                                              \
    2u                    /* word: events pushed (a hint; the event's index decides) \
                           */
#define KMD_HOST_ERROR 3u /* word: KMD_ERR_* once the state is final */
#define KMD_HOST_READY 4u /* words 4 to 7: kmd_ready_t, one 16-byte store at READY */
#define KMD_HOST_RING 64u /* byte offset of the event ring */

/* Kernel states, in the result block and in the host buffer. */
#define KMD_STATE_BOOT 1u   /* waiting for the hello and every READY */
#define KMD_STATE_READY 2u  /* taking commands */
#define KMD_STATE_DONE 3u   /* finished without error */
#define KMD_STATE_FAILED 4u /* finished with an error: see the result block */

/* Error codes. The hart result is the ticks of the run, or ~code. */
#define KMD_ERR_NONE 0u
#define KMD_ERR_ARGS 1u      /* the argument block is not this run's or out of range */
#define KMD_ERR_WINDOW 2u    /* a window did not read back as programmed */
#define KMD_ERR_ALIGN 3u     /* control block address unaligned or outside L1 */
#define KMD_ERR_DEADLINE 4u  /* a record's team missed its deadline */
#define KMD_ERR_START 5u     /* the hello or a READY did not arrive in time */
#define KMD_ERR_IDLE 6u      /* no command and nothing in flight for t_idle */
#define KMD_ERR_STOPPED 7u   /* the runtime asked the kernel to stop */
#define KMD_ERR_STOP_BUSY 9u /* a STOP command arrived with records in flight */
#define KMD_ERR_ENGINE 10u   /* core 1 did not stop within KMD_ENGINE_STOP_TICKS */

/* Reasons of a KM_EV_REFUSED event (its aux): the command was consumed, not run. */
#define KMD_REFUSE_SEG 1u    /* segment 240 to 249 */
#define KMD_REFUSE_FLAGS 2u  /* reserved flag bit 0 set */
#define KMD_REFUSE_TEAM 3u   /* count 0, or the team does not lie inside the cores */
#define KMD_REFUSE_EXITED 4u /* a record after EXIT */
#define KMD_REFUSE_REGION \
    5u /* a range not 64-byte aligned or outside the upload region */
#define KMD_REFUSE_STREAMS 6u  /* KM_SEG_RUN while KM_STREAM_MAX_ACTIVE streams run */
#define KMD_REFUSE_UNKNOWN 7u  /* KM_SEG_CANCEL of a req no active stream has */
#define KMD_REFUSE_DEADLINE 8u /* KM_SEG_DEADLINE of 0: a deadline is never zero */
#define KMD_REFUSE_QUEUE 9u    /* a KM_F_QUEUE record while KM_QUEUE_JOBS jobs wait */
#define KMD_REFUSE_DROPPED 10u /* to the engine: a queued job of its stream dropped */
#define KMD_REFUSE_BUSY 11u    /* engine, work cluster: the record cannot go now */
#define KMD_REFUSE_DYN 12u     /* a dynamic run with KM_F_QUEUE */

/* Engine to kernel: POST host event {a0 req, a1 status, a2 ticks}. Kernel to engine
 * events: RUN {req offset, ticks bytes} of an admitted upload, CANCEL {ticks req}. */
#define KMD_SEG_POST 248u
/* Engine to kernel: drop its queued jobs of stream tag a0 (the header's slot bits). */
#define KMD_SEG_DROP 247u
#define KMD_EV_RUN 0x10u
#define KMD_EV_CANCEL 0x11u
/* Kernel to engine, once after a KMD_REFUSE_BUSY: a slot or the queue has room. */
#define KMD_EV_WAKE 0x12u

/* Test builds with KMD_VERIFY_LINES: a0 = offset, a1 = bytes of the upload region to
 * check against kmd_verify_word(a2, word index from the region's start). */
#define KMD_SEG_CHECK_UPLOAD 246u

/* KMD_TRACE builds, in the region's lower half: kmd_trace_t, then per slot taken or
 * freed {kind | slot << 8 | source << 16, req, team, mtime}. */
#define KMD_TRACE_PART(upload_bytes) ((upload_bytes) / 2u & ~63u)
#define KMD_TRACE_ISSUE 1u
#define KMD_TRACE_FREE 2u
#define KMD_TRACE_QUEUED 3u        /* an issue from the job queue */
#define KMD_TRACE_MARK 0x54524345u /* "TRCE" */

typedef struct {
    uint32_t mark;
    uint32_t entries;  /* entries written */
    uint32_t untraced; /* entries that did not fit */
    uint32_t pad;
} kmd_trace_t;

/* Where codes: the loop the kernel was in. */
#define KMD_WHERE_ARGS 1u
#define KMD_WHERE_HELLO 2u
#define KMD_WHERE_READY 3u
#define KMD_WHERE_LOOP 4u
#define KMD_WHERE_FINISH 5u

/* Result block words. Pairs named _LO / _HI hold a 64-bit count. */
enum {
    KMD_R_STATE = 0,     /* KMD_STATE_* */
    KMD_R_ERROR = 1,     /* KMD_ERR_* */
    KMD_R_WHERE = 2,     /* KMD_WHERE_* */
    KMD_R_SLOT = 3,      /* the slot of a missed deadline */
    KMD_R_SEQ = 4,       /* its sequence number */
    KMD_R_REQ = 5,       /* its host command number */
    KMD_R_SEEN = 6,      /* the column that had not reported READY, or a bad word */
    KMD_R_COMMANDS = 7,  /* commands taken from the host ring */
    KMD_R_RECORDS = 8,   /* records issued to the grid */
    KMD_R_EVENTS = 9,    /* events pushed to the host */
    KMD_R_BUSY = 10,     /* busy slots when the kernel finished (bit j = slot j) */
    KMD_R_MAX_BUSY = 11, /* most slots busy at once */
    KMD_R_CTL = 12,      /* control block address from the hello */
    KMD_R_TICKS_LO = 13, /* ticks from the first command taken to the finish */
    KMD_R_TICKS_HI = 14,
    KMD_R_IDLE_LO = 15, /* loop passes with a free slot and no command */
    KMD_R_IDLE_HI = 16,
    KMD_R_FULL_LO = 17, /* loop passes with every slot busy */
    KMD_R_FULL_HI = 18,
    KMD_R_PASSES_LO = 19, /* loop passes */
    KMD_R_PASSES_HI = 20,
    KMD_R_BOOT_TICKS = 21, /* ticks from the start to READY */
    KMD_R_LAST_LO = 22,    /* ticks from the first command taken to the last event */
    KMD_R_LAST_HI = 23,
    KMD_R_REFUSED = 24, /* commands answered with KM_EV_REFUSED */
    KMD_R_HELD = 25,    /* commands that waited at least one pass (EXIT, KM_F_EXCL) */
    KMD_R_NO_ROOM = 26, /* commands that waited for room in the event ring */
    KMD_R_CHECKED = 27, /* KMD_VERIFY_LINES builds: records whose words were checked */
    KMD_R_BAD_LINES = 28, /* of those, records with a word that did not match */
    KMD_R_BAD_FIRST = 29, /* the first such command's number */
    KMD_R_RESULTS = 30,   /* KM_F_RESULT records completed */
    KMD_R_RESULT = 31,    /* the latest one's result */
    KMD_R_WORDS = 32
};

/* Hart block in LIM: each word has one writer, the hart it names. */
enum {
    KMD_H_STAMP0 = 0,      /* core 0: its loop passes / 1024, every 1024 passes */
    KMD_H_STAMP1 = 1,      /* core 1: the same for the engine's loop */
    KMD_H_ENGINE = 2,      /* core 1: KMD_ENGINE_* */
    KMD_H_PASSES = 3,      /* core 1: engine loop passes, low word, at its end */
    KMD_H_RECORDS = 4,     /* core 1: records it sent */
    KMD_H_POSTS = 5,       /* core 1: host events it posted */
    KMD_H_EVENTS = 6,      /* core 1: events it read */
    KMD_H_RUNS = 7,        /* core 1: streams it started */
    KMD_H_LOST = 8,        /* core 0: posts dropped at the finish for lack of room */
    KMD_H_SOURCE = 9,      /* core 0: the source of the error's record (KMD_R_REQ) */
    KMD_H_DYN_GAVE = 10,   /* core 0: 1 + the core whose status line said gave up */
    KMD_H_DYN_CHECKS = 11, /* core 0: two-pass checks of dynamic runs */
    KMD_H_DYN_QUIET = 12,  /* core 0: mtime, low word, of the last quiet end */
    KMD_H_WORDS = 16
};

/* The engine's state (KMD_H_ENGINE). */
#define KMD_ENGINE_WAITING 1u /* for the kernel's go */
#define KMD_ENGINE_RUNNING 2u
#define KMD_ENGINE_STOPPED 3u /* left its loop; the kernel may finish */
#define KMD_ENGINE_UNUSED 4u  /* the kernel did not start it */

/* Window slots of the kernel (16 to 191 are free for an image). */
#define KMD_WIN_PANEL0 16u /* panel r uses KMD_WIN_PANEL0 + r */
#define KMD_WIN_HOST 18u
#define KMD_WIN_WORK 19u /* the work cluster's ring area (work_port.h) */

/* Argument block, at the start of the layout. Deadlines are in mtime ticks (50 MHz). */
typedef struct {
    uint64_t run_value;     /* the run this block belongs to */
    uint64_t depth;         /* D: mailbox slots in use, 1 to KM_RING_SLOTS */
    uint64_t host_depth;    /* host command ring slots: a power of two, at most 64 */
    uint64_t event_depth;   /* event ring slots in the host buffer: a power of two */
    uint64_t columns;       /* column leaders that report READY, 1 to KMD_READY_MAX */
    uint64_t cores;         /* cores in core order; a team must lie inside */
    uint64_t panels;        /* multicast rectangles, 1 or 2 */
    uint32_t rect[2][4];    /* x_start, y_start, x_end, y_end of each */
    uint64_t host_noc_addr; /* NoC address of the host buffer, 64-byte aligned */
    uint64_t pcie_x;        /* the PCIe tile that reaches it */
    uint64_t pcie_y;
    uint64_t t_start;   /* deadline for the hello and every READY, nonzero */
    uint64_t t_record;  /* a record's deadline until a KM_SEG_DEADLINE command */
    uint64_t t_idle;    /* deadline with no command and nothing in flight; 0 = none */
    uint64_t state_ddr; /* offset in this cluster's DRAM bank of KMD_STATE_BYTES */
    uint64_t
        upload_bytes; /* upload region at KMD_LIM_UPLOAD: a multiple of 64, may be 0 */
    /* A work cluster's ring, source 2 (work_port.h); work_cmds 0: none. */
    uint64_t work_cmds;        /* LIM offset from this block of its command lines */
    uint64_t work_depth;       /* command lines, a power of two, at most 64 */
    uint64_t work_event_depth; /* events, a power of two, at most 1024 */
    uint64_t work_noc;         /* the work cluster's tile: x | y << 16 */
    uint64_t work_noc_addr;    /* NoC address there of KMD_WORK_* (64-byte aligned) */
    uint64_t dyn_seg;          /* the program's dynamic segment (dynamic.h); 0: none */
} kmd_args_t;

/* The ready block: the host waits for `mark` (its buffer is zeroed before the run), as
 * a separate store's order is not kept (one store per item written for the host). */
#define KMD_READY_MARK 0x594452u /* "RDY" */

typedef struct {
    uint32_t mark;
    uint32_t upload_addr;  /* the upload region's LIM address */
    uint32_t upload_bytes; /* its size */
    uint32_t lim_bytes;    /* KMD_LIM_BYTES: the layout before it */
} kmd_ready_t;

/* Test builds with KMD_VERIFY_LINES check every user record against these: argument
 * word w (0 to 9) of host command i, its team (first | count << 16) and header 0. */
static inline uint32_t kmd_verify_word(uint32_t i, uint32_t w) {
    uint32_t x = i * 0x9E3779B1u ^ (w + 1u) * 0x85EBCA77u;
    x ^= x >> 15;
    return x * 0x2C1B3C6Du;
}

static inline uint32_t kmd_verify_team(uint32_t i, uint32_t cores) {
    const uint32_t count = 1u + kmd_verify_word(i, 10u) % cores;
    return kmd_verify_word(i, 11u) % (cores - count + 1u) | count << 16;
}

/* The kernel keeps its state in the cluster's DRAM bank, through the cached Memory
 * Port: the host gives it KMD_STATE_BYTES there, 64-byte aligned (LIM is uncached). */
#define KMD_STATE_BYTES 0x8000u
/* Its parts: core 0's state, core 1's state, then the engine's ring (both cores). */
#define KMD_STATE_KERNEL 0x0000u
#define KMD_STATE_ENGINE 0x2000u
#define KMD_STATE_RING 0x4000u
#define KMD_STATE_QUEUE 0x6000u
/* The engine's ring, from KMD_STATE_RING: commands, events, then one word per line. */
#define KMD_ENGINE_DEPTH 64u   /* command lines */
#define KMD_ENGINE_EVENTS 128u /* events */
#define KMD_ERING_CMDS 0x0000u
#define KMD_ERING_EVENTS 0x1000u
#define KMD_ERING_CONSUMED 0x1800u /* core 0: commands taken */
#define KMD_ERING_TOLD 0x1840u     /* core 0: events pushed */
#define KMD_ERING_READ 0x1880u     /* core 1: events read */
#define KMD_ERING_STOP 0x18C0u     /* core 0: nonzero, the engine leaves its loop */
#define KMD_ERING_GO 0x1900u       /* core 0: {run value, 1 run or 2 do not} */
#define KMD_ERING_QUEUED 0x1940u   /* core 0: jobs waiting in the queue */
#define KMD_ERING_TAKEN 0x1980u    /* core 0: the engine's jobs taken into it, ever */
#define KMD_ERING_BYTES 0x19C0u
/* A work cluster's ring: its command lines, then its read line, in this cluster's LIM
 * at work_cmds; these, from work_noc_addr in the work cluster's LIM. */
#define KMD_WORK_CONSUMED 0x00u /* core 0: commands taken */
#define KMD_WORK_TOLD 0x40u     /* core 0: events pushed */
#define KMD_WORK_EVENTS 0x80u   /* work_event_depth events */
#define KMD_WORK_DEPTH_MAX 64u
#define KMD_WORK_EVENTS_MAX 1024u

/* Ticks core 0 waits for core 1 to stop at the finish (10 ms). */
#define KMD_ENGINE_STOP_TICKS 500000u

KMD_STATIC_ASSERT(sizeof(kmd_args_t) == 200, "kmd_args_t is 200 bytes");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, work_cmds) == 152, "work_cmds at 152");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, upload_bytes) == 144, "upload_bytes at 144");
KMD_STATIC_ASSERT(sizeof(kmd_ready_t) == 16, "the ready block is one 16-byte store");
KMD_STATIC_ASSERT(KMD_HOST_READY * 4u + 16u <= KMD_HOST_RING, "host buffer: ready");
KMD_STATIC_ASSERT(KMD_LIM_UPLOAD % 64u == 0, "the upload region starts on a line");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, rect) == 56, "kmd_args_t.rect at 56");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, host_noc_addr) == 88, "host_noc_addr at 88");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, t_start) == 112, "t_start at 112");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, state_ddr) == 136, "state_ddr at 136");
KMD_STATIC_ASSERT(KMD_LIM_ARGS + sizeof(kmd_args_t) <= KMD_LIM_MAILBOX, "LIM: args");
KMD_STATIC_ASSERT(KMD_LIM_MAILBOX + 8u <= KMD_LIM_RESULT, "LIM: mailbox");
KMD_STATIC_ASSERT(KMD_LIM_RESULT + 4u * KMD_R_WORDS <= KMD_LIM_READ, "LIM: result");
KMD_STATIC_ASSERT(KMD_LIM_READ + 64u <= KMD_LIM_READY, "LIM: read count");
KMD_STATIC_ASSERT(KMD_LIM_READY + 16u * KMD_READY_MAX <= KMD_LIM_HARTS, "LIM: READY");
KMD_STATIC_ASSERT(KMD_LIM_HARTS + 4u * KMD_H_WORDS <= KMD_LIM_ACKS, "LIM: harts");
KMD_STATIC_ASSERT(
    KMD_ERING_CMDS + KMD_ENGINE_DEPTH * 64u <= KMD_ERING_EVENTS, "engine ring: cmds"
);
KMD_STATIC_ASSERT(
    KMD_ERING_EVENTS + KMD_ENGINE_EVENTS * 16u <= KMD_ERING_CONSUMED,
    "engine ring: events"
);
KMD_STATIC_ASSERT(
    KMD_STATE_RING + KMD_ERING_BYTES <= KMD_STATE_QUEUE, "state: engine ring"
);
KMD_STATIC_ASSERT(
    KMD_LIM_ACKS + KM_RING_SLOTS * KM_ACK_STRIDE <= KMD_LIM_HRING, "LIM: ack words"
);
KMD_STATIC_ASSERT(
    KMD_LIM_HRING + KMD_HOST_DEPTH_MAX * 64u <= KMD_LIM_DYN, "LIM: host ring"
);
KMD_STATIC_ASSERT(KMD_LIM_DYN + 16u * KM_CORES_MAX <= KMD_LIM_BYTES, "LIM: lines");
KMD_STATIC_ASSERT((KMD_HOST_DEPTH_MAX & (KMD_HOST_DEPTH_MAX - 1u)) == 0, "power of 2");
