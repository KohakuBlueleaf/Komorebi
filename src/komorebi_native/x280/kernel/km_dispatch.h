/* The dispatcher's interface to the host, in plain C: the argument block, the
 * layouts of the cluster's local memory (LIM) and the host buffer, the result block. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "komorebi/abi/km_abi.h"

#if defined(__cplusplus)
#define KMD_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define KMD_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

/* LIM layout, in bytes from the argument block (64-byte aligned, written by the host).
 * The host zeroes the layout up to KMD_LIM_ZEROED before every run. */
#define KMD_LIM_ARGS 0x000u /* kmd_args_t */
/* hello from core 0: word 0 = 1, word 1 = control block */
#define KMD_LIM_MAILBOX 0x100u
#define KMD_LIM_RESULT 0x140u /* result block, KMD_R_WORDS words */
#define KMD_LIM_READ 0x1C0u   /* line from the host: word 0 = events it has read */
#define KMD_LIM_READY 0x200u  /* KM_READY of column leader c at + c * 16 */
#define KMD_LIM_CORES 0x300u  /* core block, KMD_H_WORDS words */
#define KMD_LIM_ACKS 0x400u   /* ack word of slot j at + j * KM_ACK_STRIDE */
#define KMD_LIM_HRING 0x800u  /* host command ring: host_depth km_command_t */
/* status line of core c in a dynamic run at + c * 16 */
#define KMD_LIM_MESSAGES 0x1800u
/* program field each core holds, a byte per core (0xFF: none), if args programs */
#define KMD_LIM_PROGRAMS 0x2000u
/* core index -> NoC x | y << 16 (the host's core table) */
#define KMD_LIM_COORDS 0x2080u
/* the inspect block: a header line, then a record per core of the group, in order */
#define KMD_LIM_INSPECT 0x2280u
#define KMD_INSPECT_HEAD 64u
#define KMD_INSPECT_RECORD 0x340u /* a line, the state words, live words, trace */
#define KMD_INSPECT_STATE 0x40u   /* in a record: KM_STATE_WORDS words */
#define KMD_INSPECT_LIVE 0x280u   /* KM_LIVE_READ bytes of live words */
#define KMD_INSPECT_TRACE 0x300u  /* KMD_INSPECT_ENTRIES trace entries */
#define KMD_INSPECT_ENTRIES 4u
#define KMD_LIM_ZEROED (KMD_LIM_INSPECT + KMD_INSPECT_HEAD)
#define KMD_LIM_BYTES (KMD_LIM_ZEROED + KM_CORES_MAX * KMD_INSPECT_RECORD)
/* upload region: args upload_bytes, not zeroed */
#define KMD_LIM_UPLOAD KMD_LIM_BYTES
#define KMD_READY_MAX 16u
/* A hello's control block lies in [KMD_CONTROL_FLOOR, KMD_L1_BYTES - KM_CONTROL_BYTES].
 */
#define KMD_CONTROL_FLOOR 0x10000u /* below: the Tensix firmware and its mailboxes */
#define KMD_L1_BYTES 0x180000u     /* Blackhole Tensix L1, 1536 KiB */
#define KMD_HOST_DEPTH_MAX 64u

/* Host buffer (pinned host memory the kernel writes through a posted window): four
 * state words, then the event ring of event_depth km_event_t. */
#define KMD_HOST_STATE 0u    /* word: KMD_STATE_* */
#define KMD_HOST_CONSUMED 1u /* word: commands taken from the host ring */
/* word: events pushed (a hint; the event's index decides) */
#define KMD_HOST_EVENTS 2u
#define KMD_HOST_ERROR 3u /* word: KMD_ERROR_* once the state is final */
#define KMD_HOST_READY 4u /* words 4 to 7: kmd_ready_t, one 16-byte store at READY */
#define KMD_HOST_RING 64u /* byte offset of the event ring */

/* Kernel states, in the result block and in the host buffer. */
#define KMD_STATE_BOOT 1u   /* waiting for the hello and every READY */
#define KMD_STATE_READY 2u  /* taking commands */
#define KMD_STATE_DONE 3u   /* finished without error */
#define KMD_STATE_FAILED 4u /* finished with an error: see the result block */

/* Error codes. The core result is the ticks of the run, or ~code. */
#define KMD_ERROR_NONE 0u
#define KMD_ERROR_ARGS 1u     /* the argument block is not this run's or out of range */
#define KMD_ERROR_WINDOW 2u   /* a window did not read back as programmed */
#define KMD_ERROR_ALIGN 3u    /* control block address unaligned or outside L1 */
#define KMD_ERROR_DEADLINE 4u /* a command's group missed its deadline */
#define KMD_ERROR_START 5u    /* the hello or a READY did not arrive in time */
#define KMD_ERROR_IDLE 6u     /* no command and nothing in flight for t_idle */
#define KMD_ERROR_STOPPED 7u  /* the runtime asked the kernel to stop */
#define KMD_ERROR_STOP_BUSY 9u /* a STOP command arrived with commands in flight */
#define KMD_ERROR_ENGINE 10u   /* core 1 did not stop within KMD_ENGINE_STOP_TICKS */
/* Exception on core `core` (0 or 1): slot = mcause, sequence = mepc, seen = mtval (low
 * words). The kernel ended without its state's pointers; the cluster restores. */
#define KMD_ERROR_TRAP 11u

/* Reasons of a KM_EVENT_REJECTED event (its auxiliary): the command was consumed, not
 * run. */
#define KMD_REJECT_HANDLER 1u /* handler 240 to 249 (240 to 242, 245: see take()) */
#define KMD_REJECT_FLAGS 2u   /* reserved flag bit 0 set */
#define KMD_REJECT_GROUP 3u   /* count 0, or the group does not lie inside the cores */
#define KMD_REJECT_EXITED 4u  /* a command after EXIT */
/* a range not 64-byte aligned or outside the upload region */
#define KMD_REJECT_REGION 5u
#define KMD_REJECT_STREAMS 6u /* KM_HANDLER_RUN beyond KM_STREAM_MAX_ACTIVE streams */
/* KM_HANDLER_CANCEL of a request no active stream has */
#define KMD_REJECT_UNKNOWN 7u
#define KMD_REJECT_DEADLINE 8u /* KM_HANDLER_DEADLINE of 0 (a deadline is never 0) */
/* a KM_FLAG_QUEUE command while KM_QUEUE_JOBS jobs wait */
#define KMD_REJECT_QUEUE 9u
#define KMD_REJECT_DROPPED 10u  /* to the engine: a queued job of its stream dropped */
#define KMD_REJECT_BUSY 11u     /* engine, work cluster: the command cannot go now */
#define KMD_REJECT_MESSAGES 12u /* a dynamic run with KM_FLAG_QUEUE */
/* the group's cores do not hold the command's program (header bits 24 to 31) */
#define KMD_REJECT_PROGRAM 13u
/* the host's command to a work cluster whose thread has KMD_WORK_IN_DEPTH unanswered */
#define KMD_REJECT_THREAD 14u

/* Engine to kernel: POST host event {a0 request, a1 status, a2 ticks}. Kernel to
 * engine: RUN {request offset, ticks bytes} of an upload, CANCEL {ticks request}. */
#define KMD_HANDLER_POST 248u
/* Engine to kernel: drop its queued jobs of stream tag a0 (the group's first half). */
#define KMD_HANDLER_DROP 247u
/* Work cluster to kernel: its thread's answer to command line a0, a1 the word that
 * kmx_program returned; no event. */
#define KMD_HANDLER_ANSWER 243u
/* Host to kernel: read cores [first, first + count) into the inspect block, then PONG
 * {ticks: the mtime ticks it took} (the same number as ANSWER, from another source). */
#define KMD_HANDLER_INSPECT 243u
/* The inspect block's header words, its kinds, and a record's first line's words. */
#define KMD_INSPECT_MARK 0x50534E49u /* "INSP": the block is complete */
#define KMD_INSPECT_HOST 1u          /* a host's inspect */
#define KMD_INSPECT_TIMEOUT 2u       /* a command's deadline: read before its abort */
#define KMD_INSPECT_READ 1u          /* record flag: the core's words were read */
#define KMD_INSPECT_TRACED 2u        /* record flag: its trace's entries were read */
/* After a timeout's abort, the X280 timer ticks (20 ms) its group's cores have to leave
 * the command; then the run ends as at any other deadline. */
#define KMD_GIVE_UP_TICKS 1000000u
/* slot_handler of a command that may not be given up */
#define KMD_HANDLER_KEPT 0xFFu
#define KMD_EVENT_RUN 0x10u
#define KMD_EVENT_CANCEL 0x11u
/* Kernel to engine, once after a KMD_REJECT_BUSY: a slot or the queue has room. */
#define KMD_EVENT_WAKE 0x12u

/* Host to kernel: reset core `first` (group of one) by the write list of a1 bytes at a0
 * in the upload region; PONG {KMD_RESET_*, ticks}. DROP's number: engine-only. */
#define KMD_HANDLER_RESET 247u
/* A write list entry: four words {kind, address, bytes or mask, value}, then for a
 * WRITE its bytes, padded to 16. ASSERT and RELEASE write the soft-reset register. */
#define KMD_RESET_END 0u
#define KMD_RESET_WRITE 1u  /* `bytes` (a multiple of 4) to L1 `address` */
#define KMD_RESET_ASSERT 2u /* the five RISC CPUs in reset */
/* BRISC out of reset, its firmware releases the others */
#define KMD_RESET_RELEASE 3u
#define KMD_RESET_WAIT 4u /* until (L1 word at address & mask) == value, 1 s */
#define KMD_RESET_ENTRY 16u
/* The Tensix soft-reset register and its RISC bits (blackhole tensix.h:
 * RISCV_DEBUG_REG_SOFT_RESET_0; RISCV_SOFT_RESET_0_BRISC, _TRISCS, _NCRISC). */
#define KMD_SOFT_RESET_ADDR 0xFFB121B0u
#define KMD_SOFT_RESET_RISCS 0x47800u
/* What the runtime's release of BRISC leaves: the others in reset, bit 31 asking for
 * the staggered start (the UMD's deassert with staggered_start). */
#define KMD_SOFT_RESET_RELEASE 0x80047000u
#define KMD_RESET_WAIT_TICKS 50000000u
/* PONG auxiliary of a reset: 0 done, else the step that failed. */
#define KMD_RESET_OK 0u
#define KMD_RESET_BAD_LIST 1u    /* an entry outside the region, L1 or alignment */
#define KMD_RESET_NO_WINDOW 2u   /* a window did not read back as programmed */
#define KMD_RESET_NO_FIRMWARE 3u /* a WAIT ran out */

/* Test builds with KMD_VERIFY_LINES: a0 = offset, a1 = bytes of the upload region to
 * check against kmd_verify_word(a2, word index from the region's start). */
#define KMD_HANDLER_CHECK_UPLOAD 246u
/* Test builds with KMD_VERIFY_LINES: core 0 loads 8 bytes from address a0 (a trap). */
#define KMD_HANDLER_TRAP 244u

/* KMD_TRACE builds, in the region's lower half: kmd_trace_t, then per slot taken or
 * freed {kind | slot << 8 | source << 16, request, group, mtime}. */
#define KMD_TRACE_PART(upload_bytes) ((upload_bytes) / 2u & ~63u)
#define KMD_TRACE_SUBMIT 1u
#define KMD_TRACE_FREE 2u
#define KMD_TRACE_QUEUED 3u        /* a submit from the job queue */
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
    KMD_R_STATE = 0,         /* KMD_STATE_* */
    KMD_R_ERROR = 1,         /* KMD_ERROR_* */
    KMD_R_WHERE = 2,         /* KMD_WHERE_* */
    KMD_R_SLOT = 3,          /* the slot of a missed deadline */
    KMD_R_SEQUENCE = 4,      /* its sequence number */
    KMD_R_REQ = 5,           /* its host command number */
    KMD_R_SEEN = 6,          /* the column that had not reported READY, or a bad word */
    KMD_R_HOST_COMMANDS = 7, /* commands taken from the host ring */
    KMD_R_COMMANDS = 8,      /* commands submitted to the grid */
    KMD_R_EVENTS = 9,        /* events pushed to the host */
    KMD_R_BUSY = 10,         /* busy slots 0 to 31 at the finish (KMD_H_BUSY_HIGH) */
    KMD_R_MAX_BUSY = 11,     /* most slots busy at once */
    KMD_R_CONTROL = 12,      /* control block address from the hello */
    KMD_R_TICKS_LO = 13,     /* ticks from the first command taken to the finish */
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
    KMD_R_REJECTED = 24, /* commands answered with KM_EVENT_REJECTED */
    KMD_R_HELD = 25, /* commands that waited at least one pass (EXIT, KM_FLAG_EXCLUSIVE)
                      */
    KMD_R_NO_ROOM = 26, /* commands that waited for room in the event ring */
    KMD_R_CHECKED = 27, /* KMD_VERIFY_LINES builds: commands whose words were checked */
    KMD_R_BAD_LINES = 28, /* of those, commands with a word that did not match */
    KMD_R_BAD_FIRST = 29, /* the first such command's number */
    KMD_R_RESULTS = 30,   /* KM_FLAG_RESULT commands completed */
    KMD_R_RESULT = 31,    /* the latest one's result */
    KMD_R_WORDS = 32
};

/* Core block in LIM: each word has one writer, the core it names. */
enum {
    KMD_H_MARKER0 = 0,  /* core 0: its loop passes / 1024, every 1024 passes */
    KMD_H_MARKER1 = 1,  /* core 1: the same for the engine's loop */
    KMD_H_ENGINE = 2,   /* core 1: KMD_ENGINE_* */
    KMD_H_PASSES = 3,   /* core 1: engine loop passes, low word, at its end */
    KMD_H_COMMANDS = 4, /* core 1: commands it sent */
    KMD_H_POSTS = 5,    /* core 1: host events it posted */
    KMD_H_EVENTS = 6,   /* core 1: events it read */
    KMD_H_RUNS = 7,     /* core 1: streams it started */
    KMD_H_LOST = 8,     /* core 0: posts dropped at the finish for lack of room */
    KMD_H_SOURCE = 9,   /* core 0: the source of the error's command (KMD_R_REQ) */
    KMD_H_MESSAGES_GAVE = 10, /* core 0: 1 + the core whose status line said gave up */
    KMD_H_MESSAGES_CHECKS = 11, /* core 0: two-pass checks of dynamic runs */
    KMD_H_MESSAGES_QUIET = 12,  /* core 0: mtime, low word, of the last quiet end */
    KMD_H_ABANDONED = 13,       /* core 0: commands in flight or queued at the finish */
    KMD_H_TRAP_CAUSE = 14,      /* core 1, after a trap: mcause */
    KMD_H_TRAP_PC = 15,         /* core 1, after a trap: mepc, low word */
    KMD_H_BUSY_HIGH = 16,       /* core 0: busy slots 32 to 63 at the finish */
    KMD_H_GIVEN_UP = 17,        /* core 0: commands given up at their deadline */
    KMD_H_WORDS = 18
};

/* The inspect block's header (KMD_LIM_INSPECT), words; `mark` is stored last. */
enum {
    KMD_I_MARK = 0,
    KMD_I_KIND = 1,    /* KMD_INSPECT_HOST or KMD_INSPECT_TIMEOUT */
    KMD_I_REQUEST = 2, /* the host command number it answers or gave up */
    KMD_I_FIRST = 3,   /* the group */
    KMD_I_COUNT = 4,
    KMD_I_DONE = 5, /* records written */
    KMD_I_SLOT = 6, /* a timeout's slot and sequence; a host's: the last sequence */
    KMD_I_SEQUENCE = 7,
    KMD_I_START = 8, /* mtime, low word, at the start */
    KMD_I_TICKS = 9, /* mtime ticks from the start to the last record */
    KMD_IR_CORE = 0, /* a record's first line: the core, its coordinates, flags */
    KMD_IR_COORDINATES = 1,
    KMD_IR_FLAGS = 2,
    KMD_IR_MTIME = 3,     /* mtime, low word, as it was read */
    KMD_IR_SEQUENCES = 4, /* the sequences of its KMD_INSPECT_ENTRIES trace entries */
};

/* The engine's state (KMD_H_ENGINE). */
#define KMD_ENGINE_WAITING 1u /* for the kernel's go */
#define KMD_ENGINE_RUNNING 2u
#define KMD_ENGINE_STOPPED 3u /* left its loop; the kernel may finish */
#define KMD_ENGINE_UNUSED 4u  /* the kernel did not start it */
#define KMD_ENGINE_TRAPPED 5u /* an exception ended it (KMD_H_TRAP_CAUSE) */

/* Window slots of the kernel (16 to 191 are free for an image). */
#define KMD_WIN_PANEL0 16u /* panel r uses KMD_WIN_PANEL0 + r */
#define KMD_WIN_HOST 18u
#define KMD_WIN_WORK 19u /* work cluster c's ring area: KMD_WIN_WORK + c - 1 */
/* one core's L1, re-aimed for each core an inspect reads */
#define KMD_WIN_INSPECT 22u
/* a reset's core: its L1, then its soft-reset register (cleared after each reset) */
#define KMD_WIN_RESET 23u
#define KMD_WIN_RESET_REGISTER 24u

/* The X280 ranks of a dynamic program in messages_handler's bits 32 to 39: work cluster
 * c's X280 cores 0 to count - 1 (count 1 to KM_X280_RANKS_MAX; 0: none). */
#define KMD_X280_RANKS(cluster, count) \
    (((uint64_t)(cluster) | (uint64_t)(count) << 4) << 32)
#define KMD_X280_RANKS_CLUSTER(word) ((uint32_t)((word) >> 32) & 0xFu)
#define KMD_X280_RANKS_COUNT(word) ((uint32_t)((word) >> 36) & 0xFu)

/* The argument block's layout; the kernel rejects a block of another version. */
#define KMD_ARGS_VERSION 3u
/* Work clusters with a ring to the kernel (clusters 1 to 3). */
#define KMD_WORK_CLUSTERS 3u

/* Argument block, at the start of the layout. Deadlines are in mtime ticks (50 MHz). */
typedef struct {
    uint64_t run_value;     /* the run this block belongs to */
    uint64_t depth;         /* D: mailbox slots in use, 1 to KM_RING_SLOTS */
    uint64_t host_depth;    /* host command ring slots: a power of two, at most 64 */
    uint64_t event_depth;   /* event ring slots in the host buffer: a power of two */
    uint64_t columns;       /* column leaders that report READY, 1 to KMD_READY_MAX */
    uint64_t cores;         /* cores in core order; a group must lie inside */
    uint64_t panels;        /* multicast rectangles, 1 or 2 */
    uint32_t rect[2][4];    /* x_start, y_start, x_end, y_end of each */
    uint64_t host_noc_addr; /* NoC address of the host buffer, 64-byte aligned */
    uint64_t pcie_x;        /* the PCIe tile that reaches it */
    uint64_t pcie_y;
    uint64_t t_start;   /* deadline for the hello and every READY, nonzero */
    uint64_t t_command; /* a command's deadline until a KM_HANDLER_DEADLINE command */
    uint64_t t_idle;    /* deadline with no command and nothing in flight; 0 = none */
    uint64_t state_ddr; /* offset in this cluster's DRAM bank of KMD_STATE_BYTES */
    uint64_t
        upload_bytes; /* upload region at KMD_LIM_UPLOAD: a multiple of 64, may be 0 */
    /* The work clusters' rings, sources 2 to 4 (work_port.h); work_cmds 0: none. Ring
     * c - 1 (cluster c) is in use when work_noc_addr[c - 1] is nonzero. */
    uint64_t work_cmds;  /* LIM offset from this block of ring 0's command lines */
    uint64_t work_depth; /* command lines per ring, a power of two, at most 64 */
    uint64_t work_event_depth; /* events per ring, a power of two, at most 1024 */
    uint64_t work_noc[KMD_WORK_CLUSTERS];      /* each cluster's tile: x | y << 16 */
    uint64_t work_noc_addr[KMD_WORK_CLUSTERS]; /* NoC address there of KMD_WORK_* */
    /* dynamic handlers, one per byte (dynamic.h), 0: none; bits 32 to 39 the X280
     * ranks, KMD_X280_RANKS(cluster, count) */
    uint64_t messages_handler;
    uint64_t version;          /* KMD_ARGS_VERSION */
    uint64_t programs; /* 1: KMD_LIM_PROGRAMS holds each core's program; 0: none */
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

/* Test builds with KMD_VERIFY_LINES check each user command against these: argument
 * `word` of host command `index`, its group (first | count << 16) and header 0. */
static inline uint32_t kmd_verify_word(uint32_t index, uint32_t word) {
    uint32_t mixed = index * 0x9E3779B1u ^ (word + 1u) * 0x85EBCA77u;
    mixed ^= mixed >> 15;
    return mixed * 0x2C1B3C6Du;
}

static inline uint32_t kmd_verify_group(uint32_t index, uint32_t cores) {
    const uint32_t count = 1u + kmd_verify_word(index, 10u) % cores;
    return kmd_verify_word(index, 11u) % (cores - count + 1u) | count << 16;
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
/* Ring r: command lines, then its read line, here at work_cmds + r * ring bytes;
 * these at work_noc_addr[r] in the work cluster's LIM. */
#define KMD_WORK_RING_BYTES(depth) ((depth) * 64u + 64u)
#define KMD_WORK_CONSUMED 0x00u /* core 0: commands taken */
#define KMD_WORK_TOLD 0x40u     /* core 0: events pushed */
#define KMD_WORK_EVENTS 0x80u   /* work_event_depth events */
#define KMD_WORK_DEPTH_MAX 64u
#define KMD_WORK_EVENTS_MAX 1024u
/* Commands to the thread (kmx_program): KMD_WORK_IN_DEPTH lines of 64 bytes after the
 * events, the core 0's; line n (from 1) in slot n % depth until the thread answers. */
#define KMD_WORK_IN(event_depth) \
    (KMD_WORK_EVENTS + (((event_depth) * 16u + 63u) & ~63u))
#define KMD_WORK_IN_DEPTH 8u
/* The run line of a dynamic run with X280 ranks on the cluster (KM_X280_RUN_*): one
 * 16-byte store on its own 64-byte line after the command lines. */
#define KMD_WORK_RUN(event_depth) (KMD_WORK_IN(event_depth) + KMD_WORK_IN_DEPTH * 64u)
#define KMD_WORK_RUN_BYTES 64u

/* Ticks core 0 waits for core 1 to stop at the finish (10 ms). */
#define KMD_ENGINE_STOP_TICKS 500000u

KMD_STATIC_ASSERT(sizeof(kmd_args_t) == 248, "kmd_args_t is 248 bytes");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, work_cmds) == 152, "work_cmds at 152");
KMD_STATIC_ASSERT(offsetof(kmd_args_t, version) == 232, "version at 232");
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
KMD_STATIC_ASSERT(KMD_LIM_READY + 16u * KMD_READY_MAX <= KMD_LIM_CORES, "LIM: READY");
KMD_STATIC_ASSERT(KMD_LIM_CORES + 4u * KMD_H_WORDS <= KMD_LIM_ACKS, "LIM: X280 cores");
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
    KMD_LIM_HRING + KMD_HOST_DEPTH_MAX * 64u <= KMD_LIM_MESSAGES, "LIM: host ring"
);
KMD_STATIC_ASSERT(
    KMD_LIM_MESSAGES + 16u * KM_CORES_MAX <= KMD_LIM_PROGRAMS, "LIM: lines"
);
KMD_STATIC_ASSERT(
    KMD_LIM_PROGRAMS + KM_CORES_MAX <= KMD_LIM_COORDS &&
        KMD_LIM_COORDS + 4u * KM_CORES_MAX <= KMD_LIM_INSPECT &&
        KMD_LIM_INSPECT % 64u == 0 && KMD_INSPECT_RECORD % 64u == 0,
    "LIM: programs, coordinates, the inspect block"
);
KMD_STATIC_ASSERT(
    KMD_INSPECT_STATE + 4u * KM_STATE_WORDS <= KMD_INSPECT_LIVE &&
        KMD_INSPECT_LIVE + KM_LIVE_READ <= KMD_INSPECT_TRACE &&
        KMD_INSPECT_TRACE + KMD_INSPECT_ENTRIES * KM_TRACE_ENTRY <= KMD_INSPECT_RECORD,
    "an inspect record's parts"
);
KMD_STATIC_ASSERT((KMD_HOST_DEPTH_MAX & (KMD_HOST_DEPTH_MAX - 1u)) == 0, "power of 2");
KMD_STATIC_ASSERT(
    KMD_HANDLER_KEPT > KM_HANDLER_USER_MAX, "a kept slot is no handler's"
);
