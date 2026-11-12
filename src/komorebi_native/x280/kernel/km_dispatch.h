/* The dispatch kernel's interface to the host: the argument block, the layout of the
 * cluster's local memory (LIM) and of the host buffer, the result block and its codes.
 * Plain C; the X280 kernel and the host include it. */
#pragma once

#include <stdint.h>

/* LIM layout, in bytes from the argument block (64-byte aligned, written by the host).
 * The host zeroes the whole layout before every run. */
#define KMD_LIM_ARGS 0x000u /* kmd_args_t */
#define KMD_LIM_MAILBOX \
    0x100u /* hello from core 0: word 0 = 1, word 1 = control block */
#define KMD_LIM_RESULT 0x140u /* result block, KMD_R_WORDS words */
#define KMD_LIM_READY 0x200u  /* KM_READY of column leader c at + c * 16 */
#define KMD_LIM_ACKS 0x400u   /* ack word of slot j at + j * KM_ACK_STRIDE */
#define KMD_LIM_HRING 0x800u  /* host command ring: host_depth km_record_t */
#define KMD_LIM_BYTES 0x1800u
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
#define KMD_ERR_COMMAND 8u   /* a command the kernel refuses: see the result block */
#define KMD_ERR_STOP_BUSY 9u /* a STOP command arrived with records in flight */

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
    KMD_R_REQ = 5,       /* its host command number, or the refused command's */
    KMD_R_SEEN = 6,      /* the column that had not reported READY, or a refused word */
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
    KMD_R_WORDS = 32
};

/* Window slots of the kernel (16 to 191 are free for an image). */
#define KMD_WIN_PANEL0 16u /* panel r uses KMD_WIN_PANEL0 + r */
#define KMD_WIN_HOST 18u

/* Argument block, at the start of the layout. Deadlines are in mtime ticks (50 MHz). */
typedef struct {
    uint64_t run_value;     /* the run this block belongs to */
    uint64_t depth;         /* D: mailbox slots in use, 1 to KM_RING_SLOTS */
    uint64_t host_depth;    /* host command ring slots, 1 to KMD_HOST_DEPTH_MAX */
    uint64_t event_depth;   /* event ring slots in the host buffer */
    uint64_t columns;       /* column leaders that report READY, 1 to KMD_READY_MAX */
    uint64_t cores;         /* cores in core order; a team must lie inside */
    uint64_t panels;        /* multicast rectangles, 1 or 2 */
    uint32_t rect[2][4];    /* x_start, y_start, x_end, y_end of each */
    uint64_t host_noc_addr; /* NoC address of the host buffer */
    uint64_t pcie_x;        /* the PCIe tile that reaches it */
    uint64_t pcie_y;
    uint64_t t_start;   /* deadline for the hello and every READY */
    uint64_t t_record;  /* a record's deadline until a KM_SEG_DEADLINE command */
    uint64_t t_idle;    /* deadline with no command and nothing in flight */
    uint64_t state_ddr; /* offset in this cluster's DRAM bank of KMD_STATE_BYTES */
} kmd_args_t;

/* The kernel keeps its state in the cluster's DRAM bank, through the cached Memory
 * Port: the host gives it KMD_STATE_BYTES there, 64-byte aligned (LIM is uncached). */
#define KMD_STATE_BYTES 1024u
