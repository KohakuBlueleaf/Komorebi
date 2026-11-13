/* For both Tensix kernels: CB indices, the broadcast buffer layout, the block contents
 * and the counter indices (collision.py mirrors the CB indices and counter names). */
#pragma once

#include <stdint.h>

/* Circular buffers (the template's control block is CB 8). */
#define kRecCntCb 9u      /* command kernel counters, 32 words */
#define kShareCb 11u      /* word 0: commands done, written by the command kernel */
#define kBroadcastCnt 12u /* broadcast kernel counters, 32 words */
#define kBroadcastCb 13u  /* broadcast buffer, laid out below */

/* Broadcast buffer (same L1 address on every core): byte offsets. */
#define BROADCAST_VALID 0u   /* VALID word, written by the sender's multicast */
#define BROADCAST_SRC 64u    /* source word of this core's marker at its sender */
#define BROADCAST_DONE 128u  /* sender only: 16 B per receiver rank 1..9, its marker */
#define BROADCAST_STATE 320u /* phase, block, tag: read by peers after a give-up */
#define BROADCAST_PEER 384u  /* a peer's state words, read after a give-up */
#define BROADCAST_BLOCK 512u /* the block, B bytes */
#define BROADCAST_MAX_RANKS 10u

/* VALID and marker words: 15 bits of tag, a stop bit, 16 bits of block number. Block
 * k's VALID equals its marker; stop(k) tells receivers that block k never comes. */
static inline uint32_t broadcast_marker(uint32_t k, uint32_t tag) {
    return ((tag & 0x7FFFu) << 17) | (k & 0xFFFFu);
}

static inline uint32_t broadcast_stop(uint32_t k, uint32_t tag) {
    return broadcast_marker(k, tag) | (1u << 16);
}

static inline uint32_t broadcast_commands_done(uint32_t tag) {
    return 0xD0000000u | (tag & 0xFFFFu);
}

/* Word w of block k: a per-block hash of k and the tag plus w times an odd constant,
 * so every word differs per block, per word and per run and costs one add to check. */
static inline uint32_t broadcast_base(uint32_t k, uint32_t tag) {
    uint32_t x = k * 0x9E3779B1u ^ tag * 0xC2B2AE3Du ^ 0x6C616E65u;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    return x;
}

#define BROADCAST_STEP 0x85EBCA77u

/* Phases in the state words (BROADCAST_STATE + 0), with the block number at + 4. */
enum {
    LP_START = 1,
    LP_WAIT_MARKERS = 2,   /* sender: waiting for every receiver's marker of k */
    LP_FILL = 3,           /* sender: writing block k */
    LP_BARRIER_BLOCK = 4,  /* sender: about to barrier after block k's multicast */
    LP_BARRIER_VALID = 5,  /* sender: about to barrier after VALID k's multicast */
    LP_WAIT_VALID = 6,     /* receiver: waiting for VALID k */
    LP_VERIFY = 7,         /* receiver: checking block k */
    LP_BARRIER_MARKER = 8, /* receiver: about to barrier after its marker of k */
    LP_END = 9,
    LP_GAVE_UP = 10
};

/* Where a timed wait gave up (LN_WHERE). */
enum {
    LW_READY = 1,  /* sender: start-up markers (block 0) */
    LW_MARKER = 2, /* sender: markers of block k */
    LW_VALID = 3   /* receiver: VALID k */
};

/* Broadcast kernel counters: 32 words, written to L1 and then to page `me` of a DRAM
 * tensor. */
enum {
    LN_MAGIC = 0,  /* BROADCAST_MAGIC | core index: the kernel reached its end */
    LN_ROLE = 1,   /* 0 idle, 1 sender, 2 receiver */
    LN_BLOCKS = 2, /* blocks completed (sender: all marked; receiver: checked) */
    LN_BAD_BLOCKS = 3,
    LN_BAD_WORDS = 4,
    LN_BAD_K = 5, /* first bad block: block, word index, value seen */
    LN_BAD_W = 6,
    LN_BAD_V = 7,
    LN_WHERE = 8, /* LW_* of the timed wait that gave up, 0 if none */
    LN_WHERE_K = 9,
    LN_MISSING = 10,    /* sender: markers still missing when it gave up */
    LN_PEER = 11,       /* rank of the peer whose state words were read */
    LN_PEER_PHASE = 12, /* its phase, or all ones if the read did not complete */
    LN_PEER_K = 13,
    LN_T_LO = 14, /* wall-clock cycles from block 1 to the end of the last block */
    LN_T_HI = 15,
    LN_STOPPED = 16, /* 1: ended by the commands-done stop */
    LN_TAG = 17,
    LN_WAIT_MAX = 18, /* longest single timed wait, cycles */
    LN_BUSY_LO = 19,  /* stream sender: cycles inside block multicast + barrier */
    LN_BUSY_HI = 20,
    LN_CHECK_LO = 21, /* stream receiver: cycles of full body checks and poisoning */
    LN_CHECK_HI = 22,
    LN_FULL = 23, /* stream receiver: blocks whose whole body was checked */
    LN_WORDS = 32
};

#define BROADCAST_MAGIC 0x1A9E0000u

/* Stream mode: slot s at BROADCAST_BLOCK + s * B; a 64 B header (marker, block number),
 * then a body filled once per run (receivers poison it with the words inverted). */
#define STREAM_HEADER 64u

static inline uint32_t stream_body(uint32_t w, uint32_t tag) {
    return broadcast_base(0xFFFFFFFFu, tag) + w * BROADCAST_STEP;
}
