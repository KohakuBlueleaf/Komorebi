/* Landing experiment: layout and record contents shared by the X280 image (C) and the
 * Tensix kernel (C++). The host mirrors the counter indices in landing.py. */
#pragma once

#include <stdint.h>

/* Control block (CB kCtlCb, 1024 B, same L1 address on every core): bytes 0..255 are
 * four 64 B record slots; protocol B's sequence word is at LANDING_SEQ_OFF. */
#define LANDING_SLOT_BYTES 64u
#define LANDING_SLOTS 4u
#define LANDING_SEQ_OFF 0x280u

/* Payload word w (0..15) of record seq: a mix of seq, w and the run tag, so a core can
 * recompute every word and a record left in L1 by an earlier run never verifies. */
static inline uint32_t landing_word(uint32_t seq, uint32_t w, uint32_t tag) {
    uint32_t x = seq * 0x9E3779B1u ^ (w + 1u) * 0x85EBCA77u ^ tag * 0xC2B2AE3Du;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    return x;
}

/* Echo word (protocol A, words 0, 4, 8, 12) and sequence word (protocol B). */
static inline uint32_t landing_echo(uint32_t seq, uint32_t tag) {
    return (tag << 16) | (seq & 0xFFFFu);
}

/* Per-core counters: 32 words, written to L1 and then to page `me` of a DRAM tensor. */
enum {
    LC_MAGIC = 0,    /* LANDING_MAGIC | core index: the kernel reached its end */
    LC_ACCEPTED = 1, /* records accepted */
    LC_TORN = 2,     /* accepted records with a payload word that did not verify */
    LC_EARLY = 3,    /* A: polls that saw 1 to 3 of the 4 echo words */
    LC_IDLE_LO = 4,  /* polls that saw nothing of the record yet (64 bit) */
    LC_IDLE_HI = 5,
    LC_CHECK_LO = 6, /* sum of wall-clock cycles of the successful check (STAMP) */
    LC_CHECK_HI = 7,
    LC_CHECK_MAX = 8,
    LC_CHECK_MIN = 9,
    LC_CHECKS = 10,   /* number of stamped checks */
    LC_TORN_SEQ = 11, /* first torn record: seq, word index, value seen */
    LC_TORN_WORD = 12,
    LC_TORN_VALUE = 13,
    LC_HEALED = 14,     /* torn records whose payload verified on a later look */
    LC_HEAL_POLLS = 15, /* most looks a torn record needed to verify */
    LC_LOST = 16,       /* bounded wait gave up (record never accepted) */
    LC_LOST_SEQ = 17,
    LC_STAMP_COST = 18, /* cycles between two back-to-back wall-clock reads */
    LC_TAG = 19,
    LC_LAST_SEQ = 20, /* last seq completed */
    LC_WORDS = 32
};

#define LANDING_MAGIC 0x1A4D0000u
