/* The decode over two cards (form A: every product split by its output columns), what
 * its thread, the crossing and Python share, beside decode.h. Plain C. */
#pragma once

#include <stdint.h>

/* CROSS, SPLIT's op on the leader: a0 the activation's word, a1 this card's first
 * page, a2 its pages, a7/a8 the squares' word and pages, a9 the far leader. */
enum { DC2_OVERLAY_CROSS = 5 };

/* The thread's words past decode.h's: the cross plan and op word, this island's first
 * rank, the far and the local leader's ranks on their islands, this card (0, 1). */
enum {
    DC2_WORD_PLAN_CROSS = 24,
    DC2_WORD_CROSS,
    DC2_WORD_FIRST,
    DC2_WORD_FAR_LEADER,
    DC2_WORD_LEADER,
    DC2_WORD_CARD,
    DC2_WORD_PLAN_O,     /* o's own plan (attend's is DC_WORD_PLAN_ATTEND) */
    DC2_WORD_GATE_TILES, /* a card's gated tiles (gate_up's output) */
    DC2_WORDS
};

/* Region `tensors`: this island's bank addresses (a half: its pages of the whole, whole
 * pages a bank) and the crossings' table words. */
enum {
    DC2_TENSOR_X,
    DC2_TENSOR_X_HALF,
    DC2_TENSOR_X1,
    DC2_TENSOR_X1_HALF,
    DC2_TENSOR_A,
    DC2_TENSOR_A_HALF,
    DC2_TENSOR_GATED,
    DC2_TENSOR_GATED_HALF,
    DC2_TENSOR_QKV, /* this card's q, k, v: its 8 query heads' and 4 KV heads' */
    DC2_TENSOR_SQUARES1,
    DC2_TENSOR_SQUARES2,
    DC2_TENSOR_CONSTANTS,
    DC2_TENSOR_ATTEND,
    DC2_TENSOR_QK,
    DC2_WORD_X, /* the table words: x, x1, a, gated, squares1, squares2 */
    DC2_WORD_X1,
    DC2_WORD_A,
    DC2_WORD_GATED,
    DC2_WORD_SQUARES1,
    DC2_WORD_SQUARES2,
    DC2_TENSORS = 24
};

/* The squares a card's groups write (pages 0 to 3 of its copy); the far card's land at
 * 4 to 7 of it, so each card sums its own first. */
enum { DC2_SQUARES_HALF = 4 };

/* A cross record's a9 bit: serial, island 1 posts once island 0's half is there. */
#define DC2_CROSS_SERIAL 0x80000000u

/* LAYERS2' a4 (past decode.h's chain bits): 0x10000 each crossing local (its leader
 * marks without crossing: the layer's records alone). */
enum { DC2_LAYERS_LOCAL = 0x10000 };

/* A layer's records in order; a crossing's stamp row is its own (5 to 8). */
enum {
    DC2_RECORD_QKV,
    DC2_RECORD_ATTEND,
    DC2_RECORD_G1,
    DC2_RECORD_O,
    DC2_RECORD_G2,
    DC2_RECORD_GATE_UP,
    DC2_RECORD_G3,
    DC2_RECORD_DOWN,
    DC2_RECORD_G4,
    DC2_RECORDS
};
