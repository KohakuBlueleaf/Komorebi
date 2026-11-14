/* Thread split example: what the X280 thread, the grid's kernels and the overlays
 * share. Plain C. */
#pragma once

#include <stdint.h>

/* PASS (the thread's, to the whole group): a0 units, a1 the tiles, a2 the pass's first
 * page, a3 the run, a4 the penalty, a5 the lines, a6 the pass, a7 1: it only pulls. */
enum { TS_HANDLER_PASS = 1 };

/* The overlays, by the kind of unit each is fast on; the word of a pull names one. */
enum {
    TS_OVERLAY_PLAIN = 1,  /* reader part: writes the result tiles value by value */
    TS_OVERLAY_SPARSE = 2, /* reader part: writes them two values a word */
    TS_OVERLAY_DENSE = 3,  /* reader, compute and writer parts: the vector unit */
    TS_OVERLAYS = 3
};

/* Buffers: the dense overlay's input and output tiles, a reader's result tile, the jobs
 * the reader hands compute and the writer, the reader's line. */
enum {
    TS_CB_IN = 0,
    TS_CB_OUT = 16,
    TS_CB_TILE = 1,
    TS_CB_JOB_COMPUTE = 2,
    TS_CB_JOB_WRITER = 3,
    TS_CB_LINE = 4,
    TS_TILE_VALUES = 1024,
    TS_TILE_BYTES = 2048
};

/* A job: the overlay (0: the pass ended), the range's first unit and count. */
enum { TS_JOB_OVERLAY, TS_JOB_FIRST, TS_JOB_COUNT, TS_JOB_WORDS = 16 };

/* Unit u's kind, 0 to 2: runs of `run` units of one kind, mixed from the run's number.
 * Overlay k is fast on kind k - 1 and pays `penalty` cycles on the others. */
#define TS_MIX(x) ((((x) * 0x9E3779B1u) ^ (((x) * 0x9E3779B1u) >> 15)) * 0x85EBCA77u)
#define TS_KIND(unit, run) ((TS_MIX((unit) / (run)) >> 16) % 3u)

/* Value j of unit u's result tile is 2 * TS_SMALL(u, j): 2 to 14, exact in bfloat16. */
#define TS_SMALL(unit, index) (1u + ((unit) + (index)) % 7u)

/* The argument of every overlay's reader part. */
typedef struct {
    uint32_t first, count; /* the range */
    uint32_t tiles, page;  /* the tiles' DRAM address, the pass's first page */
    uint32_t run, penalty;
} ts_range;

/* The dense overlay's compute and writer arguments. */
typedef struct {
    uint32_t count;
} ts_scale;

typedef struct {
    uint32_t first, count, tiles, page;
} ts_drain;

/* The thread's words. Pass modes: one overlay for everything, the host's split (a
 * fixed overlay per core), the thread's choice per range, and two pull-only passes. */
enum {
    TS_WORD_UNITS,
    TS_WORD_PER,
    TS_WORD_RUN,
    TS_WORD_PENALTY,
    TS_WORD_TILES,
    TS_WORD_LINES,
    TS_WORD_PULL_UNITS, /* the pull-only passes: ranges of one unit, no work */
    TS_WORD_SPLIT,      /* + c / 16: core c's overlay in bits 2 (c % 16), the host's */
    TS_WORDS = TS_WORD_SPLIT + 8
};

enum {
    TS_PASS_PLAIN,  /* every range plain */
    TS_PASS_SPARSE, /* every range sparse */
    TS_PASS_DENSE,  /* every range dense */
    TS_PASS_HOST,   /* each core the overlay the host chose for it */
    TS_PASS_THREAD, /* each range the overlay the thread chose from the core's tells */
    TS_PASS_PICK,   /* pull only, a pick without a word */
    TS_PASS_WORD,   /* pull only, a pick with a word (TS_PULL_WORD) */
    TS_PASSES
};

enum {
    TS_WORK_PASSES = 5, /* the passes with work; their tiles are checked */
    TS_PULL_WORD = 0x5EED
};

/* A core's line per pass in the lines tensor (page pass * cores + core). */
enum {
    TS_LINE_PASS,  /* 1 + the pass: the line is this run's */
    TS_LINE_UNITS, /* units run */
    TS_LINE_PLAIN, /* units per overlay */
    TS_LINE_SPARSE,
    TS_LINE_DENSE,
    TS_LINE_LOADS,  /* load() calls that loaded (the reader's part was not held) */
    TS_LINE_PULLS,  /* ranges pulled */
    TS_LINE_WAITED, /* cycles in next() */
    TS_LINE_SPAN,   /* cycles from the command's start to the end of its pulls */
    TS_LINE_WORDS,  /* words that were TS_PULL_WORD (pull-only passes) */
    TS_LINE_BAD,    /* words that named no overlay in a work pass */
    TS_LINE = 16
};
