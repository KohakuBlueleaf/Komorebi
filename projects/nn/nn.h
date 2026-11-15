/* Komorebi-nn: a list of kernel calls in DRAM and the X280 thread that walks it
 * (thread/executor.c); shared by the thread and Python (nn.py). Plain C. */
#pragma once

/* An entry: 32 words, 128 bytes, at a multiple of 128 of the list region. Words 0 to
 * 15 are a chain's record as the split's chain reads it (chain.h), unchanged. */
enum {
    NN_ENTRY_ARGUMENTS = 0,    /* words 0 to 9: the command's a0 to a9 */
    NN_ENTRY_GROUP = 10,       /* its cores: first | count << 16 */
    NN_ENTRY_RECORD = 11,      /* words 11 to 14: the chain's info and range words */
    NN_ENTRY_RECORD_NEXT = 15, /* the next record of the same chain; 0: its last */
    NN_ENTRY_HEADER = 16,      /* the command's: handler | flags << 8 | program << 24 */
    NN_ENTRY_KIND = 17,        /* NN_KIND_* */
    NN_ENTRY_NEXT = 18,        /* the list's next entry: its bank address; 0: the end */
    NN_ENTRY_CHAIN_MODE = 19,  /* a chain: its command's a4 (the kernel's chain mode) */
    NN_ENTRY_CHAIN_GROUP = 20, /* a chain: its command's cores, first | count << 16 */
    NN_ENTRY_OVERLAY = 21,     /* the operation overlay it calls (the host's; 0 none) */
    NN_ENTRY_SEQUENCE = 22, /* an open list's number (DRAIN's a3); a closed list's 0 */
    NN_ENTRY_CHECK = 23,    /* NN_CHECK of the entry's words but this and the next */
    NN_ENTRY_CHAIN_TOKEN = 24, /* a chain: its a0, ORed where a record's info names */
    NN_ENTRY_CHAIN_AHEAD = 25, /* a chain: its a1 (kChainAheadOff or 0, chain.h) */
    NN_ENTRY_WORDS = 32,
    NN_ENTRY_BYTES = 128
};

/* An entry's check: the sum of word i x (2 i + 1) over its words, the link and the
 * check left out. An open list's entry is new once its sequence and its check hold. */
#define NN_CHECK_FACTOR(word) (2u * (word) + 1u)

/* An entry's kind: a command; a chain from its record on; an open list's END; an
 * anchor, never run; a PUSH or PULL whose nonce word is XORed with RUN's a3. */
enum {
    NN_KIND_NONE = 0,
    NN_KIND_COMMAND = 1,
    NN_KIND_CHAIN = 2,
    NN_KIND_END = 3,
    NN_KIND_TRANSFER = 4
};

/* The executor's handlers: RUN a closed list, DRAIN an open one; their words are in
 * the README ("The thread"). Each returns its ticks, or NN_FAILED | its status. */
enum { NN_HANDLER_RUN = 1, NN_HANDLER_DRAIN = 2 };

#define NN_FAILED 0xFFFFFFF0u

/* The thread's regions, in this order. */
enum { NN_REGION_LIST = 0, NN_REGION_STATS = 1 };

enum { NN_ENTRIES_MAX = 2048, NN_LIST_BYTES = NN_ENTRIES_MAX * NN_ENTRY_BYTES };

/* A stats slot a run: status, entries run, polls and stale reads, then the terms'
 * ticks (low, high) and each entry's command ticks. */
enum {
    NN_ST_STATUS = 0,
    NN_ST_ENTRIES = 1,
    NN_ST_POLLS = 2,
    NN_ST_STALE = 3,
    NN_ST_TERMS = 4,
    NN_TERM_TOTAL = 0,   /* the run, start to end */
    NN_TERM_FETCH = 1,   /* reading the entry and building its command */
    NN_TERM_SUBMIT = 2,  /* kmx_submit */
    NN_TERM_WAIT = 3,    /* the submit's return to the event in hand */
    NN_TERM_COMMAND = 4, /* the events' ticks: the dispatcher's submit to completion */
    NN_TERM_IDLE = 5,    /* an open list: polling for the next link */
    NN_TERM_CHECK = 6, /* an open list: reading the linked entries until they are new */
    NN_TERMS = 7,
    NN_ST_ENTRY_TICKS = 32,
    NN_SLOT_WORDS = NN_ST_ENTRY_TICKS + NN_ENTRIES_MAX,
    NN_SLOTS = 128
};
