/* SPLIT's chain (SPLIT_CHAIN): a core runs a layer's commands from records in DRAM by
 * itself, each record one command and the next one's address. Plain C (the X280
 * writes). */
#pragma once

/* SPLIT's a4 kChainMode: the first record at a7, its DRAM bank in a8 as a block's. */
enum { kChainMode = 3 };

/* A record: 16 words, 64 bytes, 64-byte aligned. */
enum {
    kChainArguments = 0, /* words 0 to 9: the command's a0 to a9 (a1: its weight) */
    kChainGroup = 10,    /* the command's cores: first | count << 16 */
    kChainInfo = 11,     /* flags | needs << 8 | output tiles a group writes << 16 */
    kChainRanges = 12,   /* three: base | step << 12 | width << 20 | root only << 31 */
    kChainNext = 15,     /* the next record's address in the same bank; 0: the last */
    kChainWords = 16,
    kChainRecordBytes = 64,
    kChainRecordsMax = 16
};

/* The info word: its input is marked by the record before's roots once written; its
 * parts go up its group's tree (its receivers send credits). */
enum { kChainMarked = 1, kChainSends = 2 };

/* What a record's core reads of the record before's output (tiles of its row): its K
 * range under its plan; or the range words, range i base + step x group, width wide. */
enum { kChainNeedsK = 0, kChainNeedsRanges = 1, kChainRangeCount = 3 };

/* The buffer of a core's chain words and two record slots: the runtime's trace
 * buffer's index (0 to 28 are taken), so a chain's program runs untraced. */
enum { kChainBuffer = 31 };
