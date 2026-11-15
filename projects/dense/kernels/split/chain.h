/* SPLIT's chain (SPLIT_CHAIN): a core runs a layer's commands by itself from records
 * in DRAM, each one command and the next one's address. Plain C: the X280 writes. */
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

/* The stamps (SPLIT_CHAIN_STAMPS): SPLIT's a4 kChainStampsMode writes each core's to
 * page `me` of the tensor at a0 (pages of kChainStampPage bytes) and zeroes them. */
enum { kChainStampsMode = 4, kChainStampPage = 1024 };

/* At byte kChainStamps of the chain's buffer, a row a kernel and record: word 0 its
 * runs, the others sums of the wall clock from that kernel's chain start. */
enum {
    kChainStamps = 256,
    kChainStampKernels = 3, /* the reader, the writer, compute's pack */
    kChainStampRecords = 8,
    kChainStampWords = 8,
    kChainStampBytes = 768
};

/* A mover's (Full, the reader's Last: cycles): W's half full before the mark; Done the
 * writer's root output ready, Last its writes acknowledged, the reader's next fetch. */
enum {
    kChainStampTaken = 1,
    kChainStampWeight = 2,
    kChainStampFull = 3,
    kChainStampMarked = 4,
    kChainStampMoved = 5,
    kChainStampDone = 6,
    kChainStampLast = 7
};

/* Compute's (pack): its first chunk in, its last product, the sum, the output written;
 * an operation's start and end (attend's record). */
enum {
    kChainStampFirst = 2,
    kChainStampParts = 3,
    kChainStampSum = 4,
    kChainStampOut = 5,
    kChainStampOpStart = 6,
    kChainStampOpEnd = 7
};
