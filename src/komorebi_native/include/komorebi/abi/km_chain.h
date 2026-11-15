/* The chain: a core runs a list of commands by itself from records in DRAM, each one
 * command and the next one's address. Plain C: the device, the X280 and the host. */
#pragma once

/* A record: 16 words, 64 bytes, 64-byte aligned, at most kChainRecordsMax a chain. */
enum {
    kChainArguments = 0,  /* words 0 to 9: the command's a0 to a9 */
    kChainGroup = 10,     /* the command's cores: first | count << 16 */
    kChainInfo = 11,      /* the info word (below) */
    kChainRanges = 12,    /* three: base | step << 12 | width << 20 | root only << 31 */
    kChainNext = 15,      /* the next record's address (same bank; 0 the last) | row */
    kChainRowMask = 0x3F, /* the next word's bits 0 to 5: the record's stamp row */
    kChainWords = 16,
    kChainRecordBytes = 64,
    kChainRecordsMax = 256
};

/* The info word's flags: its input is marked by the record before's roots once
 * written; its parts go up its group's tree (credits); its input is sent (direct). */
enum { kChainMarked = 1, kChainSends = 2, kChainDataMark = 4 };

/* Info bits 4 to 7: the command's word the chain command's a0 is ORed into (km_abi.h:
 * a0 3 to a9 15; the per-submit word); 0: none. */
enum { kChainTokenShift = 4, kChainTokenMask = 0xF };

/* Info bits 8 to 31: what a core reads of the record before's output, the group's
 * size and a group's output tiles (0: the program says them), the record's handler. */
enum {
    kChainNeedsK = 0,      /* bit 8 clear: its rank's share of range 0 (a K range) */
    kChainNeedsRanges = 1, /* bit 8 set: range i from base + step x its group */
    kChainNeedsShift = 8,
    kChainRangeCount = 3,
    kChainSizeShift = 9, /* bits 9 to 15 */
    kChainSizeMask = 0x7F,
    kChainWidthShift = 16, /* bits 16 to 23 */
    kChainWidthMask = 0xFF,
    kChainHandlerShift = 24 /* bits 24 to 31; 0: the chain command's handler */
};

/* The chain's words and two record slots on each core: an index the runtime reserves
 * for it, past a program's 0 to 28 and the runtime's 29 to 31 (the trace's). */
enum { kChainBuffer = 32 };

/* A direct input's writes go under this NoC transaction id. */
enum { kChainInputTrid = 15 };

/* Stamps (KM_CHAIN_STAMPS): at byte kChainStamps of the chain's buffer a row a kernel
 * and stamp row, word 0 its runs, the others sums of the wall clock from its start. */
enum {
    kChainStamps = 256,
    kChainStampKernels = 3, /* the reader, the writer, compute's pack */
    kChainStampRecords = 8,
    kChainStampWords = 16,
    kChainStampBytes = 1536
};

/* The stamp words the chain's own calls take (a program takes the others). */
enum {
    kChainStampTaken = 1,  /* the record taken */
    kChainStampMarked = 4, /* its input marked */
    kChainStampDone = 6,   /* the reader's record done */
    kChainStampLast = 7,   /* the next fetch; the writer's writes acknowledged */
    kChainStampMarks = 11, /* its marks issued */
    kChainStampCredit = 14 /* its wait for a credit */
};
