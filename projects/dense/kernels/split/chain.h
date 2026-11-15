/* SPLIT's chain (SPLIT_CHAIN): the framework's chain (komorebi/abi/km_chain.h) with
 * the split's commands as records. Plain C: the X280 writes. */
#pragma once

#include "komorebi/abi/km_chain.h"

/* SPLIT's a4 kChainMode: the first record at a7, its DRAM bank in a8 as a block's; a1
 * kChainAheadOff: no W read a record early; kChainDataDram: sent inputs' DRAM acked. */
enum { kChainMode = 3, kChainAheadOff = 1, kChainDataDram = 2 };

/* kChainDataMark: the roots before write a core's K range of input to its arrival
 * buffer from this page on (attend's merge takes 0 to 47). */
enum { kChainInputPage = 48 };

/* The stamps (SPLIT_CHAIN_STAMPS): SPLIT's a4 kChainStampsMode writes each core's to
 * page `me` of the tensor at a0 (pages of kChainStampPage bytes) and zeroes them. */
enum { kChainStampsMode = 4, kChainStampPage = 2048 };

/* A mover's (Full: cycles): W's first block, its half full before the mark, the moves
 * done; the writer's (Done) root output ready. The chain's own: km_chain.h. */
enum { kChainStampWeight = 2, kChainStampFull = 3, kChainStampMoved = 5 };

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

/* The hand-over's: round r's arrival (reader) and add (compute) at 8 + r, an emit, the
 * root's writes issued, an operation's overlay loaded and part called. */
enum {
    kChainStampRound = 8,
    kChainStampRounds = 4,
    kChainStampEmit = 8,
    kChainStampEmitted = 9,
    kChainStampWritten = 10,
    kChainStampOpLoaded = 12,
    kChainStampOpCalled = 13,
    kChainStampFirstProduct = 13, /* compute's: its first chunk's product done */
    kChainStampTakes = 14, /* compute's rounds summed: taking, then adding (15) */
    kChainStampAdds = 15
};
