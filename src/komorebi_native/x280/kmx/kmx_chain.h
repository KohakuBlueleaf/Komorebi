/* The chain from an X280 thread: a call's description written as a record into a
 * region, and the chain command that runs them (komorebi/abi/km_chain.h). */
#pragma once

#include <stdint.h>

#include "kmx.h"
#include "komorebi/abi/km_chain.h"

/* A call of a chain; what its cores read of the call before's output: a K range over
 * the group's ranks, or up to three ranges. */
typedef struct {
    const km_command_t* command;       /* handler, cores, a0 to a9 */
    uint32_t flags;                    /* KMX_CHAIN_* */
    uint32_t word;                     /* 3 to 15: ORed with the chain's a0; 0 none */
    uint32_t by_ranges;                /* 1: `ranges`; 0: k_base, k_tiles */
    uint32_t k_base, k_tiles;          /* the K range, shared out over the ranks */
    uint32_t ranges[kChainRangeCount]; /* kmx_chain_range words */
    uint32_t size,
        width;        /* the group's size, a group's output tiles (0: the program) */
    uint32_t row;     /* its stamp row, 0 to 63 */
    uint32_t handler; /* its handler; 0: the chain command's */
} kmx_chain_call;

enum {
    KMX_CHAIN_MARKED = kChainMarked,  /* waits for the marks of the call before */
    KMX_CHAIN_SENDS = kChainSends,    /* its parts go up its group's tree */
    KMX_CHAIN_DIRECT = kChainDataMark /* its input sent into its cores' L1 */
};

/* A range word: tiles base + step x group, `width` wide; `root_only`: only rank 0. */
static inline uint32_t kmx_chain_range(
    uint32_t base, uint32_t step, uint32_t width, uint32_t root_only
) {
    return (
        (base & 0xFFFu) | (step & 0xFFu) << 12 | (width & 0x7FFu) << 20 |
        (root_only ? 1u : 0u) << 31
    );
}

/* `call` as record `index` of region `region` (64 bytes each), next `next` (0: last),
 * `*address` its address as records name it; KMX_OK or KMX_BAD. The caller flushes. */
static inline int kmx_chain_record(
    int region,
    uint32_t index,
    const kmx_chain_call* call,
    uint32_t next,
    uint32_t* address
) {
    uint64_t bytes = 0;
    uint32_t* records = (uint32_t*)kmx_region(region, &bytes);
    km_command_t place;
    if (
        records == 0 || (uint64_t)(index + 1u) * kChainRecordBytes > bytes ||
        call->word > kChainTokenMask || call->size > kChainSizeMask ||
        call->width > kChainWidthMask || call->row > kChainRowMask ||
        call->handler > 0xFFu || next % kChainRecordBytes != 0u ||
        kmx_block(
            &place, region, (uint64_t)index * kChainRecordBytes, kChainRecordBytes
        ) != KMX_OK
    ) {
        return KMX_BAD;
    }
    uint32_t* record = records + kChainWords * index;
    const km_command_t* command = call->command;
    const uint32_t arguments[10] = {
        command->a0,
        command->a1,
        command->a2,
        command->a3,
        command->a4,
        command->a5,
        command->a6,
        command->a7,
        command->a8,
        command->a9
    };
    for (uint32_t word = 0u; word < 10u; word++) {
        record[kChainArguments + word] = arguments[word];
    }
    record[kChainGroup] = command->group;
    record[kChainInfo] = (
        call->flags | call->word << kChainTokenShift |
        (call->by_ranges ? (uint32_t)kChainNeedsRanges : 0u) << kChainNeedsShift |
        call->size << kChainSizeShift | call->width << kChainWidthShift |
        call->handler << kChainHandlerShift
    );
    for (uint32_t range = 0u; range < kChainRangeCount; range++) {
        record[kChainRanges + range] = (
            call->by_ranges
                ? call->ranges[range]
                : range == 0u
                    ? kmx_chain_range(call->k_base, 0u, call->k_tiles, 0u)
                    : 0u
        );
    }
    record[kChainNext] = next | call->row;
    *address = place.a7;
    return KMX_OK;
}

/* The chain command: `handler` on cores [0, cores), a4 `mode` (the program's), a0 the
 * per-submit word, a1 `a1`, from region `region`'s record 0. KMX_OK, or KMX_BAD. */
static inline int kmx_chain_command(
    km_command_t* chain,
    int region,
    uint32_t handler,
    uint32_t cores,
    uint32_t mode,
    uint32_t a0,
    uint32_t a1
) {
    kmx_command(chain, 0u, handler, 0u, cores);
    if (kmx_block(chain, region, 0u, kChainRecordBytes) != KMX_OK) {
        return KMX_BAD;
    }
    chain->a0 = a0;
    chain->a1 = a1;
    chain->a4 = mode;
    return KMX_OK;
}

/* A chain command of `records` records: past kChainRecordsMax it takes one sequence
 * number for each kChainRecordsMax (KM_FLAG_SPAN, a9), else the reader refuses it. */
static inline void kmx_chain_span(km_command_t* chain, uint32_t records) {
    const uint32_t span = (records + kChainRecordsMax - 1u) / kChainRecordsMax;
    if (span > 1u) {
        chain->hdr |= KM_FLAG_SPAN << 8;
        chain->a9 = span - 1u;
    }
}
