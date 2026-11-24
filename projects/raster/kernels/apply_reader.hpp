// APPLY's mover A (the reader): the table pages to compute, its share of x's and the
// weights' streams; the includer gives kApplyArgsAt, the tensors' first compile arg.
#pragma once

#include "apply.h"
#include "apply_window.hpp"

// APPLY of one command on mover A. False: gave up (recorded).
bool apply_reader_command(const km::Reader& reader, const km_command_t& command) {
    using T = ApplyTensors<kApplyArgsAt>;
    const typename T::X x(command.a0);
    const typename T::W w(command.a1);
    const typename T::Tables tables(command.a3);
    const uint32_t rank = reader.me() - km::first(command);
    const uint32_t base = rank * kApplyPages;
    apply_windows();
    for (uint32_t p = 0; p < kApplyPages; p++) {
        if (
            !reader.reserve(kCbApplyEntries, 1) ||
            !km::read_pages(
                reader, command, tables, base + p, 1, get_write_ptr(kCbApplyEntries)
            )
        ) {
            return false;
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* page = (
            km::l1(get_write_ptr(kCbApplyEntries))
        );
        cb_push_back(kCbApplyEntries, 1);
        const uint32_t count = page[0] <= kApplyEntriesMost ? page[0] : 0;
        for (uint32_t at = 0; at < count; at++) {
            const volatile tt_l1_ptr uint32_t*
                e = (page + kApplyHead + kApplyEntryWords * at);
            const uint32_t src = e[kApplySrc];
            const uint32_t k = e[kApplyK] < kApplyKmax ? e[kApplyK] : kApplyKmax;
            KM_STREAM(
                xs,
                km::kMoveRead,
                x,
                src,
                k,
                kApplyKmax,
                1,
                kApplyXPair,
                kApplyReadPattern
            );
            if (k == 0) {  // an empty block: compute takes it all the same
                if (!reader.reserve(kApplyXPair.mover_a, kApplyKmax)) {
                    return false;
                }
                cb_push_back(kApplyXPair.mover_a, kApplyKmax);
            } else if (!km::move(reader, xs)) {
                return false;
            }
            KM_STREAM(
                vt,
                km::kMoveRead,
                w,
                e[kApplyW],
                e[kApplyA],
                kApplyBlock,
                1,
                kApplyWPair,
                kApplyReadPattern
            );
            if (!km::move(reader, vt)) {
                return false;
            }
            // the same stream armed again: the entry's U^T pages after its V's
            KM_STREAM(
                ut,
                km::kMoveRead,
                w,
                e[kApplyU],
                e[kApplyB],
                kApplyBlock,
                1,
                kApplyWPair,
                kApplyReadPattern
            );
            if (!km::move(reader, ut)) {
                return false;
            }
        }
    }
    if (
        !reader.reserve(kCbApplyHeadPage, 1) ||
        !km::read_pages(
            reader, command, tables, base, 1, get_write_ptr(kCbApplyHeadPage)
        )
    ) {
        return false;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* head = km::l1(get_write_ptr(kCbApplyHeadPage));
    cb_push_back(kCbApplyHeadPage, 1);
    return true;
}
