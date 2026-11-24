// APPLY's mover B (the writer): its own copy of the table pages, its share of the read
// streams, y's tiles and the profile page written; kApplyArgsAt from the includer.
#pragma once

#include "apply.h"
#include "apply_window.hpp"

// APPLY of one command on mover B. False: gave up (recorded).
bool apply_writer_command(const km::Writer& writer, const km_command_t& command) {
    using T = ApplyTensors<kApplyArgsAt>;
    const typename T::X x(command.a0);
    const typename T::W w(command.a1);
    const typename T::Y y(command.a2);
    const typename T::Tables tables(command.a3);
    const typename T::Profile profile(command.a4);
    const uint32_t rank = writer.me() - km::first(command);
    const uint32_t base = rank * kApplyPages;
    apply_windows();
    for (uint32_t p = 0; p < kApplyPages; p++) {
        if (!km::read_pages(
            writer, command, tables, base + p, 1, get_write_ptr(kCbApplyEntriesB)
        )) {
            return false;
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* page = (
            km::l1(get_write_ptr(kCbApplyEntriesB))
        );
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
            if (k != 0 && !km::move(writer, xs)) {
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
            if (!km::move(writer, vt)) {
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
            if (!km::move(writer, ut)) {
                return false;
            }
        }
    }
    if (!km::read_pages(
        writer, command, tables, base, 1, get_write_ptr(kCbApplyHeadPageB)
    )) {
        return false;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* head = km::l1(get_write_ptr(kCbApplyHeadPageB));
    const uint32_t first = head[kApplyHeadFirst];
    const uint32_t own = (
        head[kApplyHeadOwn] < kApplyOwn ? head[kApplyHeadOwn] : kApplyOwn
    );
    // y's tiles and the profile page as pack fills them, each written and popped
    for (uint32_t j = 0; j < own; j++) {
        if (
            !writer.wait_front(kCbApplyY, 1) ||
            !km::write_pages(writer, command, y, first + j, 1, get_read_ptr(kCbApplyY))
        ) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbApplyY, 1);
    }
    if (
        !writer.wait_front(kCbApplyProf, 1) ||
        !km::write_pages(writer, command, profile, rank, 1, get_read_ptr(kCbApplyProf))
    ) {
        return false;
    }
    noc_async_write_barrier();
    cb_pop_front(kCbApplyProf, 1);
    return true;
}
