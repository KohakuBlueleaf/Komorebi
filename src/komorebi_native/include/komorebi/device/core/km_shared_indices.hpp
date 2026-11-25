// Shared indices (komorebi.dsl.compile.shared): handlers give an index other pages; a
// core sets a handler's pages at its entry when they are not the ones it is set for.
#pragma once

#include <stdint.h>

#include "../km_worker.hpp"

namespace km {

#if !defined(KM_SHARE_WORD)
#define KM_SHARE_WORD uint32_t
#endif

    // One index of a handler's pages: the index, its pages and a page's bytes, each
    // of KM_SHARE_WORD (16 bits when the program's fit them, the header says).
    struct ShareRow {
        KM_SHARE_WORD index;
        KM_SHARE_WORD pages;
        KM_SHARE_WORD page_bytes;
    };

    // One set of pages: its rows this CPU sets, their count and the set's indices'
    // mask; set 0, the load's, has none.
    struct ShareTable {
        const ShareRow* rows;
        uint32_t count;
        uint32_t mask;
    };

    // The set of pages this CPU's buffer interfaces are set for (0: the load's).
    inline uint32_t share_now = 0;

    // This CPU's interfaces of `table`'s indices, empty, each at its index's own base
    // (limit less size), in this CPU's units (cb_addr_shift).
    inline void share_point(const ShareTable& table) {
        for (uint32_t r = 0; r < table.count; r++) {
            const ShareRow& row = table.rows[r];
            LocalCBInterface& view = get_local_cb_interface(row.index);
            const uint32_t base = view.fifo_limit - view.fifo_size;
            view.fifo_page_size = row.page_bytes >> cb_addr_shift;
            view.fifo_size = row.pages * view.fifo_page_size;
            view.fifo_limit = base + view.fifo_size;
            view.fifo_num_pages = row.pages;
            view.fifo_rd_ptr = base;
            view.fifo_wr_ptr = base;
            view.fifo_wr_tile_ptr = 0;
            view.tiles_acked_received_init = 0;  // a known value: the counters zeroed
        }
    }

    // Whether every index of `mask` is empty: its pages received all acknowledged.
    inline bool share_empty(uint32_t mask) {
        for (uint32_t cb = 0; cb < 32; cb++) {
            if (
                ((mask >> cb) & 1u) != 0 &&
                (
                    (*get_cb_tiles_received_ptr(cb) ^ *get_cb_tiles_acked_ptr(cb)) &
                    0xFFFFu
                ) != 0
            ) {
                return false;
            }
        }
        return true;
    }

    // A change is one function a CPU, out of line and never cloned: its locals are gone
    // before the handler's lines run (a handler's entry keeps the compare alone).
#if !defined(COMPILE_FOR_TRISC)
    // The reader's gate before it hands on a command of other pages: every command
    // completed, both sets' indices empty (late stores landed). False: gave up.
    template <typename Mover>
    __attribute__((noinline, noclone)) bool share_gate(
        const Mover& mover,
        const km_command_t& command,
        const ShareTable* tables,
        uint32_t pages
    ) {
        const uint32_t both = tables[share_now].mask | tables[pages].mask;
        volatile tt_l1_ptr uint32_t* state = mover.state();
        const uint32_t code = mover.poll_at(
            kWhereShare,
            [&] {
                return state[kStCompleted] == state[kStRuns] && share_empty(both);
            },
            mover.args().wait_limit
        );
        if (code != kGaveNone) {
            mover.gave_up(
                kStReader, code, kWhereShare, slot(command), command.sequence
            );
            return false;
        }
        for (uint32_t cb = 0; cb < 32; cb++) {
            if (((both >> cb) & 1u) != 0) {
                *get_cb_tiles_received_ptr(cb) = 0;
                *get_cb_tiles_acked_ptr(cb) = 0;
            }
        }
        share_point(tables[pages]);
        mover.forget(both);
        move_filled_word() &= ~both;
        move_drained_word() &= ~both;
        share_now = pages;
        return true;
    }

    // Every other CPU at a handler's entry when its pages differ: its interfaces set,
    // its counters' copies zero (the reader zeroed the counters), no wait; math none.
    template <typename Mover>
    __attribute__((noinline, noclone)) void share_set(
        const Mover& mover, const ShareTable* tables, uint32_t pages
    ) {
        const uint32_t both = tables[share_now].mask | tables[pages].mask;
        share_point(tables[pages]);
        mover.forget(both);
        move_filled_word() &= ~both;
        move_drained_word() &= ~both;
        share_now = pages;
    }
#else
    __attribute__((noinline, noclone)) inline void share_set(
        const Compute& compute, const ShareTable* tables, uint32_t pages
    ) {
#if !defined(TRISC_MATH)
        const uint32_t both = tables[share_now].mask | tables[pages].mask;
        share_point(tables[pages]);
        compute.forget(both);
#else
        static_cast<void>(compute);
        static_cast<void>(tables);
#endif
        share_now = pages;
    }
#endif

}  // namespace km
