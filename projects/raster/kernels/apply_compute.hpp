// APPLY's compute: a copy of projects/transport/transport.hpp's steps and of the loops
// its class compiles to (README.md "APPLY").
#pragma once

#include "apply.h"
#include "apply_window.hpp"

namespace apply {

    // Each CPU's place in the entry's stream: the tile in the column, the column, the
    // product's set-up (0 none, 1 x read transposed, 2 t read as it is).
    struct Place {
        uint32_t tile = 0;
        uint32_t column = 0;
        uint32_t setup = 0;
    };

    inline Place place;
    // Pack's bit j: own sums tile j written this command.
    inline uint32_t touched[kApplyOwnWords];
    // This CPU's wall clock at the handler's start, and its cycles inside the steps
    // below this command.
    inline uint32_t started = 0;
    inline uint32_t busy = 0;

    // A step's cycles, added to busy when it returns.
    struct Busy {
        uint32_t from = km::wall();

        ~Busy() {
            busy += km::wall() - from;
        }
    };

    // At the handler's start: no sum written yet; the products' set-up made again.
    inline void begin() {
        for (uint32_t i = 0; i < kApplyOwnWords; i++) {
            touched[i] = 0;
        }
        place = Place{};
        busy = 0;
        started = km::wall();
    }

    // The set-up for products of `weights` (in0) by `vector` (in1, transposed or not).
    inline void setup(uint32_t weights, uint32_t vector, uint32_t form) {
        if (place.setup != form) {
            reconfig_data_format(vector, weights);
            matmul_block_init(weights, vector, form == 1 ? 1 : 0, 1, 1, 1);
            place.setup = form;
        }
    }

    // The column's sum (DST 0) packed onto own sums tile `at`: the first write of a
    // tile overwrites it, later ones add to it in L1.
    inline void pack_sum(uint32_t at) {
        const uint32_t bit = 1u << (at % 32);
        const bool again = (touched[at / 32] & bit) != 0;
        pack_reconfig_data_format(kCbApplySums);
        pack_reconfig_l1_acc(again ? 1 : 0);
        pack_tile<true>(0, kCbApplySums, at);
        pack_reconfig_l1_acc(0);
        touched[at / 32] |= bit;
    }

    // One tile of an entry's first stream: a dense slice's columns (K tiles each, into
    // the sums from tile r) or V's (K tiles each, into t). `count` its tiles.
    inline void first(
        uint32_t weights,
        uint32_t index,
        uint32_t xs,
        uint32_t k,
        uint32_t count,
        uint32_t r,
        uint32_t rank
    ) {
        const Busy timed;
        setup(weights, xs, 1);
        if (place.tile == 0) {
            tile_regs_acquire();
        }
        matmul_block(weights, xs, index, place.tile, 0, 1, 1, 1, 1);
        if (++place.tile < k) {
            return;
        }
        tile_regs_commit();
        tile_regs_wait();
        if (rank == 0) {
            pack_sum(r + place.column);
        } else {
            pack_reconfig_data_format(kCbApplyHidden);
            pack_tile<true>(0, kCbApplyHidden, place.column);
        }
        tile_regs_release();
        place.tile = 0;
        if (++place.column * k == count) {
            place.column = 0;
            if (rank != 0) {  // t is whole: pack hands it to unpack
                cb_reserve_back(kCbApplyReady, 1);
                cb_push_back(kCbApplyReady, 1);
            }
        }
    }

    // One tile of a low-rank entry's U^T stream: R tiles a column, into the sums
    // from tile r.
    inline void second(
        uint32_t weights, uint32_t index, uint32_t rank, uint32_t count, uint32_t r
    ) {
        const Busy timed;
        if (place.tile == 0 && place.column == 0) {  // t packed by pack
            cb_wait_front(kCbApplyReady, 1);
            cb_pop_front(kCbApplyReady, 1);
        }
        setup(weights, kCbApplyHidden, 2);
        if (place.tile == 0) {
            tile_regs_acquire();
        }
        matmul_block(weights, kCbApplyHidden, index, place.tile, 0, 0, 1, 1, 1);
        if (++place.tile < rank) {
            return;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_sum(r + place.column);
        tile_regs_release();
        place.tile = 0;
        if (++place.column * rank == count) {
            place.column = 0;
        }
    }

    // The command's end: pack has packed every sum; unpack may read them.
    inline void finish() {
        cb_reserve_back(kCbApplyReady, 1);
        cb_push_back(kCbApplyReady, 1);
        cb_wait_front(kCbApplyReady, 1);
        cb_pop_front(kCbApplyReady, 1);
    }

    // Own sums tile j, turned back, packed into y's page (`out`, a block of one tile).
    inline void write(uint32_t out, uint32_t j) {
        const Busy timed;
        reconfig_data_format_srca(kCbApplySums);
        copy_init(kCbApplySums);
        tile_regs_acquire();
        copy_tile(kCbApplySums, j, 0);
        transpose_dest_init<DST_ACCUM_MODE, true>();
        transpose_dest<DST_ACCUM_MODE, true>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(out);
        pack_tile(0, out);
        tile_regs_release();
        place.setup = 0;  // the copy and the turn changed the units' set-up
    }

    // The command's stamps (wall-clock cycles): unpack's span, busy, start and end,
    // handed to pack by mailbox, then pack's own, into the core's profile page.
    inline void hand() {
        const uint32_t ended = km::wall();
        const uint32_t words[4] = {ended - started, busy, started, ended};
        for (uint32_t word : words) {
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, word);
        }
    }

    inline void report(uint32_t page) {
        const uint32_t ended = km::wall();
        volatile tt_l1_ptr uint32_t* words = km::l1(page);
        for (uint32_t i = 0; i < 4; i++) {
            const uint32_t word = (
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            );
            if (page != 0) {
                words[i] = word;
            }
        }
        if (page != 0) {
            words[4] = ended - started;
            words[5] = busy;
            words[6] = started;
            words[7] = ended;
        }
    }

}  // namespace apply

// APPLY of one command on the three compute CPUs: the loops of the class's compute.
inline void apply_compute_command(
    const km::Compute& compute, const km_command_t& command
) {
    static_cast<void>(command);
    apply_windows();
#if defined(TRISC_MATH)
    // Dst cleared whole, as pack's release clears a section: another handler may have
    // written it directly (km::dst) and left it, and the first products add to it
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}
        .set(ADDR_MOD_1);
    TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_1, 0);
    TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH);
#endif
#if defined(TRISC_PACK)
    cb_reserve_back(kCbApplySums, kApplyOwn);  // pack writes both in place
    cb_reserve_back(kCbApplyHidden, kApplyRank);
#endif
    bool ok = true;
    apply::begin();
    for (uint32_t p = 0; ok && p < kApplyPages; p++) {
        if (!compute.wait_front(kCbApplyEntries, 1)) {
            return;
        }
        const volatile tt_l1_ptr uint32_t* page = (
            km::l1(get_tile_address(kCbApplyEntries, 0))
        );
        invalidate_l1_cache();
        const uint32_t count = page[0] <= kApplyEntriesMost ? page[0] : 0;
        for (uint32_t at = 0; ok && at < count; at++) {
            const volatile tt_l1_ptr uint32_t*
                e = (page + kApplyHead + kApplyEntryWords * at);
            const uint32_t k = e[kApplyK];
            const uint32_t a = e[kApplyA];
            const uint32_t b = e[kApplyB];
            const uint32_t r = e[kApplyR];
            const uint32_t rank = e[kApplyRankWord];
            const uint32_t held = km::block_half(kApplyXPair, kApplyReadPattern, 0);
            ok = compute.wait_front(held, kApplyKmax);
            const uint32_t blocks_a = (a + kApplyBlock - 1) / kApplyBlock;
            for (uint32_t block = 0; ok && block < blocks_a; block++) {
                const uint32_t half = (
                    km::block_half(kApplyWPair, kApplyReadPattern, block)
                );
                ok = compute.wait_front(half, kApplyBlock);
                if (!ok) {
                    break;
                }
                const uint32_t valid = (
                    block + 1 == blocks_a ? a - block * kApplyBlock : kApplyBlock
                );
                for (uint32_t tile = 0; tile < valid; tile++) {
                    apply::first(half, tile, held, k, a, r, rank);
                }
                cb_pop_front(half, kApplyBlock);
            }
            const uint32_t blocks_b = (b + kApplyBlock - 1) / kApplyBlock;
            for (uint32_t block = 0; ok && block < blocks_b; block++) {
                const uint32_t half = (
                    km::block_half(kApplyWPair, kApplyReadPattern, block)
                );
                ok = compute.wait_front(half, kApplyBlock);
                if (!ok) {
                    break;
                }
                const uint32_t valid = (
                    block + 1 == blocks_b ? b - block * kApplyBlock : kApplyBlock
                );
                for (uint32_t tile = 0; tile < valid; tile++) {
                    apply::second(half, tile, rank, b, r);
                }
                cb_pop_front(half, kApplyBlock);
            }
            if (ok) {  // the held block freed at the entry's end
                cb_pop_front(held, kApplyKmax);
            }
        }
        cb_pop_front(kCbApplyEntries, 1);
    }
    if (!compute.wait_front(kCbApplyHeadPage, 1)) {
        return;
    }
    const volatile tt_l1_ptr uint32_t* head = (
        km::l1(get_tile_address(kCbApplyHeadPage, 0))
    );
    invalidate_l1_cache();
    const uint32_t own = head[kApplyHeadOwn];
    apply::finish();
    for (uint32_t j = 0; ok && j < own; j++) {
        const bool room = compute.reserve_back(kCbApplyY, 1);
        apply::write(kCbApplyY, j);
        if (room) {
            cb_push_back(kCbApplyY, 1);
        }
    }
    cb_pop_front(kCbApplyHeadPage, 1);
#if defined(TRISC_UNPACK)
    apply::hand();
#endif
    const bool room = compute.reserve_back(kCbApplyProf, 1);
#if defined(TRISC_PACK)
    apply::report(room ? get_local_cb_interface(kCbApplyProf).fifo_wr_ptr << 4 : 0);
#endif
    if (room) {
        cb_push_back(kCbApplyProf, 1);
    }
}
