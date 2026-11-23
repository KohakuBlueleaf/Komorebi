// The transport apply's compute steps (transport.py's bodies): the turned products of
// each entry a column at a time, packed onto the own sums; README.md, "The handler".
#pragma once

#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "komorebi/device/core/km_core.hpp"

namespace transport {

    // Each CPU's place in the entry's stream: the tile in the column, the column, the
    // product's set-up (0 none, 1 x read transposed, 2 t read as it is).
    struct Place {
        uint32_t tile = 0;
        uint32_t column = 0;
        uint32_t setup = 0;
    };

    inline Place place;
    // Pack's bit j: own sums tile j written this command.
    inline uint32_t touched = 0;
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
        touched = 0;
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
    template <uint32_t sums>
    inline void pack_sum(uint32_t at) {
        const bool again = (touched >> at) & 1;
        pack_reconfig_data_format(sums);
        pack_reconfig_l1_acc(again ? 1 : 0);
        pack_tile<true>(0, sums, at);
        pack_reconfig_l1_acc(0);
        touched |= 1u << at;
    }

    // One tile of an entry's first stream: a dense slice's columns (K tiles each, into
    // the sums from tile r) or V's (K tiles each, into t). `count` its tiles.
    template <uint32_t sums, uint32_t hidden, uint32_t ready>
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
            pack_sum<sums>(r + place.column);
        } else {
            pack_reconfig_data_format(hidden);
            pack_tile<true>(0, hidden, place.column);
        }
        tile_regs_release();
        place.tile = 0;
        if (++place.column * k == count) {
            place.column = 0;
            if (rank != 0) {  // t is whole: pack hands it to unpack
                cb_reserve_back(ready, 1);
                cb_push_back(ready, 1);
            }
        }
    }

    // One tile of a low-rank entry's U^T stream: R tiles a column, into the sums
    // from tile r.
    template <uint32_t sums, uint32_t hidden, uint32_t ready>
    inline void second(
        uint32_t weights, uint32_t index, uint32_t rank, uint32_t count, uint32_t r
    ) {
        const Busy timed;
        if (place.tile == 0 && place.column == 0) {  // t packed by pack
            cb_wait_front(ready, 1);
            cb_pop_front(ready, 1);
        }
        setup(weights, hidden, 2);
        if (place.tile == 0) {
            tile_regs_acquire();
        }
        matmul_block(weights, hidden, index, place.tile, 0, 0, 1, 1, 1);
        if (++place.tile < rank) {
            return;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_sum<sums>(r + place.column);
        tile_regs_release();
        place.tile = 0;
        if (++place.column * rank == count) {
            place.column = 0;
        }
    }

    // The command's end: pack has packed every sum; unpack may read them.
    template <uint32_t ready>
    inline void finish() {
        cb_reserve_back(ready, 1);
        cb_push_back(ready, 1);
        cb_wait_front(ready, 1);
        cb_pop_front(ready, 1);
    }

    // Own sums tile j, turned back, packed into y's page (`out`, a block of one tile).
    template <uint32_t sums>
    inline void write(uint32_t out, uint32_t j) {
        const Busy timed;
        reconfig_data_format_srca(sums);
        copy_init(sums);
        tile_regs_acquire();
        copy_tile(sums, j, 0);
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

}  // namespace transport
