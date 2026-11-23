// Rasterizer compute: the unpack thread waits for list pages and hands each to the pack
// thread, which fills tiles (or runs the arithmetic check) on the vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef RASTER_SPREAD
#include "api/compute/tile_move_copy.h"
#endif
#include "komorebi/device/km_worker.hpp"
#include "raster.h"
#ifdef TRISC_PACK
#include "fill.hpp"
#endif

namespace {

    // A give-up into the writer's gave-up word (the first one is kept) and the stop
    // word.
    void give_up(uint32_t code, uint32_t where) {
        const uint32_t control = get_arg_val<uint32_t>(0);
        volatile uint32_t* state = (
            reinterpret_cast<volatile uint32_t*>(control + KM_CONTROL_STATE)
        );
        if (state[km::kStWriter] == 0) {
            state[km::kStWriter] = code | where << 8;
        }
        *reinterpret_cast<volatile uint32_t*>(control + KM_CONTROL_STOP) = 1;
    }

    // Polls ready() for at most kWaitCycles. False: gave up (recorded at `where`).
    template <typename Ready>
    bool timed(Ready ready, uint32_t where) {
        const uint32_t started = km::wall();
        while (!ready()) {
            if (km::wall() - started > kWaitCycles) {
                give_up(kGaveWait, where);
                return false;
            }
        }
        return true;
    }

    bool mail_from(ckernel::ThreadId thread, uint32_t where) {
        return timed(
            [&] {
                return ckernel::mailbox_not_empty(thread);
            },
            where
        );
    }

#ifdef RASTER_SPREAD
    // The spread tile's hand-over: landed (posted by the unpacker when it has written
    // Dst tile 7), free (posted by the pack thread when it has loaded it).
    constexpr uint32_t kLanded = ckernel::semaphore::UNPACK_TO_DEST;
    constexpr uint32_t kFree = ckernel::semaphore::MATH_DONE;
#endif

}  // namespace

#ifdef TRISC_PACK
namespace {

    // The pack thread's view of the list: pages from the unpack thread, slot by slot
    // (nullptr: the wait for a page gave up).
    struct Pages {
        const volatile uint32_t* page = nullptr;
        uint32_t slot = kSlotsPerPage;
        uint32_t wait = 0;

        const volatile uint32_t* next() {
            if (slot == kSlotsPerPage) {
                if (page != nullptr) {
                    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
                }
                const uint32_t asked = fill::wall();
                if (!mail_from(ckernel::ThreadId::UnpackThreadId, kWhereWaitPage)) {
                    return nullptr;
                }
                page = (
                    reinterpret_cast<const volatile uint32_t*>(
                        ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
                    )
                );
                asm volatile("fence" ::: "memory");
                wait += fill::wall() - asked;
                slot = 0;
            }
            return page + kSlotWords * slot++;
        }

        // The page holding the end slot goes back with "no more".
        void end() {
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 0);
        }
    };

    uint32_t out_ptr() {
        return get_local_cb_interface(kCbOut).fifo_wr_ptr << 4;
    }

    // An item of `pages` at `destination` must end inside the output buffer; otherwise
    // the give-up goes to the writer's gave-up word and the stop flag is set.
    bool item_fits(uint32_t destination, uint32_t pages) {
        if (
            destination + pages * kOutPageBytes <=
            (get_local_cb_interface(kCbOut).fifo_limit << 4)
        ) {
            return true;
        }
        give_up(kGaveBadItem, kWhereBadOut);
        return false;
    }

    // Timed cb_reserve_back on the output buffer.
    bool reserve_out(uint32_t pages) {
        const bool reserved = timed(
            [&] {
                const uint16_t acked = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_acked_ptr(kCbOut)))
                );
                const uint16_t used = (
                    get_local_cb_interface(kCbOut).tiles_received - acked
                );
                return get_local_cb_interface(kCbOut).fifo_num_pages - used >= pages;
            },
            kWhereWaitOut
        );
        if (reserved) {
            cb_reserve_back(kCbOut, pages);
        }
        return reserved;
    }

    // Stores to L1 finished (a load of the last word written, address + 4 (words - 1),
    // used), then pushed.
    void push_out(uint32_t address, uint32_t words, uint32_t pages) {
        asm volatile("fence" ::: "memory");
        const uint32_t last_word = (
            reinterpret_cast<volatile uint32_t*>(address)[words - 1]
        );
        asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
        cb_push_back(kCbOut, pages);
    }

    // Dst words [first_word, first_word + words) to L1 at `destination`.
    void copy_out(uint32_t destination, uint32_t first_word, uint32_t words) {
        km::dst::read(
            reinterpret_cast<volatile uint32_t*>(destination), first_word, words
        );
    }

    // Rows [first_row, end_row) of each plane to the same place in the item at
    // `destination`; without the window whole planes (the writer sends the band's).
    void copy_band(uint32_t destination, uint32_t first_row, uint32_t end_row) {
#if KM_DST_WINDOW
        const uint32_t first_word = kRowBytes / 4 * first_row;
        const uint32_t words = kRowBytes / 4 * (end_row - first_row);
        for (uint32_t plane = 0; plane < kPlanes; ++plane) {
            copy_out(
                destination + 4 * (plane * kPlaneWords + first_word),
                plane * kPlaneWords + first_word,
                words
            );
        }
#else
        fill::packing(true);
        copy_out(destination, 0, kPlanes * kPlaneWords);
        fill::packing(false);
#endif
    }

    // False: an output item did not fit (recorded); the list is abandoned. `across`:
    // the grid's tiles a row (grid.h).
    bool fill_list(fill::ProfileCounters& counters, uint32_t across) {
        constexpr uint32_t background[3] = {0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A};
        Pages input;
        const uint32_t started = fill::wall();
        for (;;) {
            const volatile uint32_t* header = input.next();
            if (header == nullptr) {
                return false;
            }
            if (header[0] == kMarkEnd) {
                break;
            }
            const uint32_t tile = header[kHdrTile], records = header[kHdrCount];
            const int32_t first_row = header[kHdrRow0], end_row = header[kHdrRow1];
            if (
                !(
                    0 <= first_row && first_row < end_row && end_row <= 32 &&
                    tile < kTileCapacity
                )
            ) {
                give_up(kGaveBadItem, kWhereBadHeader);
                input.end();
                return false;
            }
            const uint32_t tile_x = tile % across, tile_y = tile / across;
            uint32_t before = fill::wall();
            fill::init_tile(tile_x, tile_y, background);
            uint32_t after = fill::wall();
            counters.init += after - before;
            for (uint32_t index = 0; index < records; ++index) {
                const volatile uint32_t* record = input.next();
                if (record == nullptr) {
                    return false;
                }
                before = fill::wall();
                uint32_t record_words[kSlotWords];
#if defined(__riscv_vector)
                fill::copy_words_rvv(
                    record_words, const_cast<const uint32_t*>(record), kSlotWords
                );
#else
                fill::fetch(record_words, record);
#endif
                const int32_t box_first_row = (
                    static_cast<int32_t>(record_words[kRecY0]) - 32 * tile_y
                );
                const int32_t box_last_row = (
                    static_cast<int32_t>(record_words[kRecY1]) - 32 * tile_y
                );
                const int32_t first_covered = (
                    box_first_row < first_row ? first_row : box_first_row
                );
                const int32_t last_covered = (
                    box_last_row > end_row - 1 ? end_row - 1 : box_last_row
                );
#ifdef RASTER_SPREAD
                // A record of kSpreadRows rows or more: its spread tile in Dst (the
                // unpack thread's post), taken.
                if (last_covered - first_covered + 1 < kSpreadRows) {
                    fill::load_triangle(record_words);
                } else {
                    const uint32_t asked = fill::wall();
                    if (!timed(
                        [] {
                            return ckernel::semaphore_read(kLanded) > 0;
                        },
                        kWhereWaitLanding
                    )) {
                        input.end();
                        return false;
                    }
                    ckernel::semaphore_get(kLanded);
                    counters.landing_wait += fill::wall() - asked;
                    fill::load_spread(record_words, kFree);
                }
#else
                fill::load_triangle(record_words);
#endif
                after = fill::wall();
                counters.triangle_setup += after - before;
                for (
                    int32_t pixel_row = first_covered; pixel_row <= last_covered;
                    ++pixel_row
                ) {
                    fill::row(pixel_row);
                    counters.row_count++;
                }
                counters.rows += fill::wall() - after;
                counters.triangles++;
            }
            before = fill::wall();
            ckernel::tensix_sync();
            after = fill::wall();
            counters.drain += after - before;
            if (!reserve_out(kPlanes)) {
                input.end();
                return false;
            }
            before = fill::wall();
            counters.wait_out += before - after;
            const uint32_t destination = out_ptr();
            if (!item_fits(destination, kPlanes)) {
                input.end();
                return false;
            }
            copy_band(destination, first_row, end_row);
            push_out(
                destination,
                (kPlanes - 1) * kPlaneWords + kRowBytes / 4 * end_row,
                kPlanes
            );
            counters.out += fill::wall() - before;
            counters.tiles++;
        }
        input.end();
        counters.wait_in = input.wait;
        counters.total = fill::wall() - started;
        return true;
    }

    bool arith_list(fill::ProfileCounters& counters) {
#if !KM_DST_WINDOW
        // Its Dst words are not whole tiles (km::dst); refused on the host (arith.py).
        Pages refused;
        if (refused.next() != nullptr) {
            refused.end();
        }
        give_up(kGaveBadItem, kWhereBadHeader);
        return false;
#else
        Pages input;
        const uint32_t started = fill::wall();
        for (;;) {
            input.slot = kSlotsPerPage;
            const volatile uint32_t* page = input.next();
            if (page == nullptr) {
                return false;
            }
            if (page[0] == kMarkEnd) {
                break;
            }
            for (uint32_t input_page = 0; input_page < kArithIn; ++input_page) {
                if (input_page > 0) {
                    input.slot = kSlotsPerPage;
                    page = input.next();
                    if (page == nullptr) {
                        return false;
                    }
                }
                km::dst::write(page, kArithLanes * input_page, kArithLanes);
            }
            km::dst::written(kArithLanes * kArithIn - 1);
            fill::arith_chunk();
            ckernel::tensix_sync();
            if (!reserve_out(kArithOut)) {
                input.end();
                return false;
            }
            const uint32_t destination = out_ptr();
            if (!item_fits(destination, kArithOut)) {
                input.end();
                return false;
            }
            km::dst::read<false>(
                reinterpret_cast<volatile uint32_t*>(destination),
                kArithLanes * kArithIn,
                kArithOut * kPlaneWords
            );
            push_out(destination, kArithOut * kPlaneWords, kArithOut);
            counters.tiles++;
        }
        input.end();
        counters.wait_in = input.wait;
        counters.total = fill::wall() - started;
        return true;
#endif
    }

    // The profile item (`pages` pages, so items stay aligned in the output CB): the
    // pack thread's counters at words 0 to 10 of its first page.
    void profile(const fill::ProfileCounters& counters, uint32_t pages) {
        if (!reserve_out(pages)) {
            return;
        }
        const uint32_t dst = out_ptr();
        if (!item_fits(dst, pages)) {
            return;
        }
        volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(dst);
        const uint32_t values[11] = {
            counters.wait_in,
            counters.init,
            counters.triangle_setup,
            counters.rows,
            counters.drain,
            counters.out,
            counters.wait_out,
            counters.tiles,
            counters.triangles,
            counters.row_count,
            counters.total
        };
        for (uint32_t word = 0; word < 32; ++word) {
            words[word] = word < 11 ? values[word] : 0;
        }
#ifdef RASTER_SPREAD
        words[kProfLanding] = counters.landing_wait;
#endif
        push_out(dst, 32, pages);
    }

}  // namespace
#endif

#if defined(TRISC_UNPACK) && defined(RASTER_SPREAD)
namespace {

    uint16_t pages_in(uint32_t cb) {
        const uint16_t received = (
            reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb)))
        );
        return static_cast<uint16_t>(received - get_local_cb_interface(cb).tiles_acked);
    }

    // The spread tile at the front of `cb` into Dst tile 7: `_llk_unpack_A_`'s 32-bit
    // unpack to Dst, at a fixed Dst address in place of the math thread's.
    void unpack_landing(uint32_t cb) {
        using namespace ckernel;
        using namespace ckernel::unpacker;
        const uint32_t operand = get_operand_id(cb);
        const uint32_t address = get_local_cb_interface(operand).fifo_rd_ptr - 1;
        const uint32_t dst_format = unpack_dst_format[operand];
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
        wait_for_next_context(2);
        cfg[
            unp_cfg_context == 0
                ? THCON_SEC0_REG3_Base_address_ADDR32
                : THCON_SEC0_REG3_Base_cntx1_address_ADDR32
        ] = address;
        semaphore_post(semaphore::UNPACK_SYNC);
        // set_dst_write_addr's steps (the address with its fixed offset of 4 x 16).
        const uint32_t dst_byte_address = 16 * (4 + kSpreadDstRow);
        TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);
        TT_SETDMAREG(
            0,
            LOWER_HALFWORD(
                canonical_unpA_z_stride(dst_format)
                << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT
            ),
            0,
            LO_16(p_gpr_unpack::TMP_LO)
        );
        TTI_WRCFG(
            p_gpr_unpack::TMP_LO,
            p_cfg::WRCFG_32b,
            UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32
        );
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(
                dst_byte_address
            );
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(
                dst_byte_address
            );
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        unpack_to_dest_tile_done(unp_cfg_context, dst_format);  // posts kLanded
        switch_config_context(unp_cfg_context);
    }

    // FILL with the spread tiles: list pages to the pack thread as the plain fill does,
    // each spread tile into Dst once the last one is given back (kFree); waits timed.
    void unpack_spread() {
        copy_init(kCbSpread);
        // A command that gave up may have left a post; the pack thread takes neither
        // before its first list page, which comes from here.
        while (ckernel::semaphore_read(kFree) > 0) {
            ckernel::semaphore_get(kFree);
        }
        while (ckernel::semaphore_read(kLanded) > 0) {
            ckernel::semaphore_get(kLanded);
        }
        bool handed = false, last_page = false, first_tile = true;
        uint32_t tiles = 0;  // even ones from the reader's buffer, odd from the writer's
        uint32_t idle_since = km::wall();
        while (!last_page) {
            bool moved = false;
            if (!handed && pages_in(kCbIn) >= 1) {
                cb_wait_front(kCbIn, 1);
                ckernel::mailbox_write(
                    ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(kCbIn, 0)
                );
                handed = moved = true;
            }
            if (handed && ckernel::mailbox_not_empty(ckernel::ThreadId::PackThreadId)) {
                const uint32_t more = (
                    ckernel::mailbox_read(ckernel::ThreadId::PackThreadId)
                );
                cb_pop_front(kCbIn, 1);
                handed = false;
                last_page = more == 0;
                moved = true;
            }
            const uint32_t from = (tiles & 1) == 0 ? kCbSpread : kCbSpreadOdd;
            if (
                pages_in(from) >= 1 &&
                (first_tile || ckernel::semaphore_read(kFree) > 0)
            ) {
                if (!first_tile) {
                    ckernel::semaphore_get(kFree);
                }
                cb_wait_front(from, 1);
                unpack_landing(from);
                cb_pop_front(from, 1);
                tiles++;
                first_tile = false;
                moved = true;
            }
            if (moved) {
                idle_since = km::wall();
            } else if (km::wall() - idle_since > kWaitCycles) {
                give_up(kGaveWait, handed ? kWhereWaitDone : kWhereWaitList);
                return;
            }
        }
        // The pack thread's last give-back, so both semaphores are 0 between commands.
        if (!first_tile &&
            timed(
                [] {
                    return ckernel::semaphore_read(kFree) > 0;
                },
                kWhereWaitSpread
            )) {
            ckernel::semaphore_get(kFree);
        }
    }

}  // namespace
#endif

// FILL, ARITH and SPIN of one command; the frame program's compute kernel calls it as
// well.
void fill_compute_command(const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    if (handler == kHandlerSpin) {
        const uint32_t cycles = command.a0;
        const uint32_t started = km::wall();
        while (km::wall() - started < cycles) {
            ;
        }
        return;
    }
    if (handler != kHandlerFill && handler != kHandlerArith) {
        return;
    }
#if defined(TRISC_UNPACK) && defined(RASTER_SPREAD)
    if (handler == kHandlerFill) {
        unpack_spread();
        return;
    }
#endif
#ifdef TRISC_UNPACK
    // Hands each page to the pack thread; frees it when the pack thread is done.
    for (;;) {
        const bool page = timed(
            [] {
                const uint16_t received = (
                    reg_read(
                        reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(kCbIn))
                    )
                );
                const uint16_t acked = get_local_cb_interface(kCbIn).tiles_acked;
                return static_cast<uint16_t>(received - acked) >= 1;
            },
            kWhereWaitList
        );
        if (!page) {
            break;
        }
        cb_wait_front(kCbIn, 1);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(kCbIn, 0)
        );
        if (!mail_from(ckernel::ThreadId::PackThreadId, kWhereWaitDone)) {
            break;
        }
        const uint32_t more = (ckernel::mailbox_read(ckernel::ThreadId::PackThreadId));
        cb_pop_front(kCbIn, 1);
        if (more == 0) {
            break;
        }
    }
#endif
#ifdef TRISC_PACK
    fill::ProfileCounters counters;
    fill::setup();
    bool filled;
    if (handler == kHandlerFill) {
        fill::edges(true);
        filled = fill_list(counters, grid_of(command.a8).across);
    } else {
        filled = arith_list(counters);
    }
    fill::finish();
    if (filled) {
        profile(counters, handler == kHandlerFill ? kPlanes : kArithOut);
    }
#endif
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t& command) {
        fill_compute_command(command);
    });
}
#endif
