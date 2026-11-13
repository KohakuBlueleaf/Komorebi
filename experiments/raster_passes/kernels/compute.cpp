// Rasterizer compute: the unpack thread waits for list pages and hands each to the pack
// thread, which fills tiles (or runs the arithmetic check) on the vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
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

    // Words [first_word, first_word + words) of the Dst window to L1 at `destination`:
    // the vector extension when the build has it.
    void copy_out(uint32_t destination, uint32_t first_word, uint32_t words) {
#if defined(__riscv_vector)
        fill::copy_words_rvv(
            reinterpret_cast<uint32_t*>(destination),
            const_cast<const uint32_t*>(fill::win() + first_word),
            words
        );
#else
        fill::copy_words(
            reinterpret_cast<volatile uint32_t*>(destination),
            fill::win() + first_word,
            words
        );
#endif
    }

    // Rows [first_row, end_row) of each plane to the same place in the item at
    // `destination`.
    void copy_band(uint32_t destination, uint32_t first_row, uint32_t end_row) {
        const uint32_t first_word = kRowBytes / 4 * first_row;
        const uint32_t words = kRowBytes / 4 * (end_row - first_row);
        for (uint32_t plane = 0; plane < kPlanes; ++plane) {
            copy_out(
                destination + 4 * (plane * kPlaneWords + first_word),
                plane * kPlaneWords + first_word,
                words
            );
        }
    }

    // False: an output item did not fit (recorded); the list is abandoned.
    bool fill_list(fill::ProfileCounters& counters) {
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
                !(0 <= first_row && first_row < end_row && end_row <= 32 && tile < 1024)
            ) {
                give_up(kGaveBadItem, kWhereBadHeader);
                input.end();
                return false;
            }
            const uint32_t tile_x = tile % 32, tile_y = tile / 32;
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
                fill::load_triangle(record_words);
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
                fill::copy_words(
                    fill::win() + kArithLanes * input_page, page, kArithLanes
                );
            }
            fill::order(kArithLanes * kArithIn - 1);
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
            fill::copy_words(
                reinterpret_cast<volatile uint32_t*>(destination),
                fill::win() + kArithLanes * kArithIn,
                kArithOut * kPlaneWords
            );
            push_out(destination, kArithOut * kPlaneWords, kArithOut);
            counters.tiles++;
        }
        input.end();
        counters.wait_in = input.wait;
        counters.total = fill::wall() - started;
        return true;
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
        push_out(dst, 32, pages);
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
        filled = fill_list(counters);
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
