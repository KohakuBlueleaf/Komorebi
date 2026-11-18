// Rasterizer reader: streams this core's list pages from DRAM into the input buffer,
// as its directory page names them. Its cycle counts go to the end of that page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_ask.hpp"
#include "raster.h"

// Timed noc_async_read_barrier. False: gave up (recorded in the reader's word).
bool read_done(const km::Reader& reader, uint32_t sequence) {
    const uint32_t code = reader.poll(
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        reader.args().wait_limit
    );
    if (code == km::kGaveNone) {
        return true;
    }
    reader.gave_up(km::kStReader, code, kWhereReadDone, 0, sequence);
    return false;
}

#ifdef RASTER_SPREAD
#include "spread.hpp"

// The list's slots as the pack thread walks them: records left in the item, its first
// pixel row and its rows [first_row, end_row) of the tile, the end.
struct Spread {
    uint32_t left = 0;
    int32_t top = 0, first_row = 0, end_row = 0;
    bool ended = false;
    uint32_t index = 0;  // records spread so far, here and by the writer
    uint32_t cycles = 0;
    uint32_t wait = 0;
    uint32_t records = 0;
};

// Spread record `index` (in list order): an even one into a free spread page here, an
// odd one's address to the writer, which spreads it. False: gave up.
bool spread_record(
    const km::Reader& reader, Spread& spread, const volatile uint32_t* record
) {
    const uint32_t asked = km::wall();
    const bool own = (spread.index++ & 1) == 0;
    if (!reader.reserve(own ? kCbSpread : kCbSpreadTask, 1)) {
        return false;
    }
    const uint32_t reserved = km::wall();
    if (own) {
        spread_tile(km::l1(get_write_ptr(kCbSpread)), record);
        cb_push_back(kCbSpread, 1);
        spread.records++;
    } else {
        *km::l1(get_write_ptr(kCbSpreadTask)) = reinterpret_cast<uint32_t>(record);
        cb_push_back(kCbSpreadTask, 1);
    }
    spread.wait += reserved - asked;
    spread.cycles += km::wall() - asked;
    return true;
}

// The records of the list page at `page` (headers give the counts that follow), each
// spread; false: gave up.
bool spread_page(const km::Reader& reader, Spread& spread, uint32_t page) {
    for (uint32_t slot = 0; slot < kSlotsPerPage && !spread.ended; ++slot) {
        const volatile uint32_t* words = km::l1(page + 4 * kSlotWords * slot);
        if (spread.left > 0) {
            // The rows the pack thread runs, as it computes them.
            const int32_t box_first = static_cast<int32_t>(words[kRecY0]) - spread.top;
            const int32_t box_last = static_cast<int32_t>(words[kRecY1]) - spread.top;
            const int32_t first = (
                box_first < spread.first_row ? spread.first_row : box_first
            );
            const int32_t last = (
                box_last > spread.end_row - 1 ? spread.end_row - 1 : box_last
            );
            if (
                last - first + 1 >= kSpreadRows && !spread_record(reader, spread, words)
            ) {
                return false;
            }
            spread.left--;
        } else if (words[0] == kMarkTile) {
            spread.left = words[kHdrCount];
            spread.top = static_cast<int32_t>(32 * (words[kHdrTile] / 32));
            spread.first_row = static_cast<int32_t>(words[kHdrRow0]);
            spread.end_row = static_cast<int32_t>(words[kHdrRow1]);
        } else {
            spread.ended = true;
        }
    }
    return true;
}
#endif

// The directory pages and the list pages (km::Tensor): compile arguments from 0.
using Directories = km::Tensor<0>;
using Lists = km::Tensor<Directories::next_compile_time_args_offset()>;

// FILL and ARITH of one command; the frame program's reader calls it as well.
void fill_reader_command(const km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != kHandlerFill && km::handler(command) != kHandlerArith) {
        return;
    }
    const uint32_t directory_address = command.a0, lists_address = command.a1;
    const uint32_t started = km::wall();
    const uint32_t scratch = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    if (!km::read_pages(
        reader, command, Directories(directory_address), reader.me(), 1, scratch
    ) ||
        !read_done(reader, command.sequence)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t first = directory[kDirFirst], pages = directory[kDirPages];
    const Lists lists(lists_address);
    uint32_t wait = 0, read = 0;
#ifdef RASTER_SPREAD
    Spread spread;
    const bool spreads = km::handler(command) == kHandlerFill;
#endif
    for (uint32_t page = 0; page < pages; ++page) {
        const uint32_t asked = km::wall();
        if (!reader.reserve(kCbIn, 1)) {
            return;
        }
        const uint32_t reserved = km::wall();
        const uint32_t page_address = get_write_ptr(kCbIn);
        if (
            !km::read_pages(reader, command, lists, first + page, 1, page_address) ||
            !read_done(reader, command.sequence)
        ) {
            return;
        }
        cb_push_back(kCbIn, 1);
        wait += reserved - asked;
        read += km::wall() - reserved;
#ifdef RASTER_SPREAD
        invalidate_l1_cache();
        if (spreads && !spread_page(reader, spread, page_address)) {
            return;
        }
#endif
    }
#ifdef RASTER_SPREAD
    directory[kRdWords + 4] = spread.cycles;
    directory[kRdWords + 5] = spread.wait;
    directory[kRdWords + 6] = spread.records;
#endif
    directory[kRdWords + 0] = wait;
    directory[kRdWords + 1] = read;
    directory[kRdWords + 2] = km::wall() - started;
    directory[kRdWords + 3] = pages;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        fill_reader_command(reader, command);
    });
}
#endif
