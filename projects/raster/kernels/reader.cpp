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
    for (uint32_t page = 0; page < pages; ++page) {
        const uint32_t asked = km::wall();
        if (!reader.reserve(kCbIn, 1)) {
            return;
        }
        const uint32_t reserved = km::wall();
        if (!km::read_pages(
            reader, command, lists, first + page, 1, get_write_ptr(kCbIn)
        ) ||
            !read_done(reader, command.sequence)) {
            return;
        }
        cb_push_back(kCbIn, 1);
        wait += reserved - asked;
        read += km::wall() - reserved;
    }
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
