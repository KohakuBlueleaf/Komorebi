// Rasterizer writer: each item's output pages (pages_per_item) to the frame, then the
// profile page (compute's counters, then the reader's and its own) to profile_address.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_ask.hpp"
#include "raster.h"

// An item of `pages` pages at the read pointer must end inside the output buffer;
// otherwise the give-up is recorded and the stop flag set.
bool item_fits(const km::Writer& writer, uint32_t pages, uint32_t sequence) {
    if (
        get_read_ptr(kCbOut) + pages * kOutPageBytes <=
        get_local_cb_interface(kCbOut).fifo_limit
    ) {
        return true;
    }
    writer.gave_up(km::kStWriter, kGaveBadItem, kWhereBadSend, 0, sequence);
    *writer.control(KM_CONTROL_STOP) = 1;
    return false;
}

// Timed NoC waits: reads arrived, or writes sent. False: gave up (recorded).
template <typename Ready>
bool noc_done(
    const km::Writer& writer, Ready ready, uint32_t where, uint32_t sequence
) {
    const uint32_t code = writer.poll(ready, writer.args().wait_limit);
    if (code == km::kGaveNone) {
        return true;
    }
    writer.gave_up(km::kStWriter, code, where, 0, sequence);
    return false;
}

bool read_done(const km::Writer& writer, uint32_t sequence) {
    return noc_done(
        writer,
        [] {
            return ncrisc_noc_reads_flushed(noc_index);
        },
        kWhereReadDone,
        sequence
    );
}

bool writes_sent(const km::Writer& writer, uint32_t sequence) {
    return noc_done(
        writer,
        [] {
            return ncrisc_noc_nonposted_writes_sent(noc_index);
        },
        kWhereFlush,
        sequence
    );
}

// The directory pages, the frame's tensors (one accessor: each tensor's word is in the
// directory) and the profile pages (km::Tensor): compile arguments from 0.
using Directories = km::Tensor<0>;
using Frame = km::Tensor<Directories::next_compile_time_args_offset()>;
using Profiles = km::Tensor<Frame::next_compile_time_args_offset()>;

// FILL and ARITH of one command; the frame program's writer calls it as well.
bool fill_writer_command(const km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != kHandlerFill && km::handler(command) != kHandlerArith) {
        return true;
    }
    const uint32_t directory_address = command.a0, pages_per_item = command.a3;
    const uint32_t profile_address = command.a4;
    const uint32_t started = km::wall();
    const uint32_t scratch = get_write_ptr(kCbDirWriter);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    if (!km::read_pages(
        writer, command, Directories(directory_address), writer.me(), 1, scratch
    ) ||
        !read_done(writer, command.sequence)) {
        return false;
    }
    invalidate_l1_cache();
    const uint32_t items = directory[kDirItems], tensor_pages = directory[kDirFrame];
    // An item's pages lie in one of the frame's tensors, each named in the table.
    if (
        tensor_pages == 0 || tensor_pages % pages_per_item != 0 || items > kDirMaxItems
    ) {
        writer.gave_up(km::kStWriter, kGaveBadItem, kWhereBadSend, 0, command.sequence);
        *writer.control(KM_CONTROL_STOP) = 1;
        return false;
    }
    auto frame_noc = [&](uint32_t page, uint32_t byte_offset) {
        using OwnBank = InterleavedAddrGen<true>;  // km: own island
        const OwnBank frame_tensor = {
            .bank_base_address = directory[kDirFrame + 1 + page / tensor_pages],
            .page_size = kOutPageBytes
        };
        return get_noc_addr(page % tensor_pages, frame_tensor, byte_offset);
    };
    // The empty tile's planes: depth 0, the background colour, triangle -1.
    const uint32_t empty = get_write_ptr(kCbEmpty);
    if (km::handler(command) == kHandlerFill) {
        constexpr uint32_t plane_value[kPlanes] = {
            0, 0x3D4CCCCD, 0x3D4CCCCD, 0x3DA3D70A, 0xBF800000
        };
        volatile tt_l1_ptr uint32_t* empty_words = km::l1(empty);
        for (uint32_t word = 0; word < kPlanes * kPlaneWords; ++word) {
            empty_words[word] = plane_value[word / kPlaneWords];
        }
    }
    uint32_t wait = 0, write = 0;
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t id = directory[kDirIds + item];
        const bool is_empty = (id & kItemEmpty) != 0;
        const uint32_t asked = km::wall();
        if (!is_empty && !writer.wait_front(kCbOut, pages_per_item)) {
            return true;
        }
        if (!is_empty && !item_fits(writer, pages_per_item, command.sequence)) {
            return false;
        }
        const uint32_t ready = km::wall();
        const uint32_t page = (id & kItemSlot) * pages_per_item;
        if (page / tensor_pages >= kDirFrameMax) {
            writer.gave_up(
                km::kStWriter, kGaveBadItem, kWhereBadSend, 0, command.sequence
            );
            *writer.control(KM_CONTROL_STOP) = 1;
            return false;
        }
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t end_row = id >> kItemRow1Shift & kItemRowMask;
        const uint32_t byte_offset = first_row * kRowBytes;
        const uint32_t bytes = (
            end_row > 0 ? (end_row - first_row) * kRowBytes : kOutPageBytes
        );
        uint32_t source = (is_empty ? empty : get_read_ptr(kCbOut)) + byte_offset;
        if (bytes == kOutPageBytes) {  // whole pages: the page calls
            const Frame frame(directory[kDirFrame + 1 + page / tensor_pages]);
            for (uint32_t plane = 0; plane < pages_per_item; ++plane) {
                if (!km::write_pages(
                    writer, command, frame, (page + plane) % tensor_pages, 1, source
                )) {
                    return false;
                }
                source += kOutPageBytes;
            }
        } else if constexpr (Frame::kLocal) {  // a band: part of each page, by address
            for (uint32_t plane = 0; plane < pages_per_item; ++plane) {
                noc_async_write(source, frame_noc(page + plane, byte_offset), bytes);
                source += kOutPageBytes;
            }
        } else {  // a band over islands: part of each page, wherever its island is
            const Frame frame(directory[kDirFrame + 1 + page / tensor_pages]);
            for (uint32_t plane = 0; plane < pages_per_item; ++plane) {
                if (!km::put_bytes(
                    writer,
                    command,
                    frame,
                    (page + plane) % tensor_pages,
                    byte_offset,
                    bytes,
                    source
                )) {
                    return false;
                }
                source += kOutPageBytes;
            }
            if (!km::put_wait(writer, command)) {
                return false;
            }
        }
        if (!writes_sent(writer, command.sequence)) {
            return false;
        }
        if (!is_empty) {
            cb_pop_front(kCbOut, pages_per_item);
        }
        wait += ready - asked;
        write += km::wall() - ready;
    }
    // The profile item: compute filled words [0, 32); reader and writer follow.
    if (!writer.wait_front(kCbOut, pages_per_item)) {
        return true;
    }
    if (!item_fits(writer, pages_per_item, command.sequence)) {
        return false;
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_read_ptr(kCbOut));
    volatile tt_l1_ptr uint32_t* reader_words = km::l1(get_write_ptr(kCbDirReader));
    for (uint32_t counter = 0; counter < 4; ++counter) {
        profile[kProfReader + counter] = reader_words[kRdWords + counter];
    }
    profile[kProfWriter + 0] = wait;
    profile[kProfWriter + 1] = write;
    profile[kProfWriter + 2] = km::wall() - started;
    profile[kProfWriter + 3] = items;
    if (!km::write_pages(
        writer,
        command,
        Profiles(profile_address),
        writer.me(),
        1,
        get_read_ptr(kCbOut)
    ) ||
        !writes_sent(writer, command.sequence)) {
        return false;
    }
    cb_pop_front(kCbOut, pages_per_item);
    return true;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        return fill_writer_command(writer, command);
    });
}
#endif
