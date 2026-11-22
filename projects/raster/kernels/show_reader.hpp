// SHOW's reader (show.h): each item's three colour planes (a band: only its rows' bytes
// of each) from the frame into kCbShow, an item at a time; the writer converts them
// (show_writer.hpp). Included by frame_reader.cpp after reader.cpp, in its namespace.
#include "show.h"

void show_reader_command(const km::Reader& reader, const km_command_t& command) {
    const uint32_t directory_address = command.a0;
    const uint32_t scratch = get_write_ptr(kCbDirReader);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    if (!km::read_pages(
        reader, command, Directories(directory_address), reader.me(), 1, scratch
    ) ||
        !read_done(reader, command.sequence)) {
        return;
    }
    invalidate_l1_cache();
    const uint32_t items = directory[kDirItems], tensor_pages = directory[kDirFrame];
    // The writer checks the same and gives up: nothing is read for a table it refuses.
    if (tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems) {
        return;
    }
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t id = directory[kDirIds + item];
        const uint32_t page = (id & kItemSlot) * kPlanes;
        if (page / tensor_pages >= kDirFrameMax) {
            return;
        }
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t end_row = id >> kItemRow1Shift & kItemRowMask;
        const uint32_t byte_offset = first_row * kRowBytes;
        const uint32_t bytes = (
            end_row > 0 ? (end_row - first_row) * kRowBytes : kOutPageBytes
        );
        if (!reader.reserve(kCbShow, kShowPlanes)) {
            return;
        }
        const uint32_t destination = get_write_ptr(kCbShow) + byte_offset;
        // A tile's five planes lie in one tensor (its pages a multiple of five).
        const InterleavedAddrGen<true> frame_tensor = {
            .bank_base_address = directory[kDirFrame + 1 + page / tensor_pages],
            .page_size = kOutPageBytes
        };
        for (uint32_t plane = 0; plane < kShowPlanes; ++plane) {
            noc_async_read(
                get_noc_addr(
                    (page + kShowFirstPlane + plane) % tensor_pages,
                    frame_tensor,
                    byte_offset
                ),
                destination + plane * kShowPageBytes,
                bytes
            );
        }
        if (!read_done(reader, command.sequence)) {
            return;
        }
        cb_push_back(kCbShow, kShowPlanes);
    }
}
