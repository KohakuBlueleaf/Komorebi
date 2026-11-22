// SHADE's writer (shade.h): each item's colour planes back to the frame (a band: its
// rows), then the profile page. In frame_writer.cpp after writer.cpp.
#include "shade.h"

static_assert(kShadeOutPages == kPlanes && kOutPages % kShadeOutPages == 0);

bool shade_writer_command(const km::Writer& writer, const km_command_t& command) {
    const uint32_t directory_address = command.a0, profile_address = command.a4;
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
    if (tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems) {
        writer.gave_up(
            km::kStWriter, kGaveBadItem, kWhereShadeBad, 0, command.sequence
        );
        *writer.control(KM_CONTROL_STOP) = 1;
        return false;
    }
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t id = directory[kDirIds + item];
        if ((id & kItemEmpty) != 0) {
            continue;  // no triangle: FILL's background stays
        }
        const uint32_t page = (id & kItemSlot) * kPlanes;
        if (page / tensor_pages >= kDirFrameMax) {
            writer.gave_up(
                km::kStWriter, kGaveBadItem, kWhereShadeBad, 0, command.sequence
            );
            *writer.control(KM_CONTROL_STOP) = 1;
            return false;
        }
        if (!writer.wait_front(kCbOut, kShadeOutPages)) {
            return true;
        }
        if (!item_fits(writer, kShadeOutPages, command.sequence)) {
            return false;
        }
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t end_row = id >> kItemRow1Shift & kItemRowMask;
        const uint32_t byte_offset = first_row * kRowBytes;
        const uint32_t bytes = (
            end_row > 0 ? (end_row - first_row) * kRowBytes : kOutPageBytes
        );
        const InterleavedAddrGen<true> frame_tensor = {
            .bank_base_address = directory[kDirFrame + 1 + page / tensor_pages],
            .page_size = kOutPageBytes
        };
        const uint32_t source = get_read_ptr(kCbOut) + byte_offset;
        for (uint32_t plane = 0; plane < kShadeColourPages; ++plane) {
            noc_async_write(
                source + plane * kOutPageBytes,
                get_noc_addr(
                    (page + kShowFirstPlane + plane) % tensor_pages,
                    frame_tensor,
                    byte_offset
                ),
                bytes
            );
        }
        if (!writes_sent(writer, command.sequence)) {
            return false;
        }
        cb_pop_front(kCbOut, kShadeOutPages);
    }
    // The profile item: the pack thread's counters in its first words.
    if (!writer.wait_front(kCbOut, kShadeOutPages)) {
        return true;
    }
    if (!item_fits(writer, kShadeOutPages, command.sequence)) {
        return false;
    }
    if (profile_address != 0) {
        const InterleavedAddrGen<true> profiles = {
            .bank_base_address = profile_address, .page_size = kShadeProfBytes
        };
        noc_async_write(
            get_read_ptr(kCbOut), get_noc_addr(writer.me(), profiles), kShadeProfBytes
        );
        if (!writes_sent(writer, command.sequence)) {
            return false;
        }
    }
    cb_pop_front(kCbOut, kShadeOutPages);
    return true;
}
