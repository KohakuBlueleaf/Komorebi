// SHADE's writer (shade.h): each item's colour planes back to the frame (a band: its
// rows; a6: the opaque layer's; lit: its three parts), then the profile page.
#include "shade.h"

static_assert(kShadeOutPages == kPlanes && kOutPages % kShadeOutPages == 0);

bool shade_writer_command(const km::Writer& writer, const km_command_t& command) {
    const uint32_t directory_address = command.a0, profile_address = command.a4;
    const uint32_t opaque_address = command.a6;
    const uint32_t scratch = get_write_ptr(kCbDirWriter);
    volatile tt_l1_ptr uint32_t* directory = km::l1(scratch);
    // the frame written: the directory's own, or the opaque layer's
    uint32_t frame[kDirFrameMax + 1] = {};
    // lit (a7): the surface's two frames' words (shade_reader.hpp surface_words)
    const bool lit = command.a7 != 0;
    uint32_t surface[kSurfaceWords] = {};
    if (lit) {
        if (!km::read_pages(
            writer, command, Directories(command.a7), writer.me(), 1, scratch
        ) ||
            !read_done(writer, command.sequence)) {
            return false;
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i <= kDirFrameMax; ++i) {
            surface[i] = directory[kDirFrame + i];
            surface[kDirFrameMax + 1 + i] = directory[kSurfaceSecond + i];
        }
    }
    if (opaque_address != 0) {
        if (!km::read_pages(
            writer, command, Directories(opaque_address), writer.me(), 1, scratch
        ) ||
            !read_done(writer, command.sequence)) {
            return false;
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i <= kDirFrameMax; ++i) {
            frame[i] = directory[kDirFrame + i];
        }
    }
    if (!km::read_pages(
        writer, command, Directories(directory_address), writer.me(), 1, scratch
    ) ||
        !read_done(writer, command.sequence)) {
        return false;
    }
    invalidate_l1_cache();
    if (opaque_address == 0) {
        for (uint32_t i = 0; i <= kDirFrameMax; ++i) {
            frame[i] = directory[kDirFrame + i];
        }
    }
    const uint32_t items = directory[kDirItems], tensor_pages = frame[0];
    if (
        tensor_pages == 0 || tensor_pages % kPlanes != 0 || items > kDirMaxItems ||
        (
            lit &&
            (
                surface[0] == 0 || surface[0] % kPlanes != 0 ||
                surface[kDirFrameMax + 1] == 0 ||
                surface[kDirFrameMax + 1] % kPlanes != 0
            )
        )
    ) {
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
        if (
            page / tensor_pages >= kDirFrameMax ||
            (
                lit &&
                (
                    page / surface[0] >= kDirFrameMax ||
                    page / surface[kDirFrameMax + 1] >= kDirFrameMax
                )
            )
        ) {
            writer.gave_up(
                km::kStWriter, kGaveBadItem, kWhereShadeBad, 0, command.sequence
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

        // not lit: the colour planes; lit: the frame's 1/z (merging) and material word,
        // the surface's planes 0 to 4, 5 to 9
        for (uint32_t part = 0; part < (lit ? kLitParts : 1); ++part) {
            if (!writer.wait_front(kCbOut, kShadeOutPages)) {
                return true;
            }
            if (!item_fits(writer, kShadeOutPages, command.sequence)) {
                return false;
            }
            const uint32_t source = get_read_ptr(kCbOut) + byte_offset;
            uint32_t first = kShowFirstPlane, end = kShowFirstPlane + kShadeColourPages;
            if (lit) {
                first = part == 0 && opaque_address == 0 ? kPlanes - 1 : 0;
                end = kPlanes;
            }
            // part 0 the frame's, then the surface's first and second frame
            const uint32_t* words = (
                part == 0 ? frame : surface + (kDirFrameMax + 1) * (part - 1)
            );
            const InterleavedAddrGen<true> tensor = {
                .bank_base_address = words[1 + page / words[0]],
                .page_size = kOutPageBytes
            };
            for (uint32_t plane = first; plane < end; ++plane) {
                if (lit && part == 0 && plane != 0 && plane != kPlanes - 1) {
                    continue;
                }
                const uint32_t from = lit ? plane : plane - kShowFirstPlane;
                noc_async_write(
                    source + from * kOutPageBytes,
                    get_noc_addr((page + plane) % words[0], tensor, byte_offset),
                    bytes
                );
            }
            if (!writes_sent(writer, command.sequence)) {
                return false;
            }
            cb_pop_front(kCbOut, kShadeOutPages);
        }
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
