// SHOW's writer (show.h): each item's planes from kCbShow to 8-bit R, G, B, 255,
// written as its rows of the tile's display page. Included by frame_writer.cpp.
#include "show.h"

// The 8-bit value of a float32 colour by reference/png.py's rule: the thresholds at or
// under its bits (word k: the smallest float32 of value k), found from `buckets`.
inline uint32_t show_byte(
    const int32_t* thresholds, const tt_l1_ptr uint8_t* buckets, uint32_t bits
) {
    const int32_t value = static_cast<int32_t>(bits);
    if (value < static_cast<int32_t>(kShowBucketBase)) {
        return 0;
    }
    if (value >= static_cast<int32_t>(kShowOne)) {
        return kShowTableWords - 1;
    }
    uint32_t index = buckets[(bits - kShowBucketBase) >> kShowBucketShift];
    while (index < kShowTableWords - 1 && value >= thresholds[index + 1]) {
        ++index;
    }
    return index;
}

// a3: the display's pages; an item's page at or past them gives up. A tile's page is
// its number; raw, a5 + its row a6 + its column (show.h kShowRaw).
bool show_writer_command(const km::Writer& writer, const km_command_t& command) {
    const uint32_t directory_address = command.a0, display_address = command.a1;
    const uint32_t profile_address = command.a2, display_pages = command.a3;
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
    const km::Block block = writer.block(command);
    const bool raw = command.a4 == kShowRaw;  // plane 0's words, no conversion
    const uint32_t first_page = raw ? command.a5 : 0;
    const uint32_t across = raw ? command.a6 : kShowTilesAcross;
    if (
        (!raw && block.bytes < 4 * kShowTableWords) || tensor_pages == 0 ||
        tensor_pages % kPlanes != 0 || items > kDirMaxItems
    ) {
        writer.gave_up(km::kStWriter, kGaveBadItem, kWhereShowBad, 0, command.sequence);
        *writer.control(KM_CONTROL_STOP) = 1;
        return false;
    }
    // the thresholds in the writer's local memory (its stack), read at every step
    int32_t thresholds[kShowTableWords];
    for (uint32_t word = 0; word < kShowTableWords; ++word) {
        thresholds[word] = raw ? 0 : static_cast<int32_t>(block.words[word]);
    }
    // Two pages of the writer's own scratch: one converted while the other is sent;
    // the third the profile's, the fourth the buckets'.
    const uint32_t out = get_write_ptr(kCbEmpty);
    tt_l1_ptr uint8_t* buckets = (
        reinterpret_cast<tt_l1_ptr uint8_t*>(out + 3 * kShowPageBytes)
    );
    for (uint32_t bucket = 0, index = 0; bucket < kShowBuckets; ++bucket) {
        const int32_t start = (
            static_cast<int32_t>(kShowBucketBase + (bucket << kShowBucketShift))
        );
        while (index < kShowTableWords - 1 && start >= thresholds[index + 1]) {
            ++index;
        }
        buckets[bucket] = static_cast<uint8_t>(index);
    }
    const InterleavedAddrGen<true> display = {
        .bank_base_address = display_address, .page_size = kShowPageBytes
    };
    uint32_t convert = 0, wait = 0, write = 0, rows = 0;
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t id = directory[kDirIds + item];
        const uint32_t tile = id & kItemSlot;
        const uint32_t page = (
            first_page + tile / kShowTilesAcross * across + tile % kShowTilesAcross
        );
        if ((tile * kPlanes) / tensor_pages >= kDirFrameMax || page >= display_pages) {
            writer.gave_up(
                km::kStWriter, kGaveBadItem, kWhereShowBad, 0, command.sequence
            );
            *writer.control(KM_CONTROL_STOP) = 1;
            return false;
        }
        const uint32_t first_row = id >> kItemRow0Shift & kItemRowMask;
        const uint32_t row_end = id >> kItemRow1Shift & kItemRowMask;
        const uint32_t end_row = row_end > 0 ? row_end : 32;
        const uint32_t byte_offset = first_row * kRowBytes;
        const uint32_t asked = km::wall();
        if (!writer.wait_front(kCbShow, kShowPlanes)) {
            return true;
        }
        const uint32_t page_address = out + (item & 1) * kShowPageBytes;
        if (!writes_sent(writer, command.sequence)) {  // the page's last write left
            return false;
        }
        const uint32_t ready = km::wall();
        invalidate_l1_cache();
        // plain loads and stores: the planes stay put until the pop, the page until
        // its write is sent
        const tt_l1_ptr uint32_t* red = (
            reinterpret_cast<const tt_l1_ptr uint32_t*>(get_read_ptr(kCbShow))
        );
        const tt_l1_ptr uint32_t* green = red + kPlaneWords;
        const tt_l1_ptr uint32_t* blue = green + kPlaneWords;
        tt_l1_ptr uint32_t* pixels = (
            reinterpret_cast<tt_l1_ptr uint32_t*>(page_address)
        );
        for (uint32_t row = first_row; row < end_row; ++row) {
            // the Dst window's order: vector v, lane l at 64 (v / 2) + 16 (l / 8) +
            // 2 (l mod 8) + v mod 2 (raster README)
            const uint32_t row_word = 64 * (row / 2) + row % 2;
            if (raw) {  // the 1/z plane (the reader's first page) as it is
                for (uint32_t column = 0; column < 32; ++column) {
                    pixels[32 * row + column] = (
                        red [row_word + 16 * (column / 8) + 2 * (column % 8)]
                    );
                }
                continue;
            }
            for (uint32_t column = 0; column < 32; ++column) {
                const uint32_t word = row_word + 16 * (column / 8) + 2 * (column % 8);
                pixels[32 * row + column] = (
                    show_byte(thresholds, buckets, red[word]) |
                    show_byte(thresholds, buckets, green[word]) << 8 |
                    show_byte(thresholds, buckets, blue[word]) << 16 | kShowAlpha
                );
            }
        }
        cb_pop_front(kCbShow, kShowPlanes);
        const uint32_t converted = km::wall();
        noc_async_write(
            page_address + byte_offset,
            get_noc_addr(page, display, byte_offset),
            (end_row - first_row) * kRowBytes
        );
        wait += ready - asked;
        convert += converted - ready;
        write += km::wall() - converted;
        rows += end_row - first_row;
    }
    if (!writes_sent(writer, command.sequence)) {
        return false;
    }
    if (profile_address != 0) {
        const uint32_t profile_page = out + 2 * kShowPageBytes;
        volatile tt_l1_ptr uint32_t* profile = km::l1(profile_page);
        for (uint32_t word = 0; word < kShowProfBytes / 4; ++word) {
            profile[word] = 0;
        }
        profile[kShowProfConvert] = convert;
        profile[kShowProfWait] = wait;
        profile[kShowProfWrite] = write;
        profile[kShowProfTotal] = km::wall() - started;
        profile[kShowProfItems] = items;
        profile[kShowProfRows] = rows;
        const InterleavedAddrGen<true> profiles = {
            .bank_base_address = profile_address, .page_size = kShowProfBytes
        };
        noc_async_write(
            profile_page, get_noc_addr(writer.me(), profiles, 0), kShowProfBytes
        );
        if (!writes_sent(writer, command.sequence)) {
            return false;
        }
    }
    return true;
}
