// SHOW's reader (raytrace.h): each of the core's tiles from the colour stream to one
// display page of 8-bit R, G, B, 255, written whole. Included by reader.cpp.
#pragma once

namespace show {

    // The 8-bit value of a float32 colour by reference/png.py's rule, from thresholds
    // and buckets, as projects/raster/kernels/show_writer.hpp's show_byte.
    inline uint32_t show_byte(
        const tt_l1_ptr int32_t* thresholds,
        const tt_l1_ptr uint8_t* buckets,
        uint32_t bits
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

    // Per tile: its descriptor and colour pages read, its pixels converted to their
    // places, the page written; an entry not of whole tiles gives up.
    template <typename Directories>
    bool frame(km::Reader& reader, const km_command_t& command) {
        const uint32_t directory_address = command.a0, colour_address = command.a1;
        const uint32_t display_address = command.a2, table_address = command.a3;
        const uint32_t desc_address = command.a5;
        const uint32_t directory_space = get_write_ptr(kCbDirReader);
        volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
        if (!km::read_pages(
            reader,
            command,
            Directories(directory_address),
            reader.me(),
            1,
            directory_space
        ) ||
            !timed::reads_done(reader, km::kStReader)) {
            return false;
        }
        const uint32_t first = directory[kDirFirst], packets = directory[kDirCount];
        if (
            first % kShowTilePackets != 0 || packets % kShowTilePackets != 0 ||
            kPagePackets % kShowTilePackets != 0 ||
            !timed::fits(kCbShow, kShowScratchBytes)
        ) {
            reader.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 4, packets);
            return false;
        }
        const uint32_t desc_l1 = get_write_ptr(kCbShow);
        const uint32_t colour_l1 = desc_l1 + kShowDescBytes;
        const uint32_t out_l1 = colour_l1 + kShowColourBytes;
        const uint32_t table_l1 = out_l1 + kShowPageBytes;
        const uint32_t buckets_l1 = table_l1 + 4 * kShowTableWords;
        const InterleavedAddrGen<true> table = {
            .bank_base_address = table_address, .page_size = 4 * kShowTableWords
        };
        noc_async_read(get_noc_addr(0, table, 0), table_l1, 4 * kShowTableWords);
        if (!timed::reads_done(reader, km::kStReader)) {
            return false;
        }
        invalidate_l1_cache();
        const tt_l1_ptr int32_t*
            thresholds = (reinterpret_cast<const tt_l1_ptr int32_t*>(table_l1));
        tt_l1_ptr uint8_t* buckets = reinterpret_cast<tt_l1_ptr uint8_t*>(buckets_l1);
        for (uint32_t bucket = 0, index = 0; bucket < kShowBuckets; ++bucket) {
            const int32_t start = (
                static_cast<int32_t>(kShowBucketBase + (bucket << kShowBucketShift))
            );
            while (index < kShowTableWords - 1 && start >= thresholds[index + 1]) {
                ++index;
            }
            buckets[bucket] = static_cast<uint8_t>(index);
        }
        const InterleavedAddrGen<true> descs = {
            .bank_base_address = desc_address,
            .page_size = kDescVectors * kVec * 4 * kPagePackets
        };
        const InterleavedAddrGen<true> colours = {
            .bank_base_address = colour_address,
            .page_size = kColVectors * kVec * 4 * kPagePackets
        };
        const InterleavedAddrGen<true> display = {
            .bank_base_address = display_address, .page_size = kShowPageBytes
        };
        constexpr uint32_t kTilesWide = kShowWidth / kShowTile;
        for (
            uint32_t start = first; start < first + packets; start += kShowTilePackets
        ) {
            const uint32_t page = start / kPagePackets, slot = start % kPagePackets;
            noc_async_read(
                get_noc_addr(page, descs, slot * kDescVectors * kVec * 4),
                desc_l1,
                kShowDescBytes
            );
            noc_async_read(
                get_noc_addr(page, colours, slot * kColVectors * kVec * 4),
                colour_l1,
                kShowColourBytes
            );
            // the reads in, and the page of the tile before sent
            if (
                !timed::reads_done(reader, km::kStReader) ||
                !timed::writes_sent(reader, km::kStReader)
            ) {
                return false;
            }
            invalidate_l1_cache();
            const tt_l1_ptr uint32_t*
                desc = (reinterpret_cast<const tt_l1_ptr uint32_t*>(desc_l1));
            const tt_l1_ptr uint32_t*
                colour = (reinterpret_cast<const tt_l1_ptr uint32_t*>(colour_l1));
            tt_l1_ptr uint32_t* out = reinterpret_cast<tt_l1_ptr uint32_t*>(out_l1);
            const uint32_t first_pixel = desc[0];
            const uint32_t tile = (
                (first_pixel / kShowWidth / kShowTile) * kTilesWide +
                first_pixel % kShowWidth / kShowTile
            );
            for (uint32_t packet = 0; packet < kShowTilePackets; ++packet) {
                const uint32_t pixel = desc[packet * kDescVectors * kVec];
                const uint32_t x = pixel % kShowWidth, y = pixel / kShowWidth;
                if (
                    (y / kShowTile) * kTilesWide + x / kShowTile != tile ||
                    x % kShowTile + kShowBlockW > kShowTile ||
                    y % kShowTile + kVec / kShowBlockW > kShowTile
                ) {
                    reader.gave_up(
                        km::kStReader, km::kGaveLimit, km::kWhereData, 5, pixel
                    );
                    return false;
                }
                const uint32_t corner = (y % kShowTile) * kShowTile + x % kShowTile;
                const tt_l1_ptr uint32_t* red = colour + packet * kColVectors * kVec;
                const tt_l1_ptr uint32_t* green = red + kVec;
                const tt_l1_ptr uint32_t* blue = green + kVec;
                for (uint32_t lane = 0; lane < kVec; ++lane) {
                    out[
                        corner + (lane / kShowBlockW) * kShowTile + lane % kShowBlockW
                    ] =
                        (
                            show_byte(thresholds, buckets, red[lane]) |
                            show_byte(thresholds, buckets, green[lane]) << 8 |
                            show_byte(thresholds, buckets, blue[lane]) << 16 |
                            kShowAlpha
                        );
                }
            }
            noc_async_write(out_l1, get_noc_addr(tile, display, 0), kShowPageBytes);
        }
        // The reader marks the command done without a barrier: the pages land first.
        return timed::writes_landed(reader, km::kStReader);
    }

}  // namespace show
