// The passes' writer: LANES the window's records, COUNT the counters, SETUP each record
// to its list slot, HIST the heavy tiles' histograms, PLACE the directory and marks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

namespace {

    // A kept triangle's tile columns and pixel rows, as the pack thread hands them.
    struct BoxWord {
        uint32_t first_tile_x, last_tile_x, first_row, last_row;
    };

#ifdef RASTER_WIDE
    // Wide (grid.h): COUNT's box in three words, tile columns 16 bits each, then the
    // first and the last pixel row; SETUP's from its record's box words.
    constexpr uint32_t kBoxWords = 3;

    inline BoxWord box_at(const volatile tt_l1_ptr uint32_t* words) {
        return {words[0] & 0xFFFF, words[0] >> 16, words[1], words[2]};
    }

    // The integer n in [0, 2^24) whose float32 bits are `bits` (a wide record's box
    // column).
    inline uint32_t integer_of(uint32_t bits) {
        if (bits == 0) {
            return 0;
        }
        const uint32_t exponent = (bits >> 23) - 127;
        return ((bits & 0x7FFFFF) | 0x800000) >> (23 - exponent);
    }

    // A record's box (raster.h kRecX0 to kRecY1: first column and last + 1 as float32,
    // first and last row).
    inline BoxWord record_box(const volatile tt_l1_ptr uint32_t* record) {
        return {
            integer_of(record[kRawX0]) >> 5,
            (integer_of(record[kRawX1]) - 1) >> 5,
            record[kRawY0],
            record[kRawY1]
        };
    }
#else
    // Narrow: one word, tile columns in 6 bits each, pixel rows in 10.
    constexpr uint32_t kBoxWords = 1;

    inline BoxWord box_at(const volatile tt_l1_ptr uint32_t* words) {
        const uint32_t word = words[0];
        return {word & 63, word >> 6 & 63, word >> 12 & 1023, word >> 22 & 1023};
    }
#endif
    static_assert(kOutWords % kBoxWords == 0);  // a COUNT page holds whole boxes

    // Records the first give-up of this core's writer and asks the run to stop.
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

    // The writer's profile counters for one command (cycles, bytes, pages).
    struct WriterProfile {
        uint32_t wait = 0, write = 0, bytes = 0, pages = 0;
    };

    // PLACE through HIST's space (HIST is done): the region's words, then this core's
    // directory page, its end slot and 32 headers at a time.
    template <typename Arrived, typename Flushed>
    bool place(
        const km::Writer& writer,
        const km_command_t& command,
        Arrived arrived,
        Flushed flushed
    ) {
        const uint32_t items_region_address = command.a0;
        const uint32_t directory_address = command.a1, cores = command.a5;
        const uint32_t fill_cores = command.a6, lists_address = command.a7;
        const uint32_t bank = bank_word(command.a8);
        // the frame's tiles, not the capacity's
        const uint32_t tiles = grid_tiles(command.a8);
        const uint32_t space = get_write_ptr(kCbHist);
        // the parts of the region the frame's items use, each on 64 bytes: the tiles'
        // words, the items' rows, the frame words and the items' words, the cores'
        const uint32_t item_most = tiles + kBandItems;
        static_assert(kItemRows % 64 == 0 && kFrameWordsAt % 64 == 0);
        static_assert(kItemWordsAt % 64 == 0 && kCoreWordsAt % 64 == 0);
        auto read_part = [&](uint32_t at, uint32_t bytes) {
            noc_async_read(
                setup_pass::noc_address(bank, items_region_address, 0, 0, at),
                space + at,
                (bytes + 63) & ~63u
            );
        };
        read_part(0, 4 * tiles);
        read_part(kItemRows, 2 * item_most);
        read_part(
            kFrameWordsAt, kItemWordsAt - kFrameWordsAt + 4 * kItemWords * item_most
        );
        read_part(
            kCoreWordsAt,
            kCoreWords * 4 * (fill_cores < kMaxFillCores ? fill_cores : kMaxFillCores)
        );
        if (!arrived()) {
            return false;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* items = km::l1(space);
        const volatile tt_l1_ptr uint16_t*
            rows = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kItemRows));
        volatile tt_l1_ptr uint32_t* item_words = km::l1(space + kItemWordsAt);
        volatile tt_l1_ptr uint32_t* core_words = km::l1(space + kCoreWordsAt);
        volatile tt_l1_ptr uint32_t* frame_words = km::l1(space + kFrameWordsAt);
        // each item's tile, in the base words' place (unused by PLACE)
        volatile tt_l1_ptr uint16_t*
            tile_of = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(space + kBaseAt));
        uint32_t item_count = 0;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint32_t first = items[tile] & 0xFFFF;
            const uint32_t items_in_tile = items[tile] >> 16;
            for (
                uint32_t i = first; i < first + items_in_tile && i < kSlotsBytes / 4;
                ++i
            ) {
                tile_of[i] = tile;
            }
            item_count = first + items_in_tile;
        }
        if (item_count > kSlotsBytes / 4 || fill_cores > kMaxFillCores) {
            give_up(kGaveBadPlace, 1);
            return true;
        }
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = lists_address, .page_size = kListPageBytes
        };
        const uint32_t me = writer.me(), directory_space = space + kItemsRegionBytes;
        static_assert(
            kItemsRegionBytes + kDirPageBytes + 128 + kOutBytes <= kHistBytes
        );
        auto build_slot = [&](uint32_t slot_address, uint32_t mark, uint32_t i) {
            volatile tt_l1_ptr uint32_t* slot_words = km::l1(slot_address);
            for (uint32_t word = 0; word < 32; ++word) {
                slot_words[word] = 0;
            }
            slot_words[0] = mark;
            if (mark == kLayMarkTile) {
                slot_words[kLayHdrTile] = tile_of[i];
                slot_words[kLayHdrCount] = item_words[kItemWords * i] >> 8;
                slot_words[kLayHdrRow0] = rows[i] & 0xFF;
                slot_words[kLayHdrRow1] = rows[i] >> 8;
            }
        };
        if (me < fill_cores) {
            volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
            for (uint32_t word = 0; word < kDirPageBytes / 4; ++word) {
                directory[word] = 0;
            }
            uint32_t directory_items = 0;
            for (
                uint32_t i = 0;
                i < item_count && directory_items < kLayDirFrame - kLayDirIds;
                ++i
            ) {
                const uint32_t item_word = item_words[kItemWords * i];
                if ((item_word & 0xFF) != me) {
                    continue;
                }
                uint32_t id = tile_of[i];
                if (rows[i] != (32u << 8)) {
                    id |= (
                        (rows[i] & 0xFF) << kLayRow0Shift |
                        (rows[i] >> 8) << kLayRow1Shift
                    );
                }
                directory[kLayDirIds + directory_items++] = (
                    (item_word >> 8) == 0 ? id | kLayItemEmpty : id
                );
            }
            directory[0] = core_words[kCoreWords * me];
            directory[1] = core_words[kCoreWords * me + 1];
            directory[2] = directory_items;
            for (uint32_t word = 0; word < 16; ++word) {
                directory[kLayDirFrame + word] = frame_words[word];
            }
            asm volatile(
                "fence" ::: "memory"
            );  // the page's stores before the NoC reads it
            noc_async_write(
                directory_space,
                setup_pass::noc_address(0, directory_address, me, kDirPageBytes),
                kDirPageBytes
            );
            const uint32_t end_slot = core_words[kCoreWords * me + 3];
            const uint32_t slot_address = directory_space + kDirPageBytes;
            build_slot(slot_address, kLayMarkEnd, 0);
            asm volatile("fence" ::: "memory");
            noc_async_write(
                slot_address,
                get_noc_addr(end_slot / 16, lists, (end_slot % 16) * 128),
                128
            );
            if (!flushed()) {
                return false;
            }
        }
        // headers, 32 at a time
        uint32_t batched = 0;
        const uint32_t header_space = directory_space + kDirPageBytes + 128;
        for (uint32_t i = me; i < item_count; i += cores) {
            if ((item_words[kItemWords * i] >> 8) == 0) {
                continue;
            }
            const uint32_t slot_address = header_space + 128 * batched;
            const uint32_t slot = item_words[kItemWords * i + 1];
            build_slot(slot_address, kLayMarkTile, i);
            asm volatile("fence" ::: "memory");
            noc_async_write(
                slot_address, get_noc_addr(slot / 16, lists, (slot % 16) * 128), 128
            );
            if (++batched == kOutBytes / 128) {
                if (!flushed()) {
                    return false;
                }
                batched = 0;
            }
        }
        return flushed();
    }

}  // namespace

namespace {

    // COUNT's second stage: per tile entries and rows (kCbCount's space) and the
    // 2-row histograms (kCbHist's, kept for HIST) from the pack thread's box words.
    template <typename Flushed, typename Arrived>
    bool count_stage(
        const km::Writer& writer,
        const km_command_t& command,
        WriterProfile& profile,
        Flushed flushed,
        Arrived arrived
    ) {
        const uint32_t counters_address = command.a1, blocks_word = command.a2;
        const uint32_t camera_address = command.a3, bank = bank_word(command.a8);
        const uint32_t across = grid_of(command.a8).across;  // tile ty TX + tx
        // the frame's tiles in SUM's slices, not the capacity's
        const uint32_t tiles = slice_tiles(command.a8);
        volatile tt_l1_ptr uint32_t* counters = km::l1(get_read_ptr(kCbCount));
        volatile tt_l1_ptr uint32_t* histogram_words = km::l1(get_read_ptr(kCbHist));
        volatile tt_l1_ptr uint16_t*
            histogram = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(histogram_words);
        const uint32_t zero_started = km::wall();
#ifdef RASTER_RECOUNT
        // COUNT counts (a4 0), or again its heavy tiles' histograms (a4 the heavy
        // tiles' page, README.md): through a map, 1 + heavy index a tile (kCbCount)
        const uint32_t heavy_address = command.a4;
        volatile tt_l1_ptr uint16_t*
            heavy_of = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(counters);
        if (heavy_address != 0) {
            const uint32_t heavy_space = get_read_ptr(kCbCount) + 4 * kTiles;
            noc_async_read(
                setup_pass::noc_address(bank, heavy_address, 0, kHeavyBytes),
                heavy_space,
                kHeavyBytes
            );
            for (uint32_t i = 0; i < tiles / 2; ++i) {
                counters[i] = 0;
            }
            if (!arrived()) {
                return false;
            }
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* heavy_words = km::l1(heavy_space);
            const uint32_t heavy_count = heavy_words[0];
            if (heavy_count > kHistTiles) {
                give_up(kGaveBadHeavy, heavy_count);
                return true;
            }
            for (uint32_t heavy_index = 0; heavy_index < heavy_count; ++heavy_index) {
                heavy_of[heavy_words[1 + heavy_index] % kTiles] = heavy_index + 1;
            }
            for (uint32_t i = 0; i < 16 * heavy_count; ++i) {
                histogram_words[i] = 0;
            }
        } else {
            for (uint32_t i = 0; i < tiles; ++i) {  // entries, then rows at kTiles
                counters[i] = counters[kTiles + i] = 0;
            }
        }
#else
        for (uint32_t i = 0; i < tiles; ++i) {  // entries, then rows at kTiles
            counters[i] = counters[kTiles + i] = 0;
        }
        for (uint32_t i = 0; i < 16 * tiles; ++i) {  // 64 bytes a tile
            histogram_words[i] = 0;
        }
#endif
        profile.write += km::wall() - zero_started;
        for (;;) {
            const uint32_t wait_start = km::wall();
            if (!writer.wait_front(kCbOut, 1, 4)) {
                return true;
            }
            const uint32_t wait_end = km::wall();
            profile.wait += wait_end - wait_start;
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* page_words = km::l1(get_read_ptr(kCbOut));
            const uint32_t head = page_words[0], words = head & ~kOutLast;
            ++profile.pages;
            for (
                uint32_t i = 0; i + kBoxWords <= words && i < kOutWords; i += kBoxWords
            ) {
                const BoxWord box = box_at(page_words + 1 + i);
                for (
                    uint32_t tile_y = box.first_row >> 5; tile_y <= box.last_row >> 5;
                    ++tile_y
                ) {
                    const uint32_t top = 32 * tile_y;
                    const uint32_t first_row = (
                        box.first_row > top ? box.first_row : top
                    );
                    const uint32_t last_row = (
                        box.last_row < top + 31 ? box.last_row : top + 31
                    );
                    for (
                        uint32_t tile_x = box.first_tile_x; tile_x <= box.last_tile_x;
                        ++tile_x
                    ) {
                        const uint32_t tile = across * tile_y + tile_x;
#ifdef RASTER_RECOUNT
                        if (heavy_address == 0) {
                            counters[tile] += 1;
                            counters[kTiles + tile] += last_row - first_row + 1;
                            continue;
                        }
                        const uint32_t heavy = heavy_of[tile];
                        if (heavy == 0) {
                            continue;
                        }
                        volatile tt_l1_ptr uint16_t*
                            tile_histogram = (histogram + 32 * (heavy - 1));
#else
                        counters[tile] += 1;
                        counters[kTiles + tile] += last_row - first_row + 1;
                        volatile tt_l1_ptr uint16_t*
                            tile_histogram = (histogram + 32 * tile);
#endif
                        const uint16_t starts = (
                            tile_histogram[(first_row - top) >> 1] + 1
                        );
                        tile_histogram[(first_row - top) >> 1] = starts;
                        const uint16_t ends = (
                            tile_histogram[16 + ((last_row - top) >> 1)] + 1
                        );
                        tile_histogram[16 + ((last_row - top) >> 1)] = ends;
                        if (starts == 0 || ends == 0) {
                            give_up(kGaveBadHist, tile);
                            cb_pop_front(kCbOut, 1);
                            return true;
                        }
                    }
                }
            }
            cb_pop_front(kCbOut, 1);
            profile.write += km::wall() - wait_end;
            if (head & kOutLast) {
                break;
            }
        }
#ifdef RASTER_RECOUNT
        if (heavy_address != 0) {  // the histograms stay for HIST; counters and costs
            return true;           // are the first COUNT's
        }
#endif
        const uint32_t send_started = km::wall();
        asm volatile(
            "fence" ::: "memory"
        );  // the counters' stores land before the read
        // the frame's tiles' entries, and their rows at kTiles (SUM's slices)
        const uint64_t counters_page = (
            setup_pass::noc_address(0, counters_address, writer.me(), kCountBytes)
        );
        profile.bytes += 8 * tiles;
        noc_async_write(get_read_ptr(kCbCount), counters_page, 4 * tiles);
        noc_async_write(
            get_read_ptr(kCbCount) + 4 * kTiles, counters_page + 4 * kTiles, 4 * tiles
        );
        if ((blocks_word & kRanges) != 0 && bank != 0) {
            // the pack thread's costs, final with its last page (setup_compute.cpp)
            noc_async_write(
                get_write_ptr(kCbWork) + kWorkCostsAt,
                setup_pass::noc_address(
                    bank, camera_address, 0, 0, kCostsAt + 128 * writer.me()
                ),
                128
            );
        }
        const bool succeeded = flushed();
        profile.write += km::wall() - send_started;
        return succeeded;
    }

    // SETUP's second stage: each record of the pack thread's pages to the next slot of
    // every item (a tile, or a band of its rows) its box reaches.
    template <typename Flushed>
    bool setup_stage(
        const km::Writer& writer,
        const km_command_t& command,
        WriterProfile& profile,
        Flushed flushed
    ) {
        const uint32_t lists_address = command.a1;
        const uint32_t list_slots = command.a6 & ~kSetupPrefix;
        const uint32_t across = grid_of(command.a8).across;  // tile ty TX + tx
        if (!writer.wait_front(kCbSlots, 1, 4) || !writer.wait_front(kCbItems, 1, 4)) {
            return true;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* next_slot = km::l1(get_read_ptr(kCbSlots));
        volatile tt_l1_ptr uint32_t* items = km::l1(get_read_ptr(kCbItems));
        const volatile tt_l1_ptr uint16_t*
            rows = (
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(
                    get_read_ptr(kCbItems) + kItemRows
                )
            );
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = lists_address, .page_size = kListPageBytes
        };
        bool succeeded = true;
        for (;;) {
            const uint32_t wait_start = km::wall();
            if (!writer.wait_front(kCbOut, 1, 4)) {
                return true;
            }
            const uint32_t wait_end = km::wall();
            profile.wait += wait_end - wait_start;
            invalidate_l1_cache();
            const uint32_t page = get_read_ptr(kCbOut);
            volatile tt_l1_ptr uint32_t* page_words = km::l1(page);
            const uint32_t head = page_words[0], entries = head & ~kOutLast;
            ++profile.pages;
            for (uint32_t i = 0; i < entries && i < kOutEntries && succeeded; ++i) {
#ifdef RASTER_WIDE
                const BoxWord box = record_box(page_words + 32 * (1 + i));
#else
                const BoxWord box = box_at(page_words + 1 + i);
#endif
                for (
                    uint32_t tile_y = box.first_row >> 5;
                    tile_y <= box.last_row >> 5 && succeeded;
                    ++tile_y
                ) {
                    const uint32_t top = 32 * tile_y;
                    const uint32_t first_row = (
                        box.first_row > top ? box.first_row : top
                    );
                    const uint32_t last_row = (
                        box.last_row < top + 31 ? box.last_row : top + 31
                    );
                    for (
                        uint32_t tile_x = box.first_tile_x;
                        tile_x <= box.last_tile_x && succeeded;
                        ++tile_x
                    ) {
                        const uint32_t tile = across * tile_y + tile_x;
                        const uint32_t first = items[tile] & 0xFFFF;
                        const uint32_t item_count = items[tile] >> 16;
                        if (first + item_count > kSlotsBytes / 4) {
                            give_up(kGaveBadSlot, tile);
                            succeeded = false;
                            break;
                        }
                        for (uint32_t item = first; item < first + item_count; ++item) {
                            // the item's rows [item_first_row, item_end_row) against
                            // the entry's 2-row groups
                            const uint32_t item_first_row = rows[item] & 0xFF;
                            const uint32_t item_end_row = rows[item] >> 8;
                            if (
                                ((first_row - top) >> 1) >= (item_end_row >> 1) ||
                                ((last_row - top) >> 1) < (item_first_row >> 1)
                            ) {
                                continue;
                            }
                            const uint32_t entry_slot = next_slot[item];
                            next_slot[item] = entry_slot + 1;
                            if (entry_slot >= list_slots) {
                                give_up(kGaveBadSlot, tile);
                                succeeded = false;
                                break;
                            }
                            noc_async_write(
                                page + 128 * (1 + i),
                                get_noc_addr(
                                    entry_slot / 16, lists, (entry_slot % 16) * 128
                                ),
                                128
                            );
                            profile.bytes += 128;
                        }
                    }
                }
            }
            succeeded = flushed() && succeeded;
            cb_pop_front(kCbOut, 1);
            profile.write += km::wall() - wait_end;
            if (!succeeded) {
                return false;
            }
            if (head & kOutLast) {
                break;
            }
        }
        cb_pop_front(kCbSlots, 1);
        cb_pop_front(kCbItems, 1);
        return true;
    }

}  // namespace

bool setup_writer_pass(
    const km::Writer& writer, const km_command_t& command, WriterProfile& profile
);

// The passes' writing for one command; the frame program's writer calls it as well.
// With profile_address (a9), the threads' counters go to the core's profile page.
bool setup_writer_command(const km::Writer& writer, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    const uint32_t cores = command.a5, profile_address = command.a9;
    if (
        (
            handler != kHandlerLanes && handler != kHandlerCount &&
            handler != kHandlerSetup && handler != kHandlerHist &&
            handler != kHandlerPlace && handler != kHandlerSum &&
            handler != kHandlerHsum
        ) ||
        cores == 0 || writer.me() >= cores
    ) {
        return true;
    }
    const uint32_t started = km::wall();
    WriterProfile profile;
    const bool succeeded = setup_writer_pass(writer, command, profile);
    if (
        !succeeded || profile_address == 0 || handler < kHandlerCount ||
        handler > kHandlerPlace
    ) {
        return succeeded;
    }
    invalidate_l1_cache();
    const uint32_t page = get_write_ptr(kCbSProf);
    volatile tt_l1_ptr uint32_t* words = km::l1(page);
    words[kSProfWriter + 0] = profile.wait;
    words[kSProfWriter + 1] = profile.write;
    words[kSProfWriter + 2] = km::wall() - started;
    words[kSProfWriter + 3] = profile.bytes;
    words[kSProfWriter + 4] = started;
    words[kSProfWriter + 5] = profile.pages;
    noc_async_write(
        page,
        setup_pass::noc_address(
            0,
            profile_address,
            writer.me(),
            kSProfPageBytes,
            128 * (handler - kHandlerCount)
        ),
        128
    );
    return writer.poll(
        [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        },
        writer.args().wait_limit
    ) == km::kGaveNone;
}

// One command's writing, its counters in profile.
bool setup_writer_pass(
    const km::Writer& writer, const km_command_t& command, WriterProfile& profile
) {
    const uint32_t handler = km::handler(command);
    const uint32_t cores = command.a5, bank = bank_word(command.a8);
    auto flushed = [&] {
        return writer.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    auto arrived = [&] {
        return writer.poll(
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    if (handler == kHandlerCount) {
        return count_stage(writer, command, profile, flushed, arrived);
    }
    if (handler == kHandlerHist) {
        const uint32_t heavy_address = command.a0, histograms_address = command.a1;
        const uint32_t histogram_page_bytes = command.a4;
        // the heavy tiles into the out buffer's space (unused by HIST)
        const uint32_t heavy_space = get_write_ptr(kCbOut);
        noc_async_read(
            setup_pass::noc_address(bank, heavy_address, 0, kHeavyBytes),
            heavy_space,
            kHeavyBytes
        );
        if (!arrived()) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* heavy_words = km::l1(heavy_space);
        const uint32_t heavy_count = heavy_words[0];
        if (heavy_count >= kHeavyBytes / 4 || 64 * heavy_count > histogram_page_bytes) {
            give_up(kGaveBadHeavy, heavy_count);
            return true;
        }
        const uint32_t histogram_space = get_read_ptr(kCbHist);
        for (uint32_t heavy_index = 0; heavy_index < heavy_count; ++heavy_index) {
#ifdef RASTER_RECOUNT
            const uint32_t tile = heavy_index;  // the second COUNT's table
#else
            const uint32_t tile = heavy_words[1 + heavy_index] % kTiles;
#endif
            noc_async_write(
                histogram_space + 64 * tile,
                setup_pass::noc_address(
                    0,
                    histograms_address,
                    writer.me(),
                    histogram_page_bytes,
                    64 * heavy_index
                ),
                64
            );
        }
        return flushed();
    }
    if ((handler == kHandlerSum || handler == kHandlerHsum) && bank == 0) {
        give_up(kGaveBadPlace, handler);
        return true;
    }
    if (handler == kHandlerHsum) {
        const uint32_t heavy_tiles = command.a0, histogram_page_bytes = command.a4;
        if (
            heavy_tiles > kHprefixPageBytes / 128 ||
            64 * heavy_tiles > histogram_page_bytes
        ) {
            give_up(kGaveBadHeavy, heavy_tiles);
            return true;
        }
    }
    if (handler == kHandlerSum) {
        const uint32_t counters_address = command.a1, prefixes_address = command.a2;
        const uint32_t totals_address = command.a3, counting_cores = command.a6;
        // counters_address counters, prefixes_address slots pages, totals_address
        // totals, counting_cores cores; 32 cores at a time through one out page.
        constexpr uint32_t kBatch = 32;
        const uint32_t staging = get_write_ptr(kCbOut);
        // the frame's tiles; the totals laid for them: entries, then rows from 4 x
        // tiles
        const uint32_t tiles = slice_tiles(command.a8), slices = tiles / 16;
        for (uint32_t slice = writer.me(); slice < slices; slice += cores) {
            uint32_t entries[16] = {}, row_counts[16] = {};
            for (
                uint32_t first_core = 0; first_core < counting_cores;
                first_core += kBatch
            ) {
                const uint32_t batch_cores = (
                    counting_cores - first_core < kBatch
                        ? counting_cores - first_core
                        : kBatch
                );
                for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
                    const uint64_t address = (
                        setup_pass::noc_address(
                            0,
                            counters_address,
                            first_core + batch_core,
                            kCountBytes,
                            64 * slice
                        )
                    );
                    noc_async_read(address, staging + 64 * batch_core, 64);
                    noc_async_read(
                        address + 4 * kTiles, staging + 2048 + 64 * batch_core, 64
                    );
                }
                if (!arrived()) {
                    return false;
                }
                invalidate_l1_cache();
                for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
                    volatile tt_l1_ptr uint32_t* entry_words = (
                        km::l1(staging + 64 * batch_core)
                    );
                    volatile tt_l1_ptr uint32_t* row_words = (
                        km::l1(staging + 2048 + 64 * batch_core)
                    );
                    for (uint32_t j = 0; j < 16; ++j) {
                        const uint32_t entry = entry_words[j];
                        entry_words[j] = entries[j];
                        entries[j] += entry;
                        row_counts[j] += row_words[j];
                    }
                    asm volatile("fence" ::: "memory");
                    noc_async_write(
                        staging + 64 * batch_core,
                        setup_pass::noc_address(
                            0,
                            prefixes_address,
                            first_core + batch_core,
                            kSlotsBytes,
                            64 * slice
                        ),
                        64
                    );
                }
                if (!flushed()) {
                    return false;
                }
            }
            volatile tt_l1_ptr uint32_t* total_words = km::l1(staging);
            for (uint32_t j = 0; j < 16; ++j) {
                total_words[j] = entries[j];
                total_words[16 + j] = row_counts[j];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(
                staging,
                setup_pass::noc_address(bank, totals_address, 0, 0, 64 * slice),
                64
            );
            noc_async_write(
                staging + 64,
                setup_pass::noc_address(
                    bank, totals_address, 0, 0, 4 * tiles + 64 * slice
                ),
                64
            );
            if (!flushed()) {
                return false;
            }
        }
        return true;
    }
    if (handler == kHandlerHsum) {
        const uint32_t heavy_tiles = command.a0, histograms_address = command.a1;
        const uint32_t prefixes_address = command.a2, totals_address = command.a3;
        const uint32_t histogram_page_bytes = command.a4, counting_cores = command.a6;
        // 16 cores at a time: their counts, then the 32 words each one's page gets
        // (README.md)
        constexpr uint32_t kBatch = 16;
        const uint32_t staging = get_write_ptr(kCbOut);
        const uint32_t prefix_space = staging + 64 * kBatch;
        for (
            uint32_t heavy_tile = writer.me(); heavy_tile < heavy_tiles;
            heavy_tile += cores
        ) {
            uint32_t totals[32] = {};
            for (
                uint32_t first_core = 0; first_core < counting_cores;
                first_core += kBatch
            ) {
                const uint32_t batch_cores = (
                    counting_cores - first_core < kBatch
                        ? counting_cores - first_core
                        : kBatch
                );
                for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
                    noc_async_read(
                        setup_pass::noc_address(
                            0,
                            histograms_address,
                            first_core + batch_core,
                            histogram_page_bytes,
                            64 * heavy_tile
                        ),
                        staging + 64 * batch_core,
                        64
                    );
                }
                if (!arrived()) {
                    return false;
                }
                invalidate_l1_cache();
                for (uint32_t batch_core = 0; batch_core < batch_cores; ++batch_core) {
                    volatile tt_l1_ptr uint16_t*
                        count_words = (
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                staging + 64 * batch_core
                            )
                        );
                    volatile tt_l1_ptr uint32_t* prefix_words = (
                        km::l1(prefix_space + 128 * batch_core)
                    );
                    uint32_t starts = 0, ends = 0;
                    for (uint32_t i = 0; i < 16; ++i) {
                        starts += totals[i];
                        ends += totals[16 + i];
                        prefix_words[i] = starts;
                        prefix_words[16 + i] = ends;
                    }
                    for (uint32_t i = 0; i < 32; ++i) {
                        totals[i] += count_words[i];
                    }
                    asm volatile("fence" ::: "memory");
                    noc_async_write(
                        prefix_space + 128 * batch_core,
                        setup_pass::noc_address(
                            0,
                            prefixes_address,
                            first_core + batch_core,
                            kHprefixPageBytes,
                            128 * heavy_tile
                        ),
                        128
                    );
                }
                if (!flushed()) {
                    return false;
                }
            }
            volatile tt_l1_ptr uint32_t* total_words = km::l1(staging);
            for (uint32_t i = 0; i < 32; ++i) {
                total_words[i] = totals[i];
            }
            asm volatile("fence" ::: "memory");
            noc_async_write(
                staging,
                setup_pass::noc_address(
                    bank,
                    totals_address,
                    0,
                    0,
                    8 * slice_tiles(command.a8) + 128 * heavy_tile  // after the rows
                ),
                128
            );
            if (!flushed()) {
                return false;
            }
        }
        return true;
    }
    if (handler == kHandlerPlace) {
        if (bank == 0) {
            give_up(kGaveBadPlace, 0);
            return true;
        }
        return place(writer, command, arrived, flushed);
    }
    if (handler == kHandlerSetup) {
        return setup_stage(writer, command, profile, flushed);
    }
    const uint32_t output_address = command.a1, blocks_word = command.a2;
    const uint32_t first_written = command.a4, written_blocks = command.a6;
    const uint32_t first_block = writer.me() * blocks_word / cores;
    const uint32_t end_block = (writer.me() + 1) * blocks_word / cores;
    const InterleavedAddrGen<true> records = {
        .bank_base_address = output_address, .page_size = kRecBytes
    };
    for (uint32_t block = first_block; block < end_block; ++block) {
        if (!writer.wait_front(kCbRec, 1)) {
            return true;
        }
        if (
            block - first_written >= written_blocks
        ) {  // outside the window (block < first_written wraps)
            cb_pop_front(kCbRec, 1);
            continue;
        }
        noc_async_write(
            get_read_ptr(kCbRec),
            get_noc_addr(block - first_written, records),
            kRecBytes
        );
        if (!flushed()) {
            return false;
        }
        cb_pop_front(kCbRec, 1);
    }
    return true;
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        return setup_writer_command(writer, command);
    });
}
#endif
