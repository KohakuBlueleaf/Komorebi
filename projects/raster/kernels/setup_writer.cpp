// The passes' writer: LANES the window's records, COUNT the counters, SETUP each record
// to its list slot, HIST the heavy tiles' histograms, PLACE the directory and marks.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

namespace {

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
        const uint32_t bank = command.a8;
        const uint32_t space = get_write_ptr(kCbHist);
        noc_async_read(
            setup_pass::noc_address(bank, items_region_address, 0, 0, 0),
            space,
            kItemsBytes
        );
        noc_async_read(
            setup_pass::noc_address(bank, items_region_address, 0, 0, kFrameWordsAt),
            space + kFrameWordsAt,
            kItemsRegionBytes - kFrameWordsAt
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
        for (uint32_t tile = 0; tile < kTiles; ++tile) {
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

bool setup_writer_pass(
    const km::Writer& writer, const km_command_t& command, WriterProfile& profile
);

// The passes' writing for one command; the frame program's writer calls it as well.
// With profile_address (a9), the threads' counters go to the core's profile page.
bool setup_writer_command(const km::Writer& writer, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    const uint32_t blocks_word = command.a2, camera_address = command.a3;
    const uint32_t cores = command.a5, bank = command.a8, profile_address = command.a9;
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
        succeeded && (handler == kHandlerCount || handler == kHandlerSetup) &&
        (blocks_word & kRanges) != 0
    ) {
        // the pass's ticks on this core, for the next frame's ranges
        const uint32_t line = get_write_ptr(kCbSProf) + 128;
        km::l1(line)[0] = km::wall() - started;
        invalidate_l1_cache();
        noc_async_write(
            line,
            setup_pass::noc_address(
                bank,
                camera_address,
                0,
                0,
                kCostsAt + 128 * writer.me() + 64 * (handler - kHandlerCount)
            ),
            64
        );
        if (writer.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        ) != km::kGaveNone) {
            return false;
        }
    }
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
    const uint32_t cores = command.a5, bank = command.a8;
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
        const uint32_t counters_address = command.a1;
        const uint32_t wait_start = km::wall();
        if (!writer.wait_front(kCbCount, 1)) {
            return true;
        }
        const uint32_t wait_end = km::wall();
        profile.wait += wait_end - wait_start;
        profile.bytes += kCountBytes;
        noc_async_write(
            get_read_ptr(kCbCount),
            setup_pass::noc_address(0, counters_address, writer.me(), kCountBytes),
            kCountBytes
        );
        const bool succeeded = flushed();
        profile.write += km::wall() - wait_end;
        cb_pop_front(kCbCount, 1);
        return succeeded;
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
            const uint32_t tile = heavy_words[1 + heavy_index] % kTiles;
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
        for (uint32_t slice = writer.me(); slice < kTiles / 16; slice += cores) {
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
                    bank, totals_address, 0, 0, 4 * kTiles + 64 * slice
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
                    bank, totals_address, 0, 0, kSumsHeavyAt + 128 * heavy_tile
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
        const uint32_t lists_address = command.a1;
        const InterleavedAddrGen<true> lists = {
            .bank_base_address = lists_address, .page_size = kListPageBytes
        };
        for (;;) {
            const uint32_t wait_start = km::wall();
            if (!writer.wait_front(kCbOut, 1, 4)) {
                return true;
            }
            const uint32_t wait_end = km::wall();
            profile.wait += wait_end - wait_start;
            const uint32_t page = get_read_ptr(kCbOut);
            volatile tt_l1_ptr uint32_t* page_words = km::l1(page);
            const uint32_t head = page_words[0], entries = head & ~kOutLast;
            profile.bytes += 128 * entries;
            ++profile.pages;
            for (uint32_t i = 0; i < entries && i < kOutEntries; ++i) {
                const uint32_t list_slot = page_words[1 + i];
                noc_async_write(
                    page + 128 * (1 + i),
                    get_noc_addr(list_slot / 16, lists, (list_slot % 16) * 128),
                    128
                );
            }
            const bool succeeded = flushed();
            profile.write += km::wall() - wait_end;
            cb_pop_front(kCbOut, 1);
            if (!succeeded) {
                return false;
            }
            if (head & kOutLast) {
                return true;
            }
        }
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
