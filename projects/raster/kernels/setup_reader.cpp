// COUNT and SETUP reader: SETUP's slots and items pages, then the camera's page and
// this core's partition of the mesh blocks into the mesh buffer, one page each.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#include "setup_noc.hpp"

namespace {

    // kSetupPrefix: base + SUM's entries before (a whole tile) or HSUM's starts before
    // end_row less ends before first_row (README.md); in place, tiles from the last.
    template <typename Arrived>
    bool slots_from_prefixes(
        const km::Reader& reader,
        const km_command_t& command,
        Arrived arrived,
        uint32_t& shade_address
    ) {
        const uint32_t items_address = command.a7, bank = command.a8;
        const uint32_t space = get_write_ptr(kCbHist);
        constexpr uint32_t kRead = kHprefixAddrAt - kBaseAt +
            64;  // base words, address
        noc_async_read(
            setup_pass::noc_address(bank, items_address, 0, 0, kBaseAt), space, kRead
        );
        if (!arrived()) {
            return false;
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* items = km::l1(get_write_ptr(kCbItems));
        volatile tt_l1_ptr uint32_t* base_words = km::l1(space);
        const uint32_t last = items[kTiles - 1];
        const uint32_t item_count = (last & 0xFFFF) + (last >> 16);
        if (item_count > kSlotsBytes / 4) {
            return false;
        }
        uint32_t heavy_tiles = 0;
        for (uint32_t item = 0; item < item_count; ++item) {
            const uint32_t heavy_index = base_words[item] >> kBaseSlotBits;
            heavy_tiles = heavy_index > heavy_tiles ? heavy_index : heavy_tiles;
        }
        const uint32_t counts = space + kRead;
        const uint32_t prefix_pages_address = (
            base_words[(kHprefixAddrAt - kBaseAt) / 4]
        );
        shade_address = base_words[(kShadeAddrAt - kBaseAt) / 4];
        if (heavy_tiles > kHprefixPageBytes / 128) {
            return false;
        }
        if (heavy_tiles != 0) {
            noc_async_read(
                setup_pass::noc_address(
                    0, prefix_pages_address, reader.me(), kHprefixPageBytes
                ),
                counts,
                128 * heavy_tiles
            );
            if (!arrived()) {
                return false;
            }
            invalidate_l1_cache();
        }
        volatile tt_l1_ptr uint32_t* slots = km::l1(get_write_ptr(kCbSlots));
        const volatile tt_l1_ptr uint16_t*
            rows = (
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    get_write_ptr(kCbItems) + kItemRows
                )
            );
        constexpr uint32_t kSlotMask = (1u << kBaseSlotBits) - 1;
        for (uint32_t tile = kTiles; tile-- > 0;) {
            const uint32_t first = items[tile] & 0xFFFF;
            const uint32_t items_in_tile = items[tile] >> 16;
            const uint32_t before = slots[tile];
            for (
                uint32_t item = first;
                item < first + items_in_tile && item < item_count;
                ++item
            ) {
                const uint32_t base_word = base_words[item];
                const uint32_t heavy_index = base_word >> kBaseSlotBits;
                uint32_t next_slot = base_word & kSlotMask;
                if (heavy_index == 0) {
                    next_slot += before;
                } else {
                    volatile tt_l1_ptr uint32_t* prefix_counts = (
                        km::l1(counts + 128 * (heavy_index - 1))
                    );
                    const uint32_t end_group = (rows[item] >> 8) >> 1;
                    const uint32_t start_group = (rows[item] & 0xFF) >> 1;
                    next_slot += (
                        (end_group ? prefix_counts[end_group - 1] : 0) -
                        (start_group ? prefix_counts[16 + start_group - 1] : 0)
                    );
                }
                slots[item] = next_slot;
            }
        }
        asm volatile("fence" ::: "memory");  // the stores land before the push
        return true;
    }

}  // namespace

// The passes' reading for one command; the frame program's reader calls it as well.
void setup_reader_command(const km::Reader& reader, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    const uint32_t cores = command.a5;
    if (
        (
            handler != kHandlerLanes && handler != kHandlerCount &&
            handler != kHandlerSetup
        ) ||
        cores == 0 || reader.me() >= cores
    ) {
        return;
    }
    const uint32_t mesh_address = command.a0, blocks_word = command.a2;
    const uint32_t camera_address = command.a3, slots_address = command.a4;
    const uint32_t list_slots = command.a6, items_address = command.a7;
    const uint32_t bank = command.a8, profile_address = command.a9;
    const uint32_t started = km::wall();
    uint32_t wait = 0, read = 0, bytes = 0, shade_address = 0;
    auto reserve = [&](uint32_t cb) {
        const uint32_t asked = km::wall();
        const bool reserved = reader.reserve(cb, 1);
        wait += km::wall() - asked;
        return reserved;
    };
    auto arrived = [&] {
        return reader.poll(
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            reader.args().wait_limit
        ) == km::kGaveNone;
    };
    if (handler == kHandlerSetup) {
        if (!reserve(kCbSlots) || !reserve(kCbItems)) {
            return;
        }
        noc_async_read(
            setup_pass::noc_address(0, slots_address, reader.me(), kSlotsBytes),
            get_write_ptr(kCbSlots),
            kSlotsBytes
        );
        noc_async_read(
            setup_pass::noc_address(bank, items_address, 0, kItemsBytes),
            get_write_ptr(kCbItems),
            kItemsBytes
        );
        if (!arrived()) {
            return;
        }
        if (
            (list_slots & kSetupPrefix) != 0 &&
            !slots_from_prefixes(reader, command, arrived, shade_address)
        ) {
            return;
        }
        cb_push_back(kCbSlots, 1);
        cb_push_back(kCbItems, 1);
    }
    const uint32_t blocks = blocks_word & ~kRanges;
    uint32_t first_block = reader.me() * blocks / cores;
    uint32_t end_block = (reader.me() + 1) * blocks / cores;
    if ((blocks_word & kRanges) != 0) {
        const uint32_t line = get_write_ptr(kCbSProf) + 192;  // scratch
        noc_async_read(
            setup_pass::noc_address(
                bank, camera_address, 0, 0, kRangesAt + 64 * reader.me()
            ),
            line,
            64
        );
        if (!arrived()) {
            return;
        }
        invalidate_l1_cache();
        first_block = km::l1(line)[0];
        end_block = km::l1(line)[1];
    }
    const InterleavedAddrGen<true> mesh = {
        .bank_base_address = mesh_address, .page_size = kMeshBytes
    };
    const InterleavedAddrGen<true> shade = {
        .bank_base_address = shade_address, .page_size = kShadeBlockBytes
    };
    for (uint32_t step = first_block; step <= end_block; ++step) {
        if (!reserve(kCbMesh)) {
            return;
        }
        const uint32_t asked = km::wall();
        const uint64_t source = (
            step == first_block
                ? setup_pass::noc_address(bank, camera_address, 0, 4 * kCamWords)
                : get_noc_addr(step - 1, mesh)
        );
        const uint32_t page_bytes = step == first_block ? 4 * kCamWords : kMeshBytes;
        noc_async_read(source, get_write_ptr(kCbMesh), page_bytes);
        if (step != first_block && shade_address != 0) {
            noc_async_read(
                get_noc_addr(step - 1, shade),
                get_write_ptr(kCbMesh) + kMeshBytes,
                kShadeBlockBytes
            );
        }
        if (!arrived()) {
            return;
        }
        read += km::wall() - asked;
        bytes += page_bytes;
        if (
            step == first_block
        ) {  // the range after the camera, for the compute threads
            volatile tt_l1_ptr uint32_t* camera = km::l1(get_write_ptr(kCbMesh));
            camera[kCamWords] = first_block;
            camera[kCamWords + 1] = end_block;
            camera[kCamWords + 2] = shade_address != 0;
            invalidate_l1_cache();
        }
        cb_push_back(kCbMesh, 1);
    }
    if (profile_address != 0) {
        volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbSProf));
        profile[kSProfReader + 0] = wait;
        profile[kSProfReader + 1] = read;
        profile[kSProfReader + 2] = km::wall() - started;
        profile[kSProfReader + 3] = bytes;
        profile[kSProfReader + 4] = started;
    }
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        setup_reader_command(reader, command);
    });
}
#endif
