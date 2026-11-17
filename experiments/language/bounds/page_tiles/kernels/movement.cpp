// Each mover reads its blocks of a core's pages of k tiles (the pattern's) as the
// movement engine does: room, the block's reads, one barrier, a push of its tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

namespace {

#if defined(COMPILE_FOR_BRISC)
    using Runtime = km::Reader;
    constexpr uint32_t kGaveUpWord = km::kStReader;
#else
    using Runtime = km::Writer;
    constexpr uint32_t kGaveUpWord = km::kStWriter;
#endif

    // Waits until `buffer` has room for `tiles`. False: gave up (recorded).
    bool wait_room(const Runtime& runtime, uint32_t buffer, uint32_t tiles) {
        const uint32_t code = runtime.poll(
            [&] {
                return cb_pages_reservable_at_back(buffer, tiles);
            },
            runtime.args().wait_limit
        );
        if (code != km::kGaveNone) {
            runtime.gave_up(kGaveUpWord, code, km::kWhereData, 0, 0);
            return false;
        }
        return true;
    }

    // READ on this mover: its blocks of every repetition, then its profile page.
    bool read(const Runtime& runtime, const km_command_t& command) {
        const uint32_t page_bytes = command.a1;
        const uint32_t pages = command.a2;
        const uint32_t block_pages = command.a3;
        const bool on_b = km::kMover == km::kMoverB;
        const uint32_t buffer = block_buffer(command.a4, on_b);
        const uint32_t repetitions = command.a5;
        const uint32_t block_tiles = command.a7;
        const uint32_t pattern = command.a8;
        const uint32_t rank = runtime.me() - km::first(command);
        const uint32_t first = rank * pages;
        const InterleavedAddrGen<true> tensor = {
            .bank_base_address = command.a0, .page_size = page_bytes
        };
        uint32_t read_pages = 0, block = 0;
        const uint32_t start = km::wall();
        for (uint32_t repetition = 0; repetition < repetitions; repetition++) {
            for (uint32_t page = first; page < first + pages; page += block_pages) {
                if (block_on_b(pattern, block++) != on_b) {
                    continue;
                }
                if (!wait_room(runtime, buffer, block_tiles)) {
                    return false;
                }
                cb_reserve_back(buffer, block_tiles);
                uint32_t address = get_write_ptr(buffer);
                for (uint32_t each = 0; each < block_pages; each++) {
                    noc_async_read(
                        tensor.get_noc_addr(page + each), address, page_bytes
                    );
                    address += page_bytes;
                }
                if (!km::move_settle(runtime, true, false)) {
                    return false;
                }
                cb_push_back(buffer, block_tiles);
                read_pages += block_pages;
            }
        }
        const uint32_t end = km::wall();
        const uint32_t own = get_write_ptr(kBufferProfile) + (on_b ? kProfileBytes : 0);
        volatile tt_l1_ptr uint32_t* profile = km::l1(own);
        profile[kProfileStart] = start;
        profile[kProfileEnd] = end;
        profile[kProfilePages] = read_pages;
        const InterleavedAddrGen<true> out = {
            .bank_base_address = command.a6, .page_size = kProfileBytes
        };
        noc_async_write(own, out.get_noc_addr(2 * rank + on_b), kProfileBytes);
        return km::move_settle(runtime, false, true, true);
    }

}  // namespace

void kernel_main() {
    Runtime runtime;
    runtime.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRead) {
            return true;
        }
        return read(runtime, command);
    });
}
