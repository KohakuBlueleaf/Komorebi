// Mover A reads a core's pages of k tiles a block at a time as the movement engine does
// (room, the block's reads, one barrier, a push of its tiles); mover B does nothing.
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

    // READ on mover A: every block of every repetition, then the profile page.
    bool read(const Runtime& runtime, const km_command_t& command) {
        const uint32_t page_bytes = command.a1;
        const uint32_t pages = command.a2;
        const uint32_t block_pages = command.a3;
        const uint32_t buffer = command.a4;
        const uint32_t repetitions = command.a5;
        const uint32_t block_tiles = command.a7;
        const uint32_t first = (runtime.me() - km::first(command)) * pages;
        const InterleavedAddrGen<true> tensor = {
            .bank_base_address = command.a0, .page_size = page_bytes
        };
        uint32_t read_pages = 0;
        const uint32_t start = km::wall();
        for (uint32_t repetition = 0; repetition < repetitions; repetition++) {
            for (uint32_t page = first; page < first + pages; page += block_pages) {
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
        volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kBufferProfile));
        profile[kProfileStart] = start;
        profile[kProfileEnd] = end;
        profile[kProfilePages] = read_pages;
        const InterleavedAddrGen<true> out = {
            .bank_base_address = command.a6, .page_size = kProfileBytes
        };
        noc_async_write(
            get_write_ptr(kBufferProfile),
            out.get_noc_addr(runtime.me() - km::first(command)),
            kProfileBytes
        );
        return km::move_settle(runtime, false, true, true);
    }

}  // namespace

void kernel_main() {
    Runtime runtime;
    runtime.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRead || km::kMover != km::kMoverA) {
            return true;
        }
        return read(runtime, command);
    });
}
