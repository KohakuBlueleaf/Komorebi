// The reader: loads and calls the overlay a command names, and writes its result and
// the cycles of load() and call() to the core's line in DRAM.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_overlay.hpp"
#include "overlays.h"

void kernel_main() {
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRun) {
            return;
        }
        km::Overlay overlay(reader, command.a0);  // names it; nothing moves
        const bool held = overlay.held();
        const uint64_t started = km::wall64();
        if (!overlay.load()) {  // into a slot of this core; nothing when held
            return;
        }
        const uint64_t loaded = km::wall64();
        uint32_t result = 0;
        if (command.a0 == kOverlayMix) {
            result = overlay.call(MixArgument{command.a1 ^ reader.me(), command.a5});
        } else {
            result = overlay.call(ScaleArgument{reader.me(), command.a3, command.a4});
        }
        const uint64_t called = km::wall64();
        // --8<-- [end:run]
        volatile tt_l1_ptr uint32_t* line = km::l1(get_write_ptr(kCbLine));
        line[0] = command.sequence;
        line[1] = result;
        line[2] = static_cast<uint32_t>(loaded - started);
        line[3] = static_cast<uint32_t>(called - loaded);
        line[4] = held;
        line[5] = command.a0;
        const InterleavedAddrGen<true> lines = {
            .bank_base_address = command.a2, .page_size = kLineWords * 4
        };
        noc_async_write(
            get_write_ptr(kCbLine), get_noc_addr(reader.me(), lines), kLineWords * 4
        );
        noc_async_write_barrier();
    });
}
