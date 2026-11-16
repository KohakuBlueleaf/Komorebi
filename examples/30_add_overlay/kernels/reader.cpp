// The reader of the load: RUN calls the overlay its a0 names (any overlay of the run,
// one added after the load too) and writes the result and its cycles to its line.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "add.h"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRun) {
            return;
        }
        uint32_t result = command.a1 ^ reader.me(), held = 0;
        uint64_t started = km::wall64(), loaded = started, called = started;
        if (command.a0 != 0) {
            km::Overlay overlay(reader, command.a0);  // names it; nothing moves
            held = overlay.held();
            started = km::wall64();
            if (!overlay.load()) {  // into a slot of this core; nothing when held
                return;
            }
            loaded = km::wall64();
            result = overlay.call(MixArgument{result, command.a3});
            called = km::wall64();
        }
        volatile tt_l1_ptr uint32_t* line = km::l1(get_write_ptr(kCbLine));
        line[0] = command.sequence;
        line[1] = result;
        line[2] = static_cast<uint32_t>(loaded - started);
        line[3] = static_cast<uint32_t>(called - loaded);
        line[4] = held;
        line[5] = command.a0;
        const km::Tensor<0> lines(command.a2);
        if (!km::write_pages(
            reader, command, lines, reader.me(), 1, get_write_ptr(kCbLine)
        )) {
            return;
        }
        noc_async_write_barrier();
    });
}
