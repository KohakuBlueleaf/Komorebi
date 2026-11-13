// The writer kernel: every core of the group stores a number in a place in its own L1.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerRemember) {
            const uint32_t notes_address = command.a0;
            const uint32_t rank = writer.me() - km::first(command);
            volatile tt_l1_ptr uint32_t* notes = km::l1(notes_address);
            notes[0] = rank * rank;
        }
        return true;
    });
    // --8<-- [end:run]
}
