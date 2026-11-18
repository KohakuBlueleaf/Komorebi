// Tree collectives, reader: every rank of TREE makes the one call (a5) on the copy.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "pages.h"

void kernel_main() {
    km::Reader reader;
    const uint32_t fold = get_write_ptr(kCbFold);  // this core's, never pushed
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerTree) {
            return;
        }
        const auto sum = static_cast<km::Sum>(command.a4);
        const km::Tree tree = (
            command.a7 != 0 ? km::tree(command.a8, command.a9) : km::tree()
        );
        const uint32_t room = kFoldPages * kFoldBytes;
        if (command.a5 == kCallGather) {
            km::all_gather(reader, command, command.a0, command.a1, command.a2);
        } else if (command.a5 == kCallReduce) {
            km::all_reduce(
                reader,
                command,
                command.a0,
                command.a1,
                command.a2,
                command.a3,
                sum,
                fold,
                room,
                tree
            );
        } else if (command.a5 == kCallReduceTo) {
            km::reduce_to(
                reader,
                command,
                command.a0,
                command.a1,
                command.a2,
                command.a3,
                sum,
                fold,
                room,
                command.a6,
                tree
            );
        } else {
            km::broadcast_from(
                reader, command, command.a0, command.a1, command.a2, command.a6, tree
            );
        }
    });
}
