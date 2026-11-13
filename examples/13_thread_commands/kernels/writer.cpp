// Thread commands example, grid side: AFFINE's leader gives a0 * 0x9E3779B1 + a1 +
// the group's count as the command's result word; every other core has nothing to do.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "handlers.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (
            km::handler(command) == kHandlerAffine && writer.me() == km::first(command)
        ) {
            writer.result(command.a0 * 0x9E3779B1u + command.a1 + km::count(command));
        }
        return true;
    });
}
