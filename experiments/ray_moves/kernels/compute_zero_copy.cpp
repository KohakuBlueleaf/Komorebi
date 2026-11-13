// The transport probe's compute kernel on pages that no core copies (the reader checks
// them).
#include <stdint.h>

#include "api/compute/common.h"
#include "zero_copy_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "bundle_body.hpp"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    const uint32_t seed_address = compute.arg(km::kMessagesCArgs + 1);
    auto forward = [](
        km::Messages & messages,
        uint32_t destination,
        volatile tt_l1_ptr uint32_t* incoming
    ) {
        messages.forward_page(destination, reinterpret_cast<uint32_t>(incoming));
    };
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerBundle) {
            bundle_run(compute, command, results_address, seed_address, false, forward);
        }
    });
}
