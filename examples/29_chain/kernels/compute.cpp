// Chain example, compute: the movers do the work; in a chain command compute only
// takes each record in turn, as every kernel of a core does.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_chain.hpp"
#include "komorebi/device/km_port.hpp"
#include "steps.h"

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerStep && command.a4 == kChainMode) {
            km::chain_compute(compute, command, [](const km_command_t&) {});
        }
    });
}
