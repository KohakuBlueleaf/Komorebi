// Chain example, compute: the movers do the work; in a chain command compute only
// takes each record in turn, as every kernel of a core does.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/movement/km_chain.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "steps.h"

// A command: a chain command's records taken in turn (nothing else on compute).
inline void command_of(km::Compute& compute, const km_command_t& command) {
    if (step_handler(km::handler(command)) && command.a4 == kChainMode) {
        km::chain_compute(compute, command, [](const km_command_t&) {});
    }
}

#if !defined(KM_OVERLAY_PART)  // the runtime's form calls command_of (runtime/)
void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t& command) {
        command_of(compute, command);
    });
}
#endif
