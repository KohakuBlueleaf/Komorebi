// The runtime's compute of a run of programs: the hardware start-up on the buffers the
// host names, then for each command the part of the command's program and its call.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"

#if !defined(KM_PROGRAM_CB_IN) || !defined(KM_PROGRAM_CB_OUT)
#error "the host names the start-up buffers (KM_PROGRAM_CB_IN and KM_PROGRAM_CB_OUT)"
#endif

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(KM_PROGRAM_CB_IN, KM_PROGRAM_CB_OUT);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) - 1u >= KM_HANDLER_USER_MAX) {
            return;  // NOP: no program's
        }
        km::Overlay program(compute, km::program_overlay(km::program(command)));
        if (!program.load()) {
            return;  // the give-up path
        }
        program.call(command);
    });
}
