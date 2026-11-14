// The runtime's compute of a run of programs: the hardware start-up (a load's, in a
// run whose groups switch), then for each command the part of its program, its call.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"

#if !defined(KM_PROGRAM_SWITCH) && \
    (!defined(KM_PROGRAM_CB_IN) || !defined(KM_PROGRAM_CB_OUT))
#error "the host names the start-up buffers (KM_PROGRAM_CB_IN and KM_PROGRAM_CB_OUT)"
#endif

void kernel_main() {
    km::Compute compute;
#if defined(KM_PROGRAM_SWITCH)
    uint32_t started = km::kSwitchNone;  // the program whose load ran the start-up
#endif
#if !defined(KM_PROGRAM_SWITCH)
    compute_kernel_hw_startup(KM_PROGRAM_CB_IN, KM_PROGRAM_CB_OUT);
#endif
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) == KM_HANDLER_LOAD) {
            // a load: parked until the reader set up, then this core's own view; the
            // program's part runs the hardware start-up on its formats when called
            const uint32_t control = compute.control();
            const uint32_t follower = 1 + km::kOverlayPartHere - km::kOverlayPartUnpack;
            if (!km::program_follow(
                control, follower, command.sequence, compute.wait_limit()
            )) {
                return;
            }
            const uint32_t mask = km::program_line(control)[km::kSwitchMask];
            UNPACK((setup_local_cb_read_write_interfaces<true, false, false>(
                km::program_buffers(control), 0, mask
            )));
            PACK((setup_local_cb_read_write_interfaces<false, true, true>(
                km::program_buffers(control), 0, mask
            )));
        } else if (km::handler(command) - 1u >= KM_HANDLER_USER_MAX) {
            return;  // NOP: no program's
        }
#if defined(KM_PROGRAM_SWITCH)
        else if (started != km::program(command)) {
            // the hardware was not started for this program: no unpack, math or pack
            UNPACK((km::command_gave_up(
                compute.control(),
                km::kStCompute,
                km::kGaveOverlay,
                km::kWhereProgramStart,
                0,
                command.sequence
            )));
            return;
        }
#endif
        km::Overlay program(compute, km::program_overlay(km::program(command)));
        if (!program.load()) {
            return;  // the give-up path
        }
        program.call(command);
#if defined(KM_PROGRAM_SWITCH)
        if (km::handler(command) == KM_HANDLER_LOAD) {
            started = km::program(command);
        }
#endif
    });
}
