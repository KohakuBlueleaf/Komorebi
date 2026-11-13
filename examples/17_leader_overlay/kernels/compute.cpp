// Compute: loads and calls the overlay of each word its reader hands it, until a 0.
// With LEADER_COMPILED scale and shift are compiled in and the word picks one.
#include "api/compute/compute_kernel_hw_startup.h"
#if defined(LEADER_COMPILED)
#include "tile_ops.h"
#else
#include "api/compute/cb_api.h"
#include "komorebi/device/km_overlay.hpp"
#include "leader_overlay.h"
#endif

void kernel_main() {
    km::Compute compute;
    const uint32_t included = compute.arg(0);
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLead || included == 0) {
            return;
        }
        // --8<-- [start:compute]
        for (;;) {
            if (!compute.wait_front(kCbChoice, 1)) {
                return;
            }
            const uint32_t word = read_tile_value(kCbChoice, 0, 0);  // on all three
            cb_pop_front(kCbChoice, 1);
            if (word == 0) {
                return;
            }
            const uint32_t scalar = scalar_of(index_of(word));
#if defined(LEADER_COMPILED)
            if (compute_tile(compute, scalar, overlay_of(word) == kOverlayScale) == 0) {
                return;
            }
#else
            km::Overlay overlay(compute, overlay_of(word));
            if (!overlay.load()) {  // parks until the reader has put the parts in place
                return;
            }
            if (overlay.call(TileArgument{scalar}) == 0) {
                return;
            }
#endif
        }
        // --8<-- [end:compute]
    });
}
