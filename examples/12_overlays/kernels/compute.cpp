// The compute kernel: for scale, loads and calls the overlay's compute parts, which
// multiply the reader's tiles by the scalar; mix has no compute part.
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "overlays.h"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRun || command.a0 != kOverlayScale) {
            return;
        }
        km::Overlay scale(compute, kOverlayScale);
        if (scale.load()) {  // parks until the reader has put the parts in place
            scale.call(ScaleArgument{compute.me(), command.a3, command.a4});
        }
    });
}
