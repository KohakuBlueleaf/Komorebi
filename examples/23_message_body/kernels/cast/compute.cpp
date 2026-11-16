// Cast compute: takes every body its broadcast brings and folds each of its words into
// the core's area (unpack), or only frees it.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/movement/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"
#include "cast.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t included = compute.arg(0);
    const uint32_t window = compute.arg(1);
    volatile tt_l1_ptr uint32_t* area = km::l1(compute.arg(2));
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCast || included == 0) {
            return;
        }
        const uint32_t casts = command.a0;
        const bool fold = command.a3 != 0;
        UNPACK(({
            for (uint32_t word = 0; word < kBodyWords; word++) {
                area[word] = 0;
            }
        }));
        uint32_t received = 0;
        for (; received < casts; received++) {
            if (!km::broadcast_receive(compute, kCbBody, 1)) {
                break;
            }
            UNPACK(({
                if (fold) {
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint32_t* body = (
                        km::l1(ckernel::get_tile_l1_byte_address(kCbBody, 0))
                    );
                    for (uint32_t word = 0; word < kBodyWords; word++) {
                        area[word] = cast_fold(area[word], body[word]);
                    }
                }
            }));
            km::broadcast_done(kCbBody, 1);
        }
        if (received == casts) {
            km::broadcast_end(compute, kCbBody, 1, window, casts);
        }
        UNPACK(({
            area[kResCasts] = received;
            area[kResRun] = command.sequence;
        }));
    });
}
