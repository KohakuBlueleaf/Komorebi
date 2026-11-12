// Mailbox probe: unpack writes k words (to math, or to math and pack in turn) before
// either reads; the readers wait `delay` cycles first. Results go to L1 at out.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_core.hpp"

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0), k = get_arg_val<uint32_t>(1);
    const uint32_t both = get_arg_val<uint32_t>(2), delay = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* o = km::l1(out);
    UNPACK(({
        const uint64_t t0 = km::wall64();
        for (uint32_t i = 0; i < k; i++) {
            const bool to_pack = both && (i & 1);
            ckernel::mailbox_write(
                to_pack ? ckernel::ThreadId::PackThreadId
                        : ckernel::ThreadId::MathThreadId,
                100 + i
            );
        }
        o[0] = static_cast<uint32_t>(km::wall64() - t0);  // cycles the writes took
    }));
    MATH(({
        const uint64_t t0 = km::wall64();
        while (km::wall64() - t0 < delay) {
            ;
        }
        for (uint32_t i = 0; i < k; i++) {
            if (!(both && (i & 1))) {
                o[2 + i] = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }
        }
        o[1] = 1;  // math done
    }));
    PACK(({
        const uint64_t t0 = km::wall64();
        while (km::wall64() - t0 < delay) {
            ;
        }
        for (uint32_t i = 0; i < k; i++) {
            if (both && (i & 1)) {
                o[2 + i] = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
            }
        }
        o[15] = 1;  // pack done
    }));
}
