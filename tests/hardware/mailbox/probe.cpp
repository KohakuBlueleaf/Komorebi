// Mailbox probe: unpack writes word_count words (to math, or to math and pack in turn)
// before either reads; the readers wait `delay` cycles first. Results go to L1 at out.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/core/km_core.hpp"

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint32_t word_count = get_arg_val<uint32_t>(1);
    const uint32_t both = get_arg_val<uint32_t>(2), delay = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* output = km::l1(out);
    UNPACK(({
        const uint64_t started = km::wall64();
        for (uint32_t i = 0; i < word_count; i++) {
            const bool to_pack = both && (i & 1);
            ckernel::mailbox_write(
                to_pack
                    ? ckernel::ThreadId::PackThreadId
                    : ckernel::ThreadId::MathThreadId,
                100 + i
            );
        }
        output[0] = static_cast<uint32_t>(
            km::wall64() - started
        );  // cycles the writes took
    }));
    MATH(({
        const uint64_t started = km::wall64();
        while (km::wall64() - started < delay) {
            ;
        }
        for (uint32_t i = 0; i < word_count; i++) {
            if (!(both && (i & 1))) {
                output[2 + i] = (
                    ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
                );
            }
        }
        output[1] = 1;  // math done
    }));
    PACK(({
        const uint64_t started = km::wall64();
        while (km::wall64() - started < delay) {
            ;
        }
        for (uint32_t i = 0; i < word_count; i++) {
            if (both && (i & 1)) {
                output[2 + i] = (
                    ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
                );
            }
        }
        output[15] = 1;  // pack done
    }));
}
