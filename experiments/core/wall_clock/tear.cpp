// Wall-clock tear check: both movers of a core read km::wall64() in a tight loop across
// a carry of the low word; each counts reads that went back or jumped by 2^31 or more.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"

namespace {

    constexpr uint32_t kWords = 16;  // each mover's words in the core's row
    constexpr uint32_t kMagic = 0x7EA20000;

}  // namespace

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint32_t mover = get_arg_val<uint32_t>(1);   // 0: BRISC, 1: NCRISC
    const uint32_t lead = get_arg_val<uint32_t>(2);    // cycles before the carry
    const uint32_t window = get_arg_val<uint32_t>(3);  // cycles of reads
    volatile tt_l1_ptr uint32_t* row = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + mover * kWords * 4)
    );
    // start `lead` cycles before the low word's next carry: the loop crosses it
    while (km::wall() < 0xFFFFFFFFu - lead || km::wall() > 0xFFFFFFFFu - lead / 2) {
        ;
    }
    uint64_t previous = km::wall64();
    const uint64_t first = previous;
    uint32_t reads = 1, back = 0, jumps = 0, largest = 0;
    uint64_t bad_from = 0, bad_to = 0;
    while (previous - first < window) {
        const uint64_t now = km::wall64();
        reads++;
        if (now < previous) {
            back++;
        } else if (now - previous >= (1ull << 31)) {
            jumps++;
        } else if (now - previous > largest) {
            largest = static_cast<uint32_t>(now - previous);
        }
        if (now < previous || now - previous >= (1ull << 31)) {
            if (bad_from == 0) {
                bad_from = previous, bad_to = now;
            }
            continue;  // a torn value is not taken: the next read is compared with
                       // the last good one
        }
        previous = now;
    }
    const uint32_t words[kWords] = {
        kMagic | mover,
        reads,
        back,
        jumps,
        largest,
        static_cast<uint32_t>(first),
        static_cast<uint32_t>(first >> 32),
        static_cast<uint32_t>(previous),
        static_cast<uint32_t>(previous >> 32),
        static_cast<uint32_t>(bad_from),
        static_cast<uint32_t>(bad_from >> 32),
        static_cast<uint32_t>(bad_to),
        static_cast<uint32_t>(bad_to >> 32),
        0,
        0,
        0,
    };
    for (uint32_t i = 0; i < kWords; i++) {
        row[i] = words[i];
    }
}
