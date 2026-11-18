// Wall-clock rate: one core spins until km::wall64() has advanced `ticks` from its
// first read, then writes the first and last reads (four words at `out`).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"

void kernel_main() {
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint64_t ticks = (
        get_arg_val<uint32_t>(1) | static_cast<uint64_t>(get_arg_val<uint32_t>(2)) << 32
    );
    const uint64_t first = km::wall64();
    uint64_t now = first;
    while (now - first < ticks) {
        now = km::wall64();
    }
    volatile tt_l1_ptr uint32_t*
        words = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out));
    words[0] = static_cast<uint32_t>(first);
    words[1] = static_cast<uint32_t>(first >> 32);
    words[2] = static_cast<uint32_t>(now);
    words[3] = static_cast<uint32_t>(now >> 32);
}
