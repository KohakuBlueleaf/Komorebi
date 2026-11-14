// Writes the wall clock at its start into its report row, then does one thing chosen
// by the host (nothing, spin, a 16-byte write to the host channel, a block write) and
// returns.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

    constexpr uint32_t kChannelLimit = 64u << 20;  // a write stays below this offset
    constexpr uint32_t kBound = 1u << 28;          // cycles of the flush wait

    inline uint64_t wall() {
        const uint32_t low = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)
        );
        const uint32_t high = (
            *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H)
        );
        return static_cast<uint64_t>(high) << 32 | low;
    }

    // Waits for the write acknowledgements for at most kBound cycles; false on a
    // timeout.
    inline bool flushed() {
        const uint64_t start = wall();
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            if (wall() - start > kBound) {
                return false;
            }
        }
        return true;
    }

}  // namespace

// Arguments: 0 report address (a row of 2048 words); 1 what to do (0 nothing, 1 spin,
// 2 one 16-byte write, 3 a block write); 2 cycles to spin; 3 bytes; 4 channel offset.
// Report words: 0 status (1 done, 2 refused, 3 timeout), 1 and 2 start stamp, 64 on the
// source bytes of a write.
void kernel_main() {
    volatile uint32_t* report = (
        reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0))
    );
    const uint32_t what = get_arg_val<uint32_t>(1);
    const uint32_t spin = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_arg_val<uint32_t>(3);
    const uint32_t offset = get_arg_val<uint32_t>(4);
    const uint64_t start = wall();
    report[1] = static_cast<uint32_t>(start);
    report[2] = static_cast<uint32_t>(start >> 32);
    report[0] = 1;
    if (what == 1) {
        while (wall() - start < spin) {
            ;
        }
    } else if (what >= 2) {
        const uint32_t count = what == 2 ? 16 : bytes;
        const uint32_t source = reinterpret_cast<uint32_t>(report + 64);
        if (
            count == 0 || count > 4096 * 2 || (count % 16) != 0 || (offset % 16) != 0 ||
            (source % 16) != 0 || static_cast<uint64_t>(offset) + count > kChannelLimit
        ) {
            report[0] = 2;
            return;
        }
        const uint64_t pcie = (
            static_cast<uint64_t>(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y))
        );
        if (what == 2) {
            noc_async_write_pcie(source, pcie | offset, 16);
        } else {
            noc_async_write_set_pcie_state(pcie | offset);
            noc_async_write_with_state(source, offset, count);
            noc_async_write_clear_pcie_state();
        }
        if (!flushed()) {
            report[0] = 3;
        }
    }
}
