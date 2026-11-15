// The reader owns the port and hosts the slot. RL_HANDLER_SWAP: per message, the body
// read from DRAM or already stored; i-cache invalidated, slot called, result told.
#include <stdint.h>

#include "../body.h"
#include "../remote_x280.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_x280.hpp"

namespace {

    constexpr uint32_t kIcBrisc = 0x1;  // RISCV_IC_INVALIDATE bit of this core

    volatile uint32_t calls = 1;  // the bodies' call counter (from 1: not in .bss)

    __attribute__((noinline)) uint32_t mix(uint32_t x) {
        return rl::mix(x);
    }

    inline uint32_t clamp16(uint32_t v) {
        return v > 0xFFFFu ? 0xFFFFu : v;
    }

}  // namespace

void kernel_main() {
    km::Reader rd;
    km::X280Port port(rd);
    const uint32_t slot = rd.arg(0), slot_bytes = rd.arg(1);
    const uint32_t read_limit = 50000000;  // cycles for one load (about 37 ms)
    rl::Args args{};
    args.mix = mix;
    args.calls = &calls;
    const rl::Entry entry = reinterpret_cast<rl::Entry>(slot);
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) != RL_HANDLER_SWAP) {
            return;
        }
        for (;;) {
            km::X280Msg m;
            if (!port.wait_told(m) || m.w[1] == RL_END) {
                return;
            }
            const uint32_t bytes = m.w[1] & RL_BYTES;
            if (bytes > slot_bytes) {
                port.tell(0, RL_FAIL, RL_ERROR_SIZE);
                continue;
            }
            uint32_t load = 0;
            if (m.w[1] & RL_LOAD) {
                const uint32_t t0 = km::wall();
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(m.w[1] >> 24, m.w[0]), slot, bytes
                );
                bool landed = true;
                while (!ncrisc_noc_reads_flushed(noc_index)) {
                    if (km::wall() - t0 > read_limit) {
                        landed = false;
                        break;
                    }
                }
                load = km::wall() - t0;
                if (!landed) {
                    port.tell(0, RL_FAIL, RL_ERROR_READ);
                    continue;
                }
            }
            volatile tt_reg_ptr uint32_t*
                cfg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE);
            cfg[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = kIcBrisc;
            args.arg = m.w[2];
            const uint32_t t0 = km::wall();
            const uint32_t got = entry(&args);
            const uint32_t run = km::wall() - t0;
            port.tell(got, calls - 1, clamp16(load) << 16 | clamp16(run));
        }
    });
}
