// The writer kernel: one 64-byte line into an X280 cluster's LIM (tile a0 = x | y <<
// 16, address a1), flushed, read back into L1; answers how many of its 16 words came
// back.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerProbe) {
            return true;
        }
        const uint32_t line = get_write_ptr(kProbeBuffer);
        volatile tt_l1_ptr uint32_t* words = km::l1(line);
        for (uint32_t k = 0; k < 32; k++) {
            words[k] = k < 16 ? command.a2 + k : 0;
        }
        const uint64_t target = (
            get_noc_addr(command.a0 & 0xFFFFu, command.a0 >> 16, command.a1)
        );
        noc_async_write(line, target, 64);
        if (!writer.flushed(km::kFlushWrites, km::kStWriter)) {
            return false;
        }
        noc_async_read(target, line + 64, 64);
        if (!writer.flushed(km::kFlushReads, km::kStWriter)) {
            return false;
        }
        uint32_t same = 0;
        for (uint32_t k = 0; k < 16; k++) {
            same += words[16 + k] == command.a2 + k ? 1 : 0;
        }
        writer.result(same);
        return true;
    });
}
