// Work-cluster example, grid side: STEP changes each core's state word (page p is
// core p's); SUM's leader reads the loop's words and gives their sum as its result.
#include <stdint.h>

#include "../decide.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"

constexpr uint32_t kCbScratch = 16;

void kernel_main() {
    using State = km::Tensor<0>;
    km::Writer writer;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile uint32_t* word = reinterpret_cast<volatile uint32_t*>(scratch);
    // NoC barriers polled with the wait limit; false (no report) if it passed.
    auto reads_done = [&] {
        const bool ok = writer.poll(
            [&] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
        invalidate_l1_cache();
        return ok;
    };
    auto writes_done = [&] {
        return writer.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    writer.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        const uint32_t state_address = command.a0;
        const State state(state_address);
        if (handler == WC_HANDLER_SUM) {
            const uint32_t cores = command.a1;
            if (writer.me() != km::first(command)) {
                return true;
            }
            uint32_t total = 0;
            for (
                uint32_t state_page = km::first(command);
                state_page < km::first(command) + cores;
                state_page++
            ) {
                if (!km::read_pages(writer, command, state, state_page, 1, scratch)) {
                    return false;
                }
                if (!reads_done()) {
                    return false;
                }
                total += word[0];
            }
            writer.result(total);
            return true;
        }
        if (handler != WC_HANDLER_STEP) {
            return true;
        }
        const uint32_t round = command.a1;
        const uint32_t page = writer.me();
        if (!km::read_pages(writer, command, state, page, 1, scratch)) {
            return false;
        }
        if (!reads_done()) {
            return false;
        }
        word[0] = word[0] + round + page;
        if (!km::write_pages(writer, command, state, page, 1, scratch)) {
            return false;
        }
        return writes_done();
    });
}
