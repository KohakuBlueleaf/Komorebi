// Graph writer: each rank reads its state word, changes it and writes it back;
// NORM's leader reads every word and gives their sum as the command's result word.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "graph.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"

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
        const uint32_t index = writer.me() - km::first(command);
        const uint32_t page = writer.me();
        const uint32_t state_address = command.a0;
        const State state(state_address);
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerStall) {
            const uint32_t stalled_rank = command.a0;
            return index != stalled_rank;
        }
        if (handler == kHandlerNorm) {
            const uint32_t pages = command.a1;
            if (index != 0) {
                return true;
            }
            uint32_t total = 0;
            for (uint32_t state_page = 0; state_page < pages; state_page++) {
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
        if (
            handler != kHandlerStep && handler != kHandlerLeft &&
            handler != kHandlerRight
        ) {
            return true;
        }
        const uint32_t round = command.a1;
        if (!km::read_pages(writer, command, state, page, 1, scratch)) {
            return false;
        }
        if (!reads_done()) {
            return false;
        }
        const uint32_t value = word[0];
        word[0] = (
            handler == kHandlerStep
                ? value + round + page
                : handler == kHandlerLeft
                    ? value >> 1
                    : value * 3u
        );
        return (
            km::write_pages(writer, command, state, page, 1, scratch) && writes_done()
        );
    });
}
