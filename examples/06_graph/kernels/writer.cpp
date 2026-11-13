// Graph writer: each rank reads its state word, changes it and writes it back;
// NORM's leader reads every word and gives their sum as the command's result word.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "graph.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto state_args = TensorAccessorArgs<0>();
    km::Writer wr;
    const uint32_t scratch = get_write_ptr(kCbScratch);
    volatile uint32_t* word = reinterpret_cast<volatile uint32_t*>(scratch);
    // NoC barriers polled with the wait limit; false (no report) if it passed.
    auto reads_done = [&] {
        const bool ok = wr.poll(
            [&] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            wr.args().wait_limit
        ) == km::kGaveNone;
        invalidate_l1_cache();
        return ok;
    };
    auto writes_done = [&] {
        return wr.poll(
            [&] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            wr.args().wait_limit
        ) == km::kGaveNone;
    };
    wr.run([&](const km_command_t& r) {
        const uint32_t index = wr.me() - km::first(r);
        const uint32_t page = wr.me();
        const auto x = TensorAccessor(state_args, r.a0);
        const uint32_t handler = km::handler(r);
        if (handler == kHandlerStall) {
            return index != r.a0;
        }
        if (handler == kHandlerNorm) {
            if (index != 0) {
                return true;
            }
            uint32_t sum = 0;
            for (uint32_t p = 0; p < r.a1; p++) {
                noc_async_read_page(p, x, scratch);
                if (!reads_done()) {
                    return false;
                }
                sum += word[0];
            }
            wr.result(sum);
            return true;
        }
        if (
            handler != kHandlerStep && handler != kHandlerLeft &&
            handler != kHandlerRight
        ) {
            return true;
        }
        noc_async_read_page(page, x, scratch);
        if (!reads_done()) {
            return false;
        }
        const uint32_t v = word[0];
        word[0] = (
            handler == kHandlerStep
                ? v + r.a1 + page
                : handler == kHandlerLeft
                    ? v >> 1
                    : v * 3u
        );
        noc_async_write_page(page, x, scratch);
        return writes_done();
    });
}
