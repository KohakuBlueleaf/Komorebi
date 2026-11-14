// Map/collect writer: MAP stores the core's tile at output page a3 first_page + rank;
// STALL reports for every core but a0 stalled_rank; HANG waits on purpose.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "mapcollect.h"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        const uint32_t index = writer.me() - km::first(command);
        switch (km::handler(command)) {
            case kHandlerMap: {
                const uint32_t output_address = command.a2;
                const uint32_t first_page = command.a3;
                if (!writer.wait_front(kCbOut, 1)) {
                    return false;  // nothing to report: the tile is stopping
                }
                const auto output = TensorAccessor(output_args, output_address);
                noc_async_write_page(first_page + index, output, get_read_ptr(kCbOut));
                noc_async_writes_flushed();
                cb_pop_front(kCbOut, 1);
                return true;
            }
            case kHandlerStall: {
                const uint32_t stalled_rank = command.a0;
                return index != stalled_rank;
            }
            case kHandlerAlign: {
                const uint32_t report_time = command.a9;
                return writer.poll(
                    [&] {
                        return static_cast<int32_t>(km::wall() - report_time) >= 0;
                    },
                    writer.args().wait_limit
                ) == km::kGaveNone;
            }
            case kHandlerLate: {
                const uint32_t late_rank = command.a0;
                const uint32_t delay = command.a1;
                if (index != late_rank) {
                    return true;
                }
                const uint32_t start = km::wall();
                return writer.poll(
                    [&] {
                        return km::wall() - start >= delay;
                    },
                    writer.args().idle_limit
                ) == km::kGaveNone;
            }
            case kHandlerResult: {
                const uint32_t result_base = command.a0;
                writer.result(result_base + index);
                return true;
            }
            case kHandlerBadItem: {
                const uint32_t faulty_rank = command.a0;
                const uint32_t faulty_role = command.a1;
                return (
                    index != faulty_rank || faulty_role != 0 ||
                    writer.wait_front(kCbOut, 2)
                );
            }
            case kHandlerHang: {
                const uint32_t hung_rank = command.a0;
                const uint32_t hang_mode = command.a1;
                if (index != hung_rank || hang_mode != kHangOutput) {
                    return true;
                }
                if (!writer.wait_front(kCbOut, 2)) {
                    return false;
                }
                cb_pop_front(kCbOut, 2);
                return true;
            }
            case kHandlerDot: {
                const uint32_t x0 = command.a0;
                const km::Block block = writer.block(command);
                uint32_t sum = 0;
                for (uint32_t word = 0; word < block.bytes / 4; word++) {
                    sum += block.words[word] * (x0 + word);
                }
                writer.result(sum);
                return true;
            }
            default:
                return true;
        }
    });
}
