// Leader writer: in CARRY, writes the core's sum of the tiles it received to the page
// of the sums that is its rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "leader.h"

void kernel_main() {
    using Sums = km::Tensor<0>;
    km::Writer writer;
    const uint32_t included = writer.arg(0);
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCarry || included == 0) {
            return true;
        }
        const uint32_t rank = writer.me() - km::first(command);
        const Sums sums(command.a7);
        if (!writer.wait_front(kCbSum, 1)) {
            return false;
        }
        if (!km::write_pages(
            writer,
            command,
            sums,
            command.a4 + rank,
            1,
            get_read_ptr(kCbSum),
            kAskPortWriter
        )) {
            return false;
        }
        const uint32_t code = writer.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        );
        if (code != km::kGaveNone) {
            writer.gave_up(
                km::kStWriter, code, km::kWhereData, km::slot(command), command.sequence
            );
            return false;
        }
        cb_pop_front(kCbSum, 1);
        return true;
    });
}
