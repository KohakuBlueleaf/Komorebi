// Chain example, writer: a step's tiles to its output; SUM's ranks other than the
// root send their part up the tree (a credit first), and a root marks the next record.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_chain.hpp"
#include "komorebi/device/km_port.hpp"
#include "steps.h"

namespace {

    using Tiles = km::Tensor<0>;

    // The tile at kCbOut's front to page `page` of the output, written and acked.
    bool write_tile(km::Writer& writer, const km_command_t& command, uint32_t page) {
        if (
            !writer.wait_front(kCbOut, 1) ||
            !km::write_pages(
                writer, command, Tiles(command.a2), page, 1, get_read_ptr(kCbOut)
            ) ||
            !writer.flushed(km::kFlushWrites, km::kStWriter)
        ) {
            return false;
        }
        cb_pop_front(kCbOut, 1);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Writer writer;
    const km::ChainDirect none = {0, 0, 0, kTileBytes};  // no direct input here
    const auto product = [&](const km_command_t& command) {
        if (km::handler(command) != kHandlerStep) {
            return true;
        }
        const km::Group group = km::group_of(command, writer.me());
        if (command.a0 == kSum && group.rank != 0) {
            const km::Arrival arrival = {kCbArrive, kArrivePages, 1};
            return (
                km::chain_credit(writer) &&
                km::reduce_emit(
                    writer, command, arrival, group, km::kReduceTree, kCbOut, 1
                )
            );
        }
        const uint32_t width = command.a0 == kSum ? 1 : command.a6;
        const uint32_t first = (writer.me() - km::first(command)) * width;
        for (uint32_t tile = first; tile < first + width; tile++) {
            if (!write_tile(writer, command, command.a0 == kSum ? 0 : tile)) {
                return false;
            }
        }
        return km::chain_mark(km::ChainShape{}, writer, command, none, false);
    };
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerStep && command.a4 == kChainMode) {
            return km::chain_writer(km::ChainShape{}, writer, command, product);
        }
        return product(command);
    });
}
