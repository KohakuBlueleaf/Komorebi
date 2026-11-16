// Chain example, writer: a step's tiles to its output; SUM's ranks other than the
// root send their part up the tree (a credit first), and a root marks the next record.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/movement/km_chain.hpp"
#include "komorebi/device/movement/km_port.hpp"
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

    // A step: its tiles out, and in a chain's record the marks after them.
    bool step(km::Writer& writer, const km_command_t& command) {
        const km::ChainDirect none = {0, 0, 0, kTileBytes};  // no direct input here
        if (!step_handler(km::handler(command))) {
            return true;
        }
        const km::Group group = step_group(command, writer.me());
        if (km::handler(command) == kHandlerTotal) {  // the sum is in once its token is
            if (!writer.wait_front(kCbOut, 1)) {
                return false;
            }
            cb_pop_front(kCbOut, 1);
            return group.rank != 0 ||
                km::chain_mark(km::ChainShape{}, writer, command, none, false);
        }
        const bool sum = km::handler(command) == kHandlerSum;
        if (sum && group.rank != 0) {
            const km::Arrival arrival = {kCbArrive, kArrivePages, 1};
            return (
                km::chain_credit(writer) &&
                km::reduce_emit(
                    writer, command, arrival, group, km::kReduceTree, kCbOut, 1
                )
            );
        }
        const uint32_t width = sum ? 1 : command.a6;
        const uint32_t first = (writer.me() - km::first(command)) * width;
        for (uint32_t tile = first; tile < first + width; tile++) {
            if (!write_tile(writer, command, sum ? 0 : tile)) {
                return false;
            }
        }
        return km::chain_mark(km::ChainShape{}, writer, command, none, false);
    }

    // A command: a chain command's records in turn, or one step.
    bool command_of(km::Writer& writer, const km_command_t& command) {
        const auto product = [&](const km_command_t& record) {
            return step(writer, record);
        };
        if (step_handler(km::handler(command)) && command.a4 == kChainMode) {
            return km::chain_writer(km::ChainShape{}, writer, command, product);
        }
#if defined(KM_CHAIN_STAMPS)
        if (
            km::handler(command) == kHandlerStamps
        ) {  // to page me of a1 (a bank address)
            return km::chain_stamps_out(writer, command.a1, kChainStampBytes);
        }
#endif
        return product(command);
    }

}  // namespace

#if !defined(KM_OVERLAY_PART)  // the runtime's form calls command_of (runtime/)
void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        return command_of(writer, command);
    });
}
#endif
