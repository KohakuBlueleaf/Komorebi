// The tree's reader: the children's sync pages, the block down into the island's copy,
// its own slice and its children's sums up (README.md, "The tree kernels").
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_program.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "komorebi/device/tensors/km_ask.hpp"
#include "common.h"

constexpr uint32_t kKids = TREE_KIDS;

namespace {

    uint32_t base_of(uint32_t cb) {
        const LocalCBInterface& buffer = get_local_cb_interface(cb);
        return buffer.fifo_limit - buffer.fifo_size;
    }

}  // namespace

void km_program(km::Reader& reader, const km_command_t& command) {
    constexpr auto pages_args = TensorAccessorArgs<0>();
    if (km::handler(command) != kHandlerTree) {
        return;
    }
    const km::Islands view = km::islands_at(reader.args().control);
    const uint32_t here = view.here();
    const uint32_t local = reader.me() - view.first();
    const tree::Word w = tree::word_of(command.a0);
    if (local >= w.senders || tree::parent(command, here) == kNone) {
        return;
    }
    const uint32_t n = TREE_PAGES, first = local * n;
    uint32_t kids[kKids];
    const uint32_t count = tree::children(command, here, kids, kKids);
    const km::Arrival tiny_up = {kCbUpTiny, kKids, 1};
    if ((w.phase & kSync) != 0 && count != 0) {
        if (!km::arrive(reader, command, tiny_up, kPortSync, count)) {
            return;
        }
    }
    const uint32_t t0 = km::wall();
    if ((w.phase & (kDown | kGo)) != 0) {
        const bool tiny = (w.phase & kGo) != 0;
        if (here == w.home) {
            if (!tiny) {  // the home's slice of x, for the writer to send
                if (
                    !reader.reserve(kCbX, n) ||
                    !km::read_pages(
                        reader,
                        command,
                        pages_args,
                        command.a7,
                        first,
                        n,
                        get_write_ptr(kCbX)
                    )
                ) {
                    return;
                }
                noc_async_read_barrier();
                cb_push_back(kCbX, n);
            }
        } else {
            const km::Arrival down = {kCbDown, n, tiny ? 1u : n};
            if (!km::arrive(reader, command, down, kPortDown, 1)) {
                return;
            }
            if (!tiny) {  // into its own copy (the writer sends the same pages on)
                if (!km::write_pages(
                    reader,
                    command,
                    pages_args,
                    command.a7,
                    first,
                    n,
                    base_of(kCbDown)
                )) {
                    return;
                }
                noc_async_write_barrier();
            }
        }
    }
    if ((w.phase & kUp) != 0) {
        if (
            !reader.reserve(kCbOwn, n) ||
            !km::read_pages(
                reader, command, pages_args, command.a7, first, n, get_write_ptr(kCbOwn)
            )
        ) {
            return;
        }
        noc_async_read_barrier();
        cb_push_back(kCbOwn, n);
    }
    if ((w.phase & (kUp | kAck)) != 0 && count != 0) {
        const bool ack = (w.phase & kAck) != 0;
        const km::Arrival up = ack ? tiny_up : km::Arrival{kCbUp, kKids * n, n};
        if (!km::arrive(reader, command, up, kPortUp, count)) {
            return;
        }
    }
    if ((w.phase & kUp) != 0 && here == w.home) {  // the total into the home's DRAM
        const uint32_t code = reader.poll_at(
            km::kWhereData,
            [&] {
                return cb_pages_available_at_front(kCbSum, n);
            },
            reader.args().wait_limit
        );
        if (code != km::kGaveNone) {
            reader.gave_up(km::kStReader, code, km::kWhereData, 0, command.sequence);
            return;
        }
        cb_wait_front(kCbSum, n);
        if (!km::write_pages(
            reader, command, pages_args, command.a8, first, n, get_read_ptr(kCbSum)
        )) {
            return;
        }
        noc_async_write_barrier();
        cb_pop_front(kCbSum, n);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            base_of(kCbStamp)
        )[4] = km::wall() - t0;
    }
}
