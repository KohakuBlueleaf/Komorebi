// The tree's writer: the sync, the block down to each child, compute's sum up to the
// parent, and the home's ticks from after the sync to its end (README.md).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_program.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "common.h"

constexpr uint32_t kKids = TREE_KIDS;

namespace {

    volatile tt_l1_ptr uint32_t* words_at(uint32_t address) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    }

    uint32_t base_of(uint32_t cb) {
        const LocalCBInterface& buffer = get_local_cb_interface(cb);
        return buffer.fifo_limit - buffer.fifo_size;
    }

}  // namespace

bool km_program(km::Writer& writer, const km_command_t& command) {
    const km::Islands view = km::islands_at(writer.args().control);
    const uint32_t here = view.here();
    const uint32_t local = writer.me() - view.first();
    const uint32_t n = TREE_PAGES;
    if (km::handler(command) == kHandlerStamp) {
        invalidate_l1_cache();
        writer.result(words_at(base_of(kCbStamp))[command.a0 == 0 ? 0 : 4]);
        return true;
    }
    if (km::handler(command) != kHandlerTree) {
        return true;
    }
    const tree::Word w = tree::word_of(command.a0);
    const uint32_t up_parent = tree::parent(command, here);
    if (local >= w.senders || up_parent == kNone) {
        return true;
    }
    uint32_t kids[kKids];
    const uint32_t count = tree::children(command, here, kids, kKids);
    const km::Arrival tiny_up = {kCbUpTiny, kKids, 1};
    auto tiny_to_parent = [&](uint32_t port) {
        cb_reserve_back(kCbTiny, 1);
        cb_push_back(kCbTiny, 1);
        return (
            km::to_island(up_parent).put(
                writer,
                command,
                tiny_up,
                kCbTiny,
                1,
                view.entry_first(up_parent) + local,
                port,
                tree::place(command, here)
            )
        );
    };
    if ((w.phase & kSync) != 0) {  // a page up the tree: every island is here
        if (count != 0) {
            if (!writer.wait_front(kCbUpTiny, kKids)) {
                return false;
            }
            cb_pop_front(kCbUpTiny, kKids);
        }
        if (here != w.home && !tiny_to_parent(kPortSync)) {
            return false;
        }
    }
    const uint32_t t0 = km::wall();
    if ((w.phase & (kDown | kGo)) != 0) {
        const bool tiny = (w.phase & kGo) != 0;
        const km::Arrival down = {kCbDown, n, tiny ? 1u : n};
        if (!tiny || here != w.home) {  // the block (or page) to send on
            const uint32_t from = here == w.home ? kCbX : kCbDown;
            if (!writer.wait_front(from, n)) {
                return false;
            }
            if (!tiny) {  // copied into the buffer it sends from
                noc_async_read(
                    get_noc_addr(my_x[noc_index], my_y[noc_index], base_of(from)),
                    base_of(kCbOut),
                    n * kPageBytes
                );
                noc_async_read_barrier();
            }
            cb_pop_front(from, n);
        }
        for (uint32_t j = 0; j < count; j++) {
            const uint32_t kid = kids[(j + local) % count];
            const uint32_t cb = tiny ? kCbTiny : kCbOut, pages = tiny ? 1u : n;
            cb_reserve_back(cb, pages);  // the buffer is `pages` long: the same place
            cb_push_back(cb, pages);
            if (!km::to_island(kid).put(
                writer,
                command,
                down,
                cb,
                pages,
                view.entry_first(kid) + local,
                kPortDown,
                0
            )) {
                return false;
            }
        }
    }
    if ((w.phase & kUp) != 0) {
        const km::Arrival up = {kCbUp, kKids * n, n};
        if (here == w.home) {
            // the home's reader writes the total
        } else if (!writer.wait_front(kCbSum, n)) {  // compute: own + the children's
            return false;
        } else if (!km::to_island(up_parent).put(
            writer,
            command,
            up,
            kCbSum,
            n,
            view.entry_first(up_parent) + local,
            kPortUp,
            tree::place(command, here)
        )) {
            return false;
        }
    } else if ((w.phase & kAck) != 0) {
        if (count != 0) {
            if (!writer.wait_front(kCbUpTiny, kKids)) {
                return false;
            }
            cb_pop_front(kCbUpTiny, kKids);
        }
        if (here != w.home && !tiny_to_parent(kPortUp)) {
            return false;
        }
    }
    words_at(base_of(kCbStamp))[0] = km::wall() - t0;
    return true;
}
