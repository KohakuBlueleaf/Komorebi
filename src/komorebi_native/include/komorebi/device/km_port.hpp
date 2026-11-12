// A value crossing to one core: emit (writer), arrive (reader), take (compute), and
// reduce by owner and by tree over them. The program supplies the combine step.
#pragma once

#include <stdint.h>

#include "km_group.hpp"
#include "km_worker.hpp"

namespace km {

    // The program's arrival buffer: circular buffer `cb` of `pages` pages (S). Slot k
    // is page k, at the same L1 address on every core (one program, one layout).
    struct Arrival {
        uint32_t cb;
        uint32_t pages;
    };

    // Bound of a wait that lies behind a port's wait (for a part that needs
    // arrivals), in wait limits: the port's give-up comes first and names the port.
    constexpr uint32_t kBehindPort = 2;

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    inline volatile tt_l1_ptr uint32_t* port_words(const Core& c, uint32_t port) {
        return c.ctl(KM_CTL_PORT + port * KM_PORT_STRIDE);
    }

    // L1 address of slot k: the buffer's base from its configuration, not from its
    // pointers, so any kernel of any core computes the same address.
    inline uint32_t slot_addr(const Arrival& a, uint32_t k) {
        const LocalCBInterface& i = get_local_cb_interface(a.cb);
        return i.fifo_limit - i.fifo_size + k * i.fifo_page_size;
    }

    // Writer: `pages` pages at the front of `cb` into slot k of core `to`, then a bump
    // of its port (each step bounded), then pop. False: gave up, port in the top byte.
    inline bool emit(
        const Writer& wr,
        const km_record_t& r,
        const Arrival& arr,
        uint32_t cb,
        uint32_t pages,
        uint32_t to,
        uint32_t port,
        uint32_t k
    ) {
        const uint32_t named = slot(r) | port << 8;
        const uint32_t bytes = pages * get_local_cb_interface(cb).fifo_page_size;
        const uint32_t room = (
            (arr.pages - k) * get_local_cb_interface(arr.cb).fifo_page_size
        );
        if (port >= KM_PORTS || k >= arr.pages || bytes > room) {
            wr.gave_up(kStWriter, kGaveBadItem, kWhereEmit, named, r.seq);
            return false;
        }
        if (!wr.wait_front(cb, pages, kBehindPort)) {
            return false;
        }
        const uint32_t limit = wr.args().wait_limit;
        noc_async_write(get_read_ptr(cb), wr.core_noc(to, slot_addr(arr, k)), bytes);
        uint32_t code = wr.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            const uint32_t at = wr.args().ctl + KM_CTL_PORT + port * KM_PORT_STRIDE;
            noc_semaphore_inc(wr.core_noc(to, at), 1);
            code = wr.poll(
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                limit
            );
        }
        if (code != kGaveNone) {
            wr.gave_up(kStWriter, code, kWhereEmit, named, r.seq);
            return false;
        }
        cb_pop_front(cb, pages);
        return true;
    }

    // Reader: waits until the port has counted n arrivals, clears it, hands all S
    // pages to compute. False: gave up; the port keeps n, so the missing count shows.
    inline bool arrive(
        const Reader& rd,
        const km_record_t& r,
        const Arrival& arr,
        uint32_t port,
        uint32_t n
    ) {
        const uint32_t named = slot(r) | port << 8;
        if (port >= KM_PORTS) {
            rd.gave_up(kStReader, kGaveBadItem, kWherePort, named, r.seq);
            return false;
        }
        volatile tt_l1_ptr uint32_t* p = port_words(rd, port);
        p[KM_PORT_WANT] = n;
        const uint32_t code = rd.poll(
            [&] {
                return static_cast<int32_t>(p[KM_PORT_COUNT] - n) >= 0;
            },
            rd.args().wait_limit
        );
        if (code != kGaveNone) {
            rd.gave_up(kStReader, code, kWherePort, named, r.seq);
            return false;
        }
        p[KM_PORT_COUNT] = 0;  // every sender of this record has bumped it
        p[KM_PORT_WANT] = 0;
        if (!rd.reserve(arr.cb, arr.pages)) {
            return false;
        }
        cb_push_back(arr.cb, arr.pages);
        return true;
    }

    // Reader side of a reduce: each hand-over this member waits for, in round order.
    inline bool reduce_arrive(
        const Reader& rd,
        const km_record_t& r,
        const Arrival& arr,
        const Group& g,
        uint32_t form
    ) {
        if (form == kReduceOwner && g.size > arr.pages + 1) {  // a slot per sender
            rd.gave_up(kStReader, kGaveBadItem, kWherePort, slot(r), r.seq);
            return false;
        }
        for (uint32_t port = 0; port < KM_PORTS; port++) {
            const uint32_t n = reduce_wants(g, form, port);
            if (n != 0 && !arrive(rd, r, arr, port, n)) {
                return false;
            }
        }
        return true;
    }

    // Writer side of a reduce: a member other than the root emits its `pages` pages
    // of `cb` to its receiver. The root sends nothing.
    inline bool reduce_emit(
        const Writer& wr,
        const km_record_t& r,
        const Arrival& arr,
        const Group& g,
        uint32_t form,
        uint32_t cb,
        uint32_t pages
    ) {
        const ReduceStep s = reduce_send(g, form);
        return s.root || emit(wr, r, arr, cb, pages, s.to, s.port, s.slot);
    }

    // Writer at the root of a reduce: waits for the result's `pages` pages of `cb`
    // (bounded behind the ports). False: gave up (recorded).
    inline bool reduce_result(const Writer& wr, uint32_t cb, uint32_t pages) {
        return wr.wait_front(cb, pages, kBehindPort);
    }

}  // namespace km

#else
namespace km {

    // Compute: waits for the S pages arrive() hands over; pages are then read by slot
    // index (copy_tile(arr.cb, k, ...)). False: the tile is stopping.
    inline bool take(const Compute& cp, const Arrival& arr) {
        return cp.wait_front(arr.cb, arr.pages, kBehindPort);
    }

    // Gives a hand-over back: all S pages, so the slot addresses never drift.
    inline void untake(const Arrival& arr) {
        cb_pop_front(arr.cb, arr.pages);
    }

    // Compute side of a reduce: each hand-over in round order; combine(k) for its
    // slots in rank order (owner: 0 to size - 2; tree: the round's slot).
    template <typename Combine>
    inline bool reduce(
        const Compute& cp,
        const Arrival& arr,
        const Group& g,
        uint32_t form,
        Combine combine
    ) {
        for (uint32_t port = 0; port < KM_PORTS; port++) {
            const uint32_t n = reduce_wants(g, form, port);
            if (n == 0) {
                continue;
            }
            if (!take(cp, arr)) {
                return false;
            }
            if (form == kReduceOwner) {
                for (uint32_t k = 0; k < n; k++) {
                    combine(k);
                }
            } else {
                combine(port);
            }
            untake(arr);
        }
        return true;
    }

}  // namespace km
#endif
