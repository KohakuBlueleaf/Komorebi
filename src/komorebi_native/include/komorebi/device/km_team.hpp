// Teams: the completion a team reports (writer side). Membership, shares and who
// reports to whom are in km_record.hpp.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Writer, outputs landed: waits (bounded) for `want` reports in DONE[slot], clears
    // it, then increments the parent's DONE[slot] (NoC1) or, as first member, acks.
    inline bool complete(const Core& c, const km_record_t& r) {
        const Args& a = c.args();
        const uint32_t j = slot(r);
        const uint32_t counter = KM_CTL_DONE + j * 16;
        const Report to = report_to(r, a.me, a.row);
        if (to.want != 0) {
            volatile tt_l1_ptr uint32_t* done = c.ctl(counter);
            const uint32_t code = c.poll(
                [&] {
                    return *done == to.want;
                },
                a.wait_limit
            );
            if (code != kGaveNone) {
                c.gave_up(kStWriter, code, kWhereComplete, j, r.seq);
                return false;
            }
            *done = 0;
        }
        if (a.me != first(r)) {
            noc_semaphore_inc(c.core_noc(to.parent, a.ctl + counter), 1);
            noc_async_atomic_barrier();  // a write barrier does not cover atomics
        } else {
            *c.ctl(kOutAck) = KM_ACK(r.seq);
            noc_async_write(
                a.ctl + kOutAck, c.disp_noc(a.ack_base + j * KM_ACK_STRIDE), 4
            );
            noc_async_writes_flushed();
        }
        c.state()[kStReported + j] = r.seq;
        return true;
    }

}  // namespace km
