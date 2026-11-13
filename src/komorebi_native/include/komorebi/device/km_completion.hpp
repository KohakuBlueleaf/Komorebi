// Groups: the completion a group reports (writer side). Group tests, partitions and who
// reports to whom are in km_command.hpp.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Writer, outputs landed: timed wait for `want` reports in DONE[slot], clears it,
    // then increments the parent's DONE[slot] (NoC1) or, as first core, acks.
    inline bool complete(const Core& c, const km_command_t& r, uint32_t result = 0) {
        const Args& a = c.args();
        if (!in_group(
            r, a.me
        )) {  // the reader never hands one to a core outside the group
            return false;
        }
        const uint32_t j = slot(r);
        const uint32_t counter = KM_CONTROL_DONE + j * 16;
        const Report to = report_to(r, a.me, a.row);
        if (to.want != 0) {
            volatile tt_l1_ptr uint32_t* done = c.control(counter);
            const uint32_t code = c.poll(
                [&] {
                    return *done == to.want;
                },
                a.wait_limit
            );
            if (code != kGaveNone) {
                c.gave_up(kStWriter, code, kWhereComplete, j, r.sequence);
                return false;
            }
            *done = 0;
        }
        invalidate_l1_cache();
        if (c.stop()) {  // this tile gave up, or the grid stops: acknowledge nothing
            c.gave_up(kStWriter, kGaveStop, kWhereComplete, j, r.sequence);
            return false;
        }
        if (a.me != first(r)) {
            noc_semaphore_inc(c.core_noc(to.parent, a.control + counter), 1);
            noc_async_atomic_barrier();  // a write barrier does not cover atomics
        } else {
            volatile tt_l1_ptr uint32_t* ack = out_slot(a.control, kOutAck);
            ack[0] = KM_ACK(r.sequence);
            // KM_FLAG_RESULT: {ack, result} as one 8-byte write.
            if (flags(r) & KM_FLAG_RESULT) {
                ack[1] = result;
                small_write<kOutAck, 8>(a.control, c.disp_noc(a.ack_base), j);
            } else {
                small_write<kOutAck, 4>(a.control, c.disp_noc(a.ack_base), j);
            }
            noc_async_writes_flushed();
        }
        c.state()[kStReported + j] = r.sequence;
        return true;
    }

}  // namespace km
