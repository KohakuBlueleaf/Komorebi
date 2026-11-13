// Groups: the completion a group reports (writer side). Group tests, partitions and who
// reports to whom are in km_command.hpp.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Writer, outputs landed: timed wait for `want` reports in DONE[slot], clears it,
    // then increments the parent's DONE[slot] (NoC1) or, as first core, acks.
    inline bool complete(
        const Core& core, const km_command_t& command, uint32_t result = 0
    ) {
        const Args& arguments = core.args();
        if (!in_group(
            command, arguments.me
        )) {  // the reader never hands one to a core outside the group
            return false;
        }
        const uint32_t slot_index = slot(command);
        const uint32_t counter = KM_CONTROL_DONE + slot_index * 16;
        const Report report = report_to(command, arguments.me, arguments.row);
        if (report.want != 0) {
            volatile tt_l1_ptr uint32_t* done = core.control(counter);
            const uint32_t code = core.poll(
                [&] {
                    return *done == report.want;
                },
                arguments.wait_limit
            );
            if (code != kGaveNone) {
                core.gave_up(
                    kStWriter, code, kWhereComplete, slot_index, command.sequence
                );
                return false;
            }
            *done = 0;
        }
        invalidate_l1_cache();
        if (core.stop()) {  // this tile gave up, or the grid stops: acknowledge nothing
            core.gave_up(
                kStWriter, kGaveStop, kWhereComplete, slot_index, command.sequence
            );
            return false;
        }
        if (arguments.me != first(command)) {
            noc_semaphore_inc(
                core.core_noc(report.parent, arguments.control + counter), 1
            );
            noc_async_atomic_barrier();  // a write barrier does not cover atomics
        } else {
            volatile tt_l1_ptr uint32_t* ack = out_slot(arguments.control, kOutAck);
            ack[0] = KM_ACK(command.sequence);
            // KM_FLAG_RESULT: {ack, result} as one 8-byte write.
            if (flags(command) & KM_FLAG_RESULT) {
                ack[1] = result;
                small_write<kOutAck, 8>(
                    arguments.control, core.disp_noc(arguments.ack_base), slot_index
                );
            } else {
                small_write<kOutAck, 4>(
                    arguments.control, core.disp_noc(arguments.ack_base), slot_index
                );
            }
            noc_async_writes_flushed();
        }
        core.state()[kStReported + slot_index] = command.sequence;
        return true;
    }

}  // namespace km
