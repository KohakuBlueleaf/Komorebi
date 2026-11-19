// Groups: the completion a group reports (writer side), an island's part through its
// first core. Group tests, partitions and who reports to whom are in km_command.hpp.
#pragma once

#include <stdint.h>

#include "km_core.hpp"

namespace km {

    // Writer, outputs landed: timed wait for `want` reports in DONE[slot], clears it,
    // then the parent's DONE[slot] or, first core, the ack; none if given up.
    inline bool complete(
        const Core& core, const km_command_t& command, uint32_t result = 0
    ) {
        const Args& arguments = core.args();
        const Islands view = islands_at(arguments.control);
        if (!in_group(
            command, name_in(view, arguments.me, command)
        )) {  // the reader never hands one to a core outside the group
            return false;
        }
        const uint32_t slot_index = slot(command);
        const uint32_t counter = KM_CONTROL_DONE + slot_index * 16;
        // the island's part completes through the first of its cores in the group
        const Part part = local_part(view, command);
        const Report report = (
            report_in(part.first, part.count, arguments.me, arguments.row)
        );
        if (report.want != 0) {
            volatile tt_l1_ptr uint32_t* done = core.control(counter);
            const uint32_t code = core.poll_at(
                kWhereComplete,
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
        if (core.given_up(command)) {
            return false;
        }
        if (arguments.me != part.first) {
            noc_semaphore_inc(
                core.core_noc(report.parent, arguments.control + counter), 1
            );
            if (!core.flushed(kFlushAtomics, kStWriter)) {  // writes do not cover these
                return false;
            }
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
#ifdef KM_TD_ACK_COUNT
            // a bench build: an add to the count the Tensix dispatcher reads
            noc_semaphore_inc(
                core.disp_noc(arguments.ack_base + KM_TD_ACK_COUNT_AT), 1
            );
            if (!core.flushed(kFlushAtomics, kStWriter)) {
                return false;
            }
#endif
            if (!core.flushed(kFlushSent, kStWriter)) {
                return false;
            }
        }
        core.state()[kStReported + slot_index] = command.sequence;
        return true;
    }

}  // namespace km
