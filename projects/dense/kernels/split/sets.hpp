// The sum's two sets of ports and arrival slots: a command with kSplitSecondSet in a3
// takes the upper half of both, so it may be in flight beside one that takes the lower
// half (the parts of one never meet the other's). A command without it takes them as
// before, every port and every slot.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_port.hpp"
#include "split.h"

// The set's first port and first arrival slot.
inline uint32_t set_port(const km_command_t& command) {
    return split_second_set(command) ? KM_PORTS / 2 : 0;
}

inline uint32_t set_slot(const km_command_t& command, const km::Arrival& arrival) {
    return split_second_set(command) ? km::arrival_slots(arrival) / 2 : 0;
}

#if !defined(COMPILE_FOR_TRISC)
// Reader: km::reduce_arrive in the command's set. False: gave up (recorded).
inline bool arrive_set(
    const km::Reader& reader,
    const km_command_t& command,
    const km::Arrival& arrival,
    const km::Group& group,
    uint32_t form
) {
    const uint32_t base = set_slot(command, arrival);
    if (km::reduce_slots(group, form) > km::arrival_slots(arrival) - base) {
        reader.gave_up(
            km::kStReader,
            km::kGaveBadItem,
            km::kWherePort,
            km::slot(command),
            command.sequence
        );
        return false;
    }
    for (uint32_t port = 0; port < KM_PORTS; port++) {
        const uint32_t wanted = km::reduce_wants(group, form, port);
        if (
            wanted != 0 &&
            !km::arrive(reader, command, arrival, set_port(command) + port, wanted)
        ) {
            return false;  // a port past the last one gives up in arrive
        }
    }
    return true;
}

// Writer: km::reduce_emit in the command's set. False: gave up (recorded).
inline bool emit_set(
    const km::Writer& writer,
    const km_command_t& command,
    const km::Arrival& arrival,
    const km::Group& group,
    uint32_t form,
    uint32_t cb,
    uint32_t pages
) {
    const km::ReduceStep step = km::reduce_send(group, form);
    return (
        step.root ||
        km::emit(
            writer,
            command,
            arrival,
            cb,
            pages,
            step.to,
            set_port(command) + step.port,
            set_slot(command, arrival) + step.slot
        )
    );
}
#else
// Compute: km::reduce in the command's set. False: stopping.
template <typename Combine>
inline bool reduce_set(
    const km::Compute& compute,
    const km_command_t& command,
    const km::Arrival& arrival,
    const km::Group& group,
    uint32_t form,
    Combine combine
) {
    const uint32_t base = set_slot(command, arrival);
    for (uint32_t port = 0; port < KM_PORTS; port++) {
        const uint32_t wanted = km::reduce_wants(group, form, port);
        if (wanted == 0) {
            continue;
        }
        if (!km::take(compute, arrival)) {
            return false;
        }
        if (form == km::kReduceOwner) {
            for (uint32_t arrival_slot = 0; arrival_slot < wanted; arrival_slot++) {
                combine(base + arrival_slot);
            }
        } else {
            combine(base + port);
        }
        km::untake(arrival);
    }
    return true;
}
#endif
