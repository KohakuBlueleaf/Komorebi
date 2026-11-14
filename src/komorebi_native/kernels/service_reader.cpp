// The runtime's service overlay, its reader part: this core's slice of a PUSH, PULL or
// COPY; a failed bound or a timed-out wait gives up (its where code).
#include <stdint.h>

#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_transfer.hpp"
#include "komorebi/device/km_worker.hpp"

void km_overlay(km::Reader& reader, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
    const bool push = handler == KM_HANDLER_PUSH;
    const bool copy = handler == KM_HANDLER_COPY;
    if (!push && !copy && handler != KM_HANDLER_PULL) {  // not assigned yet
        reader.gave_up(
            km::kStReader,
            km::kGaveBadTransfer,
            km::kWhereFromHost,
            km::slot(command),
            command.sequence
        );
        return;
    }
    uint32_t code = km::kGaveNone;
    const auto wait = [&](bool (*ready)()) __attribute__((noinline)) {
        code = reader.poll(ready, reader.args().wait_limit);
        return code == km::kGaveNone;
    };
    const uint32_t rank = reader.me() - km::first(command);
    const uint32_t count = km::count(command);
    uint32_t status = km::kTransferDone;
    if (copy) {
        const uint32_t a[10] = {
            command.a0,
            command.a1,
            command.a2,
            command.a3,
            command.a4,
            command.a5,
            command.a6,
            command.a7,
            command.a8,
            command.a9
        };
        const km_copy_t c = km::copy_resolved(km_copy_of(a));
        if (km::copy_crosses(c)) {  // a tensor placed over islands: its give-up named
            km::copy_across(reader, command, c, rank, count);
            return;
        }
        status = km::copy_pages(c, rank, count, wait);
    } else {
        const km::Transfer t = km::transfer_of(command);
        status = (
            push
                ? km::push_to_host(t, rank, count, wait)
                : km::pull_from_host(t, rank, count, wait)
        );
    }
    if (status != km::kTransferDone) {
        reader.gave_up(
            km::kStReader,
            code != km::kGaveNone ? code : km::kGaveBadTransfer,
            copy
                ? km::kWhereCopy
                : push
                    ? km::kWhereToHost
                    : km::kWhereFromHost,
            km::slot(command),
            command.sequence
        );
    }
}
