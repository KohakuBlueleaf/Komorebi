// The runtime's service overlay, reader part: PUSH, PULL, COPY, GATHER, REDUCE or ENTRY
// (KM_ENTRY_QUIET: the clear's READ or SETTLE); a failed bound or wait gives up.
#include <stdint.h>

#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "komorebi/device/tensors/km_table.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"
#include "komorebi/device/km_worker.hpp"

// with KM_ADDED_ENTRY alone: an added class's own overlays (komorebi.program.addon)
#if defined(KM_ADDED_HANDLERS) || defined(KM_ADDED_ENTRY)
// ENTRY: the table entry of an overlay added to the live run, written by this core in
// its own overlay area; with KM_ENTRY_HANDLER, its handler's bit too.
static void write_entry(km::Reader& reader, const km_command_t& command) {
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
    const uint32_t control = reader.args().control;
    const uint32_t number = a[KM_ENTRY_NUMBER] & 0xFFu;
    if (number == 0 || number >= km::kOverlayService) {
        reader.gave_up(
            km::kStReader,
            km::kGaveOverlay,
            km::kWhereOverlayFind,
            number,
            command.sequence
        );
        return;
    }
    for (uint32_t part = 0; part < km::kOverlayParts; part++) {
        volatile tt_l1_ptr uint32_t* entry = km::overlay_entry(control, number, part);
        const uint32_t lines = a[KM_ENTRY_BYTES + part / 2] >> 16 * (part & 1u);
        entry[0] = a[KM_ENTRY_ADDRESS + part];
        entry[1] = 64u * (lines & 0xFFFFu);
    }
    volatile tt_l1_ptr uint32_t* area = km::overlay_area(control);
    if (area[km::kOverlayCount] <= number) {
        area[km::kOverlayCount] = number + 1;
    }
    if ((a[KM_ENTRY_NUMBER] & KM_ENTRY_HANDLER) != 0) {
        area[km::kOverlayHandlers / 4 + (number >> 5)] |= 1u << (number & 31u);
    }
    if ((a[KM_ENTRY_NUMBER] & KM_ENTRY_MESSAGE) != 0 && number < 32) {
        area[km::kOverlayMessages / 4] |= 1u << number;
    }
}

#endif

void km_overlay(km::Reader& reader, const km_command_t& command) {
    const uint32_t handler = km::handler(command);
#if defined(KM_COLL_QUIET)  // a load whose movers call a collective
    if (handler == KM_HANDLER_ENTRY && (command.a0 & KM_ENTRY_QUIET) != 0) {
        const uint32_t code = km::quiet_rank(reader, command);  // READ or SETTLE
        if (code != km::kGaveNone) {
            reader.gave_up(
                km::kStReader,
                code,
                km::kWhereClear,
                km::slot(command),
                command.sequence
            );
        }
        return;
    }
#endif
#if defined(KM_TENSOR_TABLE)  // a tensor made after an island load (km_table.hpp)
    if (handler == KM_HANDLER_ENTRY && (command.a0 & KM_ENTRY_TABLE) != 0) {
        return km::write_table_row(reader, command);
    }
#endif
#if defined(KM_ADDED_HANDLERS) || defined(KM_ADDED_ENTRY)
    if (handler == KM_HANDLER_ENTRY) {
        write_entry(reader, command);
        return;
    }
#endif
    const bool push = handler == KM_HANDLER_PUSH;
    const bool copy = handler == KM_HANDLER_COPY;
    if (handler == KM_HANDLER_GATHER || handler == KM_HANDLER_REDUCE) {
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
        const uint32_t root = a[KM_COLL_ROOT] & ~KM_COLL_GIVEN;
        const km::Tree tree = (
            (a[KM_COLL_ROOT] & KM_COLL_GIVEN) != 0
                ? km::tree(a[KM_COLL_TREE], a[KM_COLL_TREE + 1])
                : km::tree()
        );
        if (handler == KM_HANDLER_GATHER && root != 0) {
            km::broadcast_from(
                reader,
                command,
                a[KM_COLL_TENSOR],
                a[KM_COLL_FIRST],
                a[KM_COLL_PAGES],
                root - 1,
                tree
            );
        } else if (handler == KM_HANDLER_GATHER) {
            km::all_gather(
                reader, command, a[KM_COLL_TENSOR], a[KM_COLL_FIRST], a[KM_COLL_PAGES]
            );
        } else if (root != 0) {
            km::reduce_to(
                reader,
                command,
                a[KM_COLL_TENSOR],
                a[KM_COLL_FIRST],
                a[KM_COLL_PAGES],
                a[KM_COLL_SCRATCH],
                static_cast<km::Sum>(a[KM_COLL_SUM]),
                a[KM_COLL_BUFFER],
                a[KM_COLL_ROOM],
                root - 1,
                tree
            );
        } else {
            km::all_reduce(
                reader,
                command,
                a[KM_COLL_TENSOR],
                a[KM_COLL_FIRST],
                a[KM_COLL_PAGES],
                a[KM_COLL_SCRATCH],
                static_cast<km::Sum>(a[KM_COLL_SUM]),
                a[KM_COLL_BUFFER],
                a[KM_COLL_ROOM],
                tree
            );
        }
        return;  // a give-up is recorded where it happened
    }
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
