// The test feeder made live: commands from a host ring in this core's L1, one event a
// command into the host channel, the workers' ring slots by unicast or multicast.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"
#include "live.h"

namespace {

    uint32_t arg(uint32_t i) {
        return get_arg_val<uint32_t>(i);
    }

#if defined(LIVE_PARTS)
    // Adds the cycles since `since` to a part's sum and returns now.
    inline uint64_t part(uint64_t& sum, uint64_t since) {
        const uint64_t now = km::wall64();
        sum += now - since;
        return now;
    }
#endif

}  // namespace

void kernel_main() {
    const uint32_t control = arg(kLvControl);
    const uint32_t workers = arg(kLvWorkers), block = arg(kLvBlock);
    const uint32_t host_depth = arg(kLvHostDepth), event_depth = arg(kLvEventDepth);
    uint32_t depth = arg(kLvDepth);
    depth = depth == 0 || depth > KM_RING_SLOTS ? KM_RING_SLOTS : depth;
    const uint64_t deadline = static_cast<uint64_t>(arg(kLvDeadline)) << 10;
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kLvBootLimit)) << 10;
    const uint64_t idle_limit = static_cast<uint64_t>(arg(kLvIdleLimit)) << 10;
    const uint32_t trace_cap = arg(kLvTraceCap);
    const uint32_t results_address = get_write_ptr(0);
    const uint64_t host = km::pcie_noc() | arg(kLvChannel);
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* ack = km::l1(control + KM_CONTROL_DONE);
    volatile tt_l1_ptr uint32_t* ring = km::l1(block + kLbRing);
    volatile tt_l1_ptr uint32_t* read_line = km::l1(block + kLbRead);
    volatile tt_l1_ptr uint32_t* counts = km::l1(block + kLbCounts);
    volatile tt_l1_ptr uint32_t* events = km::l1(block + kLbEvents);
    volatile tt_l1_ptr uint32_t* pending = km::l1(block + kLbPending);
    volatile tt_l1_ptr uint32_t* hand = km::l1(block + kLbHand);
    volatile tt_l1_ptr uint32_t* summary = km::l1(results_address);
    volatile tt_l1_ptr uint32_t* trace = summary + kLsRows * 16;
    auto core = [&](uint32_t worker, uint32_t address) {
        return get_noc_addr(cores[worker] & 0xFFFFu, cores[worker] >> 16, address);
    };

    // The multicast rectangles: NoC 0 order in the arguments, swapped on NoC 1.
    const uint32_t rects = arg(kLvRects);
    const bool helper = arg(kLvHelper) != 0;
    uint64_t rect_target[kLvRectsMax];
    uint32_t rect_dests[kLvRectsMax];
    for (uint32_t r = 0; r < rects && r < kLvRectsMax; r++) {
        const uint32_t a = arg(kLvRect + 3 * r), b = arg(kLvRect + 3 * r + 1);
        const uint32_t start = noc_index == 0 ? a : b, end = noc_index == 0 ? b : a;
        rect_target[r] = (
            get_noc_multicast_addr(
                start & 0xFFFFu, start >> 16, end & 0xFFFFu, end >> 16, 0
            )
        );
        rect_dests[r] = arg(kLvRect + 3 * r + 2);
    }

    const InterleavedAddrGen<true> core_table = {
        .bank_base_address = arg(kLvCores), .page_size = KM_CORES_MAX * 4
    };
    noc_async_read(
        get_noc_addr(0, core_table), control + KM_CONTROL_CORES, KM_CORES_MAX * 4
    );
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t word = 0; word < kLsRows * 16 + trace_cap * kLtWords; word++) {
        summary[word] = 0;
    }
    summary[kLsMagic] = kLiveMagic;
    uint64_t parts_take = 0, parts_slot = 0, parts_write = 0, parts_sweep = 0;
    uint64_t parts_event = 0, parts_idle = 0, helper_wait = 0;
    uint32_t passes = 0, empty = 0, held_passes = 0, counts_written = 0, rejected = 0;
    uint32_t room_passes = 0;

    // Ends the run: `status` and 1 in the stop flag of every worker.
    auto stop_all = [&](uint32_t status) {
        summary[kLsStatus] = status;
        *km::out_slot(control, 0) = 1;
        for (uint32_t worker = 0; worker < workers; worker++) {
            km::small_write<0, 4>(control, core(worker, control + KM_CONTROL_STOP), 0);
        }
        noc_async_write_barrier();
    };

    // Start-up: each group's READY and the hello {1, control}, as the feeder.
    const uint32_t groups = arg(kLvGroups);
    volatile tt_l1_ptr uint32_t* ready_words = km::l1(control + KM_CONTROL_BOOT);
    volatile tt_l1_ptr uint32_t* mailbox = km::l1(control + KM_CONTROL_JOB);
    const uint64_t boot_started = km::wall64();
    uint32_t ready = 0;
    bool booted = false;
    while (km::wall64() - boot_started < boot_limit) {
        invalidate_l1_cache();
        ready = 0;
        for (uint32_t group_index = 0; group_index < groups; group_index++) {
            ready |= (ready_words[group_index * 4] == KM_READY) << group_index;
        }
        if (ready == (1u << groups) - 1 && mailbox[0] == 1 && mailbox[1] == control) {
            booted = true;
            break;
        }
    }
    summary[kLsBoot] = static_cast<uint32_t>((km::wall64() - boot_started) >> 4);
    summary[kLsReady] = ready;

    uint32_t busy_request[KM_RING_SLOTS];  // 0: free
    uint32_t busy_sequence[KM_RING_SLOTS];
    uint32_t busy_group[KM_RING_SLOTS];
    bool busy_result[KM_RING_SLOTS];
    uint64_t submitted_at[KM_RING_SLOTS];
    for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
        busy_request[slot] = 0;
    }
    uint32_t sequence = 0, next = 1, done = 0, in_flight = 0, max_busy = 0;
    uint32_t pushed = 0, told = 0, consumed = 0, published = 0, read_count = 0;
    uint32_t pending_request = 0;  // 0: no command taken and waiting
    uint32_t exit_request = 0;     // the EXIT's request once submitted
    bool exit_acked = false;
    uint64_t first_take = 0, last_event = 0;

    // Writes event ++pushed {index, request, status, ticks} to the host's ring.
    auto push = [&](uint32_t request, uint32_t status, uint32_t ticks) {
#if defined(LIVE_PARTS)
        const uint64_t started = km::wall64();
#endif
        const uint32_t index = ++pushed;
        const uint32_t at = (index & (event_depth - 1u)) * 16;
        volatile tt_l1_ptr uint32_t* event = events + at / 4;
        event[0] = index;
        event[1] = request;
        event[2] = status;
        event[3] = ticks;
        noc_async_write_pcie(block + kLbEvents + at, host + kLhEvents + at, 16);
#if defined(LIVE_PARTS)
        part(parts_event, started);
#endif
    };
    // Once a pass: the counts line {consumed, told} when either moved.
    auto publish = [&]() {
        if (consumed != published || pushed != told) {
            noc_async_writes_flushed();  // the line's last write has left
            counts[kLhConsumed] = consumed;
            counts[kLhTold] = pushed;
            noc_async_write_pcie(block + kLbCounts, host, 16);
#if defined(LIVE_BARRIER)
            noc_async_write_barrier();
#endif
            published = consumed;
            told = pushed;
            counts_written++;
        }
    };
    auto trace_row = [&](uint32_t request)->volatile tt_l1_ptr uint32_t* {
        return request <= trace_cap ? trace + (request - 1) * kLtWords : nullptr;
    };

    // Writes the slot's line to every worker: unicast, or the rectangles (the helper
    // issues rectangle 1 on the other mover's NoC while this one issues the rest).
    auto submit = [&](uint32_t slot) {
        const uint32_t line = control + KM_CONTROL_RING + slot * 64;
        if (rects == 0) {
            for (uint32_t worker = 0; worker < workers; worker++) {
                noc_async_write(line, core(worker, line), 64);
            }
            noc_async_writes_flushed();
            return;
        }
        const uint32_t token = sequence << 8 | slot;
        if (helper) {
            hand[kHandLine] = token;
        }
        for (uint32_t r = 0; r < rects; r++) {
            if (helper && r == 1) {
                continue;
            }
#if defined(LIVE_LOOPBACK)
            noc_async_write_multicast_loopback_src(
                line, rect_target[r] | line, 64, rect_dests[r]
            );
#else
            noc_async_write_multicast(line, rect_target[r] | line, 64, rect_dests[r]);
#endif
        }
        noc_async_writes_flushed();
        if (helper) {
            const uint64_t waited = km::wall64();
            invalidate_l1_cache();
            while (hand[kHandDone] != token) {
                live_pace();
                invalidate_l1_cache();
            }
            helper_wait += km::wall64() - waited;
        }
    };

    if (!booted) {
        stop_all(kLeBoot);
    } else {
        summary[kLsStatus] = kLeDone;
        bool failed = false;
        uint64_t last_activity = km::wall64();
        while (!failed && !(exit_acked && in_flight == 0)) {
            invalidate_l1_cache();
            uint64_t now = km::wall64();
            passes++;
#if defined(LIVE_PARTS)
            uint64_t mark = now;
            const uint64_t events_before = parts_event;
#endif
            // The sweep: every busy slot's ack word, then its deadline.
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (busy_request[slot] == 0) {
                    continue;
                }
                if (ack[slot * 4] == KM_ACK(busy_sequence[slot])) {
                    const uint32_t request = busy_request[slot];
                    const uint32_t ticks = (
                        busy_result[slot]
                            ? ack[slot * 4 + 1]
                            : static_cast<uint32_t>(now - submitted_at[slot])
                    );
                    const uint32_t first_core = busy_group[slot] & 0xFFu;
                    push(
                        request,
                        KM_EVENT_DONE | first_core << 8 |
                            (busy_sequence[slot] & KM_EVENT_SEQUENCE_MASK) << 16,
                        ticks
                    );
                    if (volatile tt_l1_ptr uint32_t* row = trace_row(request)) {
                        row[1] |= kLeDone << 8;
                        row[3] = static_cast<uint32_t>(now);
                    }
                    last_event = now;
                    exit_acked |= request == exit_request;
                    busy_request[slot] = 0;
                    in_flight--;
                    done++;
                    last_activity = now;
                } else if (now - submitted_at[slot] > deadline) {
                    summary[kLsSlot] = slot;
                    summary[kLsSequence] = busy_sequence[slot];
                    summary[kLsRequest] = busy_request[slot];
                    push(
                        busy_request[slot],
                        KM_EVENT_TIMEOUT | slot << 8 |
                            (busy_sequence[slot] & KM_EVENT_SEQUENCE_MASK) << 16,
                        0
                    );
                    if (volatile tt_l1_ptr uint32_t* row = trace_row(
                        busy_request[slot]
                    )) {
                        row[1] |= kLeTimeout << 8;
                    }
                    stop_all(kLeTimeout);
                    failed = true;
                    break;
                }
            }
#if defined(LIVE_PARTS)
            mark = part(parts_sweep, mark);
            parts_sweep -= parts_event - events_before;  // its events are their own
#endif
            if (failed) {
                break;
            }
            // Take: the next line once its four landing words show its number and its
            // event fits the ring with every busy slot's, the pending's and one more.
            if (pending_request == 0 && exit_request == 0) {
                bool room = pushed - read_count + in_flight + 1 <= event_depth;
                if (!room) {
                    read_count = read_line[0];
                    room = pushed - read_count + in_flight + 1 <= event_depth;
                }
                volatile tt_l1_ptr uint32_t*
                    line = (ring + (next & (host_depth - 1u)) * 16);
                if (!room) {
                    room_passes++;
                } else if (
                    line[0] == next && line[4] == next && line[8] == next &&
                    line[12] == next
                ) {
                    for (uint32_t word = 0; word < 16; word++) {
                        pending[word] = line[word];
                    }
                    pending_request = next;
                    consumed = next;
                    next++;
                    first_take = first_take == 0 ? km::wall64() : first_take;
                    last_activity = now;
#if defined(LIVE_PARTS)
                    mark = part(parts_take, mark);
#endif
                } else {
                    empty++;
#if defined(LIVE_PARTS)
                    mark = part(parts_idle, mark);
#endif
                }
            }
            // Submit the pending command: its checks, then a free slot.
            if (pending_request != 0) {
                const uint32_t request = pending_request;
                const uint32_t handler = pending[1] & 0xFFu;
                const uint32_t flags = (pending[1] >> 8) & 0xFFu;
                const bool is_exit = handler == KM_HANDLER_EXIT;
                uint32_t group = is_exit ? KM_GROUP_ALL(workers) : pending[2];
                const uint32_t first_core = group & 0xFFFFu, count = group >> 16;
                bool go = true;
                uint32_t reason = 0;
                if (flags & 1u) {
                    reason = 2;
                } else if (
                    count == 0 || first_core > workers || count > workers - first_core
                ) {
                    reason = 3;
                }
                if (reason != 0) {
                    push(request, KM_EVENT_REJECTED | reason << 8, 0);
                    pending_request = 0;
                    rejected++;
                    go = false;
                } else if (is_exit && in_flight != 0) {
                    go = false;
                } else if (flags & KM_FLAG_EXCLUSIVE) {
                    for (uint32_t slot = 0; slot < depth && go; slot++) {
                        go = (
                            !(
                                busy_request[slot] != 0 &&
                                km::groups_overlap(busy_group[slot], group)
                            )
                        );
                    }
                }
                uint32_t slot = 0;
                while (go && slot < depth && busy_request[slot] != 0) {
                    slot++;
                }
                go = go && slot < depth;
                if (volatile tt_l1_ptr uint32_t* row = trace_row(request)) {
                    row[1] |= go ? 0u : 1u << 16;
                }
#if defined(LIVE_PARTS)
                mark = part(parts_slot, mark);
#endif
                if (go) {
                    volatile tt_l1_ptr km_command_t*
                        slot_command = (
                            reinterpret_cast<volatile tt_l1_ptr km_command_t*>(
                                control + KM_CONTROL_RING + slot * 64
                            )
                        );
                    km::copy_command(
                        reinterpret_cast<volatile tt_l1_ptr km_command_t*>(pending),
                        slot_command
                    );
                    sequence++;
                    slot_command->sequence = (
                        slot_command->land1 = slot_command
                                                  ->land2 = slot_command
                                                                ->land3 = sequence
                    );
                    slot_command->group = group;
                    slot_command->hdr = (slot_command->hdr & 0xFF00FFFFu) | slot << 16;
                    const uint64_t started = km::wall64();
                    submit(slot);
                    busy_request[slot] = request;
                    busy_sequence[slot] = sequence;
                    busy_group[slot] = group;
                    busy_result[slot] = flags & KM_FLAG_RESULT;
                    submitted_at[slot] = started;
                    in_flight++;
                    max_busy = in_flight > max_busy ? in_flight : max_busy;
                    exit_request = is_exit ? request : 0;
                    pending_request = 0;
                    if (volatile tt_l1_ptr uint32_t* row = trace_row(request)) {
                        row[0] = sequence;
                        row[1] |= slot;
                        row[2] = static_cast<uint32_t>(started);
                    }
#if defined(LIVE_PARTS)
                    mark = part(parts_write, mark);
#endif
                } else {
                    held_passes++;
                }
            }
            publish();
#if defined(LIVE_PARTS)
            part(parts_event, mark);
#endif
            if (last_activity != now) {  // nothing taken nor acked this pass
                live_pace();
            }
            if (
                in_flight == 0 && pending_request == 0 &&
                now - last_activity > idle_limit
            ) {
                stop_all(kLeIdle);
                failed = true;
            }
        }
        exit_acked = exit_acked && !failed;
        publish();
    }
    if (helper) {
        hand[kHandLine] = kHandStop;
    }
    noc_async_write_barrier();

    summary[kLsTaken] = consumed;
    summary[kLsDone] = done;
    summary[kLsFirstLo] = static_cast<uint32_t>(first_take);
    summary[kLsFirstHi] = static_cast<uint32_t>(first_take >> 32);
    summary[kLsLastLo] = static_cast<uint32_t>(last_event);
    summary[kLsLastHi] = static_cast<uint32_t>(last_event >> 32);
    summary[kLsMaxBusy] = max_busy;
    summary[kLsExitAcked] = exit_acked;
    summary[kLsPushed] = pushed;
    summary[kLpTake] = static_cast<uint32_t>(parts_take >> 4);
    summary[kLpSlot] = static_cast<uint32_t>(parts_slot >> 4);
    summary[kLpWrite] = static_cast<uint32_t>(parts_write >> 4);
    summary[kLpSweep] = static_cast<uint32_t>(parts_sweep >> 4);
    summary[kLpEvent] = static_cast<uint32_t>(parts_event >> 4);
    summary[kLpPasses] = passes;
    summary[kLpEmpty] = empty;
    summary[kLpHeld] = held_passes;
    summary[kLpCounts] = counts_written;
    summary[kLpRejected] = rejected;
    summary[kLpRoom] = room_passes;
    summary[kLpHelperWait] = static_cast<uint32_t>(helper_wait >> 4);
    summary[kLpIdle] = static_cast<uint32_t>(parts_idle >> 4);
    const InterleavedAddrGen<true> results_table = {
        .bank_base_address = arg(kLvResults), .page_size = 64
    };
    const uint32_t pages = (kLsRows * 16 + trace_cap * kLtWords + 15) / 16;
    for (uint32_t page = 0; page < pages; page++) {
        noc_async_write(
            results_address + page * 64, get_noc_addr(page, results_table), 64
        );
    }
    noc_async_write_barrier();
}
