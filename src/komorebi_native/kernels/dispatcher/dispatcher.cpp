// The Tensix dispatcher (BRISC): commands from the host ring in this core's L1, the
// workers' ring slots by multicast, one event a command into the host's area.

// No table of each core's program: the host's is the guard. It comes in with the first
// slice giving this a second source of commands (dynamic runs, engines, links).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"

namespace {

    uint32_t arg(uint32_t i) {
        return get_arg_val<uint32_t>(i);
    }

    // Empty cycles between two polls of L1 (the movers share L1 with the TRISCs).
    inline void pace() {
        for (uint32_t i = 0; i < kTdPace; i++) {
            asm volatile("nop");
        }
    }

    // The busy slots and their sweep (each ack word, then each deadline): state of its
    // own, so that a later slice may run the sweep on another processor.
    struct Slots {
        uint32_t request[KM_RING_SLOTS];  // the host's command number; 0: free
        uint32_t sequence[KM_RING_SLOTS];
        uint32_t group[KM_RING_SLOTS];
        bool result[KM_RING_SLOTS];
        uint64_t submitted[KM_RING_SLOTS];
        uint32_t depth = 0, in_flight = 0, most = 0;
        volatile tt_l1_ptr uint32_t* ack = nullptr;

        void init(uint32_t slots, volatile tt_l1_ptr uint32_t* acks) {
            depth = slots;
            ack = acks;
            for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
                request[slot] = 0;
            }
        }

        // The first free slot, or depth when every slot is busy.
        uint32_t free_slot() const {
            uint32_t slot = 0;
            while (slot < depth && request[slot] != 0) {
                slot++;
            }
            return slot;
        }

        // Whether a busy slot's group shares a core with `other`.
        bool overlaps(uint32_t other) const {
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (request[slot] != 0 && km::groups_overlap(group[slot], other)) {
                    return true;
                }
            }
            return false;
        }

        void take(
            uint32_t slot,
            uint32_t req,
            uint32_t seq,
            uint32_t grp,
            bool res,
            uint64_t at
        ) {  // fmt: skip
            request[slot] = req;
            sequence[slot] = seq;
            group[slot] = grp;
            result[slot] = res;
            submitted[slot] = at;
            in_flight++;
            most = in_flight > most ? in_flight : most;
        }

        // Each acknowledged slot: done(request, status, ticks, result flag), then
        // freed. The first busy slot past its deadline: late(slot), and false.
        template <typename Done, typename Late>
        bool sweep(uint64_t now, uint64_t deadline, Done done, Late late) {
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (request[slot] == 0) {
                    continue;
                }
                if (ack[slot * 4] == KM_ACK(sequence[slot])) {
                    const uint32_t ticks = (
                        result[slot]
                            ? ack[slot * 4 + 1]
                            : static_cast<uint32_t>(now - submitted[slot])
                    );
                    const uint32_t status = (
                        KM_EVENT_DONE | (group[slot] & 0xFFu) << 8 |
                        (sequence[slot] & KM_EVENT_SEQUENCE_MASK) << 16
                    );
                    const uint32_t req = request[slot];
                    request[slot] = 0;
                    in_flight--;
                    done(req, status, ticks, result[slot]);
                } else if (now - submitted[slot] > deadline) {
                    late(slot);
                    return false;
                }
            }
            return true;
        }

        // Busy slots 0 to 31 as bits.
        uint32_t busy_low() const {
            uint32_t bits = 0;
            for (uint32_t slot = 0; slot < depth && slot < 32; slot++) {
                bits |= (request[slot] != 0 ? 1u : 0u) << slot;
            }
            return bits;
        }
    };

}  // namespace

void kernel_main() {
    const uint32_t control = arg(kTdControl), ranks = arg(kTdRanks);
    const uint32_t block = arg(kTdBlock);
    const uint32_t host_depth = arg(kTdHostDepth), event_depth = arg(kTdEventDepth);
    const uint64_t area = km::pcie_noc() | arg(kTdArea);
    uint32_t depth = arg(kTdDepth);
    depth = depth == 0 || depth > KM_RING_SLOTS ? KM_RING_SLOTS : depth;
    const uint64_t deadline = static_cast<uint64_t>(arg(kTdDeadline)) << 10;
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kTdBootLimit)) << 10;
    const uint64_t idle_limit = static_cast<uint64_t>(arg(kTdIdleLimit)) << 10;
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* stop_word = km::l1(control + KM_CONTROL_STOP);
    volatile tt_l1_ptr uint32_t* ring = km::l1(block + kTbRing);
    volatile tt_l1_ptr uint32_t* read_line = km::l1(block + kTbRead);
    volatile tt_l1_ptr uint32_t* words = km::l1(block + kTbWords);
    volatile tt_l1_ptr uint32_t* events = km::l1(block + kTbEvents);
    volatile tt_l1_ptr uint32_t* pending = km::l1(block + kTbPending);
    volatile tt_l1_ptr uint32_t* result = km::l1(block + kTbResult);
    auto core = [&](uint32_t worker, uint32_t address) {
        return get_noc_addr(cores[worker] & 0xFFFFu, cores[worker] >> 16, address);
    };

    // The multicast rectangles: NoC 0 order in the arguments, swapped on NoC 1.
    const uint32_t rects = arg(kTdRects) < kTdRectsMax ? arg(kTdRects) : kTdRectsMax;
    uint64_t rect_target[kTdRectsMax];
    uint32_t rect_dests[kTdRectsMax];
    for (uint32_t r = 0; r < rects; r++) {
        const uint32_t a = arg(kTdRect + 3 * r), b = arg(kTdRect + 3 * r + 1);
        const uint32_t start = noc_index == 0 ? a : b, end = noc_index == 0 ? b : a;
        rect_target[r] = (
            get_noc_multicast_addr(
                start & 0xFFFFu, start >> 16, end & 0xFFFFu, end >> 16, 0
            )
        );
        rect_dests[r] = arg(kTdRect + 3 * r + 2);
    }
    const bool loopback = arg(kTdLoopback) != 0;

    const InterleavedAddrGen<true> core_table = {
        .bank_base_address = arg(kTdCores), .page_size = KM_CORES_MAX * 4
    };
    noc_async_read(
        get_noc_addr(0, core_table), control + KM_CONTROL_CORES, KM_CORES_MAX * 4
    );
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t word = 0; word < kTrWords; word++) {
        result[word] = 0;
    }
    for (uint32_t word = 0; word < 16; word++) {
        words[word] = 0;
    }

    uint32_t sequence = 0, next = 1, pushed = 0, consumed = 0, read_count = 0;
    uint32_t told = 0, published = 0, rejected = 0, held = 0, no_room = 0;
    uint32_t submitted = 0, results = 0, last_result = 0;
    uint32_t pending_request = 0;  // 0: no command taken and waiting
    uint32_t exit_request = 0;     // the EXIT's request once submitted
    uint64_t passes = 0, idle_passes = 0, full_passes = 0;
    uint64_t first_submit = 0, last_event = 0;
    Slots slots;
    slots.init(depth, km::l1(control + KM_CONTROL_DONE));

    // The host words {state, consumed, events written, error}: one 16-byte write
    // after the writes before it have left (an event before the count that tells it).
    auto put_words = [&](uint32_t state, uint32_t error) {
        noc_async_writes_flushed();
        words[KMD_HOST_STATE] = state;
        words[KMD_HOST_CONSUMED] = consumed;
        words[KMD_HOST_EVENTS] = pushed;
        words[KMD_HOST_ERROR] = error;
        noc_async_write_pcie(block + kTbWords, area, 16);
        published = consumed;
        told = pushed;
    };
    // Writes event ++pushed {index, request, status, ticks} into the host's ring.
    auto push = [&](uint32_t request, uint32_t status, uint32_t ticks) {
        const uint32_t index = ++pushed;
        const uint32_t at = (index & (event_depth - 1u)) * 16;
        volatile tt_l1_ptr uint32_t* event = events + at / 4;
        event[0] = index;
        event[1] = request;
        event[2] = status;
        event[3] = ticks;
        noc_async_write_pcie(block + kTbEvents + at, area + KMD_HOST_RING + at, 16);
    };
    // 1 in every worker's stop flag.
    auto stop_all = [&]() {
        *km::out_slot(control, 0) = 1;
        for (uint32_t worker = 0; worker < ranks; worker++) {
            km::small_write<0, 4>(control, core(worker, control + KM_CONTROL_STOP), 0);
        }
        noc_async_write_barrier();
    };
    // The result block, then the final host words (the host reads the block once the
    // state is final).
    auto finish = [&](uint32_t error, uint32_t where, uint64_t now) {
        result[KMD_R_STATE] = (
            error == KMD_ERROR_NONE ? KMD_STATE_DONE : KMD_STATE_FAILED
        );
        result[KMD_R_ERROR] = error;
        result[KMD_R_WHERE] = where;
        result[KMD_R_HOST_COMMANDS] = consumed;
        result[KMD_R_COMMANDS] = submitted;
        result[KMD_R_EVENTS] = pushed;
        result[KMD_R_BUSY] = slots.busy_low();
        result[KMD_R_MAX_BUSY] = slots.most;
        result[KMD_R_CONTROL] = control;
        const uint64_t ticks = first_submit != 0 ? now - first_submit : 0;
        const uint64_t last = first_submit != 0 ? last_event - first_submit : 0;
        result[KMD_R_TICKS_LO] = static_cast<uint32_t>(ticks);
        result[KMD_R_TICKS_HI] = static_cast<uint32_t>(ticks >> 32);
        result[KMD_R_IDLE_LO] = static_cast<uint32_t>(idle_passes);
        result[KMD_R_IDLE_HI] = static_cast<uint32_t>(idle_passes >> 32);
        result[KMD_R_FULL_LO] = static_cast<uint32_t>(full_passes);
        result[KMD_R_FULL_HI] = static_cast<uint32_t>(full_passes >> 32);
        result[KMD_R_PASSES_LO] = static_cast<uint32_t>(passes);
        result[KMD_R_PASSES_HI] = static_cast<uint32_t>(passes >> 32);
        result[KMD_R_LAST_LO] = static_cast<uint32_t>(last);
        result[KMD_R_LAST_HI] = static_cast<uint32_t>(last >> 32);
        result[KMD_R_REJECTED] = rejected;
        result[KMD_R_HELD] = held;
        result[KMD_R_NO_ROOM] = no_room;
        result[KMD_R_RESULTS] = results;
        result[KMD_R_RESULT] = last_result;
        result[kTrAbandoned] = slots.in_flight;
        noc_async_write_pcie(block + kTbResult, area + kThResult, kTrWords * 4);
        noc_async_write_barrier();
        put_words(result[KMD_R_STATE], error);
        noc_async_write_barrier();
    };

    // Start-up: each group's READY and core 0's hello {1, control}.
    const uint32_t groups = arg(kTdGroups);
    volatile tt_l1_ptr uint32_t* ready_words = km::l1(control + KM_CONTROL_BOOT);
    volatile tt_l1_ptr uint32_t* mailbox = km::l1(control + KM_CONTROL_JOB);
    const uint64_t boot_started = km::wall64();
    uint32_t ready = 0;
    bool booted = false, hello = false;
    while (km::wall64() - boot_started < boot_limit && stop_word[0] == 0) {
        invalidate_l1_cache();
        ready = 0;
        for (uint32_t group_index = 0; group_index < groups; group_index++) {
            ready |= (ready_words[group_index * 4] == KM_READY) << group_index;
        }
        hello = mailbox[0] == 1 && mailbox[1] == control;
        if (ready == (1u << groups) - 1 && hello) {
            booted = true;
            break;
        }
        pace();
    }
    result[KMD_R_BOOT_TICKS] = static_cast<uint32_t>(km::wall64() - boot_started);
    if (!booted) {
        result[KMD_R_SEEN] = ready;
        stop_all();
        finish(
            stop_word[0] != 0 ? KMD_ERROR_STOPPED : KMD_ERROR_START,
            hello ? KMD_WHERE_READY : KMD_WHERE_HELLO,
            km::wall64()
        );  // fmt: skip
        return;
    }
    words[KMD_HOST_READY] = KMD_READY_MARK;
    noc_async_write_pcie(block + kTbWords + 16, area + 16, 16);  // the ready block
    put_words(KMD_STATE_READY, KMD_ERROR_NONE);

    uint32_t error = KMD_ERROR_NONE, where = KMD_WHERE_LOOP;
    bool stopping = false;
    uint64_t last_activity = km::wall64();
    while (!stopping) {
        invalidate_l1_cache();
        const uint64_t now = km::wall64();
        bool active = false;
        passes++;
        // The sweep: every busy slot's ack word, then its deadline.
        const bool in_time = slots.sweep(
            now,
            deadline,
            [&](uint32_t request, uint32_t status, uint32_t ticks, bool with_result) {
                push(request, status, ticks);
                if (with_result) {
                    results++;
                    last_result = ticks;
                }
                last_event = now;
                active = true;
            },
            [&](uint32_t slot) {
                result[KMD_R_SLOT] = slot;
                result[KMD_R_SEQUENCE] = slots.sequence[slot];
                result[KMD_R_REQ] = slots.request[slot];
                push(
                    slots.request[slot],
                    KM_EVENT_TIMEOUT | slot << 8 |
                        (slots.sequence[slot] & KM_EVENT_SEQUENCE_MASK) << 16,
                    0
                );
            }
        );
        if (!in_time) {
            stop_all();
            error = KMD_ERROR_DEADLINE;
            break;
        }
        // Take: the next line once its four landing words show its number and its
        // event fits the ring with every busy slot's and one more.
        if (pending_request == 0) {
            bool room = pushed - read_count + slots.in_flight + 1 <= event_depth;
            if (!room) {
                read_count = read_line[0];
                room = pushed - read_count + slots.in_flight + 1 <= event_depth;
            }
            volatile tt_l1_ptr uint32_t* line = ring + (next & (host_depth - 1u)) * 16;
            if (
                line[0] == next && line[4] == next && line[8] == next &&
                line[12] == next
            ) {
                if (room) {
                    for (uint32_t word = 0; word < 16; word++) {
                        pending[word] = line[word];
                    }
                    pending_request = next;
                    consumed = next;
                    next++;
                    active = true;
                } else {
                    no_room++;
                }
            }
        }
        // The pending command: refused, served here, held, or submitted to a slot.
        if (pending_request != 0) {
            const uint32_t request = pending_request;
            const uint32_t handler = pending[1] & 0xFFu;
            const uint32_t flags = (pending[1] >> 8) & 0xFFu;
            const bool is_exit = handler == KM_HANDLER_EXIT;
            const uint32_t group = is_exit ? KM_GROUP_ALL(ranks) : pending[2];
            const uint32_t first_core = group & 0xFFFFu, count = group >> 16;
            uint32_t reason = 0;
            if (handler == KM_HANDLER_STOP) {
                // the run's end: its EXIT done, nothing in flight
                pending_request = 0;
                if (slots.in_flight != 0) {
                    stop_all();
                    error = KMD_ERROR_STOP_BUSY;
                }
                where = KMD_WHERE_FINISH;
                stopping = true;
                continue;
            }
            // LOAD (run.load) and ENTRY (run.addon) pass; 247 to 252 are not served
            if (handler > KM_HANDLER_ENTRY && handler <= KM_HANDLER_DEADLINE) {
                reason = KMD_REJECT_HANDLER;
            } else if (
                flags & (1u | KM_FLAG_QUEUE | KM_FLAG_PART | KMD_FLAG_DISPATCHER)
            ) {
                reason = KMD_REJECT_FLAGS;
            } else if (exit_request != 0) {
                reason = KMD_REJECT_EXITED;
            } else if (
                handler != KM_HANDLER_PING &&
                (count == 0 || first_core > ranks || count > ranks - first_core)
            ) {
                reason = KMD_REJECT_GROUP;
            }
            if (reason != 0) {
                push(request, KM_EVENT_REJECTED | reason << 8, 0);
                pending_request = 0;
                rejected++;
            } else if (handler == KM_HANDLER_PING) {
                push(request, KM_EVENT_PONG, 0);
                pending_request = 0;
            } else {
                bool go = !(is_exit && slots.in_flight != 0);
                // exclusive, and a LOAD whatever its flags: no busy slot shares a core
                const bool alone = (
                    (flags & KM_FLAG_EXCLUSIVE) || handler == KM_HANDLER_LOAD
                );
                go = go && !(alone && slots.overlaps(group));
                const uint32_t slot = go ? slots.free_slot() : depth;
                if (slot < depth) {
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
                    // never 0, never what the slot's ack word still holds
                    do {
                        sequence++;
                    } while (sequence == 0 || KM_ACK(sequence) == slots.ack[slot * 4]);
                    slot_command->sequence = sequence;
                    slot_command->land1 = sequence;
                    slot_command->land2 = sequence;
                    slot_command->land3 = sequence;
                    slot_command->group = group;
                    slot_command->hdr = (slot_command->hdr & 0xFF00FFFFu) | slot << 16;
                    const uint64_t started = km::wall64();
                    const uint32_t line = control + KM_CONTROL_RING + slot * 64;
                    if (rects == 0) {
                        for (uint32_t worker = 0; worker < ranks; worker++) {
                            noc_async_write(line, core(worker, line), 64);
                        }
                    }
                    for (uint32_t r = 0; r < rects; r++) {
                        if (loopback) {  // its own line onto itself: the same bytes
                            noc_async_write_multicast_loopback_src(
                                line, rect_target[r] | line, 64, rect_dests[r]
                            );
                        } else {
                            noc_async_write_multicast(
                                line, rect_target[r] | line, 64, rect_dests[r]
                            );
                        }
                    }
                    noc_async_writes_flushed();
                    slots.take(
                        slot,
                        request,
                        sequence,
                        group,
                        flags & KM_FLAG_RESULT,
                        started
                    );  // fmt: skip
                    first_submit = first_submit == 0 ? started : first_submit;
                    submitted++;
                    exit_request = is_exit ? request : 0;
                    pending_request = 0;
                    active = true;
                } else {
                    held++;
                    full_passes += go ? 1 : 0;
                }
            }
        }
        if (consumed != published || pushed != told) {
            put_words(KMD_STATE_READY, KMD_ERROR_NONE);
        }
        if (active) {
            last_activity = now;
        } else {
            idle_passes++;
            pace();
        }
        if ((!active || passes % kTdStopLook == 0) && stop_word[0] != 0) {
            error = KMD_ERROR_STOPPED;  // the host's stop (an abort): the grid's too
            break;
        }
        if (
            idle_limit != 0 && slots.in_flight == 0 && pending_request == 0 &&
            now - last_activity > idle_limit
        ) {
            stop_all();
            error = KMD_ERROR_IDLE;
            break;
        }
    }
    finish(error, where, km::wall64());
}
