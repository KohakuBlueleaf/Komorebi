// The Tensix dispatcher's "out" loop (BRISC): commands from the host ring in this
// core's L1 and from the job queue into the workers' ring slots by multicast.

// "in" (td_in.cpp, NCRISC) sweeps the slots and writes every event; with KM_TD_ALONE
// its pass runs inside this loop instead. No table of each core's program yet.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "komorebi/device/tensors/km_transfer.hpp"
#include "td_engines.hpp"
#include "td_in.hpp"
#include "td_inspect.hpp"
#include "td_lines.hpp"
#include "td_link.hpp"
#include "td_reset.hpp"
#include "td_queue.hpp"
#include "td_slots.hpp"
#include "td_sources.hpp"

namespace {

    using td::arg;

    constexpr uint32_t kNoFirst = 0xFFFFFFFFu;
    constexpr uint32_t kBoundStep = 8;  // the X280's KMD_BOUND_STEP

    // Out's large state, in L1 (kTbOut) and not on its stack: Wormhole's BRISC has
    // 4 KiB of local memory for the firmware's data, out's data and out's stack.
    struct OutState {
        td::Lines lines;  // the host's commands to the thread (kmx_program)
        td::OutSlots slots;
        td::Engines engines;
        td::Queue queue;
        uint32_t pending[16];         // the command taken, not yet submitted
        uint8_t holds[KM_CORES_MAX];  // each core's program field
    };

    static_assert(sizeof(OutState) <= kTbInspect - kTbOut, "out's state in L1");

}  // namespace

// KM_TD_PARTS (a bench build): the loop's ticks by part, into the result's kTrParts.
#ifdef KM_TD_PARTS
#define TD_PART(i)                          \
    do {                                    \
        const uint64_t mark = km::wall64(); \
        parts[i] += mark - part_at;         \
        part_at = mark;                     \
    } while (0)
#else
#define TD_PART(i) ((void)0)
#endif

#if defined(KM_TD_SELF_FEED) && !defined(KM_TD_SELF_SPREAD)
#define KM_TD_SELF_SPREAD 1  // the bench's commands all on the line's own group
#endif

void kernel_main() {
    const uint32_t control = arg(kTdControl), ranks = arg(kTdRanks);
    const uint32_t block = arg(kTdBlock), tag = arg(kTdTag);
    const uint32_t messages = arg(kTdMessages);  // the dynamic handlers (0: none)
    if (!td::block_placed(block)) {
        return;  // the host never sees READY: its wait for it fails by name
    }
    for (uint32_t i = 0; i < kTbBytes / 4; i++) {  // an earlier load's ring and words
        km::l1(block)[i] = 0;
    }
    // in the zeroed block, constructed once
    OutState& state = *new (reinterpret_cast<void*>(block + kTbOut)) OutState;
    // A thread on TRISC 1 (td_thread.hpp): source 1, its ring and events in its area
    const bool thread_on = arg(kTdThreadScale) != 0;
    const uint32_t thread_area = block + kTbBytes;
    for (uint32_t i = kTaRing / 4; thread_on && i < kTaState / 4; i++) {
        km::l1(thread_area)[i] = 0;  // an earlier load's lines and events
    }
    const uint32_t host_depth = arg(kTdHostDepth), event_depth = arg(kTdEventDepth);
    const uint64_t area = td::pcie() | arg(kTdArea);
    const uint32_t depth = td::depth_arg();
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kTdBootLimit)) << 10;
    const uint64_t idle_limit = static_cast<uint64_t>(arg(kTdIdleLimit)) << 10;
    const uint64_t ticks_us = arg(kTdTicksUs);          // times 65536
    const uint64_t bound_hold = (20 * ticks_us) >> 16;  // 20 us, the X280's hold
    uint64_t deadline = static_cast<uint64_t>(arg(kTdDeadline)) << 10;
    // An island's flat ranks and first core (0, 0: one island); a queued job takes
    // `workers` cores at most, its ranks, never its link cores.
    const uint32_t island_first = arg(kTdIsland) & 0xFFFFu;
    const uint32_t island_count = arg(kTdIsland) >> 16;
    const uint32_t island_base = arg(kTdIslandBase);
    const uint32_t workers = island_count != 0 ? island_base + island_count : ranks;
    // The resident engines on TRISC 0 (td_engine.hpp): source 4, after the thread's
    // area
    td::Engines& engines = state.engines;
    engines.init(thread_area + kTaBytes, deadline);
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* stop_word = km::l1(control + KM_CONTROL_STOP);
    volatile tt_l1_ptr uint32_t* beat = km::l1(block + kTbWatch + 4 * kTkOut);
    volatile tt_l1_ptr uint32_t* bound_word = km::l1(control + KM_CONTROL_BOUND);
    volatile tt_l1_ptr uint32_t* read_line = km::l1(block + kTbRead);
    volatile tt_l1_ptr uint32_t* words = km::l1(block + kTbWords);
    uint32_t* pending = state.pending;  // the command taken and not yet submitted
    volatile tt_l1_ptr uint32_t* result = km::l1(block + kTbResult);
    td::Hand hand;
    hand.base = block + kTbHand;
    auto core = [&](uint32_t worker, uint32_t address) {
        return get_noc_addr(cores[worker] & 0xFFFFu, cores[worker] >> 16, address);
    };

    // Each core's program field (a run whose groups switch): a command goes only to
    // cores that hold its program, a LOAD moves them to its own (the X280's table).
    const bool table = arg(kTdPrograms) != 0;
    uint8_t* holds = state.holds;
    for (uint32_t c = 0; c < KM_CORES_MAX; c++) {
        holds[c] = static_cast<uint8_t>(arg(kTdHeld + c / 4) >> (8 * (c % 4)));
    }
    auto hold = [&](uint32_t field, uint32_t first, uint32_t count) {
        for (uint32_t c = first; c < first + count; c++) {
            if (holds[c] != field) {
                return false;
            }
        }
        return true;
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

    uint32_t sequence = 0, consumed = 0, read_count = 0, caused = 0, posted = 0;
    uint32_t rejected = 0, held = 0, no_room = 0, submitted = 0;
    uint32_t pending_request = 0;  // 0: no command taken and waiting
    // The dynamic run in flight (the X280's dynamic.h): its RUN's slot, sequence,
    // group and request; and 1 + its end command's slot while that is busy.
    bool run_on = false;
    uint32_t run_slot = 0, run_sequence = 0, run_group = 0, run_request = 0;
    uint32_t run_flat = 0;  // its line's group, flat ranks
    uint32_t end_slot = 0;
    uint32_t exit_request = 0;  // the EXIT's request once submitted
    uint64_t passes = 0, idle_passes = 0, full_passes = 0;
    uint64_t first_submit = 0;
#ifdef KM_TD_PARTS
    uint64_t parts[kTrPartsCount] = {}, part_at = 0;
#endif
    uint64_t before = 0;  // the queue's places taken when the pending line was
    uint32_t bound = depth, bound_writes = 0, bound_high = 0, bound_low = 0;
    uint64_t bound_at = 0;
    td::OutSlots& slots = state.slots;
    slots.init(depth, hand);
    td::Queue& queue = state.queue;
    queue.base = block + kTbQueue;
    td::Source sources[td::kSources];
    sources[td::kSourceHost].init(km::l1(block + kTbRing), host_depth);
    td::ThreadArea work;  // the thread's events and consumed word (in writes events)
    work.base = thread_area + kTaWork;
    work.event_depth = kTaEventDepth;
    work.place(arg(kTdThreadTile), arg(kTdThreadWork));
    if (thread_on) {
        sources[td::kSourceThread].init(km::l1(thread_area + kTaRing), kTaRingDepth);
    }
    volatile tt_l1_ptr uint32_t* thread_read = km::l1(thread_area + kTaRead);
    td::Lines& lines = state.lines;  // the host's commands to the thread
    lines.init();
    sources[td::kSourceEngine].init(km::l1(engines.area + kTeRing), kTeRingDepth);
    uint32_t thread_caused = 0, turn = 0, pending_source = td::kSourceHost;
    // Commands across islands (td_link.hpp): its state and ring in the link block
    const uint32_t link_base = arg(kTdLinkBase);
    td::Link* link = reinterpret_cast<td::Link*>(link_base + kTlState);
    const bool link_on = arg(kTdLinkIsland) != 0 && link->open(link_base);
    uint32_t link_caused = 0;  // its events, bounded by its origins, crossings, stages
    if (link_on) {
        const uint32_t bytes = kTlInbox + (link->islands - 1u) * link->part_bytes;
        for (uint32_t i = 0; i < kTlState / 4; i++) {
            km::l1(link_base)[i] = 0;
        }
        for (uint32_t i = kTlInbox / 4; i < bytes / 4; i++) {
            km::l1(link_base)[i] = 0;
        }
        sources[td::kSourceLink].init(km::l1(link_base + kTlRing), kTlDepth);
    }
#ifdef KM_TD_ALONE
    td::In in;
    in.init();
#endif
#ifdef KM_TD_SELF_FEED
    uint32_t self_left = KM_TD_SELF_FEED;  // the bench's own commands still to submit
#endif

    // The host words {state, consumed, events written, error}: out writes them at the
    // start and at the end, in between.
    auto put_words = [&](uint32_t state, uint32_t error, uint32_t events) {
        noc_async_writes_flushed();
        words[KMD_HOST_STATE] = state;
        words[KMD_HOST_CONSUMED] = consumed;
        words[KMD_HOST_EVENTS] = events;
        words[KMD_HOST_ERROR] = error;
        noc_async_write_pcie(block + kTbWords, area, 16);
    };
    // An event of no slot (a rejection, a PONG), for in to write to its source.
    auto post = [&](
        uint32_t request, uint32_t status, uint32_t ticks, uint32_t source = 0
    ) {  // fmt: skip
        volatile tt_l1_ptr uint32_t* event = hand.event(posted);
        event[0] = request;
        event[1] = status;
        event[2] = ticks;
        event[3] = source;
        asm volatile("" ::: "memory");
        hand.word(kTwPosted) = ++posted;
    };
    // A line's event to its source: DONE or TIMEOUT, `ticks` its time or its word.
    auto post_line = [&](uint32_t slot, uint32_t kind, uint32_t ticks) {
        post(
            lines.request[slot],
            kind | lines.auxiliary(slot) << 8,
            ticks,
            lines.source[slot]
        );
    };
    // `bytes` at `address` of this core into the same address of every worker.
    auto to_workers = [&](uint32_t address, uint32_t bytes) {
        if (rects == 0) {
            for (uint32_t worker = 0; worker < ranks; worker++) {
                noc_async_write(address, core(worker, address), bytes);
            }
        }
        for (uint32_t r = 0; r < rects; r++) {
            if (loopback) {  // its own bytes onto itself: the same bytes
                noc_async_write_multicast_loopback_src(
                    address, rect_target[r] | address, bytes, rect_dests[r]
                );
            } else {
                noc_async_write_multicast(
                    address, rect_target[r] | address, bytes, rect_dests[r]
                );
            }
        }
    };
    // The ring bound to every worker: landed before a line it must cover goes out.
    auto store_bound = [&](uint32_t value, uint64_t now) {
        *bound_word = value;
        to_workers(control + KM_CONTROL_BOUND, 4);
        noc_async_write_barrier();
        bound = value;
        bound_at = now;
        bound_writes++;
        bound_high = value > bound_high ? value : bound_high;
        bound_low = bound_low == 0 || value < bound_low ? value : bound_low;
    };
    // 1 in every worker's stop flag.
    auto stop_all = [&]() {
        *km::out_slot(control, 0) = 1;
        for (uint32_t worker = 0; worker < ranks; worker++) {
            km::small_write<0, 4>(control, core(worker, control + KM_CONTROL_STOP), 0);
        }
        noc_async_write_barrier();
    };
    // A flat group as this island's part in its cores, first | count << 16 (0: none);
    // link cores k on are after its ranks, never cut (the X280's island_part).
    auto island_part = [&](uint32_t group) -> uint32_t {
        if (island_count == 0) {
            return group;
        }
        const uint32_t first = group & 0xFFFFu, end = first + (group >> 16);
        if (KM_GROUP_IS_LINK(first)) {
            const uint32_t at = island_base + island_count + first - KM_GROUP_LINK_BASE;
            return at | (group >> 16) << 16;
        }
        const uint32_t high = island_first + island_count;
        const uint32_t start = first > island_first ? first : island_first;
        const uint32_t stop = end < high ? end : high;
        return (
            stop > start
                ? (start - island_first + island_base) | (stop - start) << 16
                : 0u
        );
    };
    // A group of this island's cores as the flat group its line carries (the cores
    // test it): one island, itself.
    auto flat_of = [&](uint32_t group) -> uint32_t {
        return group + island_first - island_base;
    };
    // The lowest first core of `count` cores that overlap no busy slot's group.
    auto group_alloc = [&](uint32_t count) {
        uint32_t first = 0;
        while (first <= workers && count <= workers - first) {
            uint32_t end = 0;
            for (uint32_t slot = 0; slot < depth; slot++) {
                const uint32_t group = slots.group[slot];
                const uint32_t busy_first = group & 0xFFFFu;
                const uint32_t busy_end = busy_first + (group >> 16);
                if (
                    slots.busy(slot) && busy_first < first + count &&
                    first < busy_end && busy_end > end
                ) {
                    end = busy_end;
                }
            }
            if (end == 0) {
                return first;
            }
            first = end;
        }
        return kNoFirst;
    };
    // The inspect (a deadline's give-up, in's ask): a core read a pass, the abort word,
    // each core's leave; the answer to in: kTwGivenUp or kTwGiveKept.
    td::Inspect inspect;
    inspect.block = block;
    inspect.control = control;
    uint32_t give_seen = 0;
    const uint64_t leave_limit = (20000 * ticks_us) >> 16;  // the X280's 20 ms
    td::Resets resets;  // the kept lists (load(recover=True)), in DRAM
    resets.block = block;
    resets.beat = beat;
    resets.wait_ticks = (1000000 * ticks_us) >> 16;
    // Whether a busy slot other than `slot` has core `worker` in its group: its
    // command is in the core's ring, which a reset clears.
    auto held_by_other = [&](uint32_t worker, uint32_t slot) {
        for (uint32_t other = 0; other < depth; other++) {
            const uint32_t group = slots.group[other];
            const uint32_t first = group & 0xFFFFu;
            if (
                other != slot && slots.busy(other) && worker >= first &&
                worker - first < (group >> 16)
            ) {
                return true;
            }
        }
        return false;
    };
    // Whether a kernel of core `worker` ended (its own wait gave up).
    auto ended = [&](uint32_t worker) {
        const uint32_t line = block + kTbScratch;
        const uint32_t at = control + KM_CONTROL_STATE + 4 * KMD_STATE_ENDS;
        noc_async_read(core(worker, at & ~15u), line, 16);
        noc_async_read_barrier();
        invalidate_l1_cache();
        return km::l1(line)[(at % 16) / 4] != 0;
    };
    // A give-up's end with kept lists (the X280's inspect_reset): each core still in
    // the command, or ended, reset and loaded again; false if one could not be.
    auto reset_left = [&]() {
        volatile tt_l1_ptr uint32_t* head = inspect.header();
        for (uint32_t index = 0; index < inspect.count; index++) {
            const uint32_t worker = inspect.first + index;
            const bool reset = (
                !inspect.left(worker, core, block + kTbScratch) || ended(worker)
            );
            if (reset && held_by_other(worker, inspect.slot)) {
                head[KMD_I_RESET_FAILED] = (worker + 1) | KMD_RESET_HELD << 16;
                return false;
            }
        }
        for (uint32_t index = 0; index < inspect.count; index++) {
            const uint32_t worker = inspect.first + index;
            if (inspect.left(worker, core, block + kTbScratch) && !ended(worker)) {
                continue;
            }
            uint32_t offset = 0, count = 0;
            const uint32_t status = (
                resets.pick(worker, offset, count)
                    ? resets.run(worker, core, offset, count)
                    : KMD_RESET_NO_LIST
            );
            if (status != KMD_RESET_OK) {
                head[KMD_I_RESET_FAILED] = (worker + 1) | status << 16;
                return false;
            }
            km::l1(inspect.record(index))[KMD_IR_FLAGS] |= KMD_INSPECT_RESET;
            head[KMD_I_RESET]++;
        }
        return true;
    };
    auto inspect_turn = [&](uint64_t now) {
        if (
            inspect.kind == KMD_INSPECT_TIMEOUT &&
            (!slots.busy(inspect.slot) || slots.last(inspect.slot) != inspect.sequence)
        ) {
            inspect.active = false;  // the command completed meanwhile
            return;
        }
        if (inspect.leaving) {
            const uint32_t worker = inspect.first + inspect.next;
            if (
                inspect.next < inspect.count &&
                inspect.left(worker, core, block + kTbScratch)
            ) {
                inspect.next++;
            }
            if (inspect.next == inspect.count && resets.bytes == 0) {
                inspect.active = false;
                hand.word(kTwGivenUp) = inspect.sequence;
            } else if (inspect.next == inspect.count || now > inspect.until) {
                inspect.active = false;  // with kept lists: those in it or ended reset
                hand.word(reset_left() ? kTwGivenUp : kTwGiveKept) = inspect.sequence;
            }
            return;
        }
        const uint32_t worker = inspect.first + inspect.next;
        inspect.read_core(inspect.next, worker, core, cores[worker]);
        inspect.header()[KMD_I_DONE] = ++inspect.next;
        if (inspect.next < inspect.count) {
            return;
        }
        inspect.mark(now);
        if (inspect.kind == KMD_INSPECT_HOST) {
            inspect.active = false;
            post(
                inspect.request,
                KM_EVENT_PONG,
                static_cast<uint32_t>(now - inspect.start)
            );
            return;
        }
        // the abort: the slot's word on every core (its aligned 16 bytes)
        const uint32_t at = control + KM_CONTROL_ABORT + 4 * inspect.slot;
        km::l1(at)[0] = inspect.sequence;
        to_workers(at & ~15u, 16);
        noc_async_write_barrier();
        inspect.leaving = true;
        inspect.next = 0;
        inspect.until = now + leave_limit;
    };
    // Command `line` to free `slot` for `request`, on cores `group` (the next sequence,
    // the landing words); its event comes from in once the group's leader acks.
    auto submit = [&](
        const auto* line,
        uint32_t slot,
        uint32_t request,
        uint32_t group,
        uint64_t ticks,
        uint32_t silent = 0,
        uint32_t flat = 0
    ) {  // fmt: skip
        do {                                // never 0, never the slot's last
            sequence++;
        } while (sequence == 0 || sequence == slots.last(slot));
        // Straight into the slot's line, word by word (the stack is in L1 too): the
        // landing words of km_command_t, its slot and its group (`flat`, else `group`).
        const uint32_t at = control + KM_CONTROL_RING + slot * 64;
        volatile tt_l1_ptr uint32_t* to = km::l1(at);
        const uint32_t header = (line[1] & 0xFF00FFFFu) | slot << 16;
        // a program's command may be given up at its deadline (inspect_may_give_up)
        const uint32_t handler = line[1] & 0xFFu;
        const bool give_up = (
            (silent & td::kRecordSilent) == 0 && handler <= KM_HANDLER_USER_MAX &&
            !KM_IS_DYNAMIC(messages, handler)
        );
        to[0] = sequence;
        to[1] = header;
        to[2] = flat != 0 ? flat : group;
        to[3] = line[3];
        for (uint32_t chunk = 4; chunk < 16; chunk += 4) {
            to[chunk] = sequence;
            to[chunk + 1] = line[chunk + 1];
            to[chunk + 2] = line[chunk + 2];
            to[chunk + 3] = line[chunk + 3];
        }
        TD_PART(3);
        const uint64_t started = km::wall64();
        if (slot >= bound) {  // raised over it, and landed, before its line goes out
            const uint32_t raised = (slot + kBoundStep) / kBoundStep * kBoundStep;
            store_bound(raised < depth ? raised : depth, started);
        }
        to_workers(at, 64);  // no wait: the line is not written again before its ack
        TD_PART(4);
        slots.take(
            slot,
            request,
            sequence,
            group,
            ((header >> 8) & KM_FLAG_RESULT ? td::kRecordResult : 0) | silent |
                (give_up ? td::kRecordGiveUp : 0),
            started,
            ticks
        );  // fmt: skip
        first_submit = first_submit == 0 ? started : first_submit;
        submitted++;
    };
    // The link's pass (the X280's link_turn): the inbox, the link ring's events, the
    // parts queued and sent, the crossings ended or late posted. Whether it did any.
    auto link_turn = [&](uint64_t now) {
        bool did = false;
        for (uint32_t from = 0; from < link->islands; from++) {
            const uint32_t before_read = link->read[from];
            if (from != link->island) {
                link->receive_from(
                    from,
                    sources[td::kSourceLink].next,
                    km::l1(link_base + kTlTold)[kTlSamples]
                );  // fmt: skip
                did = did || link->read[from] != before_read;
            }
        }
        const uint32_t told = km::l1(link_base + kTlTold)[0];
        while (link->events_read != told) {
            const uint32_t index = link->events_read + 1u;
            volatile tt_l1_ptr uint32_t* event = (
                km::l1(link_base + kTlEvents + (index % kTlEventDepth) * 16)
            );
            const uint32_t request = event[1], status = event[2], ticks = event[3];
            if (request & td::kRequestCrossing) {
                link->part(request & 0xFFu, link->island, status, ticks);
            } else if (request & td::kRequestStage) {
                const uint32_t stage = request & 0xFFu;
                const uint32_t ticket = link->stage_ticket[stage];
                const uint32_t lost = ticket != 0 ? (ticket - 1u) & 0xFFu : 0u;
                if (
                    (status & 0xFFu) != KM_EVENT_DONE && ticket != 0 &&
                    (link->busy & (1u << lost)) &&
                    link->crossing[lost].generation == (ticket - 1u) >> 8
                ) {  // its LINE given up: the crossing late
                                               // at the next look
                    link->crossing[lost].deadline = 0;
                }
                link->stages &= ~(1u << stage);
                link->stage_ticket[stage] = 0;
            } else {
                if (link->out_room() == 0) {
                    break;
                }
                for (uint32_t k = 0; k < td::kLinkOrigins; k++) {
                    td::Origin& origin = link->origin[k];
                    if (origin.number == request) {
                        link->queue(
                            origin.island,
                            nullptr,
                            KMD_CROSS_DONE,
                            origin.ticket,
                            status,
                            ticks
                        );  // fmt: skip
                        origin.number = 0;
                        link->answered++;
                        break;
                    }
                }
            }
            link->events_read = index;
            did = true;
        }
        link->feed(now);
        // each packet staged on link core 0 (three lines), then its PAGES LINE command
        while (link->out_head != link->out_tail) {
            uint32_t stage = 0;
            while (stage < KM_LINK_STAGES && (link->stages & (1u << stage))) {
                stage++;
            }
            const uint32_t slot = slots.free_slot();
            if (stage == KM_LINK_STAGES || slot == depth) {
                break;
            }
            td::Packet& packet = link->out[link->out_head % td::kLinkOut];
            const uint32_t to = link->out_to[link->out_head % td::kLinkOut];
            const uint32_t number = ++link->sent[to];
            for (uint32_t w = 0; w < 16; w += 4) {
                packet.word[w] = number;
            }
            packet.word[16 + KMD_X_NUMBER] = number;
            packet.word[16 + KMD_X_ROW] = number;
            volatile tt_l1_ptr uint32_t* count_line = km::l1(link_base + kTlCount);
            for (uint32_t w = 0; w < 16; w++) {
                count_line[w] = w < 2 ? number : 0u;
            }
            const uint32_t staged = (
                link->area + KM_LINK_STAGE + stage * KM_LINK_STAGE_BYTES
            );
            noc_async_write(
                reinterpret_cast<uint32_t>(&packet),
                core(link->core, staged),
                KM_LINK_PACKET_BYTES
            );  // fmt: skip
            noc_async_write(link_base + kTlCount, core(link->core, staged + 128), 64);
            noc_async_write_barrier();
            // the far island's inbox is at this island's place: the same block address
            const uint32_t far = link_base + kTlInbox + link->part_of(to, link->island);
            uint32_t send[16] = {0};
            send[1] = KM_HANDLER_PAGES;
            send[2] = KM_GROUP_LINK_BASE | 1u << 16;
            send[3] = stage;
            send[5] = number;
            send[6] = link->entries[3 * to + 2];
            send[7] = far + KMD_LINK_COUNT_BYTES + (number % link->slots) * 128;
            send[9] = far;
            send[10] = link->area;
            send[11] = to;
            send[13] = 1u << 16 | KM_PAGES_LINE << 24;  // one hop, the LINE mode
            submit(
                send,
                slot,
                td::kRequestStage | stage,
                island_part(send[2]),
                deadline,
                td::kRecordLink,
                send[2]
            );  // fmt: skip
            link->stages |= 1u << stage;
            link->stage_ticket[stage] = (
                (packet.word[16 + KMD_X_KIND] & 0xFFu) == KMD_CROSS_COMMAND
                    ? packet.word[16 + KMD_X_TICKET] + 1u
                    : 0u
            );
            link->out_head++;
            did = true;
        }
        for (uint32_t index = 0; index < td::kLinkCrossings; index++) {
            td::Crossing& crossing = link->crossing[index];
            const bool room = posted - hand.word(kTwTaken) < kTxEventsMax;
            if (!(link->busy & (1u << index)) || !room) {
                continue;
            }
            if (crossing.ended) {  // every part back: its event to its source
                uint32_t ticks = 0;
                const uint32_t status = link->event_status(index, now, ticks);
                post(crossing.request, status, ticks, crossing.source);
                crossing.ended = 0;
                link->busy &= ~(1u << index);
                did = true;
                continue;
            }
            if (crossing.late || now <= crossing.deadline) {
                continue;
            }
            const uint32_t ticket = (index | crossing.generation << 8) + 1u;
            bool staged = false;
            for (uint32_t stage = 0; stage < KM_LINK_STAGES; stage++) {
                staged = (
                    staged ||
                    (
                        (link->stages >> stage & 1u) &&
                        link->stage_ticket[stage] == ticket
                    )
                );
            }
            if (staged) {
                continue;  // its LINE's give-up ends it
            }
            uint32_t missing = link->island;  // the lowest island not back
            for (uint32_t island = 0; island < link->islands; island++) {
                if (crossing.pending & (1u << island)) {
                    missing = island;
                    break;
                }
            }
            crossing.late = 1;
            link->gave_up++;
            post(
                crossing.request,
                KM_EVENT_TIMEOUT |
                    (
                        (KM_EVENT_ISLAND + missing) |
                        (crossing.sequence & KM_EVENT_SEQUENCE_MASK)
                            << KM_EVENT_SEQUENCE_SHIFT
                    )
                        << 8,
                static_cast<uint32_t>(now - crossing.submitted),
                crossing.source
            );  // fmt: skip
            did = true;
        }
        return did;
    };
    // A dynamic run across islands (the X280's waves.h): link core 0's run entry, open
    // with the run's id or closed (0); it delivers a line only while open with its id.
    auto run_entry = [&](uint32_t id) {
        volatile tt_l1_ptr uint32_t* line = km::l1(link_base + kTlCount);
        line[0] = id;
        line[1] = link->waves.sequence;
        line[2] = link->waves.group;
        line[3] = ~id;
        noc_async_write(
            link_base + kTlCount, core(link->core, link->area + KM_DELIVER_RUN), 16
        );
        noc_async_write_barrier();
    };
    // A wave packet of `kind` with `command` to every other island of the run.
    auto wave_out = [&](uint32_t kind, const uint32_t* command) {
        for (uint32_t island = link->waves.low; island <= link->waves.high; island++) {
            if (island != link->island) {
                link->queue(island, command, kind, 0, 0, 0);
            }
        }
    };
    const uint64_t wave_ticks = (10 * ticks_us) >> 16;  // between two rounds, 10 us
    // The waves, a step a pass, true once ended: the leader gives the run its id, then
    // asks rounds; two in a row all quiet, equal sums and sent == consumed end it.
    auto waves_turn = [&](uint64_t now) {
        td::Waves& w = link->waves;
        volatile tt_l1_ptr uint32_t* words = km::l1(link_base + kTlTold);
        const uint32_t others = w.high - w.low;
        uint32_t command[16] = {0};
        words[kTlWaited] = link->waited() + 1u;  // in names it at the RUN's deadline
        if (w.ended) {
            return true;
        }
        if (w.id == 0) {
            if (w.leader) {
                if (link->out_room() < others) {
                    return false;
                }
                w.id = w.sequence << 8 | link->island;
                w.id = w.id != 0 ? w.id : 0x100u;
                command[10] = w.id;
                command[11] = w.group;
                wave_out(KMD_CROSS_START, command);
            } else if (w.told_id != 0 && w.told_group == w.group) {
                w.id = w.told_id;
                w.told_id = 0;
            } else {
                return false;
            }
            run_entry(w.id);
        }
        const uint32_t samples = words[kTlSamples];
        if (!w.leader) {  // a ROUND answered with a sample taken after it
            if (w.asked != 0 && link->out_room() != 0 && samples != w.asked_at) {
                command[2] = words[kTlSent];
                command[3] = words[kTlSent + 1];
                command[6] = words[kTlConsumed];
                command[7] = words[kTlConsumed + 1];
                command[10] = w.asked_id;
                command[11] = w.asked;
                command[14] = words[kTlQuiet] != 0 && w.asked_id == w.id ? 1u : 0u;
                link->queue(w.low, command, KMD_CROSS_COUNT, 0, 0, 0);
                w.asked = 0;
            }
            return false;
        }
        if (w.round != 0 && w.answered != link->others()) {
            return false;
        }
        if (w.round != 0) {  // a whole round
            if (
                w.quiet && w.have_last && w.sent == w.last_sent &&
                w.consumed == w.last_consumed && w.sent == w.consumed
            ) {
                if (link->out_room() < others) {
                    return false;
                }
                command[10] = w.id;
                wave_out(KMD_CROSS_END, command);
                w.ended = 1;
                return true;
            }
            w.have_last = w.quiet;
            w.last_sent = w.sent;
            w.last_consumed = w.consumed;
            w.round = 0;
        }
        if (now < w.next_round || link->out_room() < others || samples == w.used) {
            return false;
        }
        w.used = samples;
        w.quiet = words[kTlQuiet];
        if (!w.quiet) {
            w.have_last = 0;
            return false;  // this island is not quiet: no round yet
        }
        w.sent = td::wide(words[kTlSent], words[kTlSent + 1]);
        w.consumed = td::wide(words[kTlConsumed], words[kTlConsumed + 1]);
        w.round = w.round_number = w.round_number + 1u;
        w.answered = 0;
        w.next_round = now + wave_ticks;
        command[10] = w.id;
        command[11] = w.round;
        wave_out(KMD_CROSS_ROUND, command);
        return false;
    };
    // in's end: its last pass, then its final words (out waits for them).
    auto end_in = [&]() {
        hand.word(kTwEnd) = tag;
#ifdef KM_TD_ALONE
        invalidate_l1_cache();
        in.pass(km::wall64());
        in.over();
#else
        const uint64_t started = km::wall64();
        do {
            invalidate_l1_cache();
        } while (hand.word(kTwOver) != tag && km::wall64() - started < boot_limit);
#endif
    };
    // The result block, then the final host words (the host reads the block once the
    // state is final).
    auto finish = [&](uint32_t error, uint32_t where, uint64_t now) {
        end_in();
        const uint32_t pushed = hand.word(kTwPushed);
        const uint64_t last_event = (
            td::wide(hand.word(kTwLastLo), hand.word(kTwLastHi))
        );
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
        result[KMD_R_RESULTS] = hand.word(kTwResults);
        result[KMD_R_RESULT] = hand.word(kTwLastResult);
        result[kTrAbandoned] = slots.in_flight + queue.length;
        result[kTrBoundWrites] = bound_writes;
        result[kTrBoundHigh] = bound_high;
        result[kTrBoundLow] = bound_low;
        result[kTrInPasses] = hand.word(kTwPasses);
        result[kTrInSweeps] = hand.word(kTwSweeps);
        result[kTrRunGave] = hand.word(kTwRunGave);
        result[kTrRunChecks] = hand.word(kTwRunChecks);
        result[kTrRunQuietAt] = hand.word(kTwRunQuietAt);
#ifdef KM_TD_PARTS
        for (uint32_t i = 0; i < kTrPartsCount; i++) {
            result[kTrParts + i] = static_cast<uint32_t>(parts[i]);
        }
#endif
        noc_async_write_pcie(block + kTbResult, area + kThResult, kTrWords * 4);
        noc_async_write_barrier();
        put_words(result[KMD_R_STATE], error, pushed);
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
        td::pace();
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
    words[KMD_HOST_READY + 1] = engines.upload();  // the engines' upload region
    words[KMD_HOST_READY + 2] = kTeUploadBytes;
    noc_async_write_pcie(block + kTbWords + 16, area + 16, 16);  // the ready block
    put_words(KMD_STATE_READY, KMD_ERROR_NONE, 0);
    noc_async_write_barrier();  // landed before in's first host words
    hand.word(kTwGo) = tag;

    uint32_t error = KMD_ERROR_NONE, where = KMD_WHERE_LOOP;
    bool stopping = false;
    uint64_t last_activity = km::wall64();
    while (!stopping) {
        invalidate_l1_cache();
        const uint64_t now = km::wall64();
        bool active = false;
        passes++;
        beat[0] = static_cast<uint32_t>(passes);  // the watch's (td_watch.hpp)
#ifdef KM_TD_PARTS
        part_at = now;
#endif
#ifdef KM_TD_ALONE
        active = in.pass(now);
#endif
        if (hand.word(kTwLate) != 0) {  // in sent the TIMEOUT and the result's words
            stop_all();
            error = KMD_ERROR_DEADLINE;
            break;
        }
        // The slots in freed; the bound lowered a step once it has held (the X280's
        // rule).
        if (slots.refresh() != 0) {
            // the dynamic run's RUN done, or its end command (no event) freed
            run_on = (
                run_on && slots.busy(run_slot) && slots.last(run_slot) == run_sequence
            );
            if (!run_on && link_on && link->waves.on) {  // its RUN ended otherwise
                if (link->waves.id != 0) {
                    run_entry(0);
                }
                link->waves.on = 0;
            }
            end_slot = end_slot != 0 && !slots.busy(end_slot - 1) ? 0 : end_slot;
            if (bound >= 2 * kBoundStep && now - bound_at >= bound_hold) {
                bool above = false;
                for (uint32_t slot = bound - 2 * kBoundStep; slot < depth; slot++) {
                    above = above || slots.busy(slot);
                }
                if (!above) {
                    store_bound(bound - kBoundStep, now);
                }
            }
        }
        TD_PART(0);
        if (link_on) {
            active = link_turn(now) || active;
        }
        // Take: a source's next line, in turn, once landed and its event fits its ring
        // with every event owed and one more (a POST: the host's), and posts have room.
        if (pending_request == 0) {
            bool room = caused - read_count + 1 <= event_depth;
            if (!room) {
                read_count = read_line[0];
                room = caused - read_count + 1 <= event_depth;
            }
            const bool posts = posted - hand.word(kTwTaken) < kTxEventsMax;
            const uint32_t engine_handler = (
                sources[td::kSourceEngine].line()[1] & 0xFFu
            );
            bool rooms[td::kSources] = {};
            rooms[td::kSourceHost] = room && posts;
            rooms[td::kSourceThread] = (
                posts && thread_caused - thread_read[0] < kTaEventDepth
            );
            rooms[td::kSourceEngine] = (
                posts &&
                (td::Engines::own(td::kSourceEngine, engine_handler) || engines.room())
            );
            rooms[td::kSourceLink] = posts;
            for (uint32_t k = 0; k < td::kSources && pending_request == 0; k++) {
                const uint32_t source = (turn + k) % td::kSources;
                td::Source& from = sources[source];
                if (from.depth == 0 || !from.landed()) {
                    continue;
                }
                if (!rooms[source]) {
                    no_room++;
                    continue;
                }
                volatile tt_l1_ptr uint32_t* next_line = from.line();
                if (
                    source != td::kSourceHost && lines.full() &&
                    (next_line[2] & 0xFFFFu) >= KM_GROUP_CLUSTER_BASE &&
                    (next_line[1] & 0xFFu) <= KM_HANDLER_USER_MAX &&
                    !(
                        source == td::kSourceEngine &&
                        (next_line[1] >> 8) & KM_FLAG_QUEUE
                    )
                ) {
                    held++;  // a command to the thread, from the thread or the engines:
                    continue;  // it waits in its ring for a line
                }
                pending_request = from.take(pending);
                pending_source = source;
                if (source == td::kSourceHost) {
                    consumed = pending_request;
                    hand.word(kTwConsumed) = consumed;
                } else if (source == td::kSourceThread) {
                    *work.consumed() = pending_request;  // its line free again
                    work.mirror(KMD_WORK_CONSUMED, 4);
                } else if (source == td::kSourceEngine) {
                    *engines.consumed() = pending_request;
                }
                turn = source + 1;
                before = queue.used;
                active = true;
            }
        }
        TD_PART(1);
        // The pending command: refused, served here, queued, held, or submitted.
        if (pending_request != 0) {
            const uint32_t request = pending_request;
            const uint32_t handler = pending[1] & 0xFFu;
            const uint32_t flags = (pending[1] >> 8) & 0xFFu;
            const uint32_t argument = pending[3];
            const bool is_exit = handler == KM_HANDLER_EXIT;
            const bool queued = !is_exit && (flags & KM_FLAG_QUEUE) != 0;
            // the line's group (flat ranks) and its part here, in this island's cores
            const uint32_t line_group = (
                is_exit ? flat_of(KM_GROUP_ALL(ranks)) : pending[2]
            );
            const bool cut = !is_exit && !queued && !(flags & KMD_FLAG_DISPATCHER);
            const uint32_t group = (
                is_exit
                    ? KM_GROUP_ALL(ranks)
                    : cut
                        ? island_part(line_group)
                        : line_group
            );
            const uint32_t first_core = group & 0xFFFFu, count = group >> 16;
            // an island's PAGES command (km_pages.hpp): to its link cores, or a rank's
            // wait for pages; the X280 takes it from the host
            const bool pages = (
                island_count != 0 && handler == KM_HANDLER_PAGES &&
                pending_source != td::kSourceEngine  // the engines' 248: POST
            );
            const uint32_t field = pending[1] >> 24;  // the command's program
            uint32_t reason = 0;
            // the thread's and the engines': their events to their rings; none of the
            // host's own commands
            const bool from_host = pending_source == td::kSourceHost;
            const bool from_thread = pending_source == td::kSourceThread;
            const bool from_engine = pending_source == td::kSourceEngine;
            // another island's part (td_link.hpp): its event to the link, its deadline
            // the time it had left
            const bool from_link = pending_source == td::kSourceLink;
            uint32_t& owed = (
                from_thread
                    ? thread_caused
                    : from_engine
                        ? engines.caused
                        : from_link
                            ? link_caused
                            : caused
            );
            const uint64_t ticks = (
                from_engine
                    ? engines.deadline
                    : from_link
                        ? link->left[request % kTlDepth]
                        : deadline
            );
            // a line that reaches other islands, from here: its parts cross
            const bool crossing = (
                link_on && !from_link && !from_engine && !is_exit && !queued &&
                (flags & (KMD_FLAG_DISPATCHER | KM_FLAG_PART)) == 0 &&
                link->reaches_out(line_group)
            );
            if (from_thread && handler == KMD_HANDLER_ANSWER) {  // its line, no event
                pending_request = 0;
                const uint32_t slot = lines.answered(work, argument);
                if (slot < KMD_WORK_IN_DEPTH) {
                    post_line(slot, KM_EVENT_DONE, lines.ticks(slot, pending[5], now));
                }
                active = true;
                continue;
            }
            // a command to the thread (kmx_program): its next line, not the cores'
            if (
                !is_exit && (pending[2] & 0xFFFFu) >= KM_GROUP_CLUSTER_BASE &&
                handler <= KM_HANDLER_USER_MAX && !(queued && from_engine)
            ) {
                uint32_t why = 0;
                if (flags & (1u | KM_FLAG_EXCLUSIVE | KM_FLAG_QUEUE)) {
                    why = KMD_REJECT_FLAGS;
                } else if (exit_request != 0) {
                    why = KMD_REJECT_EXITED;
                } else if (!thread_on || pending[2] != KM_GROUP_CLUSTER(1)) {
                    why = KMD_REJECT_GROUP;  // one thread, on cluster 1
                } else if (lines.full()) {
                    why = KMD_REJECT_THREAD;  // the host's (the others wait, at take)
                }
                if (why != 0) {
                    post(request, KM_EVENT_REJECTED | why << 8, 0, pending_source);
                    rejected++;
                } else {
                    do {
                        sequence++;
                    } while (sequence == 0);
                    lines.put(
                        work,
                        pending,
                        pending_source,
                        request,
                        sequence,
                        now,
                        ticks == 0 ? ~0ull >> 1 : ticks
                    );
                }
                owed++;
                pending_request = 0;
                active = true;
                continue;
            }
            if (handler == KM_HANDLER_STOP && from_host) {
                // the run's end: its EXIT done, nothing in flight; what waits is
                // dropped
                pending_request = 0;
                if (slots.in_flight != 0) {
                    stop_all();
                    error = KMD_ERROR_STOP_BUSY;
                }
                where = KMD_WHERE_FINISH;
                stopping = true;
                continue;
            }
            if (
                handler == KM_HANDLER_DEADLINE && argument != 0 && !from_thread &&
                !from_link
            ) {
                // the deadline of the commands submitted from here on, in microseconds
                (from_engine ? engines.deadline : deadline) = (argument * ticks_us) >>
                    16;
                if (from_host) {
                    engines.host_deadline(argument);  // a stream's that names none
                }
                pending_request = 0;
                continue;
            }
            // the host's inspect (run.inspect) of a group inside the cores
            const bool own_inspect = (
                from_host && (flags & KMD_FLAG_DISPATCHER) != 0 &&
                handler == KMD_HANDLER_INSPECT && count != 0 && first_core < ranks &&
                count <= ranks - first_core
            );
            // the host's reset commands: keep the lists (group count 0), reset a core
            const bool own_reset = (
                from_host && (flags & KMD_FLAG_DISPATCHER) != 0 &&
                handler == KMD_HANDLER_RESET &&
                (count == KMD_RESET_KEEP || (count == 1 && first_core < ranks))
            );
#ifdef KMD_STALL_DEBUG
            // test builds: PONG, then no pass for a0 ms (the watch's stall)
            const bool own_stall = (
                from_host && (flags & KMD_FLAG_DISPATCHER) != 0 &&
                handler == KMD_HANDLER_STALL
            );
#else
            const bool own_stall = false;
#endif
            // the engines' own: the host's RUN and CANCEL, their POST and DROP
            const bool engine_line = td::Engines::own(pending_source, handler);
            // the islands a crossing reaches, low to high
            uint32_t low = td::kIslandNone, high = td::kIslandNone;
            // LOAD (run.load) and ENTRY (run.addon) pass; 247 to 251 are not served
            if (crossing) {  // the X280's link_cross
                const uint32_t flat = line_group & 0xFFFFu, total = line_group >> 16;
                low = link->find(flat);
                high = total == 0 ? td::kIslandNone : link->find(flat + total - 1u);
                if (handler > KM_HANDLER_USER_MAX) {
                    reason = KMD_REJECT_HANDLER;
                } else if (flags & (1u | KM_FLAG_EXCLUSIVE)) {
                    reason = KMD_REJECT_FLAGS;
                } else if (exit_request != 0) {
                    reason = KMD_REJECT_EXITED;
                } else if (KM_IS_DYNAMIC(messages, handler)) {
                    reason = KMD_REJECT_MESSAGES;
                } else if (
                    low == td::kIslandNone || high == td::kIslandNone ||
                    flat + total < flat
                ) {
                    reason = KMD_REJECT_GROUP;
                } else if (
                    count != 0 && table && handler != KM_HANDLER_NOP &&
                    handler != KM_HANDLER_PING && !hold(field, first_core, count)
                ) {
                    reason = KMD_REJECT_PROGRAM;
                }
            } else if (own_inspect || own_reset || own_stall || engine_line) {
                // served below
            } else if (
                from_engine && handler > KM_HANDLER_USER_MAX &&
                handler != KM_HANDLER_PING && handler != KM_HANDLER_DEADLINE
            ) {
                reason = KMD_REJECT_HANDLER;  // theirs: plain commands, jobs and PING
            } else if (
                from_thread && handler > KM_HANDLER_USER_MAX &&
                handler != KM_HANDLER_LOAD && handler != KM_HANDLER_COPY &&
                handler != KM_HANDLER_PING
            ) {
                reason = KMD_REJECT_HANDLER;  // the host's alone: EXIT, STOP, DEADLINE
            } else if (from_thread && (queued || KM_IS_DYNAMIC(messages, handler))) {
                reason = KMD_REJECT_FLAGS;  // the thread's go to slots, one at a time
            } else if (handler == KM_HANDLER_DEADLINE) {
                reason = KMD_REJECT_DEADLINE;
            } else if (
                handler > KM_HANDLER_ENTRY && handler < KM_HANDLER_DEADLINE && !pages
            ) {
                reason = KMD_REJECT_HANDLER;
            } else if (flags & (1u | KMD_FLAG_DISPATCHER)) {
                // not a part (KM_FLAG_PART): one island takes it too, as the X280 does
                reason = KMD_REJECT_FLAGS;
            } else if (exit_request != 0) {
                reason = KMD_REJECT_EXITED;
            } else if (
                handler != KM_HANDLER_PING &&
                (
                    queued
                        ? count == 0 || count > workers
                        : count == 0 || first_core > ranks || count > ranks - first_core
                )
            ) {
                reason = KMD_REJECT_GROUP;
            } else if (queued && KM_IS_DYNAMIC(messages, handler)) {
                reason = KMD_REJECT_MESSAGES;
            } else if (queued && handler == KM_HANDLER_LOAD) {
                reason = KMD_REJECT_PROGRAM;  // the X280's: a LOAD is never queued
            } else if (
                handler == KM_HANDLER_LOAD
                    ? !table || field >= kTdProgramsMax
                    : table && handler != KM_HANDLER_NOP && !is_exit &&
                        handler != KM_HANDLER_COPY && handler != KM_HANDLER_ENTRY &&
                        handler != KM_HANDLER_PING && !pages &&
                        !hold(field, queued ? 0 : first_core, queued ? workers : count)
            ) {
                reason = KMD_REJECT_PROGRAM;  // a queued command: every core holds it
            } else if (queued && queue.length == KM_QUEUE_JOBS && !from_engine) {
                reason = KMD_REJECT_QUEUE;  // the engines' job waits for a place
            }
            if (reason != 0) {
                post(request, KM_EVENT_REJECTED | reason << 8, 0, pending_source);
                owed++;
                pending_request = 0;
                rejected++;
                engines.taken += from_engine && queued ? 1 : 0;  // as they count it
            } else if (engine_line) {
                auto host_room = [&]() {
                    read_count = read_line[0];
                    return caused - read_count + 1 <= event_depth;
                };
                const uint32_t posts = kTxEventsMax - (posted - hand.word(kTwTaken));
                if (engines.serve(
                    pending,
                    request,
                    pending_source,
                    exit_request != 0,
                    queue,
                    before,
                    posts,
                    caused,
                    post,
                    host_room
                )) {  // fmt: skip
                    pending_request = 0;
                    active = true;
                }
            } else if (queued && queue.length == KM_QUEUE_JOBS) {
                held++;  // the engines' job
            } else if (crossing) {
                // held while every crossing is in flight, or its part has no slot
                uint32_t index = 0;
                while (index < td::kLinkCrossings && (link->busy & (1u << index))) {
                    index++;
                }
                const uint32_t slot = count != 0 ? slots.free_slot() : depth;
                if (index == td::kLinkCrossings || (count != 0 && slot == depth)) {
                    held++;
                } else {
                    td::Crossing& c = link->crossing[index];
                    c.generation = (c.generation + 1u) & 0xFFFFFFu;
                    c.pending = 0;
                    uint32_t fars = 0;
                    for (uint32_t island = low; island <= high; island++) {
                        if (island != link->island) {
                            c.pending |= 1u << island;
                            fars++;
                        }
                    }
                    for (uint32_t word = 0; word < 16; word++) {
                        c.line[word] = pending[word];
                    }
                    c.next = low;
                    c.last = high;
                    c.request = request;
                    c.source = pending_source;
                    c.parts = (count != 0 ? 1u : 0u) + fars;
                    c.status = KM_EVENT_DONE;
                    c.result = (flags & KM_FLAG_RESULT) != 0 ? 1u : 0u;
                    c.word = 0;
                    c.late = 0;
                    c.ended = 0;
                    c.holder = low;
                    c.first = first_core;
                    c.submitted = now;
                    c.deadline = now + deadline;
                    link->busy |= 1u << index;
                    link->crossed++;
                    if (count != 0) {
                        submit(
                            pending,
                            slot,
                            td::kRequestCrossing | index,
                            group,
                            deadline,
                            td::kRecordLink,
                            line_group
                        );  // fmt: skip
                        c.sequence = slots.last(slot);
                    } else {
                        do {
                            sequence++;
                        } while (sequence == 0);
                        c.sequence = sequence;
                    }
                    owed++;
                    pending_request = 0;
                    active = true;
                }
            } else if (own_stall) {
                post(request, KM_EVENT_PONG, 0);
                owed++;
                pending_request = 0;
                const uint64_t until = now + ((1000 * argument * ticks_us) >> 16);
                while (km::wall64() < until && stop_word[0] == 0) {
                    invalidate_l1_cache();
                }
            } else if (own_reset) {
                const uint64_t started = km::wall64();
                const uint32_t status = (
                    count == KMD_RESET_KEEP
                        ? resets.keep(argument, pending[5], td::pcie(), pending[6])
                        : resets.run(first_core, core, argument, pending[5])
                );
                post(
                    request,
                    KM_EVENT_PONG | status << 8,
                    static_cast<uint32_t>(km::wall64() - started)
                );
                owed++;
                pending_request = 0;
            } else if (own_inspect) {
                if (!inspect.active) {  // one at a time: its PONG once the last is read
                    inspect.begin(KMD_INSPECT_HOST, request, group, 0, sequence, now);
                    owed++;
                    pending_request = 0;
                }
            } else if (handler == KM_HANDLER_PING) {
                post(request, KM_EVENT_PONG, 0, pending_source);
                owed++;
                pending_request = 0;
            } else if (queued) {
                queue.put(pending, request, count, ticks, pending_source);
                owed++;
                engines.taken += from_engine ? 1 : 0;
                pending_request = 0;
            } else {
                bool go = (
                    !(
                        is_exit &&
                        (slots.in_flight != 0 || queue.length != 0 || engines.busy())
                    )
                );
                // exclusive: every job queued before it placed, and no busy slot shares
                // a core; a LOAD whatever its flags, and once the queue is empty
                const bool alone = (
                    (flags & KM_FLAG_EXCLUSIVE) || handler == KM_HANDLER_LOAD
                );
                go = go && !(alone && (before != 0 || slots.overlaps(group)));
                go = go && !(handler == KM_HANDLER_LOAD && queue.length != 0);
                // one dynamic run at a time: the next waits for this one and its end
                const bool dynamic = KM_IS_DYNAMIC(messages, handler);
                go = go && !(dynamic && (run_on || end_slot != 0));
                const uint32_t slot = go ? slots.free_slot() : depth;
                TD_PART(2);
#ifdef KM_TD_SELF_FEED
                if (slot < depth && self_left != 0) {
                    // the bench: the line again, its group moved, before the line
                    // itself
                    const uint32_t step = KM_TD_SELF_FEED - self_left;
                    const uint32_t moved = (step % KM_TD_SELF_SPREAD) * count;
                    submit(
                        pending, slot, 0, (first_core + moved) | count << 16, deadline
                    );
                    self_left--;
                    active = true;
                    continue;
                }
#endif
                if (slot < depth) {
                    const uint32_t mark = (
                        from_thread
                            ? td::kRecordThread
                            : from_engine
                                ? td::kRecordEngine
                                : from_link
                                    ? td::kRecordLink
                                    : 0
                    );
                    submit(pending, slot, request, group, ticks, mark, line_group);
                    if (dynamic) {  // in looks at its lines from here
                        run_on = true;
                        run_slot = slot;
                        run_sequence = slots.last(slot);
                        run_group = group;
                        run_flat = line_group;
                        run_request = request;
                        hand.word(kTwRunSlot) = slot;
                        hand.word(kTwRunGroup) = group;
                        // across islands with deliveries: in samples its lines for out
                        const bool spans = (
                            link_on && link->deliver && count < (line_group >> 16)
                        );
                        if (link_on) {
                            km::l1(link_base + kTlTold)[kTlSpans] = spans ? run_sequence
                                                                          : 0;
                        }
                        if (spans) {
                            td::Waves& w = link->waves;
                            w.on = 1;
                            w.low = link->find(line_group & 0xFFFFu);
                            w.high = (
                                link->find(
                                    (line_group & 0xFFFFu) + (line_group >> 16) - 1u
                                )
                            );
                            w.leader = w.low == link->island ? 1u : 0u;
                            w.group = line_group;
                            w.sequence = run_sequence;
                            w.id = w.ended = w.round = w.round_number = w.have_last = 0;
                            w.next_round = 0;
                            w.used = km::l1(link_base + kTlTold)[kTlSamples];
                        }
                        asm volatile("" ::: "memory");
                        hand.word(kTwRunSequence) = run_sequence;
                    }
                    if (handler == KM_HANDLER_LOAD) {  // its cores take it in order
                        for (uint32_t c = first_core; c < first_core + count; c++) {
                            holds[c] = static_cast<uint8_t>(field);
                        }
                    }
                    owed++;
                    exit_request = is_exit ? request : 0;
                    pending_request = 0;
                    active = true;
                } else {
                    held++;
                    full_passes += go ? 1 : 0;
                }
            }
        }
        // The dynamic run quiet (in's word): its end command {a0: its sequence} to its
        // group under the RUN's request, freed with no event (the X280's).
        const bool waves = link_on && link->waves.on;
        if (
            run_on && (waves ? waves_turn(now) : hand.word(kTwRunQuiet) == run_sequence)
        ) {
            const uint32_t slot = slots.free_slot();
            if (slot < depth) {
                if (waves) {  // no line of this run is on its way
                    if (link->waves.id != 0) {
                        run_entry(0);
                    }
                    link->waves.on = 0;
                }
                uint32_t end_line[16] = {0};
                end_line[1] = KM_HANDLER_MESSAGES_END;
                end_line[3] = run_sequence;
                submit(
                    end_line,
                    slot,
                    run_request,
                    run_group,
                    deadline,
                    td::kRecordSilent,
                    run_flat
                );
                end_slot = slot + 1;
                run_on = false;
                active = true;
            }
        }
        // A deadline in asks to give up, once no inspect is being read.
        const uint32_t give = hand.word(kTwGiveSequence);
        if (give != give_seen && !inspect.active) {
            give_seen = give;
            const uint32_t slot = hand.word(kTwGiveSlot);
            if (slots.busy(slot) && slots.last(slot) == give) {
                inspect.begin(
                    KMD_INSPECT_TIMEOUT,
                    hand.record(slot)->request,
                    slots.group[slot],
                    slot,
                    give,
                    now
                );
            }
        }
        if ((lines.busy & ~lines.late) && posted - hand.word(kTwTaken) < kTxEventsMax) {
            const uint32_t slot = lines.late_one(now);  // one a pass
            if (slot < KMD_WORK_IN_DEPTH) {
                post_line(
                    slot,
                    KM_EVENT_TIMEOUT,
                    static_cast<uint32_t>(now - lines.submitted[slot])
                );
                active = true;
            }
        }
        if (inspect.active) {
            inspect_turn(now);
            active = true;
        }
        TD_PART(5);
        // The queue's turn, as the X280's: of the first KM_QUEUE_LOOK jobs each with a
        // free slot and room goes; none is passed over more than KM_QUEUE_PASSES times.
        if (queue.length != 0) {
            const uint32_t look = (
                queue.length < KM_QUEUE_LOOK ? queue.length : KM_QUEUE_LOOK
            );
            uint32_t position = 0, smallest = kNoFirst;
            bool blocked = false;
            for (uint32_t examined = 0; examined < look && !blocked; examined++) {
                const uint32_t slot = slots.free_slot();
                if (slot == depth) {
                    break;
                }
                volatile tt_l1_ptr uint32_t* job = queue.job(queue.order[position]);
                const uint32_t count = job[td::kJobCount];
                const uint32_t first = count < smallest ? group_alloc(count) : kNoFirst;
                if (first == kNoFirst) {
                    smallest = count < smallest ? count : smallest;
                    blocked = job[td::kJobPassed] >= KM_QUEUE_PASSES;
                    position++;
                    continue;
                }
                for (uint32_t earlier = 0; earlier < position; earlier++) {
                    volatile tt_l1_ptr uint32_t*
                        passed = (queue.job(queue.order[earlier]) + td::kJobPassed);
                    *passed = *passed + 1;
                    blocked = blocked || *passed >= KM_QUEUE_PASSES;
                }
                job[1] = job[1] & ~(KM_FLAG_QUEUE << 8);
                submit(
                    job,
                    slot,
                    job[td::kJobRequest],
                    first | count << 16,
                    td::wide(job[td::kJobDeadlineLo], job[td::kJobDeadlineHi]),
                    job[td::kJobSource] == td::kSourceEngine ? td::kRecordEngine : 0,
                    flat_of(first | count << 16)
                );  // fmt: skip
                queue.remove(position, before);
                active = true;
            }
        }
        engines.tell(queue.length);  // the engines' count of jobs on their way
        TD_PART(6);
        if (
            active || slots.in_flight != 0 || queue.length != 0 || engines.busy() ||
            (link_on && link->active())
        ) {
            last_activity = now;
        }
        if (!active) {
            idle_passes++;
            td::pace();
        }
        if ((!active || passes % kTdStopLook == 0) && stop_word[0] != 0) {
            error = KMD_ERROR_STOPPED;  // the host's stop (an abort): the grid's too
            break;
        }
        if (
            idle_limit != 0 && pending_request == 0 && now - last_activity > idle_limit
        ) {
            stop_all();
            error = KMD_ERROR_IDLE;
            break;
        }
    }
    finish(error, where, km::wall64());
}
