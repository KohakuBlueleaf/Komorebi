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
#include "td_in.hpp"
#include "td_inspect.hpp"
#include "td_reset.hpp"
#include "td_queue.hpp"
#include "td_slots.hpp"
#include "td_sources.hpp"

namespace {

    using td::arg;

    constexpr uint32_t kNoFirst = 0xFFFFFFFFu;
    constexpr uint32_t kBoundStep = 8;  // the X280's KMD_BOUND_STEP

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
    const uint32_t host_depth = arg(kTdHostDepth), event_depth = arg(kTdEventDepth);
    const uint64_t area = td::pcie() | arg(kTdArea);
    const uint32_t depth = td::depth_arg();
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kTdBootLimit)) << 10;
    const uint64_t idle_limit = static_cast<uint64_t>(arg(kTdIdleLimit)) << 10;
    const uint64_t ticks_us = arg(kTdTicksUs);          // times 65536
    const uint64_t bound_hold = (20 * ticks_us) >> 16;  // 20 us, the X280's hold
    uint64_t deadline = static_cast<uint64_t>(arg(kTdDeadline)) << 10;
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* stop_word = km::l1(control + KM_CONTROL_STOP);
    volatile tt_l1_ptr uint32_t* beat = km::l1(block + kTbWatch + 4 * kTkOut);
    volatile tt_l1_ptr uint32_t* bound_word = km::l1(control + KM_CONTROL_BOUND);
    volatile tt_l1_ptr uint32_t* read_line = km::l1(block + kTbRead);
    volatile tt_l1_ptr uint32_t* words = km::l1(block + kTbWords);
    uint32_t pending[16];  // the command taken and not yet submitted
    volatile tt_l1_ptr uint32_t* result = km::l1(block + kTbResult);
    td::Hand hand;
    hand.base = block + kTbHand;
    auto core = [&](uint32_t worker, uint32_t address) {
        return get_noc_addr(cores[worker] & 0xFFFFu, cores[worker] >> 16, address);
    };

    // Each core's program field (a run whose groups switch): a command goes only to
    // cores that hold its program, a LOAD moves them to its own (the X280's table).
    const bool table = arg(kTdPrograms) != 0;
    uint8_t holds[KM_CORES_MAX];
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
    td::OutSlots slots;
    slots.init(depth, hand);
    td::Queue queue;
    queue.base = block + kTbQueue;
    td::Source sources[td::kSources];
    sources[td::kSourceHost].init(km::l1(block + kTbRing), host_depth);
    td::Source& host = sources[td::kSourceHost];
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
    // An event of no slot (a rejection, a PONG), for in to write.
    auto post = [&](uint32_t request, uint32_t status, uint32_t ticks) {
        volatile tt_l1_ptr uint32_t* event = hand.event(posted);
        event[0] = request;
        event[1] = status;
        event[2] = ticks;
        asm volatile("" ::: "memory");
        hand.word(kTwPosted) = ++posted;
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
    // The lowest first core of `count` cores that overlap no busy slot's group.
    auto group_alloc = [&](uint32_t count) {
        uint32_t first = 0;
        while (first <= ranks && count <= ranks - first) {
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
        uint32_t silent = 0
    ) {  // fmt: skip
        do {                                  // never 0, never the slot's last
            sequence++;
        } while (sequence == 0 || sequence == slots.last(slot));
        // Straight into the slot's line, word by word (the stack is in L1 too): the
        // landing words of km_command_t, its slot and its group.
        const uint32_t at = control + KM_CONTROL_RING + slot * 64;
        volatile tt_l1_ptr uint32_t* to = km::l1(at);
        const uint32_t header = (line[1] & 0xFF00FFFFu) | slot << 16;
        // a program's command may be given up at its deadline (inspect_may_give_up)
        const uint32_t handler = line[1] & 0xFFu;
        const bool give_up = (
            silent == 0 && handler <= KM_HANDLER_USER_MAX &&
            !KM_IS_DYNAMIC(messages, handler)
        );
        to[0] = sequence;
        to[1] = header;
        to[2] = group;
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
        // Take: the next line once it has landed, its event fits the ring with every
        // event owed and one more, and the posted events have room.
        if (pending_request == 0) {
            bool room = caused - read_count + 1 <= event_depth;
            if (!room) {
                read_count = read_line[0];
                room = caused - read_count + 1 <= event_depth;
            }
            room = room && posted - hand.word(kTwTaken) < kTxEventsMax;
            if (host.landed()) {
                if (room) {
                    pending_request = host.take(pending);
                    consumed = pending_request;
                    hand.word(kTwConsumed) = consumed;
                    before = queue.used;
                    active = true;
                } else {
                    no_room++;
                }
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
            const uint32_t group = is_exit ? KM_GROUP_ALL(ranks) : pending[2];
            const uint32_t first_core = group & 0xFFFFu, count = group >> 16;
            const uint32_t field = pending[1] >> 24;  // the command's program
            uint32_t reason = 0;
            if (handler == KM_HANDLER_STOP) {
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
            if (handler == KM_HANDLER_DEADLINE && argument != 0) {
                // the deadline of the commands submitted from here on, in microseconds
                deadline = (argument * ticks_us) >> 16;
                pending_request = 0;
                continue;
            }
            // the host's inspect (run.inspect) of a group inside the cores
            const bool own_inspect = (
                (flags & KMD_FLAG_DISPATCHER) != 0 && handler == KMD_HANDLER_INSPECT &&
                count != 0 && first_core < ranks && count <= ranks - first_core
            );
            // the host's reset commands: keep the lists (group count 0), reset a core
            const bool own_reset = (
                (flags & KMD_FLAG_DISPATCHER) != 0 && handler == KMD_HANDLER_RESET &&
                (count == KMD_RESET_KEEP || (count == 1 && first_core < ranks))
            );
#ifdef KMD_STALL_DEBUG
            // test builds: PONG, then no pass for a0 ms (the watch's stall)
            const bool own_stall = (
                (flags & KMD_FLAG_DISPATCHER) != 0 && handler == KMD_HANDLER_STALL
            );
#else
            const bool own_stall = false;
#endif
            // LOAD (run.load) and ENTRY (run.addon) pass; 247 to 251 are not served
            if (own_inspect || own_reset || own_stall) {
                // served below
            } else if (handler == KM_HANDLER_DEADLINE) {
                reason = KMD_REJECT_DEADLINE;
            } else if (handler > KM_HANDLER_ENTRY && handler < KM_HANDLER_DEADLINE) {
                reason = KMD_REJECT_HANDLER;
            } else if (flags & (1u | KM_FLAG_PART | KMD_FLAG_DISPATCHER)) {
                reason = KMD_REJECT_FLAGS;
            } else if (exit_request != 0) {
                reason = KMD_REJECT_EXITED;
            } else if (
                handler != KM_HANDLER_PING &&
                (
                    queued
                        ? count == 0 || count > ranks
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
                        handler != KM_HANDLER_PING &&
                        !hold(field, queued ? 0 : first_core, queued ? ranks : count)
            ) {
                reason = KMD_REJECT_PROGRAM;  // a queued command: every core holds it
            } else if (queued && queue.length == KM_QUEUE_JOBS) {
                reason = KMD_REJECT_QUEUE;
            }
            if (reason != 0) {
                post(request, KM_EVENT_REJECTED | reason << 8, 0);
                caused++;
                pending_request = 0;
                rejected++;
            } else if (own_stall) {
                post(request, KM_EVENT_PONG, 0);
                caused++;
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
                caused++;
                pending_request = 0;
            } else if (own_inspect) {
                if (!inspect.active) {  // one at a time: its PONG once the last is read
                    inspect.begin(KMD_INSPECT_HOST, request, group, 0, sequence, now);
                    caused++;
                    pending_request = 0;
                }
            } else if (handler == KM_HANDLER_PING) {
                post(request, KM_EVENT_PONG, 0);
                caused++;
                pending_request = 0;
            } else if (queued) {
                queue.put(pending, request, count, deadline);
                caused++;
                pending_request = 0;
            } else {
                bool go = !(is_exit && (slots.in_flight != 0 || queue.length != 0));
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
                    submit(pending, slot, request, group, deadline);
                    if (dynamic) {  // in looks at its lines from here
                        run_on = true;
                        run_slot = slot;
                        run_sequence = slots.last(slot);
                        run_group = group;
                        run_request = request;
                        hand.word(kTwRunSlot) = slot;
                        hand.word(kTwRunGroup) = group;
                        asm volatile("" ::: "memory");
                        hand.word(kTwRunSequence) = run_sequence;
                    }
                    if (handler == KM_HANDLER_LOAD) {  // its cores take it in order
                        for (uint32_t c = first_core; c < first_core + count; c++) {
                            holds[c] = static_cast<uint8_t>(field);
                        }
                    }
                    caused++;
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
        if (run_on && hand.word(kTwRunQuiet) == run_sequence) {
            const uint32_t slot = slots.free_slot();
            if (slot < depth) {
                uint32_t end_line[16] = {0};
                end_line[1] = KM_HANDLER_MESSAGES_END;
                end_line[3] = run_sequence;
                submit(
                    end_line, slot, run_request, run_group, deadline, td::kRecordSilent
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
                    td::wide(job[td::kJobDeadlineLo], job[td::kJobDeadlineHi])
                );  // fmt: skip
                queue.remove(position, before);
                active = true;
            }
        }
        TD_PART(6);
        if (active || slots.in_flight != 0 || queue.length != 0) {
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
