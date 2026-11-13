// Test feeder: one Tensix core outside every group plays the dispatcher (commands from
// DRAM, unicast to every worker, acks in its own L1, stop flag on a missed deadline).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_core.hpp"
#include "feeder.h"

namespace {

    uint32_t arg(uint32_t i) {
        return get_arg_val<uint32_t>(i);
    }

}  // namespace

void kernel_main() {
    const uint32_t control = arg(kFdControl);
    const uint32_t workers = arg(kFdWorkers), n = arg(kFdCommands);
    uint32_t depth = arg(kFdDepth);
    depth = depth == 0 || depth > KM_RING_SLOTS ? KM_RING_SLOTS : depth;
    const uint64_t deadline = static_cast<uint64_t>(arg(kFdDeadline)) << 10;
    const uint64_t boot_limit = static_cast<uint64_t>(arg(kFdBootLimit)) << 10;
    const uint32_t tab = get_write_ptr(kFdCbTable), res = get_write_ptr(kFdCbResults);
    volatile tt_l1_ptr uint32_t* cores = km::l1(control + KM_CONTROL_CORES);
    volatile tt_l1_ptr uint32_t* ack = km::l1(
        control + KM_CONTROL_DONE
    );  // ack word j: ack[j * 4]
    volatile tt_l1_ptr uint32_t* out = km::l1(res);
    auto row = [&](uint32_t r) {
        return out + r * 16;
    };
    auto core = [&](uint32_t w, uint32_t addr) {
        return get_noc_addr(cores[w] & 0xFFFFu, cores[w] >> 16, addr);
    };

    // Core table and commands into L1; results cleared.
    const InterleavedAddrGen<true> core_tab = {
        .bank_base_address = arg(kFdCores), .page_size = KM_CORES_MAX * 4
    };
    noc_async_read(
        get_noc_addr(0, core_tab), control + KM_CONTROL_CORES, KM_CORES_MAX * 4
    );
    const InterleavedAddrGen<true> rec_tab = {
        .bank_base_address = arg(kFdTable), .page_size = 64
    };
    for (uint32_t p = 0; p < n; p++) {
        noc_async_read(get_noc_addr(p, rec_tab), tab + p * 64, 64);
        if ((p & 63) == 63) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t w = 0; w < (n + 2) * 16; w++) {
        out[w] = 0;
    }
    row(0)[kFrMagic] = kFeederMagic;

    // Ends the run: commands `status` and writes 1 to the stop flag of every worker.
    auto stop_all = [&](uint32_t status) {
        row(0)[kFrStatus] = status;
        *km::out_slot(control, 0) = 1;
        for (uint32_t w = 0; w < workers; w++) {
            km::small_write<0, 4>(control, core(w, control + KM_CONTROL_STOP), 0);
        }
        noc_async_write_barrier();
    };

    // Start-up: each group's READY (KM_CONTROL_BOOT) and the hello {1, control}
    // (KM_CONTROL_JOB). Ack words are zero from the host, only the grid writes them.
    const uint32_t groups = arg(kFdGroups);
    volatile tt_l1_ptr uint32_t* ready_words = km::l1(control + KM_CONTROL_BOOT);
    volatile tt_l1_ptr uint32_t* mailbox = km::l1(control + KM_CONTROL_JOB);
    const uint64_t t_boot = km::wall64();
    uint32_t ready = 0;
    bool booted = false;
    while (km::wall64() - t_boot < boot_limit) {
        invalidate_l1_cache();
        ready = 0;
        for (uint32_t g = 0; g < groups; g++) {
            ready |= (ready_words[g * 4] == KM_READY) << g;
        }
        if (ready == (1u << groups) - 1 && mailbox[0] == 1 && mailbox[1] == control) {
            booted = true;
            break;
        }
    }
    row(0)[kFrBoot] = static_cast<uint32_t>((km::wall64() - t_boot) >> 4);
    row(0)[kFrReady] = ready;

    int32_t busy[KM_RING_SLOTS];
    uint32_t busy_sequence[KM_RING_SLOTS];
    uint32_t busy_group[KM_RING_SLOTS];
    bool busy_result[KM_RING_SLOTS];
    uint64_t t_submit[KM_RING_SLOTS];
    for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
        busy[j] = -1;
    }
    uint32_t sequence = 0, next = 0, done = 0, in_flight = 0, max_busy = 0;
    bool exited = false;  // an EXIT command of the table was submitted: nothing after it
    // The dynamic run in flight (one at a time): its table index, sequence and group.
    const uint32_t messages_handler = arg(kFdMessagesHandler);
    int32_t messages_index = -1;
    uint32_t messages_sequence = 0, messages_group = 0;
    bool messages_ended = true;
    auto busy_slot_of = [&](int32_t index) {
        uint32_t j = 0;
        while (j < depth && busy[j] != index) {
            j++;
        }
        return j;
    };
    // The two-pass check of the group's status lines: 1 quiet (every line idle and
    // the same in both passes, sent equal to consumed), 0 not, -(1 + core) gave up.
    volatile tt_l1_ptr uint32_t* lines = km::l1(arg(kFdLines));
    volatile tt_l1_ptr uint32_t* first_pass = lines + KM_CORES_MAX * 4;
    auto quiet = [&](volatile tt_l1_ptr uint32_t* dr) -> int32_t {
        dr[kFrRecMessagesChecks] = dr[kFrRecMessagesChecks] + 1;
        const uint32_t f = messages_group & 0xFFFFu, end = f + (messages_group >> 16);
        invalidate_l1_cache();
        bool idle = true;
        for (uint32_t k = f; k < end; k++) {
            for (uint32_t w = 0; w < 4; w++) {
                first_pass[k * 4 + w] = lines[k * 4 + w];
            }
            const uint32_t state = first_pass[k * 4 + KM_MESSAGES_LINE_STATE];
            if (first_pass[k * 4 + KM_MESSAGES_LINE_RUN] != messages_sequence) {
                idle = false;
            } else if (state == KM_MESSAGES_GAVE_UP) {
                return -static_cast<int32_t>(k + 1);
            } else if (state != KM_MESSAGES_IDLE) {
                idle = false;
            }
        }
        if (!idle) {
            return 0;
        }
        invalidate_l1_cache();
        uint64_t sent = 0, consumed = 0;
        for (uint32_t k = f; k < end; k++) {
            for (uint32_t w = 0; w < 4; w++) {
                if (lines[k * 4 + w] != first_pass[k * 4 + w]) {
                    return 0;
                }
            }
            sent += first_pass[k * 4 + KM_MESSAGES_LINE_SENT];
            consumed += first_pass[k * 4 + KM_MESSAGES_LINE_CONSUMED];
        }
        dr[kFrRecMessagesSent] = static_cast<uint32_t>(sent);
        return sent == consumed ? 1 : 0;
    };
    uint64_t t_first = 0, t_last = 0, submit_cycles = 0;

    // Submits command `src` (64 bytes in L1) to slot j of every worker.
    auto submit = [&](uint32_t j, uint32_t src, bool raw) {
        volatile tt_l1_ptr km_command_t*
            st = (
                reinterpret_cast<volatile tt_l1_ptr km_command_t*>(
                    control + KM_CONTROL_RING + j * 64
                )
            );
        km::copy_command(reinterpret_cast<volatile tt_l1_ptr km_command_t*>(src), st);
        sequence++;
        st->sequence = st->land1 = st->land2 = st->land3 = sequence;
        if (!raw) {  // a raw command keeps the slot it names (test only)
            st->hdr = (st->hdr & 0xFFFFu) | j << 16;
        }
        const uint64_t t0 = km::wall64();
        if (arg(kFdRelease) != 0) {  // one release time for every core (test only)
            st->a9 = static_cast<uint32_t>(t0) + arg(kFdRelease);
        }
        for (uint32_t w = 0; w < workers; w++) {
            noc_async_write(
                control + KM_CONTROL_RING + j * 64,
                core(w, control + KM_CONTROL_RING + j * 64),
                64
            );
        }
        noc_async_writes_flushed();
        const uint64_t t1 = km::wall64();
        submit_cycles += t1 - t0;
        return t0;
    };

    if (!booted) {
        stop_all(kFsBoot);
    } else {
        row(0)[kFrStatus] = kFsDone;
        bool failed = false;
        while (!failed && (exited || done >= n ? in_flight != 0 : true)) {
            invalidate_l1_cache();
            const uint64_t now = km::wall64();
            for (uint32_t j = 0; j < depth; j++) {
                if (busy[j] < 0) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* r = row(1 + busy[j]);
                if (ack[j * 4] == KM_ACK(busy_sequence[j])) {
                    if (static_cast<uint32_t>(busy[j]) == n) {  // a dynamic run's end
                        row(
                            1 + messages_index
                        )[kFrRecMessagesEndAck] = static_cast<uint32_t>(now);
                        busy[j] = -1;
                        in_flight--;
                        continue;
                    }
                    r[kFrRecAcked] = static_cast<uint32_t>(now);
                    if (busy_result[j]) {  // one 8-byte write with the ack word
                        r[kFrRecResult] = ack[j * 4 + 1];
                    }
                    r[kFrRecStatus] = kFsDone;
                    t_last = now;
                    busy[j] = -1;
                    in_flight--;
                    done++;
                } else if (now - t_submit[j] > deadline) {
                    r[kFrRecStatus] = kFsTimeout;
                    row(0)[kFrSlot] = j;
                    row(0)[kFrSequence] = busy_sequence[j];
                    row(0)[kFrIndex] = busy[j];
                    stop_all(kFsTimeout);
                    failed = true;
                    break;
                }
            }
            // A dynamic run in flight: its end command goes out once the lines are
            // quiet.
            if (!failed && messages_index >= 0 && !messages_ended) {
                volatile tt_l1_ptr uint32_t* dr = row(1 + messages_index);
                const int32_t q = quiet(dr);
                if (q < 0) {
                    dr[kFrRecMessagesGave] = static_cast<uint32_t>(-q);
                    row(0)[kFrSlot] = busy_slot_of(messages_index);
                    row(0)[kFrSequence] = messages_sequence;
                    row(0)[kFrIndex] = messages_index;
                    stop_all(kFsMessagesGave);
                    failed = true;
                    continue;
                }
                uint32_t j = 0;
                while (j < depth && busy[j] >= 0) {
                    j++;
                }
                if (q > 0 && j < depth) {
                    dr[kFrRecMessagesEnd] = km::wall();
                    volatile tt_l1_ptr uint32_t* e = km::l1(tab + (n + 1) * 64);
                    for (uint32_t w = 0; w < 16; w++) {
                        e[w] = 0;
                    }
                    e[1] = km::make_hdr(KM_HANDLER_MESSAGES_END, 0, 0);
                    e[2] = messages_group;
                    e[3] = messages_sequence;  // a0: the run it ends
                    t_submit[j] = submit(j, tab + (n + 1) * 64, false);
                    busy[j] = static_cast<int32_t>(n);
                    busy_sequence[j] = sequence;
                    busy_group[j] = messages_group;
                    in_flight++;
                    dr[kFrRecMessagesEndSequence] = sequence;
                    messages_ended = true;
                }
            }
            if (failed || next >= n || exited) {
                continue;
            }
            const uint32_t src = tab + next * 64;
            volatile tt_l1_ptr uint32_t* r = row(1 + next);
            volatile tt_l1_ptr uint32_t* w = km::l1(src);
            const bool is_exit = (w[1] & 0xFFu) == KM_HANDLER_EXIT;
            const bool raw = arg(kFdRaw) == next + 1;
            if (is_exit) {  // to every worker, and only when no slot is busy
                w[2] = raw ? w[2] : KM_GROUP_ALL(workers);
                if (in_flight != 0) {
                    r[kFrRecHeld] = 1;
                    continue;
                }
            }
            if (
                (w[1] >> 8) & KM_FLAG_EXCLUSIVE
            ) {  // held while an overlapping group is busy
                bool held = false;
                for (uint32_t k = 0; k < depth; k++) {
                    held |= busy[k] >= 0 && km::groups_overlap(busy_group[k], w[2]);
                }
                if (held) {
                    r[kFrRecHeld] = 1;
                    continue;
                }
            }
            uint32_t j = 0;
            while (j < depth && busy[j] >= 0) {
                j++;
            }
            if (j == depth) {
                continue;
            }
            const uint64_t t0 = submit(j, src, raw);
            t_first = next == 0 ? t0 : t_first;
            r[kFrRecSequence] = sequence;
            r[kFrRecSlot] = j;
            r[kFrCmdSubmit] = static_cast<uint32_t>(t0);
            r[kFrCmdSubmitted] = km::wall();
            busy[j] = static_cast<int32_t>(next);
            busy_sequence[j] = sequence;
            busy_group[j] = w[2];
            busy_result[j] = (w[1] >> 8) & KM_FLAG_RESULT;
            t_submit[j] = t0;
            in_flight++;
            max_busy = in_flight > max_busy ? in_flight : max_busy;
            r[kFrRecBusy] = in_flight;
            if (
                messages_handler != 0 && (w[1] & 0xFFu) == messages_handler
            ) {  // a dynamic run starts
                messages_index = static_cast<int32_t>(next);
                messages_sequence = sequence;
                messages_group = w[2];
                messages_ended = false;
            }
            next++;
            exited = is_exit;
        }
        if (exited && !failed) {
            row(0)[kFrExitAcked] = row(next)[kFrRecStatus] == kFsDone;
        } else if (!failed && arg(kFdExit) != 0) {
            volatile tt_l1_ptr uint32_t* x = km::l1(tab + n * 64);  // spare command
            for (uint32_t w = 0; w < 16; w++) {
                x[w] = 0;
            }
            x[1] = km::make_hdr(KM_HANDLER_EXIT, 0, 0);
            x[2] = KM_GROUP_ALL(workers);
            const uint64_t t0 = submit(0, tab + n * 64, false);  // every slot is free
            for (;;) {  // EXIT completes like any command
                invalidate_l1_cache();
                if (ack[0] == KM_ACK(sequence)) {
                    row(0)[kFrExitAcked] = 1;
                    break;
                }
                if (km::wall64() - t0 > deadline) {
                    row(0)[kFrSlot] = 0;
                    row(0)[kFrSequence] = sequence;
                    row(0)[kFrIndex] = n;
                    stop_all(kFsTimeout);
                    break;
                }
            }
        }
    }

    row(0)[kFrSubmitted] = next;
    row(0)[kFrDone] = done;
    row(0)[kFrFirst] = static_cast<uint32_t>(t_first);
    row(0)[kFrLast] = static_cast<uint32_t>(t_last);
    const uint64_t span = done != 0 ? t_last - t_first : 0;  // 0: nothing acknowledged
    row(0)[kFrSpanLo] = static_cast<uint32_t>(span);
    row(0)[kFrSpanHi] = static_cast<uint32_t>(span >> 32);
    row(0)[kFrSubmitCycles] = static_cast<uint32_t>(submit_cycles);
    row(0)[kFrMaxBusy] = max_busy;
    const InterleavedAddrGen<true> res_tab = {
        .bank_base_address = arg(kFdResults), .page_size = 64
    };
    for (uint32_t p = 0; p <= n + 1; p++) {
        noc_async_write(res + p * 64, get_noc_addr(p, res_tab), 64);
    }
    noc_async_write_barrier();
}
