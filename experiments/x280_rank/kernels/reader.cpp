// An X280 core as a rank, grid side (xr.h): rank cores send to an X280 core's queue,
// take replies from their inbox; a stand-in tells it the run's start and end.
#include <stdint.h>

#include "../xr.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_x280.hpp"

namespace {

    // The probe's own mark: position + 1 (0 is empty), the runtime's before laps.
    inline uint32_t mark(uint32_t position) {
        return position + 1;
    }

    constexpr uint32_t kRoleRank = 0;
    constexpr uint32_t kRoleStandIn = 1;

    // The page of one RUN in L1, kept until the host reads it.
    struct Stats {
        volatile tt_l1_ptr uint32_t* w;

        void begin() {
            for (uint32_t i = 0; i < XR_PAGE_BYTES / 4; i++) {
                w[i] = 0;
            }
            w[XR_ST_MIN] = 0xFFFFFFFFu;
        }

        void time(uint32_t dt) {
            w[XR_ST_COUNT] = w[XR_ST_COUNT] + 1;
            w[XR_ST_MIN] = dt < w[XR_ST_MIN] ? dt : w[XR_ST_MIN];
            w[XR_ST_MAX] = dt > w[XR_ST_MAX] ? dt : w[XR_ST_MAX];
            const uint32_t lo = w[XR_ST_SUM_LO] + dt;
            w[XR_ST_SUM_HI] = w[XR_ST_SUM_HI] + (lo < dt ? 1 : 0);
            w[XR_ST_SUM_LO] = lo;
            uint32_t bin = dt;
            if (dt >= 16) {
                uint32_t e = 4;
                while (e < 31 && (dt >> (e + 1)) != 0) {
                    e++;
                }
                bin = 16 + 8 * (e - 4) + ((dt >> (e - 3)) & 7);
                bin = bin < XR_HIST_BINS - 1 ? bin : XR_HIST_BINS - 1;
            }
            w[XR_ST_HIST + bin] = w[XR_ST_HIST + bin] + 1;
        }

        void stamp(uint32_t word, uint64_t wall) {
            w[word] = static_cast<uint32_t>(wall);
            w[word + 1] = static_cast<uint32_t>(wall >> 32);
        }
    };

}  // namespace

void kernel_main() {
    km::Reader rd;
    km::X280Port port(rd);
    const km::MessagesArgs messages_arguments = km::messages_args(rd);
    const uint32_t role = rd.arg(km::kMessagesArgs + 0);
    const uint32_t stand_in = rd.arg(km::kMessagesArgs + 1);
    const uint32_t inbox = rd.arg(km::kMessagesArgs + 2);
    const uint32_t work = rd.arg(km::kMessagesArgs + 3);
    const uint32_t stats = rd.arg(km::kMessagesArgs + 4);
    const uint32_t me = rd.me();
    const km::Args& arguments = rd.args();
    const uint64_t wait = static_cast<uint64_t>(arguments.wait_limit) << 10;
    volatile tt_l1_ptr uint32_t* messages = rd.control(KM_CONTROL_MESSAGES);
    volatile tt_l1_ptr uint32_t* out = messages + KM_MESSAGES_OUT / 4;
    volatile tt_l1_ptr uint32_t* returned = km::l1(work + XR_WORK_RETURNED);
    const uint64_t status_line = rd.disp_noc(messages_arguments.lines + 16 * me);
    // Kept over the load's runs: the X280 core's queue and this core's inbox head.
    uint32_t queue = 0, queue_x = 0, queue_y = 0, in_head = 0, credited = 0;
    bool hello = false;

    auto flushed_writes = [&] {
        return km::noc_timeout(rd, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        });
    };

    // One 16-byte store of {run, state, sent, consumed} at the dispatcher, landed.
    auto status = [&](uint32_t run, uint32_t state, uint32_t sent, uint32_t consumed) {
        out[0] = run;
        out[1] = state;
        out[2] = sent;
        out[3] = consumed;
        invalidate_l1_cache();  // a fence: the NoC reads the stored line
        noc_async_write(
            arguments.control + KM_CONTROL_MESSAGES + KM_MESSAGES_OUT, status_line, 16
        );
        return flushed_writes();
    };

    // True when the next command of this core in the ring is the end of run `run`.
    auto end_command = [&](uint32_t run) {
        const volatile km_command_t*
            ring = (
                reinterpret_cast<const volatile km_command_t*>(
                    rd.control(KM_CONTROL_RING)
                )
            );
        volatile tt_l1_ptr uint32_t* last = rd.state() + km::kStLast;
        for (uint32_t slot_index = 0; slot_index < arguments.depth; slot_index++) {
            const uint32_t sequence = ring[slot_index].sequence;
            if (!km::newer(sequence, last[slot_index])) {
                continue;
            }
            km_command_t candidate;
            km::copy_command(&ring[slot_index], &candidate);
            if (
                candidate.sequence == sequence && KM_LANDED(&candidate) &&
                km::handler(candidate) == KM_HANDLER_MESSAGES_END &&
                candidate.a0 == run && km::in_group(candidate, me)
            ) {
                return true;
            }
        }
        return false;
    };

    // `count` staged messages to the queue: reserved on the stand-in's tail line, then
    // written, the first mark 0, and committed by a 16-byte write. 0, or where.
    auto send = [&](uint32_t count) -> uint32_t {
        returned[0] = km::kMessagesEnd;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            rd.core_noc(stand_in, work + XR_WORK_TAIL),
            NOC_UNICAST_WRITE_VC,
            count,
            31,
            false,
            false,
            work + XR_WORK_RETURNED
        );
        if (!km::noc_timeout(rd, [] {
            return ncrisc_noc_nonposted_atomics_flushed(noc_index);
        })) {
            return XR_GAVE_ATOMIC;
        }
        if (!km::noc_timeout(rd, [&] {
            invalidate_l1_cache();
            return returned[0] != km::kMessagesEnd;
        })) {
            return XR_GAVE_ATOMIC;
        }
        const uint32_t first_position = returned[0];
        for (uint32_t i = 0; i < count; i++) {
            km::l1(
                work + XR_WORK_STAGE + 64 * i
            )[XR_W_MARK] = (i == 0 ? 0 : mark(first_position + i));
        }
        invalidate_l1_cache();  // a fence: the NoC reads the stored marks
        for (uint32_t i = 0; i < count;) {
            const uint32_t position = (first_position + i) & (XR_QUEUE - 1);
            uint32_t chunk = XR_QUEUE - position;
            chunk = chunk < count - i ? chunk : count - i;
            noc_async_write(
                work + XR_WORK_STAGE + 64 * i,
                get_noc_addr(queue_x, queue_y, queue + 64 * position),
                64 * chunk
            );
            i += chunk;
        }
        if (!flushed_writes()) {
            return XR_GAVE_WRITE;
        }
        km::l1(work + XR_WORK_STAGE)[XR_W_MARK] = mark(first_position);
        invalidate_l1_cache();
        noc_async_write(
            work + XR_WORK_STAGE,
            get_noc_addr(
                queue_x, queue_y, queue + 64 * (first_position & (XR_QUEUE - 1))
            ),
            16
        );
        return flushed_writes() ? 0 : XR_GAVE_WRITE;
    };

    // Stages round trip k of this core in slot `slot`.
    auto stage = [&](uint32_t slot, uint32_t k) {
        volatile tt_l1_ptr uint32_t* line = km::l1(work + XR_WORK_STAGE + 64 * slot);
        const uint32_t value = XR_VALUE(me, k);
        line[XR_W_FROM] = me;
        line[XR_W_INDEX] = k;
        line[XR_W_VALUE] = value;
        line[XR_W_CHECK] = XR_CHECK(me, k, value);
    };

    // The inbox head to the X280 core's credit line for this core: a 64-byte write.
    auto credit = [&] {
        volatile tt_l1_ptr uint32_t* line = km::l1(work + XR_WORK_CREDIT);
        line[0] = in_head;
        invalidate_l1_cache();
        noc_async_write(
            work + XR_WORK_CREDIT,
            get_noc_addr(queue_x, queue_y, queue + XR_CREDITS + 64 * me),
            64
        );
        credited = in_head;
        return flushed_writes();
    };

    // A rank core's run: a0 round trips, a1 in flight, a2 the mode. Returns where it
    // gave up, or 0; the core then reports idle and waits for the run's end.
    auto rank = [&](const km_command_t& command, Stats& st) -> uint32_t {
        const uint32_t run = command.sequence, trips = command.a0;
        const uint32_t window = (
            command.a1 < 1 ? 1 : command.a1 > XR_INBOX ? XR_INBOX : command.a1
        );
        uint32_t sent = 0, consumed = 0, done = 0, started = 0, where = 0;
        bool idle_shown = false;
        uint64_t since = km::wall64(), last_input = since;
        uint32_t sent_at[XR_INBOX];
        if (!hello) {
            km::X280Msg message;
            if (!port.wait_told(message) || message.w[0] != XR_TELL_HELLO) {
                return XR_GAVE_HELLO;
            }
            queue = message.w[1];
            queue_x = message.w[2] & 0xFFFFu;
            queue_y = message.w[2] >> 16;
            hello = true;
        }
        st.stamp(XR_ST_FIRST_LO, km::wall64());
        if (command.a2 == XR_MODE_ASK) {
            km::X280Msg message;
            for (uint32_t k = 1; k <= trips; k++) {
                const uint32_t t0 = km::wall();
                if (!port.call(k, me, 0, message)) {
                    return XR_GAVE_ASK;
                }
                st.time(km::wall() - t0);
                st.w[XR_ST_BAD] += message.w[0] != XR_ANSWER(k, me) ? 1 : 0;
            }
            done = trips;
        } else {
            const uint32_t first = trips < window ? trips : window;
            for (; started < first; started++) {
                stage(started, started);
                sent_at[started % window] = km::wall();
            }
            if (first != 0) {
                const uint32_t t0 = km::wall();
                where = send(first);
                st.w[XR_ST_SEND_CYCLES] += km::wall() - t0;
                st.w[XR_ST_SENDS] += 1;
                sent += where == 0 ? first : 0;
            }
        }
        st.w[XR_ST_RUN] = run;
        while (where == 0) {
            invalidate_l1_cache();
            if (rd.stop()) {
                where = XR_GAVE_STOPPED;
                break;
            }
            uint32_t valid = 0;
            while (
                valid < XR_INBOX &&
                km::l1(inbox + 64 * ((in_head + valid) & (XR_INBOX - 1)))[XR_W_MARK] ==
                    mark(in_head + valid)
            ) {
                valid++;
            }
            const uint64_t now = km::wall64();
            if (valid == 0) {
                if (done < trips && now - last_input > wait) {
                    where = XR_GAVE_REPLY;
                    break;
                }
                if (!idle_shown && now - since >= messages_arguments.idle) {
                    if (!status(run, KM_MESSAGES_IDLE, sent, consumed)) {
                        where = XR_GAVE_LINE;
                        break;
                    }
                    idle_shown = true;
                    st.w[XR_ST_IDLES] += 1;
                }
                if (idle_shown && end_command(run)) {
                    break;
                }
                continue;
            }
            if (idle_shown) {  // busy has landed before the first message is taken
                if (!status(run, KM_MESSAGES_BUSY, 0, 0)) {
                    where = XR_GAVE_LINE;
                    break;
                }
                idle_shown = false;
            }
            uint32_t count = 0;
            for (uint32_t i = 0; i < valid; i++) {
                volatile tt_l1_ptr uint32_t* reply = (
                    km::l1(inbox + 64 * ((in_head + i) & (XR_INBOX - 1)))
                );
                const uint32_t from = reply[XR_W_FROM], k = reply[XR_W_INDEX];
                const uint32_t value = reply[XR_W_VALUE];
                if (done >= trips) {
                    st.w[XR_ST_EXTRA] += 1;
                } else {
                    const bool good = (
                        k == done && value == XR_REPLY(XR_VALUE(me, k), k) &&
                        reply[XR_W_CHECK] == XR_CHECK(from, k, value)
                    );
                    st.w[XR_ST_BAD] += good ? 0 : 1;
                    st.time(km::wall() - sent_at[done % window]);
                    done++;
                }
                consumed++;
                if (started < trips) {
                    stage(count++, started);
                    sent_at[started % window] = km::wall();
                    started++;
                }
            }
            in_head += valid;
            if (count != 0) {
                const uint32_t t0 = km::wall();
                where = send(count);
                st.w[XR_ST_SEND_CYCLES] += km::wall() - t0;
                st.w[XR_ST_SENDS] += 1;
                sent += where == 0 ? count : 0;
            }
            if (where == 0 && in_head - credited >= XR_INBOX / 4) {
                where = credit() ? 0 : XR_GAVE_WRITE;
                st.w[XR_ST_CREDITS] += 1;
            }
            if (done == trips) {
                st.stamp(XR_ST_LAST_LO, km::wall64());
            }
            since = last_input = km::wall64();
        }
        st.w[XR_ST_SENT] = sent;
        st.w[XR_ST_CONSUMED] = consumed;
        st.w[XR_ST_DONE] = done;
        if (command.a2 == XR_MODE_ASK && where == 0) {
            st.stamp(XR_ST_LAST_LO, km::wall64());
        }
        return where;
    };

    rd.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == XR_HANDLER_FINISH) {
            if (role == kRoleStandIn) {
                port.tell(XR_TELL_FINISH, 0, 0);
            }
            return;
        }
        if (handler != messages_arguments.handler) {
            return;
        }
        if (role == kRoleStandIn) {  // its X280 core writes this core's status line
            port.tell(XR_TELL_START, command.sequence, command.a2);
            while (!end_command(command.sequence) && !rd.stop()) {
                invalidate_l1_cache();
            }
            port.tell(XR_TELL_END, command.sequence, 0);
            return;
        }
        if (role != kRoleRank) {
            return;
        }
        Stats st{km::l1(stats + XR_PAGE_BYTES * (command.a3 % XR_PAGES))};
        st.begin();
        const uint32_t where = rank(command, st);
        if (where != 0) {
            st.w[XR_ST_GAVE] = where;
            km::messages_gave_up(arguments.control, where, in_head, command.sequence);
            status(command.sequence, KM_MESSAGES_GAVE_UP, in_head, where);
            rd.gave_up(
                km::kStReader,
                km::kGaveStop,
                km::kWhereData,
                km::slot(command),
                command.sequence
            );
        }
    });
}
