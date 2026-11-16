// A dynamic run's messages to a rank of another island: the writer hands a bin to the
// link service (op 5); that island's link core reserves, writes and commits it there.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_island.hpp"
#include "km_messages.hpp"

#if !defined(COMPILE_FOR_TRISC) && defined(KM_PAGE_LINKS)
#include "km_far.hpp"
#include "km_pages.hpp"

namespace km {

    // Link core 0 of this island, its run entry (KM_DELIVER_RUN) read into `scratch`
    // (16 bytes): true when it says open (its id and ~id agree). kGave* in `code`.
    template <typename Kernel>
    inline bool deliver_entry(const Kernel& kernel, uint32_t scratch, uint32_t& code) {
        const Islands view = islands_at(kernel.args().control);
        const uint32_t home = view.base() + view.ranks();
        noc_async_read(
            kernel.core_noc(home, view.link_area() + KM_DELIVER_RUN), scratch, 16
        );
        code = kernel.poll_at(
            kWhereFar,
            [] {
                return ncrisc_noc_reads_flushed(noc_index);
            },
            kernel.args().wait_limit
        );
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* entry = l1(scratch);
        return code == kGaveNone && entry[0] != 0 && entry[3] == ~entry[0];
    }

    // Where a writer's bins for another island's ranks go: a delivery line each, done
    // (in that rank's queue) when it returns. The run's id, from this island's entry.
    struct DeliverOut {
        uint32_t id = 0;

        bool takes(uint32_t destination) const {
            return (destination & kMessagesAway) != 0;
        }

        template <typename WriterType>
        uint32_t put(
            WriterType& writer,
            const km_command_t& command,
            uint32_t destination,
            uint32_t base,
            uint32_t stride,
            uint32_t count,
            uint32_t body = 0,
            uint32_t body_bytes = 0
        ) {
            const uint32_t control = writer.args().control;
            const uint32_t scratch = control + KM_CONTROL_TAILS + KM_TAILS_WRITER;
            const uint64_t end = (
                wall64() + (static_cast<uint64_t>(writer.args().wait_limit) << 10)
            );
            while (id == 0) {  // the dispatcher opens it once the leader's id is in
                uint32_t code = kGaveNone;
                if (
                    deliver_entry(writer, scratch, code) &&
                    l1(scratch)[1] == command.sequence
                ) {
                    id = l1(scratch)[0];
                } else if (code != kGaveNone || writer.stop() || wall64() > end) {
                    return kWhereFar;
                }
            }
            const uint32_t flat = destination & ~kMessagesAway;
            const FarLine far = {
                KM_ASK_OP_DELIVER << KM_ASK_OP_SHIFT,
                flat,
                islands_at(control).island_of(flat),
                stride,
                count | body_bytes << 8,
                body,
                base,
                id
            };
            return far_line(writer, far, true) == kGaveNone ? 0 : kWhereFar;
        }
    };

#if defined(KM_DELIVER)
    constexpr uint32_t kDeliverIslands = KM_DELIVER_ISLANDS;  // the session's
    constexpr uint32_t kDeliverSlots = KM_DELIVER_SLOTS;
    constexpr uint32_t kDeliverSlotBytes = KM_DELIVER_SLOT_BYTES;
    constexpr uint32_t kDeliverBodyAt = KM_DELIVER_BODY_AT;  // a slot's body, after 32

    // The delivery places on a link core: after the put queue, far places and rooms.
    inline uint32_t deliver_places() {
        return link_queue() + KM_QUEUE_BYTES + kFarBytes + kForwardBytes;
    }

    constexpr uint32_t deliver_bytes() {
        return KM_DELIVER_BYTES(kDeliverIslands, kDeliverSlots, kDeliverSlotBytes);
    }

    // The places' words zeroed (link core 0, before its hello: no line comes before).
    inline void deliver_zero(uint32_t area) {
        constexpr uint32_t n = kDeliverIslands, s = kDeliverSlots;
        const uint32_t places = deliver_places();
        for (uint32_t w = 0; w < KM_DELIVER_LINE(n, s) / 4; w++) {  // and the scratch
            l1(places)[w] = 0;
        }
        for (uint32_t i = 0; i < n * s; i++) {
            l1(places + KM_DELIVER_LINE(n, s) + 64 * i)[KM_ASK_NUMBER] = 0;
        }
        for (uint32_t w = 0; w < 4; w++) {
            l1(area + KM_DELIVER_RUN)[w] = 0;
        }
    }

    // A direct line to another island's rank (far_write: each piece landed), its slot
    // at `slot` there: the body, the line with mark 0, its first 16 bytes with `mark`.
    template <typename Kernel>
    uint32_t direct_far_line(
        const Kernel& kernel,
        uint32_t rank,
        uint32_t slot,
        uint32_t source,
        uint32_t message_bytes,
        uint32_t body,
        uint32_t mark
    ) {
        uint32_t island = 0, code = kGaveNone;
        far_core(kernel, rank, island);
        const uint32_t body_bytes = body & kMessagesOutBodyBytes;
        if (body_bytes != 0) {
            const uint32_t from = (body >> kMessagesOutBodyShift) << 6;
            const uint32_t lines = messages_body_lines(body_bytes);
            code = (
                far_write(kernel, rank, island, slot + message_bytes, from, lines, -1)
            );
        }
        *message_mark_half(source) = 0;
        invalidate_l1_cache();  // a fence: the NoC reads the stored words
        if (code == kGaveNone) {
            code = far_write(kernel, rank, island, slot, source, message_bytes, -1);
        }
        *message_mark_half(source) = mark;
        invalidate_l1_cache();
        if (code == kGaveNone) {
            code = far_write(kernel, rank, island, slot, source, 16, -1);
        }
        return code;
    }

    // A direct slot given back to another island's sender: 1 added to word 0 of its
    // credit line at `line` there, posted (that sender's adds land in order).
    template <typename Kernel>
    uint32_t direct_far_credit(const Kernel& kernel, uint32_t rank, uint32_t line) {
        uint32_t island = 0;
        far_core(kernel, rank, island);
        const FarLine far = {
            KM_ASK_OP_ADD << KM_ASK_OP_SHIFT, rank, island, 1, 0, line, 0, 0
        };
        return far_line(kernel, far, false);
    }

    // The largest piece of `units` units of `unit` bytes, a whole number of units that
    // divides them and fits a packet: the pieces cover them exactly.
    inline uint32_t deliver_piece(uint32_t units, uint32_t unit) {
        const uint32_t packet = pages_packet() / 16 * 16;
        uint32_t k = unit != 0 && packet / unit < units ? packet / unit : units;
        while (k > 1 && units % k != 0) {
            k--;
        }
        return k * unit;
    }

    // The lead, a delivery line: its messages and body into a slot of the rank's island
    // (a credit of its places), then the line. `sent`: a slot was free. kGave* code.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t deliver_send(
        const Kernel& kernel, uint32_t cpu, uint32_t area, uint32_t at, bool& sent
    ) {
        constexpr uint32_t n = kDeliverIslands, s = kDeliverSlots;
        const Islands view = islands_at(kernel.args().control);
        volatile tt_l1_ptr uint32_t* line = l1(at);
        const uint32_t places = deliver_places();
        volatile tt_l1_ptr uint32_t* words = l1(places);
        const uint32_t holder = line[KM_ASK_HOLDER];
        invalidate_l1_cache();
        if (holder >= n || holder == view.here()) {
            l1(area)[KM_PUT_BAD / 4] += 1;
            return kGaveNone;  // taken, never confirmed: its asker gives up
        }
        const uint32_t count = words[KM_DELIVER_SENT(n) / 4 + holder];
        sent = count - words[KM_DELIVER_CREDIT(n) / 4 + holder] < s;
        if (!sent) {
            return kGaveNone;
        }
        uint32_t tile = view.far_tile(holder, false);
        tile = tile != 0 ? tile : pages_far0;
        const uint32_t i = view.here() * s + count % s;
        const uint32_t slot = places + KM_DELIVER_DATA(n, s) + i * kDeliverSlotBytes;
        const uint32_t bytes = line[KM_ASK_PAGE];
        const uint32_t messages = line[KM_ASK_PAGES] & 0xFFu;
        const uint32_t body = messages_body_lines(line[KM_ASK_PAGES] >> 8);
        const uint32_t piece = deliver_piece(messages, bytes);
        const uint32_t body_piece = deliver_piece(body / 64, 64);
        const uint32_t pieces = (
            (piece != 0 ? messages * bytes / piece : 0) +
            (body_piece != 0 ? body / body_piece : 0)
        );
        if (
            tile == 0 || messages == 0 || messages > kMessagesPacket || piece == 0 ||
            kMessagesPacket * bytes > kDeliverBodyAt ||
            kDeliverBodyAt + body > kDeliverSlotBytes
        ) {
            l1(area)[KM_PUT_BAD / 4] += 1;
            return kGaveNone;  // as above
        }
        const uint32_t xy = (tile & 0xFFu) | (tile >> 16 & 0xFFu) << 8;
        PagesRequest r = {
            line[KM_ASK_SOURCE],
            piece,
            0,
            messages * bytes / piece,
            slot,
            xy,
            pages_held.chunk,
            places + KM_DELIVER_COUNT(n) + 4 * i,
            xy,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            line[KM_ASK_CORE],
            0,
            0
        };
        const bool far = view.hops(holder) > 1;
        if (far) {
            forward_route(holder, KM_CROSS_PAGES);
        }
        uint32_t code = pages_push(kernel, r, cpu, Partition{0, r.pages});
        if (code == kGaveNone && body != 0) {
            r.from = line[KM_ASK_TO];
            r.page = body_piece;
            r.pages = body / body_piece;
            r.to = slot + kDeliverBodyAt;
            code = pages_push(kernel, r, cpu, Partition{0, r.pages});
        }
        if (code == kGaveNone) {
            line[KM_ASK_FIRST] = pieces;
            if (far) {
                forward_route(holder, KM_CROSS_LINES);
            }
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(
                    tile & 0xFFFFu, tile >> 16, places + KM_DELIVER_LINE(n, s) + 64 * i
                ),
                64
            );
            code = pages_packet(kernel, header, at, 64);
        }
        forward_route_off();
        if (code == kGaveNone) {
            words[KM_DELIVER_SENT(n) / 4 + holder] = count + 1;
            cross_count(area, KM_CROSS_MESSAGES, messages, messages * bytes + body);
        }
        return code;
    }

    // Page CPU j, line `taken` + 1 at `at`, a delivery: the lead sends it, the other
    // page CPUs only take it.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t deliver_put(
        const Kernel& kernel, uint32_t cpu, uint32_t area, uint32_t at, uint32_t taken
    ) {
        const uint32_t j = page_index(area);
        if (j == page_lead(area)) {
            bool sent = true;
            const uint32_t code = deliver_send(kernel, cpu, area, at, sent);
            if (code != kGaveNone || !sent) {
                return code;  // not sent yet: taken on a later pass
            }
        }
        return pages_taken(kernel, area, j, taken);
    }

    // The far lead, slot `slot`'s line for run `entry`: reserved on the rank's counter
    // tile, written into its queue, committed by its mark; `held`: no READY. kWhere*.
    template <typename Kernel>
    inline uint32_t deliver_bin(
        const Kernel& kernel,
        const volatile tt_l1_ptr uint32_t* line,
        uint32_t slot,
        const volatile tt_l1_ptr uint32_t* entry,
        uint32_t scratch,
        bool& held
    ) {
        const Args& arguments = kernel.args();
        const MessagesArgs a = messages_args(kernel);
        const Islands view = islands_at(arguments.control);
        km_command_t run = {};
        run.group = entry[2];
        const MessageRanks ranks(view, run, a.counters);
        const uint32_t flat = line[KM_ASK_TENSOR];
        const uint32_t bytes = line[KM_ASK_PAGE];
        const uint32_t count = line[KM_ASK_PAGES] & 0xFFu;
        const uint32_t body = line[KM_ASK_PAGES] >> 8;
        if (
            a.counters == 0 || flat - ranks.low() >= ranks.ranks_here() ||
            bytes != a.message_bytes || body > a.body_bytes
        ) {
            return kWhereMessagesBadDest;
        }
        const uint32_t destination = flat - view.offset();
        const uint32_t counter = ranks.counter_core(flat);
        volatile tt_l1_ptr uint32_t* words = l1(scratch);
        noc_async_read(
            kernel.core_noc(
                counter, arguments.control + KM_CONTROL_TAILS + KM_TAILS_READY
            ),
            scratch,
            16
        );
        if (!noc_timeout(kernel, [] {
            return ncrisc_noc_reads_flushed(noc_index);
        })) {
            return kWhereMessagesRead;
        }
        invalidate_l1_cache();
        if (words[0] != entry[1] || words[1] != 1) {
            held = true;  // its counter tile not READY for the run yet
            return 0;
        }
        words[4] = kMessagesEnd;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            kernel.core_noc(
                counter,
                arguments.control + KM_CONTROL_TAILS + KM_TAILS_LINE + 16 * destination
            ),
            NOC_UNICAST_WRITE_VC,
            count,
            31,
            false,
            false,
            scratch + 16
        );
        if (!noc_timeout(kernel, [&] {
            invalidate_l1_cache();
            return (
                ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                words[4] != kMessagesEnd
            );
        })) {
            return kWhereMessagesLost;
        }
        const uint32_t first_position = words[4];
        const uint32_t shift = __builtin_ctz(a.cap);
        const InterleavedAddrGen<true> queue = {
            .bank_base_address = a.queue,
            .page_size = kMessagesPageMessages * a.message_bytes
        };
        if (body != 0) {
            const InterleavedAddrGen<true> bodies = {
                .bank_base_address = a.body, .page_size = a.body_bytes
            };
            noc_async_write(
                slot + kDeliverBodyAt,
                bodies.get_noc_addr(
                    messages_body_page(destination, first_position + count - 1, a.cap)
                ),
                messages_body_lines(body)
            );
        }
        const MessageMarks marks(first_position, shift, count);
        *message_mark_half(slot) = 0;
        for (uint32_t i = 1; i < count; i++) {
            *message_mark_half(slot + i * bytes) = marks(i);
        }
        invalidate_l1_cache();  // a fence: the NoC reads the stored marks
        for (uint32_t i = 0; i < count;) {
            const uint32_t position = first_position + i;
            const uint32_t at = position & (a.cap - 1);
            uint32_t chunk = kMessagesPageMessages - at % kMessagesPageMessages;
            chunk = chunk < count - i ? chunk : count - i;
            noc_async_write(
                slot + i * bytes,
                queue.get_noc_addr(
                    messages_page(destination, position, a.cap),
                    at % kMessagesPageMessages * bytes
                ),
                chunk * bytes
            );
            i += chunk;
        }
        if (!noc_timeout(kernel, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        *message_mark_half(slot) = marks(0);
        invalidate_l1_cache();
        noc_async_write(
            slot,
            queue.get_noc_addr(
                messages_page(destination, first_position, a.cap),
                (first_position & (a.cap - 1)) % kMessagesPageMessages * bytes
            ),
            16
        );
        if (!noc_timeout(kernel, [] {
            return ncrisc_noc_nonposted_writes_flushed(noc_index);
        })) {
            return kWhereMessagesFlush;
        }
        return 0;
    }

    // The far lead: each island's next slot here once its pieces landed and this
    // island's run is the line's, delivered, confirmed to its asker and given back.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t
    deliver_confirm(const Kernel& kernel, uint32_t area) {
        constexpr uint32_t n = kDeliverIslands, s = kDeliverSlots;
        const Islands view = islands_at(kernel.args().control);
        const uint32_t places = deliver_places();
        volatile tt_l1_ptr uint32_t* words = l1(places);
        const uint32_t scratch = places + KM_DELIVER_SCRATCH(n, s);
        volatile tt_l1_ptr uint32_t* entry = l1(scratch + 32);  // the run, as last read
        invalidate_l1_cache();
        for (uint32_t from = 0; from < view.count() && from < n; from++) {
            const uint32_t taken = words[KM_DELIVER_TAKEN(n) / 4 + from];
            const uint32_t i = from * s + taken % s;
            volatile tt_l1_ptr uint32_t* line = (
                l1(places + KM_DELIVER_LINE(n, s) + 64 * i)
            );
            const uint32_t number = line[KM_ASK_NUMBER];
            const uint32_t pieces = line[KM_ASK_FIRST];
            if (
                number == 0 || line[KM_ASK_END] != number ||
                static_cast<int32_t>(words[KM_DELIVER_COUNT(n) / 4 + i] - pieces) < 0
            ) {
                continue;  // none, not whole yet, or its pieces not all landed
            }
            uint32_t code = kGaveNone;
            if (entry[0] != line[KM_ASK_PORT] || entry[3] != ~entry[0]) {
                if (!deliver_entry(kernel, scratch + 32, code)) {
                    if (code != kGaveNone) {
                        return code;
                    }
                    continue;  // no run open here yet
                }
                if (entry[0] != line[KM_ASK_PORT]) {
                    continue;  // this island is not in the line's run yet
                }
            }
            bool held = false;
            const uint32_t slot = (
                places + KM_DELIVER_DATA(n, s) + i * kDeliverSlotBytes
            );
            const uint32_t where = (
                deliver_bin(kernel, line, slot, entry, scratch, held)
            );
            if (held) {
                continue;
            }
            if (where != 0) {
                messages_gave_up(
                    kernel.args().control, where, line[KM_ASK_TENSOR], entry[1]
                );
                return kGaveBadTransfer;  // recorded above
            }
            const uint32_t asker = line[KM_ASK_CORE];
            const uint32_t owner = line[KM_ASK_OWNER] % KM_QUEUE_LINES;
            const bool far = view.hops(from) > 1;
            uint32_t back = view.far_tile(from, true);
            back = back != 0 ? back : pages_far0;
            if (far) {
                forward_route(from, KM_CROSS_PAGES);  // back to the asker
            }
            const uint32_t own = (
                view.own_words() + (owner / 2 % 2 == 1 ? KM_OWN_CPU_BYTES : 0)
            );
            auto* done = pages_header(area + KM_PUT_HEADERS + 128);
            link_inline(
                done,
                link_noc(
                    asker & 0xFFFFu,
                    asker >> 16,
                    own + KM_OWN_PUT_CONFIRMED + owner % 2 * 4
                ),
                number
            );
            code = pages_packet(kernel, done, 0, 0);
            forward_route_off();
            if (code != kGaveNone) {
                return code;
            }
            words[KM_DELIVER_COUNT(n) / 4 + i] -= pieces;
            line[KM_ASK_NUMBER] = 0;
            words[KM_DELIVER_TAKEN(n) / 4 + from] = taken + 1;
            if (far) {
                forward_route(from, KM_CROSS_LINES);  // its slot back
            }
            auto* credit = pages_header(area + KM_PUT_HEADERS + 128);
            link_inline(
                credit,
                link_noc(
                    back & 0xFFFFu,
                    back >> 16,
                    places + KM_DELIVER_CREDIT(n) + 4 * view.here()
                ),
                taken + 1
            );
            code = pages_packet(kernel, credit, 0, 0);
            forward_route_off();
            if (code != kGaveNone) {
                return code;
            }
        }
        return kGaveNone;
    }
#endif

}  // namespace km
#endif
