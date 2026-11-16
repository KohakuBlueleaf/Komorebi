// The link cores' side of an L1 write and an add (ops 1 and 2): the near page CPUs send
// the line and its pieces as a put's; the far lead does the write or add locally.
#pragma once

#include <stdint.h>

#include "km_pages.hpp"

namespace km {

    // Whether queue line `line` is an L1 write or an add (word 12's op).
    inline bool far_op(const volatile tt_l1_ptr uint32_t* line) {
        const uint32_t op = line[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        return op == KM_ASK_OP_L1 || op == KM_ASK_OP_ADD;
    }

    // An L1 write's or an add's request: the pieces (an inline word, an add: none)
    // from the asker's L1 into the rank's core, counted at `count` of tile `counter`.
    struct FarRequest {
        PagesRequest r;
        bool good;
    };

    template <typename Kernel>
    inline FarRequest far_request(
        const Kernel& kernel,
        const volatile tt_l1_ptr uint32_t* line,
        uint32_t cpu,
        uint32_t count,
        uint32_t counter
    ) {
        const Islands view = islands_at(kernel.args().control);
        const uint32_t op = line[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        const bool inline_word = (line[KM_ASK_RING] & KM_ASK_FORM) == KM_ASK_INLINE;
        const uint32_t pieces = (
            op == KM_ASK_OP_L1 && !inline_word ? line[KM_ASK_PAGES] : 0
        );
        // the rank's core: its island's first core on the card (the core tables are
        // the same on every island; a split card's is the card's)
        const uint32_t holder = line[KM_ASK_HOLDER];
        const uint32_t index = (
            holder < view.count()
                ? line[KM_ASK_TENSOR] - view.entry_first(holder)
                : KM_CORES_MAX
        );
        const uint32_t base = holder < view.count() ? view.entry_base(holder) : 0;
        const bool known = (
            holder < view.count() && holder != view.here() &&
            index < view.entry_count(holder) && base + index < KM_CORES_MAX
        );
        const uint32_t xy = (
            known
                ? core_entry(
                    l1(kernel.args().control + KM_CONTROL_CORES),
                    base + index,
                    kernel.args().control,
                    kernel.role()
                )
                : 0
        );
        const FarRequest out = {
            {
                line[KM_ASK_SOURCE],
                line[KM_ASK_PAGE],
                0,
                pieces,
                line[KM_ASK_TO],
                (xy & 0xFFu) | (xy >> 16 & 0xFFu) << 8,
                pages_held.chunk,
                count,
                (counter & 0xFFu) | (counter >> 16 & 0xFFu) << 8,
                pages_held.hops,
                KM_PAGES_SEND,
                0,
                pages_held.area - cpu * pages_held.chunk,
                line[KM_ASK_CORE],
                0,
                0
            },
            known &&
                (
                    pieces == 0 ||
                    (
                        line[KM_ASK_PAGE] != 0 && line[KM_ASK_PAGE] % 16 == 0 &&
                        line[KM_ASK_PAGE] <= pages_packet() &&
                        line[KM_ASK_SOURCE] % 16 == 0 && line[KM_ASK_TO] % 16 == 0
                    )
                )
        };
        return out;
    }

    // A line found bad before it left: across it does nothing and counts no piece.
    inline void far_spoil(
        volatile tt_l1_ptr uint32_t* line, volatile tt_l1_ptr uint32_t* words
    ) {
        line[KM_ASK_RING] = KM_ASK_OP_WRITE << KM_ASK_OP_SHIFT;
        line[KM_ASK_PAGES] = 0;
        words[KM_PUT_BAD / 4] += 1;
    }

#if defined(KM_FAR_PUTS)
    // The lead, an L1 write or an add past the next island: its pieces, then the line,
    // by the forwarding rooms into its places there (as a far put's). `sent`: a place.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t far_put_far(
        const Kernel& kernel, uint32_t cpu, uint32_t area, uint32_t at, bool& sent
    ) {
        constexpr uint32_t n = KM_ISLANDS_MAX;
        const Islands view = islands_at(kernel.args().control);
        volatile tt_l1_ptr uint32_t* line = l1(at);
        const uint32_t far = link_queue() + KM_QUEUE_BYTES;
        volatile tt_l1_ptr uint32_t* words = l1(far);
        const uint32_t holder = line[KM_ASK_HOLDER];
        invalidate_l1_cache();
        const uint32_t count = words[KM_FAR_SENT(n) / 4 + holder];
        sent = (
            holder < n && count - words[KM_FAR_CREDIT(n) / 4 + holder] < KM_FAR_LINES
        );
        if (!sent) {
            return kGaveNone;
        }
        const uint32_t tile = view.far_tile(holder, false);
        const uint32_t i = view.here() * KM_FAR_LINES + count % KM_FAR_LINES;
        const FarRequest put = (
            far_request(kernel, line, cpu, far + KM_FAR_COUNT(n) + 4 * i, tile)
        );
        const bool good = put.good && tile != 0;
        if (!good) {
            far_spoil(line, l1(area));
        }
        uint32_t code = kGaveNone;
        if (
            good && put.r.pages != 0
        ) {  // every piece by the lead: the place is its own
            forward_route(holder, KM_CROSS_PAGES);
            code = pages_push(kernel, put.r, cpu, Partition{0, put.r.pages});
        }
        if (code == kGaveNone && tile != 0) {
            forward_route(holder, KM_CROSS_LINES);
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(tile & 0xFFFFu, tile >> 16, far + KM_FAR_LINE(n) + 64 * i),
                64
            );
            code = pages_packet(kernel, header, at, 64);
        }
        forward_route_off();
        if (code == kGaveNone) {
            words[KM_FAR_SENT(n) / 4 + holder] = count + 1;
            cross_count(area, KM_CROSS_PUTS, 1, 64);
        }
        return code;
    }
#endif

    // Page CPU j, line `taken` + 1 at `at`, an L1 write or an add: to the next island
    // the lead sends the line across as a put's, the pieces go into the rank's L1.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t far_put(
        const Kernel& kernel, uint32_t cpu, uint32_t area, uint32_t at, uint32_t taken
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        volatile tt_l1_ptr uint32_t* line = l1(at);
        const uint32_t j = page_index(area);
        const bool lead = j == page_lead(area);
#if defined(KM_FAR_PUTS)
        if (islands_at(kernel.args().control).hops(line[KM_ASK_HOLDER]) > 1) {
            if (lead) {  // the other page CPUs only take it
                bool sent = true;
                const uint32_t code = far_put_far(kernel, cpu, area, at, sent);
                if (code != kGaveNone || !sent) {
                    return code;  // not sent yet: taken on a later pass
                }
            }
            return pages_taken(kernel, area, j, taken);
        }
#endif
        const uint32_t queue = link_queue();
        const uint32_t far0 = pages_far0;  // the far link core 0
        const FarRequest far = (
            far_request(
                kernel,
                line,
                cpu,
                queue + KM_QUEUE_ARRIVED + line[KM_ASK_OWNER] % KM_QUEUE_LINES * 4,
                far0
            )
        );
        uint32_t code = kGaveNone;
        if (lead) {  // on link core 0: the line is its own
            put_stamp(area, KM_STAMP_TAKEN);
            if (!far.good) {
                far_spoil(line, words);
            }
            const uint32_t sent = words[KM_PUT_DONE / 4];
            auto* header = pages_header(area + KM_PUT_HEADERS);
            link_write(
                header,
                link_noc(
                    far0 & 0xFFFFu,
                    far0 >> 16,
                    queue + KM_QUEUE_ARRIVALS + sent % KM_QUEUE_LINES * 64
                ),
                64
            );
            code = pages_packet(kernel, header, at, 64);
            if (code == kGaveNone) {
                words[KM_PUT_DONE / 4] = sent + 1;
                cross_count(area, KM_CROSS_PUTS, 1, 64);
                put_stamp(area, KM_STAMP_ACROSS);
            }
        }
        const Partition share = page_share(area, far.r.pages, j);
        if (code == kGaveNone && far.good && share.len != 0) {
            code = pages_push(kernel, far.r, cpu, share);
        }
        if (code != kGaveNone) {
            return code;
        }
        return pages_taken(kernel, area, j, taken);
    }

    // The rank's core on this island for line `line`, its op | form << 4 | port + 1 <<
    // 8 | core << 16 (KM_QUEUE_E_FAR); 0: not an L1 write or an add, or a bad one.
    template <typename Kernel>
    inline uint32_t far_word(
        const Kernel& kernel, const volatile tt_l1_ptr uint32_t* line
    ) {
        if (!far_op(line)) {
            return 0;
        }
        const Islands view = islands_at(kernel.args().control);
        const uint32_t op = line[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        const uint32_t form = line[KM_ASK_RING] & KM_ASK_FORM;
        const uint32_t index = line[KM_ASK_TENSOR] - view.first();
        const uint32_t port = op == KM_ASK_OP_L1 ? line[KM_ASK_PORT] : 0;
        if (
            index >= view.ranks() || view.base() + index >= KM_CORES_MAX ||
            port > KM_PORTS || form > 0xFu
        ) {
            l1(view.link_area())[KM_PUT_BAD / 4] += 1;
            return 0;  // confirmed, nothing done
        }
        return op | form << 4 | port << 8 | (view.base() + index) << 16;
    }

    // The far lead, line `line` arrived as pending entry `entry`: an L1 write's or an
    // add's words (the rank's core, the address, the value); any other line: none.
    template <typename Kernel>
    inline void far_arrived(
        const Kernel& kernel,
        volatile tt_l1_ptr uint32_t* entry,
        const volatile tt_l1_ptr uint32_t* line
    ) {
        entry[KM_QUEUE_E_FAR] = far_word(kernel, line);
        if (!far_op(line)) {
            return;
        }
        entry[KM_QUEUE_E_KEEPER] = 0;
        entry[KM_QUEUE_E_VALUE] = line[KM_ASK_PAGE];
        if (line[KM_ASK_PAGES] == 0) {
            entry[KM_QUEUE_E_COUNT] = line[KM_ASK_TO];  // the word's address
        }
    }

    // The add, the inline word or the port's 1 of `far` (KM_QUEUE_E_FAR) on the rank's
    // core, each acknowledged: `address` the word's, `value` the add's or the word.
    template <typename Kernel>
    inline uint32_t far_do(
        const Kernel& kernel, uint32_t far, uint32_t address, uint32_t value
    ) {
        const uint32_t op = far & 0xFu;
        const uint32_t form = far >> 4 & 0xFu;
        const uint32_t port = far >> 8 & 0xFFu;
        const uint32_t core = far >> 16;
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kGaveNone;
        if (op == KM_ASK_OP_ADD) {
            noc_semaphore_inc(kernel.core_noc(core, address), value);
        } else if (form == KM_ASK_INLINE) {
            noc_inline_dw_write(kernel.core_noc(core, address), value);
            code = kernel.poll_at(
                kWherePages,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                limit
            );
        }
        if (code == kGaveNone && port != 0) {  // after the bytes: they landed
            noc_semaphore_inc(
                kernel.core_noc(
                    core,
                    kernel.args().control + KM_CONTROL_PORT +
                        (port - 1) * KM_PORT_STRIDE + KM_PORT_COUNT * 4
                ),
                1
            );
        }
        if (code != kGaveNone) {
            return code;
        }
        return kernel.poll_at(
            kWherePages,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            limit
        );
    }

    // The far lead, entry `entry` landed (its pieces counted): far_do once. kGave*.
    template <typename Kernel>
    inline uint32_t far_land(
        const Kernel& kernel, uint32_t area, volatile tt_l1_ptr uint32_t* entry
    ) {
        static_cast<void>(area);
        const uint32_t far = entry[KM_QUEUE_E_FAR];
        entry[KM_QUEUE_E_FAR] = 0;
        return (
            far == 0
                ? kGaveNone
                : far_do(kernel, far, entry[KM_QUEUE_E_COUNT], entry[KM_QUEUE_E_VALUE])
        );
    }

    // The far lead, a line from an island past the next landed in its place (its
    // pieces counted): far_do once, then it reads as a put. kGave* code.
    template <typename Kernel>
    inline uint32_t far_land_line(
        const Kernel& kernel, volatile tt_l1_ptr uint32_t* line
    ) {
        const uint32_t far = far_word(kernel, line);
        if (!far_op(line)) {
            return kGaveNone;
        }
        line[KM_ASK_RING] = KM_ASK_OP_WRITE << KM_ASK_OP_SHIFT;  // done once
        return (
            far == 0
                ? kGaveNone
                : far_do(kernel, far, line[KM_ASK_TO], line[KM_ASK_PAGE])
        );
    }

}  // namespace km
