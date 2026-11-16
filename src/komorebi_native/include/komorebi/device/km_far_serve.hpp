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

    // Page CPU j, line `taken` + 1 at `at`, an L1 write or an add: the lead sends the
    // line across as a put's, the pieces go from the asker's L1 into the rank's.
    template <typename Kernel>
    __attribute__((noinline)) uint32_t far_put(
        const Kernel& kernel, uint32_t cpu, uint32_t area, uint32_t at, uint32_t taken
    ) {
        volatile tt_l1_ptr uint32_t* words = l1(area);
        volatile tt_l1_ptr uint32_t* line = l1(at);
        const Islands view = islands_at(kernel.args().control);
        const uint32_t j = page_index(area);
        const bool lead = j == page_lead(area);
        const uint32_t queue = link_queue();
        const uint32_t op = line[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        const bool inline_word = (line[KM_ASK_RING] & KM_ASK_FORM) == KM_ASK_INLINE;
        const uint32_t pieces = (
            op == KM_ASK_OP_L1 && !inline_word ? line[KM_ASK_PIECES] : 0
        );
        // the rank's core: the island tables are the same on every island
        const uint32_t holder = line[KM_ASK_HOLDER];
        const uint32_t rank = line[KM_ASK_TENSOR];
        const uint32_t index = (
            holder < view.count() ? rank - view.entry_first(holder) : KM_CORES_MAX
        );
        const bool known = (
            holder < view.count() && holder != view.here() &&
            index < view.entry_count(holder) && view.base() + index < KM_CORES_MAX
        );
        const uint32_t xy = (
            known
                ? core_entry(
                    l1(kernel.args().control + KM_CONTROL_CORES),
                    view.base() + index,
                    kernel.args().control,
                    kernel.role()
                )
                : 0
        );
        const uint32_t far0 = pages_far0;  // the far link core 0
        const PagesRequest r = {
            line[KM_ASK_SOURCE],
            line[KM_ASK_PAGE],
            0,
            pieces,
            line[KM_ASK_TO],
            (xy & 0xFFu) | (xy >> 16 & 0xFFu) << 8,
            pages_held.chunk,
            queue + KM_QUEUE_ARRIVED + line[KM_ASK_OWNER] % KM_QUEUE_LINES * 4,
            (far0 & 0xFFu) | (far0 >> 16 & 0xFFu) << 8,
            pages_held.hops,
            KM_PAGES_SEND,
            0,
            pages_held.area - cpu * pages_held.chunk,
            line[KM_ASK_CORE],
            0,
            0
        };
        const bool good = (
            known &&
            (
                pieces == 0 ||
                (
                    r.page != 0 && r.page % 16 == 0 && r.page <= pages_packet() &&
                    r.from % 16 == 0 && r.to % 16 == 0
                )
            )
        );
        uint32_t code = kGaveNone;
        if (lead) {  // on link core 0: the line is its own
            put_stamp(area, KM_STAMP_TAKEN);
            if (!good) {  // across, it does nothing and counts no piece
                line[KM_ASK_RING] = KM_ASK_OP_WRITE << KM_ASK_OP_SHIFT;
                line[KM_ASK_PAGES] = 0;
                words[KM_PUT_BAD / 4] += 1;
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
        const Partition share = page_share(area, pieces, j);
        if (code == kGaveNone && good && share.len != 0) {
            code = pages_push(kernel, r, cpu, share);
        }
        if (code != kGaveNone) {
            return code;
        }
        return pages_taken(kernel, area, j, taken);
    }

    // The far lead, line `line` arrived as pending entry `entry`: an L1 write's or an
    // add's words (the rank's core, its pieces, the address); any other line: none.
    template <typename Kernel>
    inline void far_arrived(
        const Kernel& kernel,
        volatile tt_l1_ptr uint32_t* entry,
        const volatile tt_l1_ptr uint32_t* line
    ) {
        entry[KM_QUEUE_E_FAR] = 0;
        if (!far_op(line)) {
            return;
        }
        const Islands view = islands_at(kernel.args().control);
        const uint32_t op = line[KM_ASK_RING] >> KM_ASK_OP_SHIFT;
        const uint32_t form = line[KM_ASK_RING] & KM_ASK_FORM;
        const uint32_t index = line[KM_ASK_TENSOR] - view.first();
        const bool pieces = op == KM_ASK_OP_L1 && form != KM_ASK_INLINE;
        const uint32_t port = op == KM_ASK_OP_L1 ? line[KM_ASK_PORT] : 0;
        entry[KM_QUEUE_E_KEEPER] = 0;
        entry[KM_QUEUE_E_PAGES] = pieces ? line[KM_ASK_PIECES] : 0;
        if (
            index >= view.ranks() || view.base() + index >= KM_CORES_MAX ||
            port > KM_PORTS
        ) {
            l1(view.link_area())[KM_PUT_BAD / 4] += 1;
            return;  // confirmed, nothing done
        }
        entry[KM_QUEUE_E_FAR] = (
            op | form << 4 | port << 8 | (view.base() + index) << 16
        );
        entry[KM_QUEUE_E_VALUE] = (
            op == KM_ASK_OP_ADD ? line[KM_ASK_PAGES] : line[KM_ASK_PAGE]
        );
        if (!pieces) {
            entry[KM_QUEUE_E_COUNT] = line[KM_ASK_TO];  // the word's address
        }
    }

    // The far lead, entry `entry` landed (its pieces counted): the add, the inline word
    // or the port's 1 on the rank's core, each acknowledged. kGave* code.
    template <typename Kernel>
    inline uint32_t far_land(
        const Kernel& kernel, uint32_t area, volatile tt_l1_ptr uint32_t* entry
    ) {
        static_cast<void>(area);
        const uint32_t far = entry[KM_QUEUE_E_FAR];
        if (far == 0) {
            return kGaveNone;
        }
        const uint32_t op = far & 0xFu;
        const uint32_t form = far >> 4 & 0xFu;
        const uint32_t port = far >> 8 & 0xFFu;
        const uint32_t core = far >> 16;
        const uint32_t limit = kernel.args().wait_limit;
        uint32_t code = kGaveNone;
        if (op == KM_ASK_OP_ADD) {
            noc_semaphore_inc(
                kernel.core_noc(core, entry[KM_QUEUE_E_COUNT]), entry[KM_QUEUE_E_VALUE]
            );
        } else if (form == KM_ASK_INLINE) {
            noc_inline_dw_write(
                kernel.core_noc(core, entry[KM_QUEUE_E_COUNT]), entry[KM_QUEUE_E_VALUE]
            );
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
        entry[KM_QUEUE_E_FAR] = 0;
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

}  // namespace km
