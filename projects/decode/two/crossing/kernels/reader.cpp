// A crossing alone, reader: roots write their block into their card's copy, then V1
// (the leader sends the card's half) or V2 (each root sends its block).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "crossing.h"
#include "komorebi/device/tensors/km_ask.hpp"

namespace {

    volatile tt_l1_ptr uint32_t* stamps;

    void stamp(uint32_t step) {
        stamps[step] = km::wall();
    }

    // The block of root `root` of island `here`: word w of page p is salt + p x 512 +
    // w.
    void fill(uint32_t block, uint32_t first, uint32_t pages, uint32_t salt) {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block);
        for (uint32_t k = 0; k < pages * kPageBytes / 4; k++) {
            words[k] = salt + first * (kPageBytes / 4) + k;
        }
    }

    // The roots meet (km::team_meet's marker, the roots alone): each root writes the
    // marker into its word of the leader's, the leader waits for all. False: gave up.
    bool roots_meet(
        km::Reader& reader,
        const km_command_t& command,
        const km::Team& t,
        uint32_t roots,
        uint32_t stride
    ) {
        const uint32_t count = km::team_now(command).meets++;
        const uint32_t marker = km::team_marker(command, count);
        const uint32_t words = km::islands_at(reader.args().control).meetings();
        if (words == 0 || t.size > km::l1(words)[0]) {
            return km::coll_gave_up(reader, command, km::kGaveBadItem);
        }
        if (t.index != 0) {
            noc_inline_dw_write(
                reader.core_noc(reader.core_of(t.leader), words + 4 + 4 * t.index),
                marker
            );
            return km::team_wrote(reader, command);
        }
        const volatile tt_l1_ptr uint32_t* met = km::l1(words + 4);
        uint32_t k = 1;  // the roots before it have met (a word keeps its marker)
        const uint32_t code = reader.poll_at(
            km::kWhereAsk,
            [&] {
                invalidate_l1_cache();
                while (k < roots && met[k * stride] == marker) {
                    k++;
                }
                return k >= roots;
            },
            reader.args().wait_limit
        );
        return code == km::kGaveNone || km::coll_gave_up(reader, command, code);
    }

    bool cross(km::Reader& reader, const km_command_t& command) {
        const km::Islands view = km::islands_at(reader.args().control);
        const uint32_t here = view.here(), far = 1 - here;
        const uint32_t half = command.a2, roots = command.a3, stride = command.a4;
        const uint32_t pages = command.a5, way = command.a6;
        km::Team t = km::team_of(view, command, reader.me());
        const bool root = t.index % stride == 0 && t.index / stride < roots;
        const uint32_t first = here * half + t.index / stride * pages;
        const km::SizedTensor x(command.a0, kPageBytes);
        // the far island receives on its port toward this one (km::all_gather's)
        const uint32_t port_in = here == 0 ? KM_PORT_RIGHT : KM_PORT_LEFT;
        const uint32_t port_out = here == 0 ? KM_PORT_LEFT : KM_PORT_RIGHT;
        // flat ranks (km::keeper): the far island's first + its team leader's place
        const uint32_t keeper = (
            km::keeper(
                view.entry_first(far) + km::team_leader(view, command, far), port_out
            )
        );
        km::Gather posted = {};
        stamp(kXsEntry);
        if (root) {
            const uint32_t block = get_write_ptr(0);
            fill(block, first, pages, command.a7);
            if (
                !km::write_pages(reader, command, x, first, pages, block) ||
                !reader.flushed(km::kFlushWrites, km::kStReader)
            ) {
                return false;
            }
            stamp(kXsWritten);
            if (way == kWayRoots) {
                if (!km::gather_pages(
                    reader, command, command.a0, first, pages, far, posted, keeper
                )) {
                    return false;
                }
                stamp(kXsPosted);
            }
            if (t.index != 0 && !roots_meet(reader, command, t, roots, stride)) {
                return false;
            }
            stamp(kXsMarked);
        }
        if (t.index == 0) {
            if (!roots_meet(reader, command, t, roots, stride)) {
                return false;
            }
            stamp(kXsMet);
            if (way == kWayLeader) {
                if (!km::gather_pages(
                    reader,
                    command,
                    command.a0,
                    here * half,
                    half,
                    far,
                    posted,
                    keeper
                )) {
                    return false;
                }
                stamp(kXsPosted);
            }
            if (!km::gathered(reader, command, port_in, half)) {
                return false;
            }
            stamp(kXsArrived);
        }
        if (!km::team_go(reader, command, t)) {
            return false;
        }
        stamp(kXsReleased);
        if (!km::gather_wait(reader, command, posted)) {
            return false;
        }
        stamp(kXsConfirmed);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Reader reader;
    stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerCross && handler != kHandlerGather) {
            return;
        }
        for (uint32_t w = 0; w < kXsWords; w++) {
            stamps[w] = 0;
        }
        stamps[kXsSalt] = command.a7;
        bool ok = true;
        if (handler == kHandlerCross) {
            ok = cross(reader, command);
        } else {
            stamp(kXsEntry);
            ok = km::all_gather(reader, command, command.a0, 0, 2 * command.a2);
            stamp(kXsReleased);
        }
        if (ok) {
            const km::SizedTensor out(command.a1, kStampBytes);
            km::write_pages(reader, command, out, reader.me(), 1, get_write_ptr(1));
            reader.flushed(km::kFlushWrites, km::kStReader);
        }
    });
}
