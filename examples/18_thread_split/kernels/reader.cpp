// The reader owns the X280 port: a PASS pulls ranges, loads (if not held) and calls the
// overlay each answer's word names, tells each range's kinds, then writes its line.
#include <stdint.h>

#include "../split.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "komorebi/device/core/km_x280.hpp"

namespace {

    // Hands compute and the writer one job (the dense overlay's range, or the end).
    bool hand(km::Reader& reader, uint32_t overlay, uint32_t first, uint32_t count) {
        for (const uint32_t cb : {TS_CB_JOB_COMPUTE, TS_CB_JOB_WRITER}) {
            if (!reader.reserve(cb, 1)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* job = km::l1(get_write_ptr(cb));
            job[TS_JOB_OVERLAY] = overlay;
            job[TS_JOB_FIRST] = first;
            job[TS_JOB_COUNT] = count;
            cb_push_back(cb, 1);
        }
        return true;
    }

}  // namespace

void kernel_main() {
    km::Reader reader;
    km::X280Port port(reader);
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != TS_HANDLER_PASS) {
            return;
        }
        const ts_range shared = {
            0,
            0,
            command.a1,
            command.a2,
            command.a3 == 0 ? 1 : command.a3,
            command.a4,
            km::slot(command),
            command.sequence
        };
        const bool pull_only = command.a7 != 0;
        uint32_t line[TS_LINE] = {};
        line[TS_LINE_PASS] = command.a6 + 1;
        const uint32_t started = km::wall();
        // --8<-- [start:pull]
        uint32_t first = 0, word = 0;
        for (;;) {
            const uint32_t asked = km::wall();
            const uint32_t count = port.next(first, word);  // the range and its overlay
            line[TS_LINE_WAITED] += km::wall() - asked;
            if (count == 0) {
                break;
            }
            line[TS_LINE_PULLS]++;
            if (pull_only) {
                line[TS_LINE_WORDS] += word == TS_PULL_WORD;
                continue;
            }
            if (word < TS_OVERLAY_PLAIN || word > TS_OVERLAYS) {
                line[TS_LINE_BAD]++;
                continue;
            }
            if (word == TS_OVERLAY_DENSE && !hand(reader, word, first, count)) {
                return;  // compute and the writer meet this load()
            }
            km::Overlay overlay(reader, word);
            line[TS_LINE_LOADS] += !overlay.held();
            if (!overlay.load()) {  // nothing moves when the slot holds it
                return;
            }
            ts_range range = shared;
            range.first = first;
            range.count = count;
            line[TS_LINE_UNITS] += overlay.call(range);
            line[TS_LINE_PLAIN + word - 1] += count;
            uint32_t kinds = 0;  // the range's kinds, a byte each: what the thread
                                 // decides on
            for (uint32_t unit = first; unit < first + count; unit++) {
                kinds += 1u << 8 * TS_KIND(unit, range.run);
            }
            port.tell(kinds, word, count);
        }
        // --8<-- [end:pull]
        line[TS_LINE_SPAN] = km::wall() - started;
        if (port.pull_gave_up() || (!pull_only && !hand(reader, 0, 0, 0))) {
            return;
        }
        if (!reader.reserve(TS_CB_LINE, 1)) {
            return;
        }
        volatile tt_l1_ptr uint32_t* out = km::l1(get_write_ptr(TS_CB_LINE));
        for (uint32_t index = 0; index < TS_LINE; index++) {
            out[index] = line[index];
        }
        // the tiles' arguments first (the overlays' reader parts write them)
        using Tiles = km::Tensor<0>;
        const km::Tensor<Tiles::next_compile_time_args_offset()> lines(command.a5);
        const uint32_t page = command.a6 * km::count(command) + reader.me();
        if (!km::write_pages(
            reader, command, lines, page, 1, get_write_ptr(TS_CB_LINE)
        )) {
            return;
        }
        noc_async_write_barrier();
    });
}
