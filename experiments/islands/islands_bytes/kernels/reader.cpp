// Ranges of a page across islands, reader: READ reads ranges of a tensor's pages with
// km::read_bytes (one in flight) or km::ask_bytes and km::bytes_wait (several).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "ranges.h"

// READ (ranges.h): the rank's ranges, checked, then its report.
template <typename Args>
void read(km::Reader& reader, const km_command_t& command, const Args& args) {
    const uint32_t from = command.a0, report = command.a1, pages = command.a2;
    const uint32_t bytes = command.a3, count = command.a4, seed = command.a6;
    const uint32_t flight = command.a5;
    const uint32_t page = TensorAccessor(args, 0).get_aligned_page_size();
    const uint32_t buffer = get_write_ptr(kCbRanges);
    const uint32_t rank = reader.me() - km::first(command);
    uint32_t done = 0, wrong = 0, guard = 0, first = kRangesNone, cycles = 0;
    uint32_t failed = 0;
    RangeAt last = {0, 0};
    uint32_t last_into = buffer;
    const bool fits = (
        bytes != 0 && bytes <= kRangesLargest && bytes <= page && flight != 0 &&
        flight <= kRangesInFlight
    );
    while (fits && done < count && failed == 0) {
        const uint32_t n = count - done < flight ? count - done : flight;
        // each slot's words the mark, its range's place after the guard line
        uint32_t into[kRangesInFlight];
        RangeAt at[kRangesInFlight];
        for (uint32_t k = 0; k < n; k++) {
            at[k] = ranges_pick(seed, rank, done + k, pages, page, bytes);
            const uint32_t slot = buffer + k * kRangesSlot;
            into[k] = slot + 64 + at[k].offset % 64;
            volatile tt_l1_ptr uint32_t* words = km::l1(slot);
            for (uint32_t w = 0; w < kRangesSlot / 4; w++) {
                words[w] = kRangesMark;
            }
        }
        const uint32_t started = km::wall();
        if (flight == 1) {
            failed = (
                !km::read_bytes(
                    reader, command, args, from, at[0].page, at[0].offset, bytes, into[0]
                )
            );
        } else {
            uint32_t asked = 0;
            for (uint32_t k = 0; k < n && failed == 0; k++) {
                failed = (
                    !km::ask_bytes(
                        reader,
                        command,
                        args,
                        from,
                        at[k].page,
                        at[k].offset,
                        bytes,
                        into[k],
                        asked
                    )
                );
            }
            failed = failed || !km::bytes_wait(reader, command, asked);
        }
        cycles += km::wall() - started;
        if (failed) {
            break;
        }
        invalidate_l1_cache();
        for (uint32_t k = 0; k < n; k++) {
            const volatile tt_l1_ptr uint32_t* got = km::l1(into[k]);
            uint32_t bad = 0;
            for (uint32_t w = 0; w < bytes / 4; w++) {
                bad += got[w] != ranges_word(at[k].page, at[k].offset / 4 + w) ? 1 : 0;
            }
            for (uint32_t w = 1; w <= 4; w++) {  // the 16 bytes either side
                guard += got[-static_cast<int32_t>(w)] != kRangesMark ? 1 : 0;
                guard += got[bytes / 4 + w - 1] != kRangesMark ? 1 : 0;
            }
            if (bad != 0 && first == kRangesNone) {
                first = done + k;
            }
            wrong += bad;
        }
        last = at[n - 1];
        last_into = into[n - 1];
        done += n;
    }
    // the report after the slots: its words, then the last range's bytes
    const uint32_t out = buffer + kRangesInFlight * kRangesSlot;
    volatile tt_l1_ptr uint32_t* words = km::l1(out);
    const volatile tt_l1_ptr uint32_t* copy = km::l1(last_into);
    const uint32_t copied = done != 0 && fits ? bytes / 4 : 0;
    for (uint32_t w = 0; w < copied; w++) {
        words[kRangesCopy / 4 + w] = copy[w];
    }
    const uint32_t head[10] = {
        kRangesMagic,
        rank,
        done,
        wrong,
        guard,
        first,
        cycles,
        failed != 0 || !fits ? 1u : 0u,
        last.page,
        last.offset
    };
    for (uint32_t w = 0; w < 10; w++) {
        words[w] = head[w];
    }
    km::write_pages(reader, command, args, report, command.a7 + rank, 1, out);
    noc_async_write_barrier();
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // the source's and the report's
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerRead) {
            read(reader, command, args);
        }
    });
}
