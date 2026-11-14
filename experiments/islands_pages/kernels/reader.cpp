// Pages across islands, reader: MOVE reads a tensor's pages with km::read_pages and
// writes them into another tensor with km::write_pages, wherever either lives.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

// SOAK (pages.h): the rank's share, a4 rounds, then its report.
template <typename Args>
void soak(km::Reader& reader, const km_command_t& command, const Args& args) {
    const uint32_t from = command.a0, to = command.a1, step = command.a3;
    const uint32_t rounds = command.a4, report = command.a5;
    const uint32_t buffer = get_write_ptr(kCbMove);
    const uint32_t page = command.a6;  // aligned page bytes: the last word's place
    const uint32_t rank = reader.me() - km::first(command);
    const km::Partition share = (
        km::partition(command.a2, rank, km::count(command))
    );
    const bool asks = km::island_of(from, 0) != km::island_of(report, 0);
    volatile tt_l1_ptr uint32_t* l1 = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer)
    );
    uint32_t calls = 0, bad = 0, first_bad = kSoakNone;
    uint32_t missing = kSoakNone, failed = kSoakNone;
    for (uint32_t round = 0; round < rounds && failed == kSoakNone; round++) {
        for (uint32_t done = 0; done < share.len;) {
            const uint32_t n = share.len - done < step ? share.len - done : step;
            const uint32_t first = share.start + done;
            if (asks) {
                for (uint32_t k = 0; k < n; k++) {
                    l1[k * page / 4] = kSoakMark | k;
                }
                if (!km::read_pages(reader, command, args, from, first, n, buffer)) {
                    invalidate_l1_cache();
                    for (uint32_t k = 0; k < n && missing == kSoakNone; k++) {
                        if (l1[k * page / 4] == (kSoakMark | k)) {
                            missing = first + k;
                        }
                    }
                    failed = round;
                    break;
                }
                invalidate_l1_cache();
                for (uint32_t k = 0; k < n; k++) {
                    const uint32_t p = first + k;
                    if (l1[k * page / 4] != p || l1[(k + 1) * page / 4 - 1] != ~p) {
                        first_bad = bad == 0 ? p : first_bad;
                        bad++;
                    }
                }
            } else {
                if (!km::read_pages(reader, command, args, from, first, n, buffer)) {
                    failed = round;
                    break;
                }
                noc_async_read_barrier();
                if (!km::write_pages(reader, command, args, to, first, n, buffer)) {
                    failed = round;
                    break;
                }
            }
            calls++;
            done += n;
        }
    }
    noc_async_write_barrier();
    const uint32_t words[kSoakWords] = {
        kSoakMagic, calls, bad, first_bad, missing, failed, rank, 0
    };
    for (uint32_t w = 0; w < kSoakWords; w++) {
        l1[w] = words[w];
    }
    km::write_pages(reader, command, args, report, rank, 1, buffer);
    noc_async_write_barrier();
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // both tensors' (one shape)
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerSoak) {
            soak(reader, command, args);
            return;
        }
        if (km::handler(command) != kHandlerMove) {
            return;
        }
        const uint32_t from = command.a0, to = command.a1, step = command.a3;
        const uint32_t buffer = get_write_ptr(kCbMove);
        const km::Partition share = (
            km::partition(
                command.a2, reader.me() - km::first(command), km::count(command)
            )
        );
        for (uint32_t done = 0; done < share.len;) {
            const uint32_t n = share.len - done < step ? share.len - done : step;
            const uint32_t first = share.start + done;
            if (!km::read_pages(reader, command, args, from, first, n, buffer)) {
                return;
            }
            noc_async_read_barrier();
            if (!km::write_pages(reader, command, args, to, first, n, buffer)) {
                return;
            }
            noc_async_write_barrier();
            done += n;
        }
    });
}
