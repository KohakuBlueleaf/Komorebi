// Collectives across islands, reader: GATHER posts a rank's share of pages to a
// neighbour island and waits once; REDUCE adds two islands' partials (collectives.h).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "collectives.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"

namespace {

    // The rank's report (collectives.h kReport*) into page `rank` of tensor `word`.
    template <typename Args>
    void report(
        km::Reader& reader,
        const km_command_t& command,
        const Args& args,
        uint32_t word,
        uint32_t rank,
        const uint32_t (&words)[kReportWords]
    ) {
        if (word == 0) {
            return;
        }
        const uint32_t buffer = get_write_ptr(kCbPages);
        volatile tt_l1_ptr uint32_t* l1 = km::l1(buffer);
        for (uint32_t w = 0; w < kReportWords; w++) {
            l1[w] = words[w];
        }
        km::write_pages(reader, command, args, word, rank, 1, buffer);
        noc_async_write_barrier();
    }

    // GATHER (collectives.h): a rank's share posted, then one wait.
    template <typename Args>
    void gather(km::Reader& reader, const km_command_t& command, const Args& args) {
        const uint32_t tensor = command.a0, to = command.a3, issuers = command.a4;
        const uint32_t target = command.a6 != 0 ? command.a6 : command.a0;
        const uint32_t rank = reader.me() - km::first(command);
        uint32_t words[kReportWords] = {kReportMagic, rank};
        words[kReportStart] = km::wall();
        if (rank >= issuers) {
            return;
        }
        const km::Partition share = km::partition(command.a2, rank, issuers);
        const uint32_t first = command.a1 + share.start;
        words[kReportPages] = share.len;
        if (command.a8 == kModeRing && share.len != 0) {
            // the linear kernel's put: a run named, its pages from L1 a piece at a time
            const uint32_t buffer = get_write_ptr(kCbPages);
            const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
            const km::PageAt at = km::gather_place(target, first, to);
            const uint32_t n = km::page_run(target, first, share.len, at);
            km::PutRun run;
            if (
                at.address == 0 || n != share.len ||
                !km::put_pages(reader, command, to, at.address, bytes, at.index, n, run)
            ) {
                return;
            }
            for (uint32_t k = 0; k < n;) {
                const uint32_t m = n - k < kPiece ? n - k : kPiece;
                if (!km::read_pages(
                    reader, command, args, tensor, first + k, m, buffer
                )) {
                    return;
                }
                noc_async_read_barrier();
                if (!km::put_piece(reader, command, run, buffer, m)) {
                    return;
                }
                k += m;
            }
            words[kReportPosted] = km::wall();
            if (!km::put_wait(reader, command, run)) {
                return;
            }
        } else if (share.len != 0) {
            km::Gather posted = {};
            if (!km::gather_pages(
                reader, command, tensor, first, share.len, to, posted, 0, command.a6
            )) {
                return;
            }
            words[kReportPosted] = km::wall();
            if (!km::gather_wait(reader, command, posted)) {
                return;
            }
        }
        words[kReportDone] = km::wall();
        words[kReportEnd] = words[kReportDone];
        report(reader, command, args, command.a5, rank, words);
    }

    // REDUCE (collectives.h): the leader's one line of partials to the far keepers,
    // each rank's fold of its share, the leader's one line of sums back.
    template <typename Args>
    void reduce(km::Reader& reader, const km_command_t& command, const Args& args) {
        const uint32_t partial = command.a0, staging_there = command.a1;
        const uint32_t staging = command.a2, result = command.a3, kept = command.a4;
        const uint32_t ranks = command.a5 & 0xFFFFu, other = command.a5 >> 16;
        const uint32_t mine = command.a8, theirs = command.a9;
        const uint32_t rank = reader.me() - km::first(command);
        const bool leader = rank == 0;
        uint32_t words[kReportWords] = {kReportMagic, rank};
        words[kReportStart] = km::wall();
        if (rank >= ranks) {
            return;
        }
        const km::Partition share = km::partition(kept, rank, ranks);
        words[kReportPages] = share.len;
        km::Gather posted = {};
        if (
            leader &&
            !km::gather_pages(
                reader,
                command,
                partial,
                theirs,
                kept,
                other,
                posted,
                km::keeper(0, kPortKept, ranks),
                staging_there
            )
        ) {
            return;
        }
        words[kReportPosted] = km::wall();
        if (!km::gathered(reader, command, kPortKept, share.len)) {
            return;
        }
        words[kReportDone] = km::wall();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        const uint32_t ours = get_write_ptr(kCbPages);
        const uint32_t theirs_at = ours + kFold * bytes;
        for (uint32_t k = 0; k < share.len;) {
            const uint32_t n = share.len - k < kFold ? share.len - k : kFold;
            const uint32_t page = mine + share.start + k;
            if (
                !km::read_pages(reader, command, args, partial, page, n, ours) ||
                !km::read_pages(reader, command, args, staging, page, n, theirs_at)
            ) {
                return;
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* a = km::l1(ours);
            const volatile tt_l1_ptr uint32_t* b = km::l1(theirs_at);
            for (uint32_t w = 0; w < n * bytes / 4; w++) {
                a[w] = a[w] + b[w];
            }
            if (!km::write_pages(reader, command, args, result, page, n, ours)) {
                return;
            }
            noc_async_write_barrier();
            k += n;
        }
        words[kReportFolded] = km::wall();
        // its fold done: one more on the leader's kPortFolded
        noc_semaphore_inc(
            reader.core_noc(
                reader.core_of(km::first(command)),
                reader.args().control + KM_CONTROL_PORT + kPortFolded * KM_PORT_STRIDE
            ),
            1
        );
        if (reader.poll_at(
            km::kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            reader.args().wait_limit
        ) != km::kGaveNone) {
            return;
        }
        if (
            leader &&
            (
                !km::gathered(reader, command, kPortFolded, ranks) ||
                !km::gather_pages(reader, command, result, mine, kept, other, posted) ||
                !km::gather_wait(reader, command, posted)
            )
        ) {
            return;
        }
        words[kReportEnd] = km::wall();
        report(reader, command, args, command.a6, rank, words);
    }

    // FILL and CHECK (collectives.h): the rank's share of the pages, kFold at a time.
    template <typename Args>
    void pattern(
        km::Reader& reader, const km_command_t& command, const Args& args, bool check
    ) {
        const uint32_t tensor = command.a0, salt = command.a3, other = command.a4;
        const uint32_t rank = reader.me() - km::first(command);
        const km::Partition share = km::partition(command.a2, rank, km::count(command));
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        const uint32_t buffer = get_write_ptr(kCbPages);
        volatile tt_l1_ptr uint32_t* l1 = km::l1(buffer);
        uint32_t bad = 0, first_bad = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < share.len;) {
            const uint32_t n = share.len - k < 2 * kFold ? share.len - k : 2 * kFold;
            const uint32_t page = command.a1 + share.start + k;
            if (check) {
                if (!km::read_pages(reader, command, args, tensor, page, n, buffer)) {
                    return;
                }
                noc_async_read_barrier();
            }
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t p = page + i;
                uint32_t want = p * kPagePrime + salt;
                if (other != 0) {
                    want += p * kPagePrime + other;
                }
                const uint32_t step = other != 0 ? 2 * kWordPrime : kWordPrime;
                volatile tt_l1_ptr uint32_t* words = l1 + i * bytes / 4;
                for (uint32_t w = 0; w < bytes / 4; w++, want += step) {
                    if (!check) {
                        words[w] = want;
                    } else if (words[w] != want) {
                        first_bad = bad == 0 ? p : first_bad;
                        bad++;
                    }
                }
            }
            if (!check) {
                if (!km::write_pages(reader, command, args, tensor, page, n, buffer)) {
                    return;
                }
                noc_async_write_barrier();
            }
            k += n;
        }
        if (check) {
            uint32_t words[kReportWords] = {
                kCheckMagic, rank, bad, first_bad, share.len
            };
            report(reader, command, args, command.a5, rank, words);
        }
    }

    // STAMPS (collectives.h): link core 0's stamps toward island a0, and this clock.
    template <typename Args>
    void stamps(km::Reader& reader, const km_command_t& command, const Args& args) {
        const km::Islands view = km::islands_at(reader.args().control);
        const uint32_t buffer = get_write_ptr(kCbPages);
        const uint32_t link = view.base() + view.ranks() + view.link(command.a0);
        noc_async_read(
            reader.core_noc(link, view.link_area() + KM_PUT_STAMPS),
            buffer,
            KM_STAMP_WORDS * 4
        );
        noc_async_read_barrier();
        km::l1(buffer)[KM_STAMP_WORDS] = km::wall();
        km::write_pages(reader, command, args, command.a1, 0, 1, buffer);
        noc_async_write_barrier();
    }

}  // namespace

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // every tensor's (one shape)
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerGather) {
            gather(reader, command, args);
        } else if (km::handler(command) == kHandlerReduce) {
            reduce(reader, command, args);
        } else if (km::handler(command) == kHandlerStamps) {
            stamps(reader, command, args);
        } else if (km::handler(command) == kHandlerFill) {
            pattern(reader, command, args, false);
        } else if (km::handler(command) == kHandlerCheck) {
            pattern(reader, command, args, true);
        }
    });
}
