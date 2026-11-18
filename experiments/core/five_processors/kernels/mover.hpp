// The handlers of the two movers: the reader on BRISC (NoC 0) and the writer on NCRISC
// (NoC 1) run the same body with their own processor number.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "five.h"
#include "komorebi/device/km_worker.hpp"
#include "loop.hpp"

namespace five {

    constexpr uint32_t kProbeLimit = 400000000;  // cycles the movers wait for the TRISC
    constexpr uint32_t kGoLimit = 100000000;     // cycles NCRISC waits for a start
    constexpr uint32_t kGoAhead = 2000;          // cycles from a start's word to it

    // This mover's software counters of `noc` set from the NoC's own counters: a TRISC
    // used the NoC, and its requests and acknowledgements are in them.
    inline void take_counters(uint32_t noc) {
        noc_reads_num_issued[noc] = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
        noc_nonposted_writes_num_issued[noc] = (
            NOC_STATUS_READ_REG(noc, NIU_MST_NONPOSTED_WR_REQ_SENT)
        );
        noc_nonposted_writes_acked[noc] = (
            NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED)
        );
        noc_nonposted_atomics_acked[noc] = (
            NOC_STATUS_READ_REG(noc, NIU_MST_ATOMIC_RESP_RECEIVED)
        );
        noc_posted_writes_num_issued[noc] = (
            NOC_STATUS_READ_REG(noc, NIU_MST_POSTED_WR_REQ_SENT)
        );
    }

    // PROBE on a mover: both NoCs' ids, flush, ready; after the TRISC's done, retake
    // the counters of the NoC it used if that NoC is this mover's.
    inline void probe(uint32_t rows, uint32_t m, const km_command_t& c) {
        volatile tt_l1_ptr uint32_t* row = words(rows);
        volatile tt_l1_ptr uint32_t* r = row + kResult + m * kResultWords + kRProbe;
        for (uint32_t noc = 0; noc < 2; noc++) {
            r[2 * noc] = NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID);
            r[2 * noc + 1] = NOC_CMD_BUF_READ_REG(noc, 0, NOC_CFG(NOC_ID_LOGICAL));
        }
        noc_async_write_barrier();
        noc_async_read_barrier();
        row[kMoverReady + 4 * m] = c.sequence;
        const uint32_t t0 = km::wall();
        const bool seen = wait_word(row + kTrisDone, c.sequence, kProbeLimit);
        r[4] = seen;
        r[5] = km::wall() - t0;
        if (seen && c.a1 == noc_index) {
            take_counters(noc_index);
        }
    }

    inline bool writes(uint32_t kind) {
        return (
            kind == kNocWriteOne || kind == kNocWriteDram || kind == kNocWriteMany ||
            kind == kNocMulticast
        );
    }

    inline bool many(uint32_t kind) {
        return kind == kNocWriteMany || kind == kNocReadMany;
    }

    // Into m's table: NoC addresses (on this mover's NoC) of `count` targets from rank
    // `first`, a DRAM bank or a multicast rectangle.
    template <typename Core>
    inline void fill_table(
        const Core& core,
        uint32_t rows,
        uint32_t m,
        uint32_t first,
        uint32_t count,
        const km_command_t& c
    ) {
        volatile tt_l1_ptr uint32_t* table = words(rows) + kTable + m * kTableWords;
        const uint32_t remote = (
            rows + 4 * (writes(c.a1) ? kLanding + 16 * m : kReadSource)
        );
        for (uint32_t i = 0; i < count; i++) {
            uint64_t address = 0;
            if (c.a1 == kNocWriteDram || c.a1 == kNocReadDram) {
                address = get_noc_addr_from_bank_id<true>(c.a4, c.a9);
            } else if (c.a1 == kNocMulticast) {
                const uint32_t start = noc_index == 0 ? c.a5 : c.a7;
                const uint32_t end = noc_index == 0 ? c.a6 : c.a8;
                address = (
                    get_noc_multicast_addr(
                        start & 0xFFFF,
                        start >> 16,
                        end & 0xFFFF,
                        end >> 16,
                        rows + 4 * kMulticastLanding
                    )
                );
            } else {
                address = core.core_noc(first + i, remote);
            }
            table[2 * i] = static_cast<uint32_t>(address);
            table[2 * i + 1] = static_cast<uint32_t>(address >> 32);
        }
    }

    // NOC for TRISC 0, on a mover: BRISC fills the TRISC's table on NoC 0; both flush,
    // say ready, wait for the TRISC's done, and BRISC takes NoC 0's counters again.
    template <typename Core>
    inline void noc_for_trisc(
        const Core& core, uint32_t rows, uint32_t m, const km_command_t& c
    ) {
        volatile tt_l1_ptr uint32_t* row = words(rows);
        if (m == kBrisc) {
            const uint32_t count = many(c.a1) ? c.a2 : 1;
            fill_table(core, rows, kTrisc0, c.a4, count < 128 ? count : 128, c);
        }
        noc_async_write_barrier();
        noc_async_read_barrier();
        row[kMoverReady + 4 * m] = c.sequence;
        if (wait_word(row + kTrisDone, c.sequence, kProbeLimit) && noc_index == 0) {
            take_counters(0);
        }
    }

    // NOC on mover m (see five.h): the targets' NoC addresses into the mover's table,
    // then a3 repetitions, each timed from its start to its last flush or barrier.
    template <typename Core>
    inline void noc(
        const Core& core, uint32_t rows, uint32_t m, const km_command_t& c
    ) {
        if (((c.a0 >> kTrisc0) & 1) != 0) {
            noc_for_trisc(core, rows, m, c);
            return;
        }
        if (((c.a0 >> m) & 1) == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* row = words(rows);
        volatile tt_l1_ptr uint32_t* r = row + kResult + m * kResultWords + kRNoc;
        volatile tt_l1_ptr uint32_t* times = row + kTimes + m * kTimesWords;
        volatile tt_l1_ptr uint32_t* table = row + kTable + m * kTableWords;
        const bool both = c.a0 == 3;
        const uint32_t kind = c.a1;
        uint32_t first = c.a4, count = many(kind) ? c.a2 : 1;
        if (both && many(kind)) {  // BRISC the first half, NCRISC the rest
            const uint32_t half = c.a2 / 2;
            first = m == 0 ? c.a4 : c.a4 + half;
            count = m == 0 ? half : c.a2 - half;
        }
        if (count > kTableWords / 2) {
            r[1] = 1;
            return;
        }
        fill_table(core, rows, m, first, count, c);
        const uint32_t source = rows + 4 * (kSource + 16 * m);
        const uint32_t into = rows + 4 * (kReadInto + 16 * m);
        for (uint32_t w = 0; w < 16; w++) {
            row[kSource + 16 * m + w] = c.sequence << 8 | m << 4 | w;
        }
        const uint32_t reps = c.a3 < kTimesWords ? c.a3 : kTimesWords;
        uint32_t failures = 0;
        for (uint32_t rep = 0; rep < reps; rep++) {
            uint32_t t0 = km::wall();
            if (both && m == 0) {  // BRISC sets the start and waits for NCRISC's end
                if (rep > 0 && !wait_word(row + kGoBack, rep, kGoLimit)) {
                    failures++;
                    break;
                }
                t0 = km::wall() + kGoAhead;
                row[kGoTime] = t0;
                row[kGo] = rep + 1;
            } else if (both) {
                if (!wait_word(row + kGo, rep + 1, kGoLimit)) {
                    failures++;
                    break;
                }
                t0 = row[kGoTime];
            }
            if (both) {
                while (static_cast<int32_t>(km::wall() - t0) < 0) {
                    ;
                }
            }
            if (kind == kNocMulticast) {
                const uint64_t address = (
                    static_cast<uint64_t>(table[1]) << 32 | table[0]
                );
                noc_async_write_multicast(source, address, 64, c.a2);
                noc_async_write_barrier();
            } else if (writes(kind)) {
                for (uint32_t i = 0; i < count; i++) {
                    const uint64_t address = (
                        static_cast<uint64_t>(table[2 * i + 1]) << 32 | table[2 * i]
                    );
                    noc_async_write(source, address, 64);
                }
                noc_async_write_barrier();
            } else {
                for (uint32_t i = 0; i < count; i++) {
                    const uint64_t address = (
                        static_cast<uint64_t>(table[2 * i + 1]) << 32 | table[2 * i]
                    );
                    noc_async_read(address, into, 64);
                }
                noc_async_read_barrier();
            }
            times[rep] = km::wall() - t0;
            if (both && m == 1) {
                row[kGoBack] = rep + 1;
            }
        }
        r[0] = reps;
        r[1] = failures;
    }

    // One mover's handlers; processor m (kBrisc or kNcrisc).
    template <typename Core>
    inline void body(const Core& core, uint32_t m, const km_command_t& c) {
        const uint32_t rows = core.arg(0), blocks = core.arg(1);
        switch (km::handler(c)) {
            case kHandlerProbe:
                probe(rows, m, c);
                break;
            case kHandlerLoop:
                loop(rows, blocks, m, c);
                break;
            case kHandlerNoc:
                noc(core, rows, m, c);
                break;
            case kHandlerStamp:
                if (m == kBrisc) {
                    const uint64_t now = km::wall64();
                    volatile tt_l1_ptr uint32_t* r = words(rows) + kResult + kRStamp;
                    r[2 * c.a0] = static_cast<uint32_t>(now);
                    r[2 * c.a0 + 1] = static_cast<uint32_t>(now >> 32);
                }
                break;
            case kHandlerPing:
                if (
                    m == kBrisc && (c.a0 == kPairBriscNcrisc || c.a0 == kPairBriscTrisc)
                ) {
                    ping(rows, m, c);
                } else if (
                    (m == kNcrisc && c.a0 == kPairBriscNcrisc) ||
                    (m == kBrisc && c.a0 == kPairTriscBrisc)
                ) {
                    pong(rows, c);
                }
                break;
            default:
                break;
        }
    }

}  // namespace five
