// The three TRISCs: LOOP and PING as processors 2, 3 and 4, and PROBE: one TRISC reads
// a NoC register, or writes or reads 16 bytes through a NoC's command buffer itself.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "five.h"
#include "komorebi/device/km_worker.hpp"
#include "loop.hpp"
#include "noc/noc_parameters.h"

namespace {

#if defined(TRISC_UNPACK)
    constexpr uint32_t kThread = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t kThread = 1;
#else
    constexpr uint32_t kThread = 2;
#endif

    constexpr uint32_t kNocLimit = 1000000;  // cycles for a command buffer or an answer
    constexpr uint32_t kWriteBuffer = 0;     // the movers' write command buffer
    constexpr uint32_t kReadBuffer = 1;      // and their read one

    // Register `offset` (a NOC_* address of NoC 0) of command buffer `buffer` of `noc`.
    volatile uint32_t* reg(uint32_t noc, uint32_t buffer, uint32_t offset) {
        return reinterpret_cast<volatile uint32_t*>(
            (buffer << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) +
            offset
        );
    }

    uint32_t status(uint32_t noc, uint32_t counter) {
        return *reinterpret_cast<volatile uint32_t*>(
            (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(counter)
        );
    }

    // Waits until counter `counter` of `noc` differs from `before`. False: it ran out.
    bool changed(uint32_t noc, uint32_t counter, uint32_t before) {
        const uint32_t t0 = km::wall();
        while (status(noc, counter) == before) {
            if (km::wall() - t0 > kNocLimit) {
                return false;
            }
        }
        return true;
    }

    // The registers a request sets, kept and put back after it (the movers keep state
    // in their command buffers between requests).
    constexpr uint32_t kKept[] = {
        NOC_CTRL,
        NOC_TARG_ADDR_LO,
        NOC_TARG_ADDR_MID,
        NOC_TARG_ADDR_HI,
        NOC_RET_ADDR_LO,
        NOC_RET_ADDR_MID,
        NOC_RET_ADDR_HI,
        NOC_AT_LEN_BE,
    };
    constexpr uint32_t kKeptCount = sizeof(kKept) / sizeof(kKept[0]);

    // PROBE on this TRISC. Result words at kRProbe: register: NOC_NODE_ID,
    // NOC_ID_LOGICAL, cycles; write and read: the counter before, after, cycles, the
    // outcome (1 answered, 2 no answer, 3 the command buffer stayed busy); read: the 4
    // words read at + 4.
    void probe(uint32_t rows, const km_command_t& c) {
        volatile tt_l1_ptr uint32_t* row = five::words(rows);
        volatile tt_l1_ptr uint32_t* r = row + kResult +
            (kTrisc0 + kThread) * kResultWords;
        volatile tt_l1_ptr uint32_t* p = r + kRProbe;
        const uint32_t noc = c.a1, mode = c.a0;
        for (uint32_t w = 0; w < 8; w++) {
            p[w] = 0;
        }
        r[kRProbe - 1] = mode;  // what was asked, before the first access
        if (mode == kProbeRegister) {
            const uint32_t t0 = km::wall();
            p[0] = *reg(noc, 0, NOC_NODE_ID);
            p[1] = *reg(noc, 0, NOC_CFG(NOC_ID_LOGICAL));
            p[2] = km::wall() - t0;
            row[kTrisDone] = c.sequence;
            return;
        }
        // the movers have flushed their own requests and wait for this one
        bool ready = true;
        for (uint32_t m = 0; m < 2; m++) {
            ready = ready &&
                five::wait_word(row + kMoverReady + 4 * m, c.sequence, kNocLimit);
        }
        const uint32_t buffer = mode == kProbeWrite ? kWriteBuffer : kReadBuffer;
        const uint32_t counter =
            (mode == kProbeWrite ? NIU_MST_WR_ACK_RECEIVED : NIU_MST_RD_RESP_RECEIVED);
        uint32_t kept[kKeptCount];
        for (uint32_t i = 0; i < kKeptCount; i++) {
            kept[i] = *reg(noc, buffer, kKept[i]);
        }
        const uint32_t t0 = km::wall();
        while (ready && *reg(noc, buffer, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY) {
            if (km::wall() - t0 > kNocLimit) {
                ready = false;
            }
        }
        if (!ready) {
            p[3] = 3;
            row[kTrisDone] = c.sequence;
            return;
        }
        const uint32_t target = NOC_XY_ENCODING(c.a2 & 0xFFFF, c.a2 >> 16);
        const uint32_t before = status(noc, counter);
        if (mode == kProbeWrite) {
            for (uint32_t w = 0; w < 4; w++) {
                row[kTrisSource + w] = 0x7A000000u | (c.sequence & 0xFFFF) << 8 |
                    noc << 4 | w;
            }
            asm volatile("fence" ::: "memory");
            *reg(noc, buffer, NOC_CTRL) =
                (NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(1) |
                 NOC_CMD_RESP_MARKED);
            *reg(noc, buffer, NOC_TARG_ADDR_LO) = rows + 4 * kTrisSource;
            *reg(noc, buffer, NOC_TARG_ADDR_MID) = 0;
            *reg(noc, buffer, NOC_RET_ADDR_LO) = rows + 4 * kTrisLanding;
            *reg(noc, buffer, NOC_RET_ADDR_MID) = 0;
            *reg(noc, buffer, NOC_RET_ADDR_HI) = target;
        } else {
            const uint32_t own = *reg(noc, 0, NOC_CFG(NOC_ID_LOGICAL)) & 0xFFF;
            *reg(noc, buffer, NOC_CTRL) =
                (NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC |
                 NOC_CMD_STATIC_VC(1));
            *reg(noc, buffer, NOC_RET_ADDR_LO) = rows + 4 * kTrisReadInto;
            *reg(noc, buffer, NOC_RET_ADDR_MID) = 0;
            *reg(noc, buffer, NOC_RET_ADDR_HI) = own;
            *reg(noc, buffer, NOC_TARG_ADDR_LO) = rows + 4 * kReadSource;
            *reg(noc, buffer, NOC_TARG_ADDR_MID) = 0;
            *reg(noc, buffer, NOC_TARG_ADDR_HI) = target;
        }
        *reg(noc, buffer, NOC_AT_LEN_BE) = 16;
        const uint32_t sent = km::wall();
        *reg(noc, buffer, NOC_CMD_CTRL) = NOC_CTRL_SEND_REQ;
        const bool answered = changed(noc, counter, before);
        p[2] = km::wall() - sent;
        p[0] = before;
        p[1] = status(noc, counter);
        p[3] = answered ? 1 : 2;
        while (*reg(noc, buffer, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY) {
            if (km::wall() - sent > kNocLimit) {
                break;
            }
        }
        for (uint32_t i = 0; i < kKeptCount; i++) {
            *reg(noc, buffer, kKept[i]) = kept[i];
        }
        if (mode == kProbeRead) {
            invalidate_l1_cache();
            for (uint32_t w = 0; w < 4; w++) {
                p[4 + w] = row[kTrisReadInto + w];
            }
        }
        row[kTrisDone] = c.sequence;
    }

    // NOC on TRISC 0 (kHandlerNoc, bit 2), on NoC 0 through the movers' write or read
    // command buffer, after BRISC filled its table: a3 repetitions, each timed from its
    // start until the NoC's counter has every acknowledgement or answer.
    void noc_rates(uint32_t rows, const km_command_t& c) {
        constexpr uint32_t noc = 0, m = kTrisc0;
        volatile tt_l1_ptr uint32_t* row = five::words(rows);
        volatile tt_l1_ptr uint32_t* r = row + kResult + m * kResultWords + kRNoc;
        volatile tt_l1_ptr uint32_t* times = row + kTimes + m * kTimesWords;
        volatile tt_l1_ptr uint32_t* table = row + kTable + m * kTableWords;
        const uint32_t kind = c.a1;
        const bool multicast = kind == kNocMulticast;
        const bool writes =
            (kind == kNocWriteOne || kind == kNocWriteDram || kind == kNocWriteMany ||
             multicast);
        const bool many = kind == kNocWriteMany || kind == kNocReadMany;
        const uint32_t count = many ? (c.a2 < 128 ? c.a2 : 128) : 1;
        const uint32_t answers = multicast ? c.a2 : count;
        bool ready = true;
        for (uint32_t k = 0; k < 2; k++) {
            ready = ready &&
                five::wait_word(row + kMoverReady + 4 * k, c.sequence, kNocLimit);
        }
        const uint32_t buffer = writes ? kWriteBuffer : kReadBuffer;
        const uint32_t counter = writes ? NIU_MST_WR_ACK_RECEIVED
                                        : NIU_MST_RD_RESP_RECEIVED;
        const uint32_t own = *reg(noc, 0, NOC_CFG(NOC_ID_LOGICAL)) & 0xFFF;
        const uint32_t source = rows + 4 * (kSource + 16 * m);
        const uint32_t into = rows + 4 * (kReadInto + 16 * m);
        for (uint32_t w = 0; w < 16; w++) {
            row[kSource + 16 * m + w] = c.sequence << 8 | m << 4 | w;
        }
        asm volatile("fence" ::: "memory");
        const uint32_t control =
            (writes
                 ? (NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_RESP_MARKED |
                    (multicast ? NOC_CMD_STATIC_VC(4) | NOC_CMD_BRCST_PACKET |
                             NOC_CMD_PATH_RESERVE
                               : NOC_CMD_STATIC_VC(1)))
                 : NOC_CMD_CPY | NOC_CMD_RD | NOC_CMD_RESP_MARKED | NOC_CMD_VC_STATIC |
                     NOC_CMD_STATIC_VC(1));
        uint32_t kept[kKeptCount];
        for (uint32_t i = 0; i < kKeptCount; i++) {
            kept[i] = *reg(noc, buffer, kKept[i]);
        }
        const uint32_t reps = c.a3 < kTimesWords ? c.a3 : kTimesWords;
        uint32_t failures = ready ? 0 : 1, rep = 0;
        for (; ready && rep < reps; rep++) {
            const uint32_t before = status(noc, counter);
            const uint32_t t0 = km::wall();
            for (uint32_t i = 0; i < count; i++) {
                while (*reg(noc, buffer, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY) {
                    ;
                }
                const uint32_t low = table[2 * i];
                const uint32_t coordinates =
                    ((table[2 * i + 1] >> (NOC_ADDR_LOCAL_BITS - 32)) & 0xFFFFFF);
                *reg(noc, buffer, NOC_CTRL) = control;
                if (writes) {
                    *reg(noc, buffer, NOC_TARG_ADDR_LO) = source;
                    *reg(noc, buffer, NOC_RET_ADDR_LO) = low;
                    *reg(noc, buffer, NOC_RET_ADDR_MID) = 0;
                    *reg(noc, buffer, NOC_RET_ADDR_HI) = coordinates;
                } else {
                    *reg(noc, buffer, NOC_RET_ADDR_LO) = into;
                    *reg(noc, buffer, NOC_RET_ADDR_MID) = 0;
                    *reg(noc, buffer, NOC_RET_ADDR_HI) = own;
                    *reg(noc, buffer, NOC_TARG_ADDR_LO) = low;
                    *reg(noc, buffer, NOC_TARG_ADDR_MID) = 0;
                    *reg(noc, buffer, NOC_TARG_ADDR_HI) = coordinates;
                }
                *reg(noc, buffer, NOC_AT_LEN_BE) = 64;
                *reg(noc, buffer, NOC_CMD_CTRL) = NOC_CTRL_SEND_REQ;
            }
            while (status(noc, counter) - before < answers) {
                if (km::wall() - t0 > kNocLimit) {
                    failures++;
                    break;
                }
            }
            times[rep] = km::wall() - t0;
            if (failures != 0) {
                break;
            }
        }
        while (*reg(noc, buffer, NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY) {
            ;
        }
        for (uint32_t i = 0; i < kKeptCount; i++) {
            *reg(noc, buffer, kKept[i]) = kept[i];
        }
        r[0] = rep;
        r[1] = failures;
        row[kTrisDone] = c.sequence;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const uint32_t rows = compute.arg(0), blocks = compute.arg(1);
    const uint32_t me = kTrisc0 + kThread;
    compute.run([&](const km_command_t& c) {
        switch (km::handler(c)) {
            case kHandlerProbe:
                if (c.a3 == kThread) {
                    probe(rows, c);
                }
                break;
            case kHandlerLoop:
                five::loop(rows, blocks, me, c);
                break;
            case kHandlerNoc:
                if (kThread == 0 && ((c.a0 >> kTrisc0) & 1) != 0) {
                    noc_rates(rows, c);
                }
                break;
            case kHandlerPing:
                if (kThread == 0 &&
                    (c.a0 == kPairTriscTrisc || c.a0 == kPairTriscBrisc)) {
                    five::ping(rows, me, c);
                } else if (
                    (kThread == 0 && c.a0 == kPairBriscTrisc) ||
                    (kThread == 2 && c.a0 == kPairTriscTrisc)
                ) {
                    five::pong(rows, c);
                }
                break;
            default:
                break;
        }
    });
}
