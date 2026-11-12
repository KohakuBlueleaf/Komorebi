// What every resident kernel of a tile shares: runtime-owned buffers, the state words,
// the start-up arguments, the clock, and the bounded wait that observes the stop word.
#pragma once

#include <stdint.h>

#include "km_record.hpp"

namespace km {

    // Runtime-owned circular buffers: the tokens the reader hands to compute and
    // writer, and the per-record trace (only when tracing).
    constexpr uint32_t kCbTokenCompute = 29;
    constexpr uint32_t kCbTokenWriter = 30;
    constexpr uint32_t kCbTrace = 31;
    constexpr uint32_t kTokenBytes = 64;

    // State words (32-bit, from KM_CTL_STATE). A gave-up word is code | where << 8 |
    // slot << 16; its sequence number is in the next word. Counters saturate.
    constexpr uint32_t kStLast = 0;        // 16 words: last sequence consumed, per slot
    constexpr uint32_t kStReported = 16;   // 16 words: last sequence reported, per slot
    constexpr uint32_t kStReader = 32;     // reader gave-up word
    constexpr uint32_t kStReaderSeq = 33;  // its sequence number
    constexpr uint32_t kStWriter = 34;     // writer gave-up word
    constexpr uint32_t kStWriterSeq = 35;  // its sequence number
    constexpr uint32_t kStCompute = 36;    // compute gave-up word
    constexpr uint32_t kStComputeSeq = 37;  // its sequence number
    constexpr uint32_t kStRuns = 38;        // records the reader took
    constexpr uint32_t kStPasses = 39;      // ring passes in next(), all calls
    constexpr uint32_t kStNextCycles = 40;  // cycles in next(), all calls, / 16
    constexpr uint32_t kStCompleted = 41;   // records the writer finished
    constexpr uint32_t kStSeen = 42;        // records compute took
    constexpr uint32_t kStWrong = 43;       // bytes unpack, math, pack: bad job copies
    constexpr uint32_t kStEnds = 44;        // bytes reader, writer, compute: kEnd*
    constexpr uint32_t kStMagic = 45;       // kStateMagic | tag << 8 | me
    constexpr uint32_t kStSlotWrong = 46;   // records not run: header slot not theirs
    constexpr uint32_t kStTraceDropped = 47;  // trace stamps beyond its capacity
    constexpr uint32_t kStateWords = 48;
    constexpr uint32_t kStateMagic = 0x4B000000;

    // Gave-up codes, where codes and how a kernel left.
    constexpr uint32_t kGaveNone = 0;
    constexpr uint32_t kGaveStop = 1;       // the stop word was set
    constexpr uint32_t kGaveLimit = 2;      // the wait's bound passed
    constexpr uint32_t kWhereCheckin = 1;   // leader collecting start-up check-ins
    constexpr uint32_t kWhereToken = 2;     // reader waiting for free token pages
    constexpr uint32_t kWhereNext = 3;      // reader waiting for a record
    constexpr uint32_t kWhereStall = 4;     // a segment waiting for the stop word
    constexpr uint32_t kWhereComplete = 5;  // leader waiting for its members
    constexpr uint32_t kWhereWriter = 6;    // writer waiting for a token
    constexpr uint32_t kWhereData = 7;      // a segment's bounded buffer wait
    constexpr uint32_t kEndExit = 1;        // left on an EXIT record
    constexpr uint32_t kEndGaveUp = 2;      // left after its own wait gave up
    constexpr uint32_t kEndAbort = 3;       // left on the EXIT handed after a give-up
    constexpr uint32_t kEndReader = 0;      // bytes of kStEnds; kStWrong: unpack, ...
    constexpr uint32_t kEndWriter = 1;
    constexpr uint32_t kEndCompute = 2;

    // Scratch for outgoing NoC writes (16-byte parts of KM_CTL_OUT): check-in, READY,
    // hello (reader); acknowledgement (writer).
    constexpr uint32_t kOutCheckin = KM_CTL_OUT;
    constexpr uint32_t kOutReady = KM_CTL_OUT + 16;
    constexpr uint32_t kOutHello = KM_CTL_OUT + 32;
    constexpr uint32_t kOutAck = KM_CTL_OUT + 48;

    // Start-up arguments of the reader and the writer: runtime args 0 to kArgs - 1. The
    // program's own arguments follow. The compute kernel gets ctl and me only.
    struct Args {
        uint32_t ctl;         // control block address, the same on every core
        uint32_t me;          // this core's index in the core order
        uint32_t cores;       // DRAM address of the core table (one page)
        uint32_t disp_xy;     // NoC x | y << 16 of the dispatcher's ack words
        uint32_t ack_base;    // ack word of slot j at ack_base + j * KM_ACK_STRIDE
        uint32_t mailbox;     // hello address at the dispatcher; 0: send none
        uint32_t leader;      // start-up group: the leader's core index
        uint32_t rank;        // this core's check-in slot at the leader
        uint32_t members;     // leader: members that check in
        uint32_t ready_slot;  // leader: its READY word, at ready_base + slot * 16
        uint32_t ready_base;  // the dispatcher's READY words (not the ack words)
        uint32_t tag;         // per-process random tag, 16 bits
        uint32_t depth;       // ring slots scanned, 1 to KM_RING_SLOTS
        uint32_t row;         // cores per grid row (completion); 0: one level always
        uint32_t wait_limit;  // bound of a wait for another core, 1024-cycle units
        uint32_t idle_limit;  // bound of a wait for the next record, same units
        uint32_t trace;       // DRAM address of the trace, a page per core; 0: none
        uint32_t trace_cap;   // sequences traced: 1 to trace_cap
    };

    constexpr uint32_t kArgs = sizeof(Args) / 4;
    constexpr uint32_t kComputeArgs = 2;

    // Trace record per sequence number: 4 words of wall-clock cycles (low 32 bits).
    constexpr uint32_t kTraceSeen = 0;   // reader took the record
    constexpr uint32_t kTraceDone = 1;   // writer: outputs written
    constexpr uint32_t kTraceSent = 2;   // writer: completion sent (leader: pushed)
    constexpr uint32_t kTraceExtra = 3;  // reader: ring passes it took; with a stamp
                                         // in a9: cycles from the stamp to taking it
    constexpr uint32_t kTraceWords = 4;

    // Wall clock: reading the low word latches the high word.
    inline uint64_t wall64() {
        const uint32_t lo = (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
        const uint32_t hi = (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_H
            )
        );
        return static_cast<uint64_t>(hi) << 32 | lo;
    }

    inline uint32_t wall() {
        return (
            *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            )
        );
    }

    inline volatile tt_l1_ptr uint32_t* l1(uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    }

    // Adds v to a counter that stops at 0xFFFFFFFF.
    inline void add_sat(volatile tt_l1_ptr uint32_t* w, uint32_t v) {
        const uint32_t x = *w + v;
        *w = x < v ? 0xFFFFFFFFu : x;
    }

    // Byte `b` of state word `word` (each byte has one writer).
    inline void put_byte(
        volatile tt_l1_ptr uint32_t* state, uint32_t word, uint32_t b, uint32_t v
    ) {
        reinterpret_cast<volatile tt_l1_ptr uint8_t*>(state + word)[b] = v;
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace km {

    // A data-movement kernel's view of the control block and of its start-up arguments.
    class Core {
    public:
        Core() {
            uint32_t* f = &a_.ctl;
            for (uint32_t k = 0; k < kArgs; k++) {
                f[k] = get_arg_val<uint32_t>(k);
            }
            a_.tag &= 0xFFFFu;
            if (a_.depth == 0 || a_.depth > KM_RING_SLOTS) {
                a_.depth = KM_RING_SLOTS;
            }
        }

        const Args& args() const {
            return a_;
        }

        uint32_t me() const {
            return a_.me;
        }

        // The program's argument i (after the start-up arguments).
        uint32_t arg(uint32_t i) const {
            return get_arg_val<uint32_t>(kArgs + i);
        }

        volatile tt_l1_ptr uint32_t* ctl(uint32_t off) const {
            return l1(a_.ctl + off);
        }

        volatile tt_l1_ptr uint32_t* state() const {
            return ctl(KM_CTL_STATE);
        }

        // Nonzero stop word; the caller has invalidated the L1 cache.
        bool stop() const {
            return *ctl(KM_CTL_STOP) != 0;
        }

        // NoC address of `addr` on core `index` of the core table.
        uint64_t core_noc(uint32_t index, uint32_t addr) const {
            const uint32_t xy = ctl(KM_CTL_CORES)[index];
            return get_noc_addr(xy & 0xFFFFu, xy >> 16, addr);
        }

        uint64_t disp_noc(uint32_t addr) const {
            return get_noc_addr(a_.disp_xy & 0xFFFFu, a_.disp_xy >> 16, addr);
        }

        // Polls ready() until it holds (kGaveNone); kGaveLimit after `limit` 1024-cycle
        // units; kGaveStop `grace` units after the stop word is set (0: at once).
        template <typename Ready>
        uint32_t poll(Ready ready, uint32_t limit, uint32_t grace = 0) const {
            uint64_t end = wall64() + (static_cast<uint64_t>(limit) << 10);
            bool stopped = false;
            for (uint32_t i = 0;; i++) {
                invalidate_l1_cache();
                if (ready()) {
                    return kGaveNone;
                }
                if (!stopped && stop()) {
                    if (grace == 0) {
                        return kGaveStop;
                    }
                    stopped = true;
                    const uint64_t e = wall64() + (static_cast<uint64_t>(grace) << 10);
                    end = e < end ? e : end;
                }
                if ((i & 15) == 15 && wall64() > end) {
                    return stopped ? kGaveStop : kGaveLimit;
                }
            }
        }

        // Records a give-up in the state word pair `word` (kStReader or kStWriter); the
        // first one is kept, as it names the cause.
        void gave_up(
            uint32_t word, uint32_t code, uint32_t where, uint32_t slot, uint32_t seq
        ) const {
            if (state()[word] != 0) {
                return;
            }
            state()[word] = code | where << 8 | slot << 16;
            state()[word + 1] = seq;
        }

    private:
        Args a_;
    };

}  // namespace km
#endif
