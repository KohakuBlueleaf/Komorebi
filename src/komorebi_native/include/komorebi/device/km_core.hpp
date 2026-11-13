// What every persistent kernel of a tile uses: runtime-owned buffers, the state words,
// the start-up arguments, the clock, and the wait (timeout, stop flag).
#pragma once

#include <stdint.h>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "km_command.hpp"

namespace km {

    // Runtime-owned circular buffers: the tokens the reader hands to compute and
    // writer, and the per-command trace (only when tracing).
    constexpr uint32_t kCbTokenCompute = 29;
    constexpr uint32_t kCbTokenWriter = 30;
    constexpr uint32_t kCbTrace = 31;
    constexpr uint32_t kTokenBytes = 64;
    // Word of each 16-byte DONE slot (word 0 counts the cores): the last sequence
    // whose compute body returned in that slot, written by unpack, read by the writer.
    constexpr uint32_t kDoneCompute = 1;

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
    constexpr uint32_t kStRuns = 38;        // commands the reader took
    constexpr uint32_t kStPasses = 39;      // ring passes in next(), all calls
    constexpr uint32_t kStNextCycles = 40;  // cycles in next(), all calls, / 16
    constexpr uint32_t kStCompleted = 41;   // commands the writer finished
    constexpr uint32_t kStSeen = 42;        // commands compute took
    constexpr uint32_t kStWrong = 43;       // bytes unpack, math, pack: bad job copies
    constexpr uint32_t kStEnds = 44;        // bytes reader, writer, compute: kEnd*
    constexpr uint32_t kStMagic = 45;       // kStateMagic | tag << 8 | me
    constexpr uint32_t kStSlotWrong = 46;   // commands not run: header slot not theirs
    constexpr uint32_t kStTraceDropped = 47;  // commands beyond the trace's capacity
    constexpr uint32_t kStateWords = 48;
    constexpr uint32_t kStateMagic = 0x4B000000;

    // Gave-up codes, where codes and how a kernel left.
    constexpr uint32_t kGaveNone = 0;
    constexpr uint32_t kGaveStop = 1;     // the stop flag was set
    constexpr uint32_t kGaveLimit = 2;    // the wait timed out
    constexpr uint32_t kGaveBadItem = 3;  // a multi-page item would pass the buffer end
    constexpr uint32_t kGaveBadBroadcast = 4;  // a broadcast's arguments do not fit the command
    constexpr uint32_t kWhereCheckin = 1;   // leader collecting start-up check-ins
    constexpr uint32_t kWhereToken = 2;     // reader waiting for free token pages
    constexpr uint32_t kWhereNext = 3;      // reader waiting for a command
    constexpr uint32_t kWhereStall = 4;     // a handler waiting for the stop flag
    constexpr uint32_t kWhereComplete = 5;  // leader waiting for its cores
    constexpr uint32_t kWhereWriter = 6;    // writer waiting for a token
    constexpr uint32_t kWhereData = 7;      // a handler's timed buffer wait
    constexpr uint32_t kWhereCToken = 8;    // compute (unpack) waiting for a token
    constexpr uint32_t kWhereCInput = 9;    // compute (unpack) waiting for input pages
    constexpr uint32_t kWhereCOutput = 10;  // compute (pack) waiting for output space
    constexpr uint32_t kWhereCDone = 11;    // writer: compute's and reader's bodies
    constexpr uint32_t kEndExit = 1;        // left on an EXIT command
    constexpr uint32_t kEndGaveUp = 2;      // left after its own wait gave up
    constexpr uint32_t kEndAbort = 3;       // left on the EXIT handed after a give-up
    constexpr uint32_t kEndReader = 0;      // bytes of kStEnds; kStWrong: unpack, ...
    constexpr uint32_t kEndWriter = 1;
    constexpr uint32_t kEndCompute = 2;

    // Where codes of broadcast (km_broadcast.hpp).
    constexpr uint32_t kWhereBroadcastBegin = 12;  // broadcast: arguments or buffer at a command
    constexpr uint32_t kWhereBroadcastDone = 13;  // broadcast sender: DONE markers, its NoC barriers
    constexpr uint32_t kWhereBroadcastValid = 14;  // broadcast receiver waiting for VALID
    constexpr uint32_t kWhereBroadcastFree = 15;  // broadcast: its compute's pops, its last markers

    // Where codes of a port's wait (reader) and of an emit (writer): the gave-up word's
    // top byte is the port; the port keeps the count it wanted (KM_PORT_WANT).
    constexpr uint32_t kWherePort = 17;
    constexpr uint32_t kWhereEmit = 18;

    // Where codes of a dynamic run (km_dyn.hpp), KM_DYN_GAVE word 0; word 1 position,
    // word 2 run, word 3 the word 0 at a gap or a lost answer's destination.
    constexpr uint32_t kWhereDynGap = 19;       // reserved message: not landed in time
    constexpr uint32_t kWhereDynOverflow = 20;  // IN_TAIL - IN_HEAD passed the capacity
    constexpr uint32_t kWhereDynAtomic = 21;   // a reservation was not answered in time
    constexpr uint32_t kWhereDynFlush = 22;    // writer: message writes did not leave
    constexpr uint32_t kWhereDynRead = 23;     // reader: message reads did not land
    constexpr uint32_t kWhereDynInPage = 24;   // reader: no free packet page
    constexpr uint32_t kWhereDynLine = 25;     // the status line write did not leave
    constexpr uint32_t kWhereDynBadDest = 26;  // forward or send outside the group
    constexpr uint32_t kWhereDynNext = 27;     // compute's wait for a packet
    constexpr uint32_t kWhereDynOutPage = 28;  // compute's wait for an outgoing page
    constexpr uint32_t kWhereDynOutbox = 29;   // writer: wait for an outgoing message
    constexpr uint32_t kWhereDynStopped = 30;  // the stop flag ended the run
    constexpr uint32_t kWhereDynAlign = 31;    // pages or messages not 64-B aligned
    constexpr uint32_t kWhereDynLost = 32;     // a reservation's answer never landed
    constexpr uint32_t kWhereDynFill = 33;     // the start-up zero fill of the queue
    constexpr uint32_t kWhereDynFull = 34;  // positions reached 2^30: reload the program
    constexpr uint32_t kWhereDynCounter = 35;  // counter tile not ready, or no cores
    static_assert(
        kWhereDynGap == kWhereEmit + 1 && kWhereDynCounter == kWhereDynGap + 16 &&
            kWhereDynCounter < 256,
        "dynamic where codes: 19 to 35, after the runtime's, one byte"
    );

    // Slots of KM_CTL_OUT (16 bytes each), the sources of small NoC writes: check-in,
    // READY, hello (reader); acknowledgement (writer).
    constexpr uint32_t kOutCheckin = 0;
    constexpr uint32_t kOutReady = 1;
    constexpr uint32_t kOutHello = 2;
    constexpr uint32_t kOutAck = 3;
    constexpr uint32_t kOutSlots = (KM_CTL_CORES - KM_CTL_OUT) / 16;
    static_assert(KM_CTL_OUT % 16 == 0 && kOutAck < kOutSlots, "KM_CTL_OUT slots");
    static_assert(KM_ACK_STRIDE == 16, "ack words are 16-byte slots");

    // Start-up arguments of the reader and the writer: runtime args 0 to kArgs - 1. The
    // program's own arguments follow. The compute kernel gets the kCArg* words.
    struct Args {
        uint32_t ctl;         // control block address, the same on every core
        uint32_t me;          // this core's index in the core order
        uint32_t cores;       // DRAM address of the core table (one page)
        uint32_t disp_xy;     // NoC x | y << 16 of the dispatcher's ack words
        uint32_t ack_base;    // ack word of slot j at ack_base + j * KM_ACK_STRIDE
        uint32_t mailbox;     // hello address at the dispatcher; 0: send none
        uint32_t leader;      // start-up group: the leader's core index
        uint32_t rank;        // this core's check-in slot at the leader
        uint32_t peers;       // leader: cores that check in
        uint32_t ready_slot;  // leader: its READY word, at ready_base + slot * 16
        uint32_t ready_base;  // the dispatcher's READY words (not the ack words)
        uint32_t tag;         // per-process random tag, 16 bits
        uint32_t workers;     // cores 0 to workers - 1 run commands (an EXIT's group)
        uint32_t depth;       // ring slots scanned, 1 to KM_RING_SLOTS
        uint32_t row;         // cores per grid row (completion); 0: one level always
        uint32_t wait_limit;  // timeout of a wait for another core, 1024-cycle units
        uint32_t idle_limit;  // timeout of a wait for the next command; 0: none
        uint32_t trace;       // DRAM address of the trace, a page per core; 0: none
        uint32_t trace_cap;   // sequences traced: 1 to trace_cap
    };

    constexpr uint32_t kArgs = sizeof(Args) / 4;
    constexpr uint32_t kCArgCtl = 0;   // compute: control block address
    constexpr uint32_t kCArgMe = 1;    // this core's index
    constexpr uint32_t kCArgWait = 2;  // wait_limit, as for reader and writer
    constexpr uint32_t kCArgIdle = 3;  // idle_limit
    constexpr uint32_t kComputeArgs = 4;

    // Trace command per sequence number: 4 words of wall-clock cycles (low 32 bits).
    constexpr uint32_t kTraceSeen = 0;  // reader took the command
    constexpr uint32_t kTraceDone = 1;  // writer: outputs written
    constexpr uint32_t kTraceSent = 2;  // writer: completion sent (leader: pushed)
    constexpr uint32_t kTraceExtra = 3;  // reader: ring passes it took; with a timestamp
                          // in a9: cycles from the timestamp to taking it
    constexpr uint32_t kTraceWords = 4;

    // A 64-bit count from a live high word read before and after the low word; a carry
    // between them retries, and a second one returns the carry's own instant.
    template <typename High, typename Low>
    inline uint64_t wall64_of(High high, Low low) {
        for (uint32_t i = 0; i < 2; i++) {
            const uint32_t hi = high();
            const uint32_t lo = low();
            if (high() == hi) {
                return static_cast<uint64_t>(hi) << 32 | lo;
            }
        }
        return static_cast<uint64_t>(high()) << 32;
    }

    // Wall clock. Not the latched high word: every core of the tile latches it with
    // its own low-word reads, so it can be torn by 2^32 cycles at a carry.
    inline uint64_t wall64() {
        return wall64_of(
            [] {
                return (
                    *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                        RISCV_DEBUG_REG_WALL_CLOCK_1
                    )
                );
            },
            [] {
                return (
                    *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                        RISCV_DEBUG_REG_WALL_CLOCK_L
                    )
                );
            }
        );
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

    // True if `pages` pages from `ptr` (this side's fifo_rd_ptr or fifo_wr_ptr, in its
    // units) end at or before the limit of buffer cb: an item is one contiguous region.
    inline bool item_fits(uint32_t cb, uint32_t ptr, uint32_t pages) {
        const LocalCBInterface& i = get_local_cb_interface(cb);
        return ptr + pages * i.fifo_page_size <= i.fifo_limit;
    }

    // Polls ready() until it holds (kGaveNone); kGaveLimit after `limit` 1024-cycle
    // units (0: no limit); kGaveStop `grace` units after *stop is set (0: at once).
    template <typename Ready>
    inline uint32_t poll_stop(
        volatile tt_l1_ptr uint32_t* stop, Ready ready, uint32_t limit, uint32_t grace
    ) {
        uint64_t end = (
            limit == 0 ? ~0ull : wall64() + (static_cast<uint64_t>(limit) << 10)
        );
        bool stopped = false;
        for (uint32_t i = 0;; i++) {
            invalidate_l1_cache();
            if (ready()) {
                return kGaveNone;
            }
            if (!stopped && *stop != 0) {
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

    // Commands a give-up in state word pair `word` (the first one is kept: it names the
    // cause) and sets the tile's own stop flag, so all three kernels leave.
    inline void command_gave_up(
        uint32_t ctl,
        uint32_t word,
        uint32_t code,
        uint32_t where,
        uint32_t slot,
        uint32_t seq
    ) {
        volatile tt_l1_ptr uint32_t* state = l1(ctl + KM_CTL_STATE);
        if (state[word] == 0) {
            state[word] = code | where << 8 | slot << 16;
            state[word + 1] = seq;
        }
        *l1(ctl + KM_CTL_STOP) = 1;
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // Word 0 of slot `slot` of KM_CTL_OUT in the control block at `ctl`.
    inline volatile tt_l1_ptr uint32_t* out_slot(uint32_t ctl, uint32_t slot) {
        return l1(ctl + KM_CTL_OUT + 16 * slot);
    }

    // A small NoC write: Bytes from slot From of KM_CTL_OUT to slot `to` above the
    // 16-byte aligned `base`, so both ends share their low four address bits.
    template <uint32_t From, uint32_t Bytes>
    inline void small_write(uint32_t ctl, uint64_t base, uint32_t to) {
        static_assert(From < kOutSlots && Bytes > 0 && Bytes <= 16, "one slot");
        noc_async_write(ctl + KM_CTL_OUT + 16 * From, base + 16 * to, Bytes);
    }

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

        // Nonzero stop flag; the caller has invalidated the L1 cache.
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

        // poll_stop on this tile's stop flag.
        template <typename Ready>
        uint32_t poll(Ready ready, uint32_t limit, uint32_t grace = 0) const {
            return poll_stop(ctl(KM_CTL_STOP), ready, limit, grace);
        }

        // command_gave_up for this tile (kStReader or kStWriter).
        void gave_up(
            uint32_t word, uint32_t code, uint32_t where, uint32_t slot, uint32_t seq
        ) const {
            command_gave_up(a_.ctl, word, code, where, slot, seq);
        }

    private:
        Args a_;
    };

}  // namespace km
#endif
