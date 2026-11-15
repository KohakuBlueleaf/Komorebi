// The probe's persistent kernel on data-movement core 0: on a command, loads what it
// names into the slots, starts the four parked cores, runs its own share, reports.
#include <stdint.h>

#include "api.h"
#include "costs.h"
#include "internal/circular_buffer_init.h"

namespace {

    // RISCV_IC_INVALIDATE bits (the firmware's): this core, compute 0..2, the other
    // data-movement core
    constexpr uint32_t kIcThis = 0x1, kIcCompute = 0x2 | 0x4 | 0x8, kIcOther = 0x10;
    constexpr uint64_t kDoneLimit = 1500000000;  // cycles (about 1.1 s) for a go
    // bfloat16 bits of 1 .. 16: an input tile's values, (j % 16) + 1
    constexpr uint16_t kBf16Small[16] = {
        0x3F80,
        0x4000,
        0x4040,
        0x4080,
        0x40A0,
        0x40C0,
        0x40E0,
        0x4100,
        0x4110,
        0x4120,
        0x4130,
        0x4140,
        0x4150,
        0x4160,
        0x4170,
        0x4180,
    };

    volatile uint32_t part_calls = 0;

    inline uint64_t wall64() {
        volatile tt_reg_ptr uint32_t*
            lo = (
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_L
                )
            );
        volatile tt_reg_ptr uint32_t*
            hi = (
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                    RISCV_DEBUG_REG_WALL_CLOCK_1
                )
            );
        for (;;) {
            const uint32_t h = *hi, l = *lo;
            if (*hi == h) {
                return static_cast<uint64_t>(h) << 32 | l;
            }
        }
    }

    __attribute__((noinline)) uint32_t mix(uint32_t x) {
        return cs::mix(x);
    }

    // The table of the runtime's calls an overlay makes (the table form).
    __attribute__((noinline)) bool api_reserve(
        uint32_t cb, uint32_t pages, volatile uint32_t* stop
    ) {
        return cs::reserve(cb, pages, stop);
    }

    __attribute__((noinline)) uint32_t api_write_ptr(uint32_t cb) {
        return get_write_ptr(cb);
    }

    __attribute__((noinline)) void api_read_page(
        uint32_t tensor, uint32_t page, uint32_t bytes, uint32_t l1
    ) {
        cs::read_page(tensor, page, bytes, l1);
    }

    __attribute__((noinline)) bool api_read_barrier() {
        return cs::read_barrier();
    }

    __attribute__((noinline)) void api_push(uint32_t cb, uint32_t pages) {
        cb_push_back(cb, pages);
    }

    // Invalidates the i-cache of this tile's cores in `mask` (the firmware's write).
    inline void invalidate_icache(uint32_t mask) {
        volatile tt_reg_ptr uint32_t*
            cfg = (reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE));
        cfg[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = mask;
    }

    // The four other cores: parked when every done word equals the go word.
    struct Others {
        volatile tt_l1_ptr uint32_t* c;

        bool parked(uint64_t limit) const {
            const uint64_t t0 = wall64();
            for (uint32_t k = 0; k < 4; k++) {
                for (;;) {
                    invalidate_l1_cache();
                    if (c[cs::kDone + k] == c[cs::kGo]) {
                        break;
                    }
                    if (wall64() - t0 > limit) {
                        return false;
                    }
                }
            }
            return true;
        }

        void go(uint32_t op) const {
            c[cs::kParkedOp] = op;
            c[cs::kGo] = c[cs::kGo] + 1;
        }

        // The first failed core's status, else kOk.
        uint32_t status() const {
            for (uint32_t k = 0; k < 4; k++) {
                if (c[cs::kOut + 8 * k + cs::kOutStatus] != cs::kOk) {
                    return c[cs::kOut + 8 * k + cs::kOutStatus];
                }
            }
            return cs::kOk;
        }

        uint32_t out(uint32_t k, uint32_t word) const {
            return c[cs::kOut + 8 * k + word];
        }
    };

    // Reads `bytes` from DRAM (bank, address) into L1 at `l1`; the barrier is apart.
    inline void load(uint32_t bank, uint32_t address, uint32_t bytes, uint32_t l1) {
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, address), l1, bytes);
    }

    // Writes the input pattern into every page of a buffer [base, base + bytes).
    void fill(uint32_t base, uint32_t bytes) {
        volatile tt_l1_ptr uint16_t*
            in = (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base));
        for (uint32_t j = 0; j < bytes / 2; j++) {
            in[j] = kBf16Small[j % 16];
        }
    }

    // Pushes `count` input tiles into kCbIn (the pages already hold the pattern);
    // counts write pointers outside [low, high). Returns a status.
    uint32_t feed(
        uint32_t count,
        volatile uint32_t* stop,
        uint32_t low,
        uint32_t high,
        uint32_t& bad
    ) {
        for (uint32_t i = 0; i < count; i++) {
            if (!cs::reserve(cs::kCbIn, 1, stop)) {
                return cs::kWaitTimeout;
            }
            const uint32_t at = get_write_ptr(cs::kCbIn);
            bad += at < low || at + cs::kTileBytes > high;
            cb_push_back(cs::kCbIn, 1);
        }
        return cs::kOk;
    }

    // Waits (timed) until every page pushed into `cb` has been popped.
    bool drained(uint32_t cb, uint64_t limit) {
        const uint64_t t0 = wall64();
        for (;;) {
            const uint32_t received = *get_cb_tiles_received_ptr(cb) & 0xFFFFu;
            const uint32_t acked = *get_cb_tiles_acked_ptr(cb) & 0xFFFFu;
            if (received == acked) {
                return true;
            }
            if (wall64() - t0 > limit) {
                return false;
            }
        }
    }

    // A configuration word of `cb` (0 address, 1 bytes, 2 pages, 3 page bytes).
    inline uint32_t layout_word(uint32_t layout, uint32_t cb, uint32_t word) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(layout)[4 * cb + word];
    }

}  // namespace

void kernel_main() {
    const uint32_t control = get_arg_val<uint32_t>(0);
    const uint32_t resp_bank = get_arg_val<uint32_t>(1);
    const uint32_t resp_addr = get_arg_val<uint32_t>(2);
    const uint32_t slots = get_arg_val<uint32_t>(3);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(4);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 20;
    const uint32_t tensor = get_arg_val<uint32_t>(6);
    const uint32_t page_bytes = get_arg_val<uint32_t>(7);
    volatile tt_l1_ptr uint32_t* c = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control)
    );
    volatile tt_l1_ptr uint32_t* r = c + cs::kResp;
    const Others others{c};
    const uint64_t resp_noc = get_noc_addr_from_bank_id<true>(resp_bank, resp_addr);
    cs::MoverApi api{};
    api.reserve = api_reserve;
    api.write_ptr = api_write_ptr;
    api.read_page = api_read_page;
    api.read_barrier = api_read_barrier;
    api.push = api_push;
    cs::Call call{};
    call.tensor = tensor;
    call.page_bytes = page_bytes;
    call.stop = c + cs::kStop;
    call.api = &api;
    call.mix = mix;
    call.calls = &part_calls;
    uint32_t held[cs::kSlots] = {};  // the overlay id each slot holds
    uint32_t held_program = 0;       // the program whose five parts are held
    uint32_t last = 0;
    uint64_t seen_at = wall64();
    for (;;) {
        invalidate_l1_cache();
        const uint32_t sequence = c[cs::kSequence];
        if (sequence == last) {
            if (wall64() - seen_at > idle) {
                c[cs::kGo] = cs::kLeave;
                r[0] = last;
                r[1] = cs::kIdle;
                noc_async_write(control + 4 * cs::kResp, resp_noc, 64);
                noc_async_write_barrier();
                return;
            }
            continue;
        }
        uint64_t t_seen = wall64();
        seen_at = t_seen;
        last = sequence;
        for (uint32_t i = 1; i < 16; i++) {
            r[i] = 0;
        }
        const uint32_t op = c[cs::kOp], flags = c[cs::kFlags];
        uint32_t status = cs::kOk;
        if (flags & cs::kFlagAt) {
            // starts at the wall-clock count in kAt: every core of a broadcast begins
            // together, and its times count from there
            while (
                static_cast<int32_t>(static_cast<uint32_t>(wall64()) - c[cs::kAt]) < 0
            ) {
                if (wall64() - t_seen > kDoneLimit) {
                    break;
                }
            }
            r[13] = static_cast<uint32_t>(t_seen);
            t_seen = wall64();
        }
        if (op == cs::kOpClock || op == cs::kOpExit) {
            r[2] = static_cast<uint32_t>(t_seen);
        } else if (!others.parked(cs::kWaitLimit)) {
            status = cs::kNotParked;
        } else if (op == cs::kOpRead) {
            // the reader loop inline or in a slot; data-movement 1 drains and sums
            const uint32_t slot_index = c[cs::kSlot] % cs::kSlots;
            const uint32_t slot = slots + slot_index * slot_bytes;
            const bool overlay = (flags & cs::kFlagOverlay) != 0;
            if (overlay && held[slot_index] != c[cs::kProgram]) {
                const uint64_t t0 = wall64();
                if (c[cs::kParts + 2] > slot_bytes) {
                    status = cs::kTooBig;
                } else {
                    load(c[cs::kParts], c[cs::kParts + 1], c[cs::kParts + 2], slot);
                    status = cs::read_barrier() ? cs::kOk : cs::kReadTimeout;
                }
                held[slot_index] = 0;
                held_program = 0;
                if (status == cs::kOk) {
                    invalidate_icache(kIcThis);
                    held[slot_index] = c[cs::kProgram];
                    r[4] = 1;
                }
                r[5] = static_cast<uint32_t>(wall64() - t0);
            }
            if (status == cs::kOk) {
                call.count = c[cs::kCount];
                call.arg = c[cs::kFirst];
                others.go(cs::kParkedDrain);
                const uint64_t t0 = wall64();
                status =

                    overlay ? reinterpret_cast<cs::Entry>(slot)(&call)
                            : cs::reader_loop(call, cs::InlineMover{});
                r[6] = static_cast<uint32_t>(wall64() - t0);
                if (!others.parked(kDoneLimit)) {
                    status = cs::kNotParked;
                } else if (status == cs::kOk) {
                    status = others.status();
                }
                r[2] = others.out(0, cs::kOutResult);
                r[8] = others.out(0, cs::kOutCycles);
            }
        } else if (op == cs::kOpCompute) {
            // input tiles into kCbIn; compute (inline or its three parts in slots)
            // multiplies; data-movement 1 checks a sample of every output tile
            const bool overlay = (flags & cs::kFlagOverlay) != 0;
            if (overlay && held_program != c[cs::kProgram]) {
                const uint64_t t0 = wall64();
                held_program = 0;
                for (uint32_t k = 2; k < 5 && status == cs::kOk; k++) {
                    volatile tt_l1_ptr uint32_t* part = c + cs::kParts + 3 * k;
                    if (part[2] > slot_bytes) {
                        status = cs::kTooBig;
                    } else {
                        load(part[0], part[1], part[2], slots + k * slot_bytes);
                    }
                }
                if (status == cs::kOk && !cs::read_barrier()) {
                    status = cs::kReadTimeout;
                }
                if (status == cs::kOk) {
                    invalidate_icache(kIcCompute);
                    held_program = c[cs::kProgram];
                    r[4] = 1;
                }
                r[5] = static_cast<uint32_t>(wall64() - t0);
            }
            if (status == cs::kOk) {
                const LocalCBInterface& in = get_local_cb_interface(cs::kCbIn);
                fill(in.fifo_limit - in.fifo_size, in.fifo_size);
                others.go(overlay ? cs::kParkedComputeSlot : cs::kParkedCompute);
                const uint64_t t0 = wall64();
                uint32_t bad = 0;
                status = feed(c[cs::kCount], call.stop, 0, 0xFFFFFFFFu, bad);
                if (!others.parked(kDoneLimit)) {
                    status = cs::kNotParked;
                } else if (status == cs::kOk) {
                    status = others.status();
                }
                r[6] = static_cast<uint32_t>(wall64() - t0);
                r[2] = others.out(0, cs::kOutResult);  // wrong tiles
                for (uint32_t k = 0; k < 3; k++) {
                    r[8 + k] = others.out(1 + k, cs::kOutCycles);
                }
            }
        } else if (op == cs::kOpSwitch) {
            // the program check; when it differs, five parts loaded and invalidated
            // while the other cores are parked; then every core runs its part
            const uint32_t program = c[cs::kProgram];
            if (program != held_program) {
                held_program = 0;
                for (uint32_t k = 0; k < cs::kSlots; k++) {
                    held[k] = 0;
                }
                for (uint32_t k = 0; k < 5 && status == cs::kOk; k++) {
                    volatile tt_l1_ptr uint32_t* part = c + cs::kParts + 3 * k;
                    if (part[2] > slot_bytes) {
                        status = cs::kTooBig;
                    } else {
                        load(part[0], part[1], part[2], slots + k * slot_bytes);
                    }
                }
                if (status == cs::kOk && !cs::read_barrier()) {
                    status = cs::kReadTimeout;
                }
                if (status == cs::kOk) {
                    invalidate_icache(kIcThis | kIcCompute | kIcOther);
                    held_program = program;
                    r[4] = 1;
                }
                r[5] = static_cast<uint32_t>(wall64() - t_seen);
            }
            if (status == cs::kOk) {
                call.arg = c[cs::kArg];
                const uint64_t t_go = wall64();
                others.go(cs::kParkedPart);
                r[6] = static_cast<uint32_t>(t_go - t_seen);
                r[2] = reinterpret_cast<cs::Entry>(slots)(&call);
                if (!others.parked(kDoneLimit)) {
                    status = cs::kNotParked;
                } else {
                    status = others.status();
                }
                uint32_t release = 0;
                for (uint32_t k = 0; k < 4; k++) {
                    r[8 + k] = others.out(k, cs::kOutResult);
                    const uint32_t started = (
                        others.out(k, cs::kOutStart) - static_cast<uint32_t>(t_go)
                    );
                    release = started > release ? started : release;
                }
                r[12] = release;
            }
        } else if (op == cs::kOpLayouts) {
            // kCount switches between two buffer configurations, each between two
            // runs of kTiles tiles through compute, every output value checked
            const uint32_t layouts[2] = {c[cs::kLayoutA], c[cs::kLayoutB]};
            const uint32_t masks[2] = {c[cs::kMaskA], c[cs::kMaskB]};
            for (uint32_t l = 0; l < 2; l++) {
                fill(
                    layout_word(layouts[l], cs::kCbIn, 0),
                    layout_word(layouts[l], cs::kCbIn, 1)
                );
            }
            uint32_t wrong = 0, bad = 0, tiles = 0;
            uint64_t switch_sum = 0;
            uint32_t switch_min = 0xFFFFFFFFu, switch_max = 0;
            for (uint32_t s = 0; s <= c[cs::kCount] + 1 && status == cs::kOk; s++) {
                // s = 0 sets up layout A; then a run, then a switch, alternately
                const uint32_t l = s & 1u;
                if (s > 0) {
                    const uint32_t low = layout_word(layouts[l ^ 1], cs::kCbIn, 0);
                    const uint32_t high = (
                        low + layout_word(layouts[l ^ 1], cs::kCbIn, 1)
                    );
                    others.go(cs::kParkedCheck);
                    status = feed(c[cs::kTiles], call.stop, low, high, bad);
                    if (!others.parked(kDoneLimit)) {
                        status = cs::kNotParked;
                    } else if (status == cs::kOk) {
                        status = others.status();
                    }
                    wrong += others.out(0, cs::kOutResult);
                    bad += others.out(0, cs::kOutBad);
                    tiles += c[cs::kTiles];
                    if (s == c[cs::kCount] + 1 || status != cs::kOk) {
                        break;
                    }
                }
                // the switch: every core idle, every page popped, the counters
                // zeroed, then each core reads the new configuration
                const uint64_t t0 = wall64();
                if (
                    !drained(cs::kCbIn, cs::kWaitLimit) ||
                    !drained(cs::kCbOut, cs::kWaitLimit)
                ) {
                    status = cs::kWaitTimeout;
                    break;
                }
                for (uint32_t cb = 0; cb < 32; cb++) {
                    if (((masks[0] | masks[1]) >> cb) & 1u) {
                        *get_cb_tiles_received_ptr(cb) = 0;
                        *get_cb_tiles_acked_ptr(cb) = 0;
                    }
                }
                setup_local_cb_read_write_interfaces<true, true, false>(
                    reinterpret_cast<uint32_t tt_l1_ptr*>(layouts[l]), 0, masks[l]
                );
                c[cs::kParkedLayout] = layouts[l];
                c[cs::kParkedMask] = masks[l];
                others.go(cs::kParkedSetup);
                if (!others.parked(cs::kWaitLimit)) {
                    status = cs::kNotParked;
                    break;
                }
                const uint32_t t = static_cast<uint32_t>(wall64() - t0);
                if (s > 0) {
                    switch_sum += t;
                    switch_min = t < switch_min ? t : switch_min;
                    switch_max = t > switch_max ? t : switch_max;
                }
                if (status == cs::kOk) {
                    status = others.status();
                }
            }
            r[2] = wrong;
            r[8] = tiles;
            r[9] = static_cast<uint32_t>(switch_sum);
            r[10] = static_cast<uint32_t>(switch_sum >> 32);
            r[11] = bad;
            r[5] = switch_min;
            r[6] = switch_max;
        } else {
            status = cs::kBadOp;
        }
        r[1] = status;
        r[3] = held_program;
        r[7] = static_cast<uint32_t>(wall64() - t_seen);
        r[14] = static_cast<uint32_t>(t_seen);
        r[15] = part_calls;
        r[0] = sequence;
        noc_async_write(control + 4 * cs::kResp, resp_noc, 64);
        const uint64_t tw = wall64();
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            if (wall64() - tw > cs::kWaitLimit) {
                break;
            }
        }
        if (op == cs::kOpExit) {
            c[cs::kGo] = cs::kLeave;
            return;
        }
    }
}
