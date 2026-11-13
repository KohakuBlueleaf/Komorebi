// The probe's persistent kernel (data-movement core 0): on a command, reads the named
// overlay from DRAM into the slot unless held, calls it, writes a response to DRAM.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "overlay.h"

__attribute__((used)) volatile uint32_t ov_calls = 0;  // C linkage: overlay.h

namespace {

    // Control words (the host writes 4..19, then the sequence word 0).
    constexpr uint32_t kSeq = 0, kOp = 4, kFlags = 5, kArg = 6, kReps = 7;
    constexpr uint32_t kA = 8, kB = 12;  // id, bank, address, bytes of overlays A, B
    constexpr uint32_t kExpectA = 16, kExpectB = 17;
    constexpr uint32_t kResp = 32;   // response line, 16 words, 64-byte aligned
    constexpr uint32_t kTrisc = 48;  // compute: go, then done per compute core
    constexpr uint32_t kOpRun = 1, kOpAlt = 2, kOpExit = 3, kOpCompute = 4,
                       kOpClock = 5;
    constexpr uint32_t kFlagInvalidate = 1;  // invalidate the i-cache after a load
    constexpr uint32_t kFlagAt = 2;          // run: start at the wall clock in kReps
    constexpr uint32_t kOk = 0, kBadOp = 1, kReadTimeout = 2, kTooBig = 3, kIdle = 4;
    constexpr uint32_t kComputeTimeout = 5;
    constexpr uint32_t kCbIn = 0, kCbOut = 16, kTileValues = 1024;  // bfloat16 tile
    // bfloat16 bits of 1 .. 16: the input tile's values, (j % 16) + 1
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
    // RISCV_IC_INVALIDATE bits (the firmware's): data-movement core 0, compute 0..2
    constexpr uint32_t kIcBrisc = 0x1, kIcTriscs = 0x2 | 0x4 | 0x8;

    inline uint64_t wall64() {
        volatile tt_reg_ptr uint32_t*
            lo = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            );
        volatile tt_reg_ptr uint32_t*
            hi = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_1
            );
        for (;;) {
            const uint32_t h = *hi, l = *lo;
            if (*hi == h) {
                return static_cast<uint64_t>(h) << 32 | l;
            }
        }
    }

    __attribute__((noinline)) uint32_t mix(uint32_t x) {
        return ov::mix(x);
    }

    // Invalidates the i-cache of the cores in `mask` (the firmware's register write).
    inline void invalidate_icache(uint32_t mask) {
        volatile tt_reg_ptr uint32_t*
            cfg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE);
        cfg[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = mask;
    }

    // The compute cores are parked when each done word equals the go word (bounded).
    bool parked(volatile tt_l1_ptr uint32_t* t, uint64_t t0, uint64_t limit) {
        for (uint32_t k = 0; k < 3; k++) {
            for (;;) {
                invalidate_l1_cache();
                if (t[1 + k] == t[0]) {
                    break;
                }
                if (wall64() - t0 > limit) {
                    return false;
                }
            }
        }
        return true;
    }

    // Reads `bytes` from DRAM (bank, address) into the slot; bounded by `limit`.
    bool load(
        uint32_t bank, uint32_t addr, uint32_t bytes, uint32_t slot, uint64_t limit
    ) {
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), slot, bytes);
        const uint64_t t0 = wall64();
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            if (wall64() - t0 > limit) {
                return false;
            }
        }
        return true;
    }

}  // namespace

void kernel_main() {
    const uint32_t ctl = get_arg_val<uint32_t>(0);
    const uint32_t resp_bank = get_arg_val<uint32_t>(1),
                   resp_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot = get_arg_val<uint32_t>(3),
                   slot_bytes = get_arg_val<uint32_t>(4);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 20;
    const uint32_t tslots = get_arg_val<uint32_t>(6);  // compute slots: 3 x tslot bytes
    const uint32_t tslot_bytes = get_arg_val<uint32_t>(7);
    const uint64_t read_limit = 50000000;  // cycles (about 37 ms) for one load
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ctl
    );
    volatile tt_l1_ptr uint32_t* r = c + kResp;
    const uint64_t resp_noc = get_noc_addr_from_bank_id<true>(resp_bank, resp_addr);
    ov::Args args{};
    args.mix = mix;
    const ov::Entry entry = reinterpret_cast<ov::Entry>(slot);
    uint32_t last = 0, held = 0;
    uint64_t seen_at = wall64();
    for (;;) {
        invalidate_l1_cache();
        const uint32_t seq = c[kSeq];
        if (seq == last) {
            if (wall64() - seen_at > idle) {
                r[0] = last;
                r[1] = kIdle;
                noc_async_write(ctl + 4 * kResp, resp_noc, 64);
                noc_async_write_barrier();
                return;
            }
            continue;
        }
        uint64_t t_seen = wall64();
        seen_at = t_seen;
        last = seq;
        for (uint32_t i = 1; i < 16; i++) {
            r[i] = 0;
        }
        const uint32_t op = c[kOp], flags = c[kFlags];
        args.arg = c[kArg];
        uint32_t status = kOk;
        if (op == kOpRun && (flags & kFlagAt)) {
            // starts at the wall-clock count in kReps (low word), bounded: every core
            // of a broadcast begins together; the response's times count from there
            while (
                static_cast<int32_t>(static_cast<uint32_t>(wall64()) - c[kReps]) < 0
            ) {
                if (wall64() - t_seen > read_limit * 20) {
                    break;
                }
            }
            r[13] = static_cast<uint32_t>(t_seen);  // when the command was seen
            t_seen = wall64();
        }
        if (op == kOpClock) {
            r[2] = static_cast<uint32_t>(t_seen);
        } else if (op == kOpRun) {
            uint64_t t_load = 0;
            if (c[kA + 3] > slot_bytes) {
                status = kTooBig;
            } else if (c[kA] != held) {
                const uint64_t t0 = wall64();
                if (!load(c[kA + 1], c[kA + 2], c[kA + 3], slot, read_limit)) {
                    status = kReadTimeout;
                    held = 0;
                } else {
                    if (flags & kFlagInvalidate) {
                        invalidate_icache(kIcBrisc);
                    }
                    held = c[kA];
                    r[4] = 1;
                }
                t_load = wall64() - t0;
            }
            if (status == kOk) {
                const uint64_t t0 = wall64();
                r[2] = entry(&args);
                r[6] = static_cast<uint32_t>(wall64() - t0);
            }
            r[5] = static_cast<uint32_t>(t_load);
        } else if (op == kOpAlt) {
            // reps x (load A, run A, load B, run B); counts wrong and stale results.
            const uint64_t t0 = wall64();
            uint32_t wrong = 0, stale = 0, runs = 0, last_wrong = 0;
            for (uint32_t i = 0; i < c[kReps] && status == kOk; i++) {
                for (uint32_t w = 0; w < 2; w++) {
                    const uint32_t o = w == 0 ? kA : kB;
                    if (c[o + 3] > slot_bytes) {
                        status = kTooBig;
                        break;
                    }
                    if (!load(c[o + 1], c[o + 2], c[o + 3], slot, read_limit)) {
                        status = kReadTimeout;
                        break;
                    }
                    if (flags & kFlagInvalidate) {
                        invalidate_icache(kIcBrisc);
                    }
                    const uint32_t got = entry(&args);
                    runs++;
                    const uint32_t want = c[kExpectA + w], other = c[kExpectA + 1 - w];
                    if (got != want) {
                        wrong++;
                        last_wrong = got;
                        stale += got == other;
                    }
                }
            }
            held = 0;
            const uint64_t t = wall64() - t0;
            r[8] = wrong;
            r[9] = stale;
            r[10] = runs;
            r[11] = static_cast<uint32_t>(t);
            r[12] = static_cast<uint32_t>(t >> 32);
            r[13] = last_wrong;
        } else if (op == kOpCompute) {
            // Parked compute cores: their parts (A, part k at k x tslot), the input
            // tile, go; then the sum of the output tile's 16-bit words.
            volatile tt_l1_ptr uint32_t* t = c + kTrisc;
            uint64_t t_load = 0;
            const uint64_t t0 = wall64();
            status = parked(t, t0, read_limit) ? kOk : kComputeTimeout;
            if (status == kOk && c[kA] != held) {
                const uint64_t t1 = wall64();
                if (c[kA + 3] > 3 * tslot_bytes) {
                    status = kTooBig;
                } else if (!load(c[kA + 1], c[kA + 2], c[kA + 3], tslots, read_limit)) {
                    status = kReadTimeout;
                }
                held = 0;
                if (status == kOk) {
                    if (flags & kFlagInvalidate) {
                        invalidate_icache(kIcTriscs);
                    }
                    held = c[kA];
                    r[4] = 1;
                }
                t_load = wall64() - t1;
            }
            if (status == kOk) {
                const uint64_t t1 = wall64();
                cb_reserve_back(kCbIn, 1);
                volatile tt_l1_ptr uint16_t*
                    in = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        get_write_ptr(kCbIn)
                    );
                for (uint32_t j = 0; j < kTileValues; j++) {
                    in[j] = kBf16Small[j % 16];
                }
                cb_push_back(kCbIn, 1);
                t[4] = args.arg;
                t[0] = t[0] + 1;  // go: every compute core runs its part once
                status = parked(t, t1, read_limit) ? kOk : kComputeTimeout;
                if (status == kOk) {
                    cb_wait_front(kCbOut, 1);
                    invalidate_l1_cache();
                    volatile tt_l1_ptr uint16_t*
                        out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                            get_read_ptr(kCbOut)
                        );
                    uint32_t sum = 0;
                    for (uint32_t j = 0; j < kTileValues; j++) {
                        sum += out[j];
                    }
                    cb_pop_front(kCbOut, 1);
                    r[2] = sum;
                }
                r[6] = static_cast<uint32_t>(wall64() - t1);
            }
            r[5] = static_cast<uint32_t>(t_load);
        } else if (op != kOpExit) {
            status = kBadOp;
        }
        r[1] = status;
        if (op != kOpCompute) {
            r[3] = held;
        }
        r[7] = static_cast<uint32_t>(wall64() - t_seen);
        r[14] = static_cast<uint32_t>(t_seen);
        r[15] = ov_calls;
        r[0] = seq;
        noc_async_write(ctl + 4 * kResp, resp_noc, 64);
        const uint64_t tw = wall64();
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            if (wall64() - tw > read_limit) {
                break;
            }
        }
        if (op == kOpExit) {
            if (tslots != 0) {
                volatile tt_l1_ptr uint32_t* t = c + kTrisc;
                t[0] = 0xFFFFFFFFu;  // compute cores leave their wait loop
            }
            return;
        }
    }
}
