// Persistent kernel on every core: as writer, stores a body into target cores' slots
// and messages them; as target, parked here, invalidates its i-cache, calls, replies.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "overlay.h"

__attribute__((used)) volatile uint32_t ov_calls = 0;  // C linkage: overlay.h

namespace {

    // Control words (the host writes 4..19, then the sequence word 0).
    constexpr uint32_t kSeq = 0, kOp = 4, kFlags = 5, kArg = 6, kCount = 7;
    constexpr uint32_t kA = 8, kB = 12;  // id, bank, address, bytes of bodies 0, 1
    constexpr uint32_t kWhich = 16;      // feed: body 0 or 1
    constexpr uint32_t kExpect = 18;     // stale: the results of bodies 0 and 1
    constexpr uint32_t kRegValue = 97;   // stale: the invalidate value, 16 B + 4 like
                                         // the register's address
    constexpr uint32_t kResp = 32;       // response line to the host, 16 words
    constexpr uint32_t kMsg = 64;        // message from a writer: id, bytes, arg, go
    constexpr uint32_t kReply = 68;      // the host's: writer x, y, reply line address
    constexpr uint32_t kOut = 80;        // the reply line as built, 16 words
    constexpr uint32_t kOpFetch = 1, kOpFeed = 2, kOpExit = 3, kOpClock = 5;
    constexpr uint32_t kOpStale = 6;
    constexpr uint32_t kFlagTable = 1;   // feed: the reply lines to DRAM afterwards
    constexpr uint32_t kFlagRemote = 2;  // stale: the writer invalidates the target's
                                         // i-cache through the NoC
    constexpr uint32_t kNoInvalidate = 0x80000000u;  // message: no self-invalidate
    constexpr uint32_t kOk = 0, kBadOp = 1, kReadTimeout = 2, kTooBig = 3, kIdle = 4;
    constexpr uint32_t kNotParked = 5, kNoReply = 6, kWriteTimeout = 7;
    // Table on the writer: reply lines (16 words each), target words, sent messages.
    constexpr uint32_t kTargetsMax = 128;
    constexpr uint32_t kTableTargets = 16 * kTargetsMax;
    constexpr uint32_t kTableSent = kTableTargets + kTargetsMax;
    constexpr uint32_t kLineGo = 15;  // the reply line's last word echoes the go
    constexpr uint32_t kIcBrisc = 0x1;  // RISCV_IC_INVALIDATE bit of data-movement core 0

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
        return ov::mix(x);
    }

    // Invalidates this core's i-cache (the firmware's register write at launch).
    inline void invalidate_icache(uint32_t mask) {
        volatile tt_reg_ptr uint32_t*
            cfg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE);
        cfg[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = mask;
    }

    // Reads `bytes` from DRAM (bank, address) into L1 at `to`; times out at `limit`.
    bool load(
        uint32_t bank, uint32_t addr, uint32_t bytes, uint32_t to, uint64_t limit
    ) {
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr), to, bytes);
        const uint64_t t0 = wall64();
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            if (wall64() - t0 > limit) {
                return false;
            }
        }
        return true;
    }

    // Waits until every NoC write of this core is acknowledged; times out at `limit`.
    bool writes_flushed(uint64_t limit) {
        const uint64_t t0 = wall64();
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            if (wall64() - t0 > limit) {
                return false;
            }
        }
        return true;
    }

    inline uint64_t at_core(uint32_t xy, uint32_t addr) {
        return get_noc_addr(xy & 0xFFFF, xy >> 16, addr);
    }

}  // namespace

void kernel_main() {
    const uint32_t ctl = get_arg_val<uint32_t>(0);
    const uint32_t resp_bank = get_arg_val<uint32_t>(1);
    const uint32_t resp_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot = get_arg_val<uint32_t>(3);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(4);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 20;
    const uint32_t store = get_arg_val<uint32_t>(6);  // two halves of slot_bytes
    const uint32_t table = get_arg_val<uint32_t>(7);
    const uint32_t table_bank = get_arg_val<uint32_t>(8);
    const uint32_t table_addr = get_arg_val<uint32_t>(9);
    const uint64_t limit = 50000000;  // cycles (about 37 ms) for one wait
    volatile tt_l1_ptr uint32_t* c = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl)
    );
    volatile tt_l1_ptr uint32_t* r = c + kResp;
    volatile tt_l1_ptr uint32_t* t = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table)
    );
    const uint64_t resp_noc = get_noc_addr_from_bank_id<true>(resp_bank, resp_addr);
    ov::Args args{};
    args.mix = mix;
    const ov::Entry entry = reinterpret_cast<ov::Entry>(slot);
    uint32_t last = 0, last_go = 0;
    uint32_t held_id[2] = {0, 0}, held_bytes[2] = {0, 0};
    uint64_t seen_at = wall64();
    for (;;) {
        invalidate_l1_cache();
        // Target: a writer's message names the body now in the slot.
        if (c[kMsg + 3] != last_go) {
            const uint64_t t_seen = wall64();
            seen_at = t_seen;
            last_go = c[kMsg + 3];
            const uint32_t id = c[kMsg], bytes = c[kMsg + 1] & ~kNoInvalidate;
            const bool self = !(c[kMsg + 1] & kNoInvalidate);
            args.arg = c[kMsg + 2];
            volatile tt_l1_ptr uint32_t* o = c + kOut;
            uint32_t status = kOk, got = 0, run = 0;
            if (bytes > slot_bytes) {
                status = kTooBig;
            } else {
                if (self) {
                    invalidate_icache(kIcBrisc);
                }
                const uint64_t t0 = wall64();
                got = entry(&args);
                run = static_cast<uint32_t>(wall64() - t0);
            }
            o[0] = got;
            o[1] = ov_calls;
            o[2] = run;
            o[3] = static_cast<uint32_t>(t_seen);
            o[4] = static_cast<uint32_t>(wall64());
            o[5] = status;
            o[6] = id;
            o[kLineGo] = last_go;
            noc_async_write(ctl + 4 * kOut, at_core(c[kReply], c[kReply + 1]), 64);
            writes_flushed(limit);
            continue;
        }
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
        const uint64_t t_seen = wall64();
        seen_at = t_seen;
        last = seq;
        for (uint32_t i = 1; i < 16; i++) {
            r[i] = 0;
        }
        const uint32_t op = c[kOp], flags = c[kFlags];
        uint32_t status = kOk;
        if (op == kOpClock) {
            r[2] = static_cast<uint32_t>(t_seen);
        } else if (op == kOpFetch) {
            // Writer: both bodies from DRAM into the two halves of the store, once.
            for (uint32_t w = 0; w < 2 && status == kOk; w++) {
                const uint32_t o = w == 0 ? kA : kB;
                if (c[o + 3] > slot_bytes) {
                    status = kTooBig;
                } else if (!load(
                    c[o + 1],
                    c[o + 2],
                    c[o + 3],
                    store + w * slot_bytes,
                    limit
                )) {
                    status = kReadTimeout;
                } else {
                    held_id[w] = c[o];
                    held_bytes[w] = c[o + 3];
                }
            }
            r[5] = static_cast<uint32_t>(wall64() - t_seen);
        } else if (op == kOpFeed) {
            // Writer: body `which` into the slot of targets 0..n-1 (each parked: its
            // last reply echoes its last go), then a message each, then their replies.
            const uint32_t n = c[kCount], w = c[kWhich] & 1;
            const uint32_t bytes = held_bytes[w], src = store + w * slot_bytes;
            volatile tt_l1_ptr uint32_t* sent = t + kTableSent;
            if (n > kTargetsMax || bytes == 0) {
                status = kBadOp;
            }
            for (uint32_t i = 0; i < n && status == kOk; i++) {
                for (;;) {
                    invalidate_l1_cache();
                    if (t[16 * i + kLineGo] == sent[4 * i + 3]) {
                        break;
                    }
                    if (wall64() - t_seen > limit) {
                        status = kNotParked;
                        break;
                    }
                }
            }
            uint64_t t_store = 0, t_sent = 0;
            if (status == kOk) {
                const uint64_t t0 = wall64();
                for (uint32_t i = 0; i < n; i++) {
                    noc_async_write(src, at_core(t[kTableTargets + i], slot), bytes);
                }
                if (!writes_flushed(limit)) {
                    status = kWriteTimeout;
                }
                t_store = wall64() - t0;
            }
            if (status == kOk) {
                for (uint32_t i = 0; i < n; i++) {
                    volatile tt_l1_ptr uint32_t* m = sent + 4 * i;
                    m[0] = held_id[w];
                    m[1] = bytes;
                    m[2] = c[kArg] + i;
                    m[3] = m[3] + 1;  // the go word, last in the 16-byte message
                    noc_async_write(
                        table + 4 * (kTableSent + 4 * i),
                        at_core(t[kTableTargets + i], ctl + 4 * kMsg),
                        16
                    );
                }
                t_sent = wall64() - t_seen;
                for (uint32_t i = 0; i < n && status == kOk; i++) {
                    for (;;) {
                        invalidate_l1_cache();
                        if (t[16 * i + kLineGo] == sent[4 * i + 3]) {
                            break;
                        }
                        if (wall64() - t_seen > limit) {
                            status = kNoReply;
                            break;
                        }
                    }
                }
            }
            r[7] = static_cast<uint32_t>(wall64() - t_seen);  // command to last reply
            r[2] = t[0];                                      // target 0: result
            r[4] = t[1];                                      // its call counter
            r[5] = static_cast<uint32_t>(t_store);
            r[6] = t[2];  // target 0's run
            r[8] = static_cast<uint32_t>(t_sent);
            r[9] = held_id[w];
            if (status == kOk && (flags & kFlagTable)) {
                noc_async_write(
                    table,
                    get_noc_addr_from_bank_id<true>(table_bank, table_addr),
                    64 * n
                );
                if (!writes_flushed(limit)) {
                    status = kWriteTimeout;
                }
            }
        } else if (op == kOpStale) {
            // Writer, target 0: bodies 0, 1 alternated, the target told not to
            // invalidate; kFlagRemote: the writer writes its invalidate register.
            volatile tt_l1_ptr uint32_t* sent = t + kTableSent;
            const uint32_t xy = t[kTableTargets];
            const uint32_t reg = (
                TENSIX_CFG_BASE + 4 * RISCV_IC_INVALIDATE_InvalidateAll_ADDR32
            );
            c[kRegValue] = kIcBrisc;
            const uint64_t t0 = wall64();
            uint32_t wrong = 0, stale = 0, runs = 0, last_wrong = 0;
            for (uint32_t i = 0; i < c[kCount] && status == kOk; i++) {
                for (uint32_t w = 0; w < 2 && status == kOk; w++) {
                    const uint64_t tw = wall64();
                    for (;;) {
                        invalidate_l1_cache();
                        if (t[kLineGo] == sent[3]) {
                            break;
                        }
                        if (wall64() - tw > limit) {
                            status = kNotParked;
                            break;
                        }
                    }
                    if (status != kOk) {
                        break;
                    }
                    noc_async_write(
                        store + w * slot_bytes, at_core(xy, slot), held_bytes[w]
                    );
                    if (!writes_flushed(limit)) {
                        status = kWriteTimeout;
                        break;
                    }
                    if (flags & kFlagRemote) {
                        noc_async_write(ctl + 4 * kRegValue, at_core(xy, reg), 4);
                        if (!writes_flushed(limit)) {
                            status = kWriteTimeout;
                            break;
                        }
                    }
                    sent[0] = held_id[w];
                    sent[1] = held_bytes[w] | kNoInvalidate;
                    sent[2] = c[kArg];
                    sent[3] = sent[3] + 1;
                    noc_async_write(
                        table + 4 * kTableSent, at_core(xy, ctl + 4 * kMsg), 16
                    );
                    const uint64_t tr = wall64();
                    for (;;) {
                        invalidate_l1_cache();
                        if (t[kLineGo] == sent[3]) {
                            break;
                        }
                        if (wall64() - tr > limit) {
                            status = kNoReply;
                            break;
                        }
                    }
                    if (status != kOk) {
                        break;
                    }
                    runs++;
                    const uint32_t got = t[0];
                    if (got != c[kExpect + w]) {
                        wrong++;
                        last_wrong = got;
                        stale += got == c[kExpect + 1 - w];
                    }
                }
            }
            const uint64_t tt = wall64() - t0;
            r[8] = wrong;
            r[9] = stale;
            r[10] = runs;
            r[11] = static_cast<uint32_t>(tt);
            r[12] = static_cast<uint32_t>(tt >> 32);
            r[13] = last_wrong;
        } else if (op != kOpExit) {
            status = kBadOp;
        }
        r[1] = status;
        if (op != kOpFeed) {
            r[7] = static_cast<uint32_t>(wall64() - t_seen);
        }
        r[14] = static_cast<uint32_t>(t_seen);
        r[15] = ov_calls;
        r[0] = seq;
        noc_async_write(ctl + 4 * kResp, resp_noc, 64);
        writes_flushed(limit);
        if (op == kOpExit) {
            return;
        }
    }
}
