// Routed messages on the simulated grid: the generic reader and writer with the forward
// example's compute, the quiet end, and the gap, overflow, stall and quota paths.
#include <stdint.h>

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "checks.hpp"
#include "forward.h"
#include "komorebi/device/km_core.hpp"
#include "komorebi/device/km_dyn.hpp"

void kmsim_dyn_reader_main();
void kmsim_dyn_writer_main();
void kmsim_forward_unpack_main();
void kmsim_forward_math_main();
void kmsim_forward_pack_main();

using namespace kmsim;
using namespace scen;

namespace {

    constexpr uint32_t kQueue = 0x200000;  // DRAM: the queues
    constexpr uint32_t kRes = 0xE800;      // ctl-relative: the example's results
    constexpr uint32_t kBins = 2;          // few bins: bins are evicted and refilled

    // `cores` cores, then `counters` counter tiles: the run's group.
    struct Forward {
        uint32_t cores = 2, counters = 1, messages = 1, forwards = 1, cap = 256,
                 max_live = 256;
        int32_t one_core = -1, stall = -1, probe = -1;
        uint32_t idle = 2000;  // cycles before a line says idle
    };

    // The forward example's grid: dynamic arguments, buffers and one RUN command.
    Config dyn_config(const Forward& h) {
        Config cfg;
        cfg.cores = h.cores + h.counters;
        cfg.row = 2;
        cfg.idle_limit = 0;
        cfg.wait_limit = 64;
        cfg.deadline = 4000000;
        cfg.timeout = 1ull << 26;
        cfg.dyn_handler = kHandlerForward;
        cfg.kernels[kReader] = kmsim_dyn_reader_main;
        cfg.kernels[kWriter] = kmsim_dyn_writer_main;
        cfg.kernels[kUnpack] = kmsim_forward_unpack_main;
        cfg.kernels[kMath] = kmsim_forward_math_main;
        cfg.kernels[kPack] = kmsim_forward_pack_main;
        cfg.cbs = {
            {km::kCbDynIn, 0xC000, km::kDynHeader + 32 * 64, 2},  // to 0xD080
            {km::kCbDynOut, 0xD100, 64, 64},
            {km::kCbDynBins, 0xE100, kBins * 32 * 64, 1},
        };
        const uint32_t quota = h.max_live / h.cores;
        const uint32_t cap = h.cap, idle = h.idle, k = h.counters;
        cfg.program_args =
            [=](uint32_t, std::vector<uint32_t>& rw, std::vector<uint32_t>& c) {
                const uint32_t ctl = rw[0], lines = rw[4] - kDispAck + kDispLines;
                rw.insert(
                    rw.end(), {kHandlerForward, kQueue, cap, 64, lines, idle, kBins, k}
                );
                c.insert(c.end(), {64, quota, k, ctl + kRes});
            };
        Job run;
        run.handler = kHandlerForward;
        run.first = 0;
        run.count = h.cores + h.counters;
        run.flags = KM_F_EXCL;
        run.a[0] = h.messages;
        run.a[1] = h.forwards;
        run.a[2] = h.one_core >= 0 ? kModeOneCore : kModeRandom;
        run.a[3] = h.one_core >= 0 ? h.one_core : 0;
        run.a[4] = h.stall + 1;
        run.a[5] = h.probe + 1;
        cfg.jobs = {run};
        return cfg;
    }

    // The replay: messages consumed, their total and the visits at each core.
    struct Expect {
        std::vector<uint32_t> consumed, visits;
        std::vector<uint64_t> total;
    };

    Expect replay(const Forward& h) {
        Expect e{
            std::vector<uint32_t>(h.cores),
            std::vector<uint32_t>(h.cores),
            std::vector<uint64_t>(h.cores)
        };
        for (uint32_t m = 0; m < h.cores; m++) {
            const km::Partition s = km::partition(h.messages, m, h.cores);
            for (uint32_t id = s.start; id < s.start + s.len; id++) {
                uint32_t st = forward_seed(id), sum = 0, at = m;
                for (uint32_t k = 0; k <= h.forwards; k++) {
                    e.visits[at]++;
                    st = forward_next(st);
                    sum += st;
                    if (k < h.forwards) {
                        at = h.one_core >= 0 ? h.one_core : st % h.cores;
                    }
                }
                e.consumed[at]++;
                e.total[at] += sum;
            }
        }
        return e;
    }

    uint32_t res(const Sim& s, uint32_t core, uint32_t word) {
        return s.word(core, kRes + 4 * word);
    }

    uint32_t gave_where(const Sim& s, uint32_t core) {
        return s.word(core, KM_CTL_DYN + KM_DYN_GAVE);
    }

    // A clean run: every message consumed once where the replay says, the quiet end
    // after the last consume, every kernel left on the EXIT.
    void exact(const Sim& s, const Forward& h, Check& c) {
        const Expect e = replay(h);
        c.expect(s.dyn.quiet_at != 0, "the quiet end was seen");
        c.expect(
            all_done(s), "the run, its end command and the EXIT were acknowledged"
        );
        for (uint32_t m = 0; m < h.cores; m++) {
            const uint64_t total =
                (res(s, m, kResTotalLo) |
                 static_cast<uint64_t>(res(s, m, kResTotalHi)) << 32);
            c.expect(
                res(s, m, kResRun) == s.jobs[0].seq, fmt("core %u ran the run", m)
            );
            c.expect(
                res(s, m, kResConsumed) == e.consumed[m],
                fmt("core %u consumed %u messages, the replay %u",
                    m,
                    res(s, m, kResConsumed),
                    e.consumed[m])
            );
            c.expect(
                total == e.total[m],
                fmt("core %u: total %x, the replay %x",
                    m,
                    static_cast<uint32_t>(total),
                    static_cast<uint32_t>(e.total[m]))
            );
            c.expect(
                res(s, m, kResVisits) == e.visits[m] + res(s, m, kResSent),
                fmt("core %u: visits as the replay", m)
            );
            c.expect(
                s.dyn.quiet_at >= res(s, m, kResLastConsume),
                fmt("core %u consumed before the quiet end", m)
            );
            c.expect(gave_where(s, m) == 0, fmt("core %u gave up nothing", m));
        }
        c.expect(
            s.dyn.sent ==
                static_cast<uint64_t>(h.messages) * (h.forwards + 1) +
                    (h.probe >= 0 ? res(s, h.probe, kResSent) : 0),
            "sent counts every send and every forward"
        );
    }

    void dyn_smoke(Outcome& o) {
        const Forward h;
        Sim s(dyn_config(h));
        s.run();
        common(s, o.check);
        exact(s, h, o.check);
        finish(s, o);
    }

    // Four cores on two counter tiles, many messages, two bins: bins are evicted,
    // packets fill and split.
    void dyn_random(Outcome& o) {
        Forward h;
        h.cores = 4;
        h.counters = 2;
        h.messages = 60;
        h.forwards = 12;
        Sim s(dyn_config(h));
        s.run();
        common(s, o.check);
        exact(s, h, o.check);
        finish(s, o);
    }

    // The idle protocol under slow and uneven delivery: lines go idle after 50 cycles
    // and back to busy many times; the end comes only after the last consume.
    void dyn_idle_flapping(Outcome& o) {
        Forward h;
        h.cores = 4;
        h.messages = 40;
        h.forwards = 8;
        h.idle = 50;
        Config cfg = dyn_config(h);
        std::mt19937 rng(7);
        cfg.delay = [&](const Chunk&) -> int64_t {
            return 32 + static_cast<int64_t>(rng() % 3000);
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        exact(s, h, o.check);
        o.check.expect(s.dyn.checks > 1, "the dispatcher checked more than once");
        finish(s, o);
    }

    // Every forward of every message to rank 0: its queue holds the whole population.
    void dyn_one_core(Outcome& o) {
        Forward h;
        h.cores = 4;
        h.messages = 60;
        h.forwards = 6;
        h.one_core = 0;
        h.cap = 64;
        h.max_live = 64;
        Sim s(dyn_config(h));
        s.run();
        common(s, o.check);
        exact(s, h, o.check);
        finish(s, o);
    }

    // Rank 1 sends until its quota is gone: send returns false once, and the
    // sent messages consume and give their quota back.
    void dyn_quota(Outcome& o) {
        Forward h;
        h.cores = 2;
        h.messages = 6;
        h.forwards = 3;
        h.max_live = 64;
        h.probe = 1;
        Sim s(dyn_config(h));
        s.run();
        common(s, o.check);
        exact(s, h, o.check);
        const uint32_t quota = h.max_live / h.cores;
        o.check.expect(
            res(s, 1, kResSent) == quota - 3,
            fmt("probe sent %u, quota %u less its 3 messages",
                res(s, 1, kResSent),
                quota)
        );
        o.check.expect(res(s, 1, kResRejected) == 1, "one send returned false");
        o.check.expect(
            res(s, 1, kResNull) == res(s, 1, kResSent), "every probe message consumed"
        );
        finish(s, o);
    }

    // The first packet commit to DRAM is lost: the reserved position never lands, its
    // reader reports a gap, the dispatcher stops the grid; nothing hangs.
    void dyn_gap(Outcome& o) {
        Forward h;
        h.cores = 2;
        h.messages = 4;
        h.forwards = 4;
        Config cfg = dyn_config(h);
        bool dropped = false;
        cfg.delay = [&](const Chunk& c) -> int64_t {
            if (c.dst == kDram && c.off >= kQueue && c.parts == 1 && !dropped) {
                dropped = true;  // the first packet's commit (one 16-byte chunk)
                return kDrop;
            }
            return 32;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        uint32_t gaps = 0;
        for (uint32_t m = 0; m < h.cores; m++) {
            gaps += gave_where(s, m) == km::kWhereDynGap;
        }
        o.check.expect(gaps == 1, fmt("one core reported a gap, %u did", gaps));
        o.check.expect(s.dyn.gave != 0, "the dispatcher saw the line that gave up");
        o.check.expect(s.timed_out, "the grid was stopped");
        finish(s, o);
    }

    // Rank 0's reader is held while every other core sends all its messages to it:
    // more messages are reserved than its queue holds, and it gives up with overflow.
    void dyn_overflow(Outcome& o) {
        Forward h;
        h.cores = 4;
        h.messages = 80;
        h.forwards = 2;
        h.one_core = 0;
        h.cap = 32;
        Config cfg = dyn_config(h);
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (!held && p.core == 0 && p.role == kReader &&
                !std::strcmp(p.name, "cb_pages_reservable_at_back") &&
                p.cb == km::kCbDynIn) {
                held = true;
                sp->freeze(0, kReader, 200000);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        o.check.expect(
            gave_where(s, 0) == km::kWhereDynOverflow,
            fmt("core 0 gave up: %u", gave_where(s, 0))
        );
        o.check.expect(s.timed_out, "the grid was stopped");
        finish(s, o);
    }

    // Rank 1 stalls in its body: its reader gives up on a free packet page (or the
    // deadline passes); either way core 1 is named and every kernel returns.
    void dyn_stall(Outcome& o) {
        Forward h;
        h.cores = 3;
        h.messages = 12;
        h.forwards = 4;
        h.stall = 1;
        Config cfg = dyn_config(h);
        cfg.deadline = 400000;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(s.timed_out, "the grid was stopped");
        o.check.expect(s.jobs[0].status != kDone, "the run was not acknowledged");
        o.check.expect(
            s.dyn.gave == 2 || s.jobs[0].status == kTimedOut,
            fmt("core 1 named: gave %u, status %u", s.dyn.gave, s.jobs[0].status)
        );
        o.check.expect(
            gave_where(s, 1) == km::kWhereDynInPage ||
                gave_where(s, 1) == km::kWhereDynStopped,
            fmt("core 1's reader: where %u", gave_where(s, 1))
        );
        finish(s, o);
    }

}  // namespace

namespace scen {

    const std::vector<Case>& dyn_cases() {
        static const std::vector<Case> cases = {
            {"dyn_smoke", dyn_smoke},
            {"dyn_random", dyn_random},
            {"dyn_idle_flapping", dyn_idle_flapping},
            {"dyn_one_core", dyn_one_core},
            {"dyn_quota", dyn_quota},
            {"dyn_gap", dyn_gap},
            {"dyn_overflow", dyn_overflow},
            {"dyn_stall", dyn_stall},
        };
        return cases;
    }

}  // namespace scen
