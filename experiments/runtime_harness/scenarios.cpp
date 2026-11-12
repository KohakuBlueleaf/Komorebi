// Failure-path and protocol scenarios: each builds a grid, runs the runtime's kernels
// on it and prints one JSON line of its checks. Usage: scenarios <name> | list
#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "api/dataflow/dataflow_api.h"
#include "harness.hpp"
#include "komorebi/device/km_core.hpp"

using namespace kmsim;

// A case's compute kernels (item_kernels.cpp), one per thread.
void kmsim_item_in_unpack_main();
void kmsim_item_in_math_main();
void kmsim_item_in_pack_main();
void kmsim_item_out_unpack_main();
void kmsim_item_out_math_main();
void kmsim_item_out_pack_main();

namespace {

    constexpr uint32_t kSegMap = 1;  // the example's segments
    constexpr uint32_t kSegStall = 2;
    constexpr uint32_t kSegBadItem = 5;  // a0: the member; a1 = 0: writer, 1: reader
    constexpr uint32_t kSegNop = 9;      // unknown to the example: every body returns
    constexpr uint32_t kCbIn = 0;

    struct Check {
        std::vector<std::string> fails;

        void expect(bool ok, const std::string& what) {
            if (!ok && fails.size() < 40) {
                fails.push_back(what);
            }
        }
    };

    std::string fmt(const char* f, unsigned a = 0, unsigned b = 0, unsigned c = 0) {
        char buf[200];
        std::snprintf(buf, sizeof buf, f, a, b, c);
        return buf;
    }

    Job map_job(uint32_t first, uint32_t count, uint32_t tiles) {
        Job j;
        j.seg = kSegMap;
        j.first = first;
        j.count = count;
        j.a[1] = tiles;
        j.map = true;
        return j;
    }

    Job nop_job(uint32_t first, uint32_t count) {
        Job j;
        j.seg = kSegNop;
        j.first = first;
        j.count = count;
        return j;
    }

    // Settings for the fault cases: short bounds, so that a stranded kernel shows fast.
    Config fault_config(uint32_t cores) {
        Config c;
        c.cores = cores;
        c.row = 2;
        c.wait_limit = 8;
        c.idle_limit = 64;
        c.deadline = 200000;
        c.bound = 1500000;
        return c;
    }

    // A record as the cores saw it, for the checks: its team and the slot it went to.
    struct Issued {
        uint32_t first, count, slot;
        bool exit;
        bool done;
    };

    std::map<uint32_t, Issued> issued(const Sim& s) {
        std::map<uint32_t, Issued> m;
        for (size_t k = 0; k < s.jobs.size(); k++) {
            const JobResult& r = s.jobs[k];
            if (r.status != kNotIssued) {
                const Job& j = s.cfg.jobs[k];
                m[r.seq] = {j.first, j.count, r.slot, false, r.status == kDone};
            }
        }
        if (s.exit_job.status != kNotIssued) {
            // EXIT is completed as a team of every core, whatever its team field says.
            m[s.exit_job.seq] = {0, s.cfg.cores, 0, true, s.exit_job.status == kDone};
        }
        return m;
    }

    bool all_done(const Sim& s) {
        for (const JobResult& r : s.jobs) {
            if (r.status != kDone) {
                return false;
            }
        }
        return s.exit_job.status == kDone;
    }

    // What every case ends with: every thread returned, no stub contract broken, state
    // words consistent, and each acknowledged record completed by its whole team.
    void common(const Sim& s, Check& c) {
        c.expect(
            s.all_returned, "every thread of every core returned within the bound"
        );
        c.expect(s.violations.empty(), "no stub contract or landing violation");
        if (all_done(s)) {
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                c.expect(
                    (s.state(core, km::kStEnds) & 0xFFFFFF) == 0x010101 * km::kEndExit,
                    fmt("core %u's reader, writer and compute left on the EXIT", core)
                );
            }
        }
        const auto recs = issued(s);
        std::map<uint32_t, std::map<uint32_t, uint32_t>> takes, reports, acks;
        std::map<uint32_t, uint32_t> runs;
        std::map<uint32_t, std::vector<Event>> by_seq;
        for (const Event& e : s.events) {
            if (e.kind != kEvStop) {
                by_seq[e.a].push_back(e);
            }
            if (e.kind == kEvTake && e.a != 0) {
                takes[e.a][e.core]++;
                runs[e.core]++;
            } else if (e.kind == kEvReport) {
                reports[e.a][e.core]++;
            } else if (e.kind == kEvAck) {
                acks[e.a][e.core]++;
                const auto it = recs.find(e.a);
                c.expect(
                    it != recs.end() && it->second.slot == e.b,
                    fmt("core %u acknowledged %u in ack word %u, the slot it was "
                        "issued to",
                        e.core,
                        e.a,
                        e.b)
                );
            }
        }
        for (uint32_t core = 0; core < s.cfg.cores; core++) {
            c.expect(
                s.state(core, km::kStMagic) == (km::kStateMagic | 0x5A5Au << 8 | core),
                fmt("core %u start-up word intact", core)
            );
            c.expect(
                s.state(core, km::kStRuns) == runs[core],
                fmt("core %u counts %u records taken, the log shows %u",
                    core,
                    s.state(core, km::kStRuns),
                    runs[core])
            );
        }
        for (const auto& [seq, r] : recs) {
            if (!r.done) {
                continue;
            }
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                const bool member = core - r.first < r.count;
                const uint32_t t = takes[seq][core];
                c.expect(
                    t == (member ? 1u : 0u),
                    fmt("record %u taken %u times by core %u", seq, t, core)
                );
                if (member) {
                    const uint32_t want = core == r.first ? 0 : 1;
                    c.expect(
                        reports[seq][core] == want,
                        fmt("record %u: core %u sent %u reports",
                            seq,
                            core,
                            reports[seq][core])
                    );
                    c.expect(
                        acks[seq][core] == (core == r.first ? 1u : 0u),
                        fmt("record %u: core %u sent %u acks",
                            seq,
                            core,
                            acks[seq][core])
                    );
                } else {
                    c.expect(
                        reports[seq][core] + acks[seq][core] == 0,
                        fmt("record %u: core %u, not in the team, reported", seq, core)
                    );
                }
            }
            // A core reports (or acks) only after every report to it for the record
            // landed, and it waits for exactly as many as its children in the tree.
            km_record_t rec = {};
            rec.team = km::make_team(r.first, r.count);
            std::map<uint32_t, std::vector<uint64_t>> landed;
            for (const Event& e : by_seq[seq]) {
                if (e.kind == kEvReportLand) {
                    landed[e.b].push_back(e.t);
                }
            }
            for (const Event& e : by_seq[seq]) {
                if (e.kind != kEvReport && e.kind != kEvAck) {
                    continue;
                }
                const uint32_t want = km::report_to(rec, e.core, s.cfg.tree_row).want;
                uint32_t before = 0;
                for (uint64_t t : landed[e.core]) {
                    before += t <= e.t;
                }
                c.expect(
                    before == want,
                    fmt("record %u: core %u sent its report after %u of the "
                        "reports to it landed",
                        seq,
                        e.core,
                        before) +
                        fmt(", not %u", want)
                );
            }
        }
        for (size_t k = 0; k < s.jobs.size(); k++) {
            c.expect(
                s.jobs[k].data_bad == 0,
                fmt("job %u: %u output elements wrong", k, s.jobs[k].data_bad)
            );
        }
        if (!s.timed_out && s.all_returned) {
            for (uint32_t core = 0; core < s.cfg.cores; core++) {
                for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
                    c.expect(
                        s.word(core, KM_CTL_DONE + j * 16) == 0,
                        fmt("core %u: completion counter %u is zero at the end",
                            core,
                            j)
                    );
                }
            }
        }
    }

    // A tile that gave up acknowledges nothing afterwards: no report or ack from `core`
    // for record `seq` or any later one.
    void silent_from(const Sim& s, Check& c, uint32_t core, uint32_t seq) {
        for (const Event& e : s.events) {
            if ((e.kind == kEvReport || e.kind == kEvAck) && e.core == core &&
                km::newer(e.a + 1, seq)) {
                c.expect(
                    false,
                    fmt("core %u reported record %u after giving up on %u",
                        core,
                        e.a,
                        seq)
                );
            }
        }
    }

    struct Outcome {
        Check check;
        uint64_t cycles = 0;
        double host_ms = 0;
        std::vector<std::string> stuck, violations;
        size_t jobs = 0, done = 0;
        uint32_t mailbox_peak = 0;
    };

    void finish(const Sim& s, Outcome& o) {
        o.cycles = s.end_cycles;
        o.host_ms = s.host_ms;
        o.stuck = s.stuck;
        o.violations = s.violations;
        o.jobs = s.jobs.size();
        o.mailbox_peak = s.mailbox_peak;
        for (const JobResult& r : s.jobs) {
            o.done += r.status == kDone;
        }
    }

    // ---- cases ----

    // A few MAP jobs on a 4-core grid, no fault: everything completes.
    void smoke(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.jobs = {
            map_job(0, 4, 20), map_job(1, 2, 5), nop_job(3, 1), map_job(0, 1, 18)
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        finish(s, o);
    }

    // A long MAP record is in flight on core 1 (compute slow, reader waiting for room)
    // when the dispatcher stops the grid because core 2 stalls.
    void stop_during_long_compute(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.wait_limit = 512;
        cfg.deadline = 100000;
        cfg.bound = 2000000;
        cfg.jobs = {map_job(1, 1, 40), Job{}};
        cfg.jobs[1].seg = kSegStall;
        cfg.jobs[1].first = 2;
        Sim* sp = nullptr;
        cfg.triggers.push_back([&](const Probe& p) {
            if (p.core == 1 && p.role == kUnpack &&
                !std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn) {
                sp->freeze(1, kUnpack, 5000);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        o.check.expect(s.timed_out, "the dispatcher's deadline stopped the grid");
        o.check.expect(
            s.jobs[0].status != kDone, "the long record is not acknowledged"
        );
        finish(s, o);
    }

    // Core 1's math thread is held for three wait bounds in its first record: the
    // writer's wait for the output gives up.
    void writer_gives_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(1, 1, 4), map_job(1, 1, 4), map_job(0, 1, 2)};
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (!held && p.core == 1 && p.role == kMath &&
                !std::strcmp(p.name, "tile_regs_acquire")) {
                held = true;
                sp->freeze(1, kMath, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 1, s.jobs[0].seq);
        o.check.expect(
            s.jobs[0].status != kDone, "the record without output is not acked"
        );
        o.check.expect(
            s.state(1, km::kStWriter) != 0, "core 1's writer recorded a give-up"
        );
        finish(s, o);
    }

    // Core 1's writer is held before its report: the leader (core 0) gives up waiting.
    void leader_gives_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2), nop_job(0, 1), nop_job(0, 1)};
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (!held && p.core == 1 && p.role == kWriter &&
                !std::strcmp(p.name, "noc_semaphore_inc")) {
                held = true;
                sp->freeze(1, kWriter, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 0, s.jobs[0].seq);
        o.check.expect(s.jobs[0].status != kDone, "the record is not acknowledged");
        o.check.expect(s.state(0, km::kStWriter) != 0, "the leader recorded a give-up");
        finish(s, o);
    }

    // Core 1's unpack thread is held at its first input tile: the reader's wait for
    // room in the input buffer gives up in a record of 24 tiles.
    void reader_gives_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(1, 1, 24), map_job(1, 1, 4), map_job(0, 1, 2)};
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (!held && p.core == 1 && p.role == kUnpack &&
                !std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn) {
                held = true;
                sp->freeze(1, kUnpack, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 1, s.jobs[0].seq);
        o.check.expect(s.jobs[0].status != kDone, "the short-fed record is not acked");
        o.check.expect(
            s.state(1, km::kStReader) != 0, "core 1's reader recorded a give-up"
        );
        finish(s, o);
    }

    // Core 1's compute is still in its first record when the reader, having handed
    // the second, waits for the token page and the stop word is set.
    void token_wait_after_stop(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(1, 1, 2), nop_job(1, 1)};
        Sim* sp = nullptr;
        bool held = false, stopped = false;
        uint32_t handed = 0;
        cfg.triggers.push_back([&](const Probe& p) {
            if (p.core != 1) {
                return;
            }
            if (!held && p.role == kMath && !std::strcmp(p.name, "tile_regs_acquire")) {
                held = true;
                sp->freeze(1, kMath, 6 * 8 * 1024);
            }
            if (p.role == kReader && !std::strcmp(p.name, "cb_push_back") &&
                p.cb == km::kCbTokenCompute) {
                handed++;
            }
            if (!stopped && handed == 2 && p.role == kReader &&
                !std::strcmp(p.name, "cb_pages_reservable_at_back") &&
                p.cb == km::kCbTokenCompute) {
                stopped = true;
                sp->set_stop(1);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        o.check.expect(stopped, "the stop word was set while the reader waited");
        finish(s, o);
    }

    // Every core counts the one refused record in its own word (not in records taken).
    void refused_counted(const Sim& s, Check& c) {
        for (uint32_t core = 0; core < s.cfg.cores; core++) {
            const uint32_t n = (
                km::kStSlotWrong == ~0u ? 0 : s.state(core, km::kStSlotWrong)
            );
            c.expect(n == 1, fmt("core %u counts %u refused records, not 1", core, n));
        }
    }

    // A record whose header names slot 20: not run, counted as refused by each core.
    void slot_out_of_range(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2)};
        cfg.jobs[0].hdr_slot = 20;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        for (const Event& e : s.events) {
            o.check.expect(
                e.kind != kEvTake || e.a != s.jobs[0].seq,
                fmt("core %u ran the record with slot field 20", e.core)
            );
        }
        o.check.expect(s.jobs[0].status != kDone, "the record is not acknowledged");
        refused_counted(s, o.check);
        finish(s, o);
    }

    // A record of both cores, found in ring slot 0, whose header names slot 3.
    void slot_not_ring_slot(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2)};
        cfg.jobs[0].hdr_slot = 3;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        for (const Event& e : s.events) {
            o.check.expect(
                e.kind != kEvTake || e.a != s.jobs[0].seq,
                fmt("core %u ran a record whose slot field is not its slot", e.core)
            );
        }
        o.check.expect(s.jobs[0].status != kDone, "the record is not acknowledged");
        refused_counted(s, o.check);
        finish(s, o);
    }

    // EXIT with team (0, 2) on four cores; the increments of cores 2 and 3 are late.
    void exit_partial_team(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.jobs = {nop_job(0, 4)};
        cfg.exit_team = km::make_team(0, 2);
        cfg.delay = [](const Chunk& c) -> int64_t {
            return c.atomic && c.src >= 2 ? 3000 : 40;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        for (uint32_t core = 0; core < cfg.cores; core++) {
            o.check.expect(
                s.state(core, km::kStWriter) == 0,
                fmt("core %u's writer did not give up", core)
            );
        }
        finish(s, o);
    }

    // Sequence numbers jump in steps below 2^31 up to the 32-bit wrap and run across
    // it (0 is skipped: it marks the synthetic EXIT); every team is the whole grid.
    void scan_across_wrap(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.wait_limit = 64;
        cfg.idle_limit = 4096;
        cfg.deadline = 400000;
        cfg.bound = 1ull << 26;
        cfg.first_seq = 0x100;
        cfg.next_seq = [](uint32_t s) -> uint32_t {
            if (s == 0x103) {
                return 0x70000000u;
            }
            if (s == 0x70000003u) {
                return 0xE0000000u;
            }
            if (s == 0xE0000003u) {
                return 0xFFFFFFF0u;
            }
            return s + 1 == 0 ? 1 : s + 1;
        };
        std::mt19937 rng(7);
        for (uint32_t k = 0; k < 48; k++) {
            cfg.jobs.push_back(map_job(0, 4, rng() % 21));
        }
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        bool wrapped = false;
        for (size_t k = 1; k < s.jobs.size(); k++) {
            wrapped = wrapped || s.jobs[k].seq < s.jobs[k - 1].seq;
        }
        o.check.expect(wrapped, "the sequence wrapped during the run");
        finish(s, o);
    }

    // Record 0 (smaller sequence) lands its last chunk late on core 1; record 1, also
    // for core 1, has fully landed: core 1 waits and runs record 0 first.
    void smallest_unless_part_landed(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.depth = 2;
        cfg.jobs = {nop_job(1, 1), nop_job(1, 1)};
        cfg.delay = [](const Chunk& c) -> int64_t {
            return c.job == 0 && c.dst == 1 && c.part == 3 ? 6000 : 40;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        std::vector<uint32_t> order;
        for (const Event& e : s.events) {
            if (e.kind == kEvTake && e.core == 1) {
                order.push_back(e.a);
            }
        }
        o.check.expect(
            order.size() >= 2 && order[0] == s.jobs[0].seq && order[1] == s.jobs[1].seq,
            "core 1 ran the smaller sequence first"
        );
        o.check.expect(
            s.jobs[1].issued + 6000 > s.jobs[0].issued + 40,
            "record 1 was complete before record 0"
        );
        finish(s, o);
    }

    // Chunk 2 of the record to core 1 is lost: core 1 never sees it land, the deadline
    // stops the grid, every kernel leaves.
    void dropped_chunk_times_out(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2)};
        cfg.delay = [](const Chunk& c) -> int64_t {
            return c.job == 0 && c.dst == 1 && c.part == 2 ? kDrop : 40;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(
            s.timed_out && s.jobs[0].status == kTimedOut,
            "the record timed out at the dispatcher"
        );
        finish(s, o);
    }

    // Every team (first, count) of an 8-core grid, rows of 3, random chunk delays.
    void completion_every_team(Outcome& o) {
        Config cfg = fault_config(8);
        cfg.row = 3;
        cfg.wait_limit = 64;
        cfg.idle_limit = 4096;
        cfg.deadline = 400000;
        cfg.bound = 1ull << 28;
        for (uint32_t f = 0; f < cfg.cores; f++) {
            for (uint32_t n = 1; f + n <= cfg.cores; n++) {
                cfg.jobs.push_back(nop_job(f, n));
                cfg.jobs.push_back(map_job(f, n, (f * 7 + n * 3) % 20));
            }
        }
        auto rng = std::make_shared<std::mt19937>(11);
        cfg.delay = [rng](const Chunk&) -> int64_t {
            return 16 + (*rng)() % 400;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        finish(s, o);
    }

    // 10^4 jobs with random teams, sizes and delays on 6 cores, no fault.
    void random_jobs(Outcome& o) {
        Config cfg;
        cfg.cores = 6;
        cfg.row = 3;
        cfg.wait_limit = 256;
        cfg.idle_limit = 4096;
        cfg.deadline = 4000000;
        cfg.bound = 1ull << 40;
        cfg.host_seconds = 120;
        std::mt19937 rng(5);
        for (uint32_t k = 0; k < 10000; k++) {
            const uint32_t f = rng() % cfg.cores;
            const uint32_t n = 1 + rng() % (cfg.cores - f);
            cfg.jobs.push_back(
                rng() % 4 == 0 ? nop_job(f, n) : map_job(f, n, rng() % 25)
            );
        }
        auto drng = std::make_shared<std::mt19937>(9);
        cfg.delay = [drng](const Chunk&) -> int64_t {
            return 16 + (*drng)() % 300;
        };
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        finish(s, o);
    }

    // Idle limit 0 means no limit: a 400,000-cycle gap between two records leaves
    // every core in place, and nothing gives up.
    void idle_zero_means_no_limit(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.idle_limit = 0;
        cfg.deadline = 600000;
        cfg.bound = 3000000;
        cfg.jobs = {map_job(1, 1, 4), map_job(0, 2, 4)};
        cfg.jobs[1].gap = 400000;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        for (uint32_t core = 0; core < cfg.cores; core++) {
            o.check.expect(
                s.state(core, km::kStReader) == 0 && s.state(core, km::kStWriter) == 0,
                fmt("core %u recorded no give-up", core)
            );
        }
        finish(s, o);
    }

    // A nonzero idle limit that passes is a give-up like any other: the reader records
    // it, the tile leaves and the record issued after the gap is not run.
    void idle_expiry_is_a_give_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(0, 2, 4), map_job(0, 2, 4)};
        cfg.jobs[1].gap = 200000;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        const uint32_t want = km::kGaveLimit | km::kWhereNext << 8;
        for (uint32_t core = 0; core < cfg.cores; core++) {
            o.check.expect(
                (s.state(core, km::kStReader) & 0xFFFF) == want,
                fmt("core %u's reader recorded the idle give-up", core)
            );
        }
        for (const Event& e : s.events) {
            o.check.expect(
                e.kind != kEvTake || e.a != s.jobs[1].seq,
                fmt("core %u ran a record after its idle give-up", e.core)
            );
        }
        o.check.expect(
            s.timed_out && s.jobs[1].status != kDone,
            "the record after the gap timed out at the dispatcher"
        );
        finish(s, o);
    }

    // ---- the compute mailboxes (unpack -> math, pack: ok, then the sequence) ----

    bool is(const Probe& p, uint32_t core, Role role, const char* name) {
        return p.core == core && p.role == role && !std::strcmp(p.name, name);
    }

    // A mailbox case runs at mailbox depths 1 and 4; it checks that every thread
    // returned, nothing broke the stub's contract and no mailbox word was left over.
    template <typename Scenario>
    void mailbox_case(Outcome& o, Scenario scenario) {
        for (uint32_t depth : {1u, 4u}) {
            Check c;
            Sim* sp = nullptr;
            bool fired = false;
            Config cfg = fault_config(2);
            cfg.mailbox_depth = depth;
            scenario(cfg, sp, fired, c, false);
            Sim s(cfg);
            sp = &s;
            s.run();
            c.expect(fired, "the fault was injected at the named point");
            c.expect(
                s.all_returned, "every thread of every core returned within the bound"
            );
            c.expect(s.violations.empty(), "no stub contract or landing violation");
            c.expect(
                s.mailbox_words() == 0,
                fmt("%u words left in compute mailboxes", s.mailbox_words())
            );
            scenario(cfg, sp, fired, c, true);
            for (const std::string& f : c.fails) {
                o.check.expect(false, fmt("depth %u: ", depth) + f);
            }
            if (depth == 4) {
                finish(s, o);
            } else {
                for (const std::string& t : s.stuck) {
                    o.stuck.push_back(fmt("depth %u: ", depth) + t);
                }
            }
        }
    }

    // Unpack is held after its "ok" for a token and before the sequence, and the stop
    // word is set there: math and pack wait for the sequence, then all leave.
    void mb_stop_between_ok_and_seq(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check&, bool after) {
            if (after) {
                return;
            }
            cfg.jobs = {map_job(1, 1, 4), map_job(1, 1, 4), nop_job(1, 1)};
            auto takes = std::make_shared<uint32_t>(0);
            auto writes = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, takes, writes](const Probe& p) {
                if (is(p, 1, kUnpack, "cb_wait_front") && p.cb == km::kCbTokenCompute) {
                    ++*takes;
                    *writes = 0;
                } else if (
                    is(p, 1, kUnpack, "mailbox_write") && ++*writes == 3 &&
                    *takes == 2 && !fired
                ) {
                    fired = true;
                    sp->set_stop(1);
                    sp->freeze(1, kUnpack, 3 * 8 * 1024);
                }
            });
        });
    }

    // Math is held at its read of the next token's "ok" while the stop word makes
    // unpack's token wait give up: math reads "not ok" later and leaves.
    void mb_math_held_while_unpack_aborts(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check&, bool after) {
            if (after) {
                return;
            }
            cfg.jobs = {map_job(1, 1, 4), nop_job(0, 1)};
            auto reads = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, reads](const Probe& p) {
                // Record 0 takes 2 reads (ok, sequence) and 1 per input tile.
                if (is(p, 1, kMath, "mailbox_read") && ++*reads == 7 && !fired) {
                    fired = true;
                    sp->set_stop(1);
                    sp->freeze(1, kMath, 4 * 8 * 1024);
                }
            });
        });
    }

    // The reader is held before its first input lands; pack is held at its first input
    // "ok"; the stop word ends unpack's input wait and later hands an EXIT.
    void mb_pack_held_while_input_wait_aborts(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check&, bool after) {
            if (after) {
                return;
            }
            cfg.jobs = {map_job(1, 1, 8)};
            auto reads = std::make_shared<uint32_t>(0);
            auto held = std::make_shared<bool>(false);
            auto barriers = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, reads, held, barriers](const Probe& p) {
                // The reader's first read barrier is start-up's; the second is input.
                if (!*held && is(p, 1, kReader, "noc_async_read_barrier") &&
                    ++*barriers == 2) {
                    *held = true;
                    sp->freeze(1, kReader, 3 * 8 * 1024);
                }
                if (is(p, 1, kPack, "mailbox_read") && ++*reads == 3 && !fired) {
                    fired = true;
                    sp->set_stop(1);
                    sp->freeze(1, kPack, 4 * 8 * 1024);
                }
            });
        });
    }

    // The input wait of record 0 gives up on its bound (reader held, no stop); record
    // 1's token is good. By the rule, the tile then acknowledges nothing.
    void mb_abort_then_good_token(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check& c, bool after) {
            if (after) {
                c.expect(!sp->jobs.empty(), "the grid started");
                if (!sp->jobs.empty()) {
                    silent_from(*sp, c, 1, sp->jobs[0].seq);
                }
                return;
            }
            cfg.jobs = {map_job(1, 1, 8), map_job(1, 1, 4)};
            auto held = std::make_shared<bool>(false);
            auto barriers = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, held, barriers](const Probe& p) {
                if (!*held && is(p, 1, kReader, "noc_async_read_barrier") &&
                    ++*barriers == 2) {
                    *held = fired = true;
                    sp->freeze(1, kReader, 12 * 1024);
                }
            });
        });
    }

    // Unpack is held at its first input tile of a 24-tile record: the reader's room
    // wait gives up on its bound, the EXIT token comes while compute is still inside.
    void mb_exit_during_input_wait(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check& c, bool after) {
            static bool inside = false;
            if (after) {
                c.expect(inside, "the EXIT token came while compute was in the record");
                c.expect(
                    sp->exit_job.status == kDone || sp->timed_out,
                    "the EXIT completed or the deadline stopped the grid"
                );
                return;
            }
            inside = false;
            cfg.jobs = {map_job(1, 1, 24)};
            auto held = std::make_shared<bool>(false);
            cfg.triggers.push_back([&, held](const Probe& p) {
                if (!*held && p.core == 1 && p.role == kUnpack &&
                    (!std::strcmp(p.name, "reg_read") ||
                     (!std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn)) &&
                    sp->popped(1, km::kCbTokenCompute) == 1) {
                    *held = true;
                    sp->freeze(1, kUnpack, 12 * 1024);
                }
                if (is(p, 1, kReader, "cb_push_back") && p.cb == km::kCbTokenCompute &&
                    sp->popped(1, km::kCbTokenCompute) == 1 && *held) {
                    fired = true;
                    inside = inside || sp->popped(1, kCbIn) < 24;
                }
            });
        });
    }

    // ---- a buffer's items against its end ----

    // Per record: n items of 5 pages, then one of `last` pages, through a 15-page
    // buffer from a compute producer to a data-movement consumer.
    constexpr uint32_t kCbItems = 24;
    constexpr uint32_t kItemRecords = 20;
    uint32_t items_n = 5, items_last = 1;

    void items_none() {
    }

    template <bool Producer>
    void items_side() {
        for (uint32_t rec = 1; rec <= kItemRecords; rec++) {
            note_record(rec);
            for (uint32_t i = 0; i <= items_n; i++) {
                const uint32_t pages = i < items_n ? 5 : items_last;
                if (Producer) {
                    cb_reserve_back(kCbItems, pages);
                    cb_push_back(kCbItems, pages);
                } else {
                    cb_wait_front(kCbItems, pages);
                    cb_pop_front(kCbItems, pages);
                }
            }
        }
    }

    // The 1-page item leaves the next record's first item straddling the end at
    // record 2 (n = 5 and 6); a 5-page last item keeps every item whole.
    void items_straddle_the_end(Outcome& o) {
        for (uint32_t n : {5u, 6u}) {
            for (uint32_t last : {1u, 5u}) {
                items_n = n;
                items_last = last;
                Config cfg;
                cfg.cores = 1;
                cfg.dispatch = false;
                cfg.exit = false;
                cfg.cbs = {{kCbItems, 0xC000, 64, 15}};
                cfg.kernels[kReader] = items_none;
                cfg.kernels[kWriter] = items_side<false>;
                cfg.kernels[kUnpack] = items_none;
                cfg.kernels[kMath] = items_none;
                cfg.kernels[kPack] = items_side<true>;
                Sim s(cfg);
                s.run();
                const std::string at = fmt("n %u, last %u: ", n, last);
                o.check.expect(s.all_returned, at + "every thread returned");
                o.check.expect(
                    s.popped(0, kCbItems) == kItemRecords * (5 * n + last),
                    at + "every page reached the consumer"
                );
                if (last == 5) {
                    o.check.expect(s.violations.empty(), at + "no buffer violation");
                    finish(s, o);
                    continue;
                }
                bool first = true;
                for (const std::string& v : s.violations) {
                    o.check.expect(
                        (v.find(first ? "record 2:" : "record 1:") !=
                         std::string::npos) == first,
                        at +
                            (first ? "the first violation is at record 2: "
                                   : "no violation at record 1: ") +
                            v
                    );
                    first = false;
                }
                o.check.expect(!s.violations.empty(), at + "the straddle is reported");
            }
        }
    }

    // ---- the bounded buffer wrappers against an item past the end ----

    void item_feed() {  // one page for the compute input buffer
        cb_reserve_back(24, 1);
        cb_push_back(24, 1);
    }

    void expect_bad_item(
        const Sim& s,
        Check& c,
        uint32_t core,
        uint32_t word,
        uint32_t where,
        const char* w
    ) {
        c.expect(
            (s.state(core, word) & 0xFFFF) == (km::kGaveBadItem | where << 8),
            std::string(w) + " recorded the bad-item give-up"
        );
        c.expect(s.word(core, KM_CTL_STOP) != 0, std::string(w) + " set the stop word");
        c.expect(s.all_returned, std::string(w) + ": every thread returned");
    }

    // Each of the four wrappers is asked for an item that starts one page before the
    // end: it gives up with the bad-item code and sets the stop word.
    void wrappers_refuse_straddling_items(Outcome& o) {
        for (uint32_t a1 : {1u, 0u}) {  // reader (after 5 input pages), writer (after
                                        // 1)
            Config cfg = fault_config(2);
            Job bad;
            bad.seg = kSegBadItem;
            bad.first = 1;
            bad.a[1] = a1;
            cfg.jobs = {map_job(1, 1, 5), bad};
            Sim s(cfg);
            s.run();
            common(s, o.check);
            const char* w = a1 ? "the reader's reserve" : "the writer's wait_front";
            expect_bad_item(
                s, o.check, 1, a1 ? km::kStReader : km::kStWriter, km::kWhereData, w
            );
            o.check.expect(
                s.jobs[1].status != kDone,
                std::string(w) + ": the bad record is not acked"
            );
            finish(s, o);
        }
        for (bool in : {true, false}) {
            Config cfg;
            cfg.cores = 1;
            cfg.dispatch = false;
            cfg.exit = false;
            cfg.cbs = {{24, 0xC000, 64, 2}, {25, 0xC100, 64, 2}};
            for (void (*&k)() : cfg.kernels) {
                k = items_none;
            }
            if (in) {
                cfg.kernels[kReader] = item_feed;
                cfg.kernels[kUnpack] = kmsim_item_in_unpack_main;
                cfg.kernels[kMath] = kmsim_item_in_math_main;
                cfg.kernels[kPack] = kmsim_item_in_pack_main;
            } else {
                cfg.kernels[kUnpack] = kmsim_item_out_unpack_main;
                cfg.kernels[kMath] = kmsim_item_out_math_main;
                cfg.kernels[kPack] = kmsim_item_out_pack_main;
            }
            Sim s(cfg);
            s.run();
            const char* w = in ? "compute's wait_front" : "compute's reserve_back";
            expect_bad_item(
                s,
                o.check,
                0,
                km::kStCompute,
                in ? km::kWhereCInput : km::kWhereCOutput,
                w
            );
            o.check.expect(
                s.violations.empty(), std::string(w) + ": no stub violation"
            );
        }
    }

    // ---- KM_F_EXCL ----

    // An EXCL record on a team disjoint from the one in flight is issued at once; one
    // that overlaps either waits until both are acknowledged.
    void excl_waits_for_overlapping_team(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.jobs = {map_job(0, 2, 24), map_job(2, 2, 4), map_job(1, 2, 4)};
        cfg.jobs[1].flags = cfg.jobs[2].flags = KM_F_EXCL;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every record and the EXIT acknowledged");
        const auto& j = s.jobs;
        o.check.expect(
            j[1].issued < j[0].acked, "the disjoint EXCL record ran beside the first"
        );
        o.check.expect(
            j[2].issued >= j[0].acked && j[2].issued >= j[1].acked,
            "the overlapping EXCL record waited for both teams"
        );
        finish(s, o);
    }

    struct Case {
        const char* name;
        void (*fn)(Outcome&);
    };

    const Case kCases[] = {
        {"smoke", smoke},
        {"stop_during_long_compute", stop_during_long_compute},
        {"writer_gives_up", writer_gives_up},
        {"leader_gives_up", leader_gives_up},
        {"reader_gives_up", reader_gives_up},
        {"token_wait_after_stop", token_wait_after_stop},
        {"slot_out_of_range", slot_out_of_range},
        {"slot_not_ring_slot", slot_not_ring_slot},
        {"exit_partial_team", exit_partial_team},
        {"scan_across_wrap", scan_across_wrap},
        {"smallest_unless_part_landed", smallest_unless_part_landed},
        {"dropped_chunk_times_out", dropped_chunk_times_out},
        {"idle_zero_means_no_limit", idle_zero_means_no_limit},
        {"idle_expiry_is_a_give_up", idle_expiry_is_a_give_up},
        {"mb_stop_between_ok_and_seq", mb_stop_between_ok_and_seq},
        {"mb_math_held_while_unpack_aborts", mb_math_held_while_unpack_aborts},
        {"mb_pack_held_while_input_wait_aborts", mb_pack_held_while_input_wait_aborts},
        {"mb_abort_then_good_token", mb_abort_then_good_token},
        {"mb_exit_during_input_wait", mb_exit_during_input_wait},
        {"items_straddle_the_end", items_straddle_the_end},
        {"wrappers_refuse_straddling_items", wrappers_refuse_straddling_items},
        {"excl_waits_for_overlapping_team", excl_waits_for_overlapping_team},
        {"completion_every_team", completion_every_team},
        {"random_jobs", random_jobs},
    };

    std::string json(const std::string& s) {
        std::string out = "\"";
        for (char ch : s) {
            if (ch == '"' || ch == '\\') {
                out += '\\';
            }
            out += ch;
        }
        return out + "\"";
    }

    std::string json(const std::vector<std::string>& v) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); i++) {
            out += (i ? ", " : "") + json(v[i]);
        }
        return out + "]";
    }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: scenarios <name> | list\n");
        return 2;
    }
    for (const Case& c : kCases) {
        if (std::strcmp(argv[1], "list") == 0) {
            std::printf("%s\n", c.name);
        } else if (std::strcmp(argv[1], c.name) == 0) {
            Outcome o;
            c.fn(o);
            std::printf(
                "{\"case\": %s, \"ok\": %s, \"fails\": %s, \"stuck\": %s, "
                "\"violations\": %s, \"cycles\": %llu, \"host_ms\": %.1f, "
                "\"jobs\": %zu, \"done\": %zu, \"mailbox_peak\": %u}\n",
                json(c.name).c_str(),
                o.check.fails.empty() ? "true" : "false",
                json(o.check.fails).c_str(),
                json(o.stuck).c_str(),
                json(o.violations).c_str(),
                static_cast<unsigned long long>(o.cycles),
                o.host_ms,
                o.jobs,
                o.done,
                o.mailbox_peak
            );
            return 0;
        }
    }
    return std::strcmp(argv[1], "list") == 0 ? 0 : 2;
}
