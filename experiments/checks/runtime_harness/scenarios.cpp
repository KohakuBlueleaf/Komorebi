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
#include "checks.hpp"
#include "komorebi/device/core/km_core.hpp"

using namespace kmsim;
using namespace scen;

namespace {

    // ---- cases ----

    // A long MAP command is in flight on core 1 (compute slow, reader waiting for room)
    // when the dispatcher stops the grid because core 2 stalls.
    void stop_during_long_compute(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.wait_limit = 512;
        cfg.deadline = 100000;
        cfg.timeout = 2000000;
        cfg.jobs = {map_job(1, 1, 40), Job{}};
        cfg.jobs[1].handler = kHandlerStall;
        cfg.jobs[1].first = 2;
        Sim* sp = nullptr;
        cfg.triggers.push_back([&](const Probe& p) {
            if (
                p.core == 1 && p.role == kUnpack &&
                !std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn
            ) {
                sp->freeze(1, kUnpack, 5000);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        o.check.expect(s.timed_out, "the dispatcher's deadline stopped the grid");
        o.check.expect(
            s.jobs[0].status != kDone, "the long command is not acknowledged"
        );
        finish(s, o);
    }

    // Core 1's math thread is held for three wait bounds in its first command: the
    // writer's wait for the output gives up.
    void writer_gives_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(1, 1, 4), map_job(1, 1, 4), map_job(0, 1, 2)};
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (
                !held && p.core == 1 && p.role == kMath &&
                !std::strcmp(p.name, "tile_regs_acquire")
            ) {
                held = true;
                sp->freeze(1, kMath, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 1, s.jobs[0].sequence);
        o.check.expect(
            s.jobs[0].status != kDone, "the command without output is not acked"
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
            if (
                !held && p.core == 1 && p.role == kWriter &&
                !std::strcmp(p.name, "noc_semaphore_inc")
            ) {
                held = true;
                sp->freeze(1, kWriter, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 0, s.jobs[0].sequence);
        o.check.expect(s.jobs[0].status != kDone, "the command is not acknowledged");
        o.check.expect(s.state(0, km::kStWriter) != 0, "the leader recorded a give-up");
        finish(s, o);
    }

    // Core 1's unpack thread is held at its first input tile: the reader's wait for
    // room in the input buffer gives up in a command of 24 tiles.
    void reader_gives_up(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {map_job(1, 1, 24), map_job(1, 1, 4), map_job(0, 1, 2)};
        Sim* sp = nullptr;
        bool held = false;
        cfg.triggers.push_back([&](const Probe& p) {
            if (
                !held && p.core == 1 && p.role == kUnpack &&
                !std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn
            ) {
                held = true;
                sp->freeze(1, kUnpack, 3 * 8 * 1024);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        silent_from(s, o.check, 1, s.jobs[0].sequence);
        o.check.expect(s.jobs[0].status != kDone, "the short-fed command is not acked");
        o.check.expect(
            s.state(1, km::kStReader) != 0, "core 1's reader recorded a give-up"
        );
        finish(s, o);
    }

    // Core 1's compute is still in its first command when the reader, having handed
    // the second, waits for the token page and the stop flag is set.
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
            if (
                p.role == kReader && !std::strcmp(p.name, "cb_push_back") &&
                p.cb == km::kCbTokenCompute
            ) {
                handed++;
            }
            if (
                !stopped && handed == 2 && p.role == kReader &&
                !std::strcmp(p.name, "cb_pages_reservable_at_back") &&
                p.cb == km::kCbTokenCompute
            ) {
                stopped = true;
                sp->set_stop(1);
            }
        });
        Sim s(cfg);
        sp = &s;
        s.run();
        common(s, o.check);
        o.check.expect(stopped, "the stop flag was set while the reader waited");
        finish(s, o);
    }

    // Every core counts the one rejected command in its own word (not in commands
    // taken).
    void rejected_counted(const Sim& s, Check& c) {
        for (uint32_t core = 0; core < s.cfg.cores; core++) {
            const uint32_t n = (
                km::kStSlotWrong == ~0u ? 0 : s.state(core, km::kStSlotWrong)
            );
            c.expect(
                n == 1, fmt("core %u counts %u rejected commands, not 1", core, n)
            );
        }
    }

    // A command whose header names slot 20: not run, counted as rejected by each core.
    void slot_out_of_range(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2)};
        cfg.jobs[0].hdr_slot = 20;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        for (const Event& e : s.events) {
            o.check.expect(
                e.kind != kEventTake || e.a != s.jobs[0].sequence,
                fmt("core %u ran the command with slot field 20", e.core)
            );
        }
        o.check.expect(s.jobs[0].status != kDone, "the command is not acknowledged");
        rejected_counted(s, o.check);
        finish(s, o);
    }

    // A command of both cores, found in ring slot 0, whose header names slot 3.
    void slot_not_ring_slot(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.jobs = {nop_job(0, 2)};
        cfg.jobs[0].hdr_slot = 3;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        for (const Event& e : s.events) {
            o.check.expect(
                e.kind != kEventTake || e.a != s.jobs[0].sequence,
                fmt("core %u ran a command whose slot field is not its slot", e.core)
            );
        }
        o.check.expect(s.jobs[0].status != kDone, "the command is not acknowledged");
        rejected_counted(s, o.check);
        finish(s, o);
    }

    // Sequence numbers jump in steps below 2^31 up to the 32-bit wrap and run across
    // it (0 is skipped: it marks the synthetic EXIT); every group is the whole grid.
    void scan_across_wrap(Outcome& o) {
        Config cfg = fault_config(4);
        cfg.wait_limit = 64;
        cfg.idle_limit = 4096;
        cfg.deadline = 400000;
        cfg.timeout = 1ull << 26;
        cfg.first_sequence = 0x100;
        cfg.next_sequence = [](uint32_t s) -> uint32_t {
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
        o.check.expect(all_done(s), "every command and the EXIT acknowledged");
        bool wrapped = false;
        for (size_t k = 1; k < s.jobs.size(); k++) {
            wrapped = wrapped || s.jobs[k].sequence < s.jobs[k - 1].sequence;
        }
        o.check.expect(wrapped, "the sequence wrapped during the run");
        finish(s, o);
    }

    // Command 0 (smaller sequence) lands its last chunk late on core 1; command 1, also
    // for core 1, has fully landed: core 1 waits and runs command 0 first.
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
        o.check.expect(all_done(s), "every command and the EXIT acknowledged");
        std::vector<uint32_t> order;
        for (const Event& e : s.events) {
            if (e.kind == kEventTake && e.core == 1) {
                order.push_back(e.a);
            }
        }
        o.check.expect(
            order.size() >= 2 && order[0] == s.jobs[0].sequence &&
                order[1] == s.jobs[1].sequence,
            "core 1 ran the smaller sequence first"
        );
        o.check.expect(
            s.jobs[1].submitted + 6000 > s.jobs[0].submitted + 40,
            "command 1 was complete before command 0"
        );
        finish(s, o);
    }

    // Chunk 2 of the command to core 1 is lost: core 1 never sees it land, the deadline
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
            "the command timed out at the dispatcher"
        );
        finish(s, o);
    }

    // Every group (first, count) of an 8-core grid, rows of 3, random chunk delays.
    void completion_every_group(Outcome& o) {
        Config cfg = fault_config(8);
        cfg.row = 3;
        cfg.wait_limit = 64;
        cfg.idle_limit = 4096;
        cfg.deadline = 400000;
        cfg.timeout = 1ull << 28;
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
        o.check.expect(all_done(s), "every command and the EXIT acknowledged");
        finish(s, o);
    }

    // 10^4 jobs with random groups, sizes and delays on 6 cores, no fault.
    void random_jobs(Outcome& o) {
        Config cfg;
        cfg.cores = 6;
        cfg.row = 3;
        cfg.wait_limit = 256;
        cfg.idle_limit = 4096;
        cfg.deadline = 4000000;
        cfg.timeout = 1ull << 40;
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
        o.check.expect(all_done(s), "every command and the EXIT acknowledged");
        finish(s, o);
    }

    // Idle limit 0 means no limit: a 400,000-cycle gap between two commands leaves
    // every core in place, and nothing gives up.
    void idle_zero_means_no_limit(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.idle_limit = 0;
        cfg.deadline = 600000;
        cfg.timeout = 3000000;
        cfg.jobs = {map_job(1, 1, 4), map_job(0, 2, 4)};
        cfg.jobs[1].gap = 400000;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(all_done(s), "every command and the EXIT acknowledged");
        for (uint32_t core = 0; core < cfg.cores; core++) {
            o.check.expect(
                s.state(core, km::kStReader) == 0 && s.state(core, km::kStWriter) == 0,
                fmt("core %u recorded no give-up", core)
            );
        }
        finish(s, o);
    }

    // A nonzero idle limit that passes is a give-up like any other: the reader commands
    // it, the tile leaves and the command submitted after the gap is not run.
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
                e.kind != kEventTake || e.a != s.jobs[1].sequence,
                fmt("core %u ran a command after its idle give-up", e.core)
            );
        }
        o.check.expect(
            s.timed_out && s.jobs[1].status != kDone,
            "the command after the gap timed out at the dispatcher"
        );
        finish(s, o);
    }

    // ---- a command given up at its deadline; the run goes on ----

    // Rank 1 of a 2-core HANG waits (`mode`) for what never comes: given up at the
    // deadline after an inspect, then right MAPs on its cores, no ack of the HANG.
    void give_up_case(Outcome& o, uint32_t mode, uint32_t role, uint32_t where) {
        Config cfg = fault_config(2);
        cfg.give_up = true;
        cfg.wait_limit = 128;  // the waits' own limit is past the deadline
        cfg.idle_limit = 1024;
        cfg.deadline = 40000;
        Job hang;
        hang.handler = kHandlerHang;
        hang.count = 2;
        hang.a[0] = 1;
        hang.a[1] = mode;
        cfg.jobs = {hang, map_job(0, 2, 20), map_job(0, 2, 7), map_job(1, 1, 3)};
        cfg.jobs[1].gap = 50000;  // after the timeout, as a host would send it
        Sim s(cfg);
        s.run();
        common(s, o.check);
        const JobResult& r = s.jobs[0];
        o.check.expect(r.status == kGivenUp, "the HANG was given up at its deadline");
        o.check.expect(
            r.live.size() == 2 && r.live[1][km::kLvWhere + role] == where,
            fmt("the inspect names where %u on rank 1", where)
        );
        o.check.expect(
            r.live.size() == 2 &&
                r.live[0][km::kLvWhere + km::kRoleWriter] == km::kWhereComplete,
            "the inspect shows rank 0, the leader, waiting for rank 1's report"
        );
        for (size_t k = 1; k < s.jobs.size(); k++) {
            o.check.expect(
                s.jobs[k].status == kDone, fmt("job %u after it acknowledged", k)
            );
        }
        o.check.expect(
            s.exit_job.status == kDone, "the EXIT acknowledged: the run went on"
        );
        o.check.expect(s.late_acks == 0, "no ack for the given-up command");
        const uint32_t word = (
            role == km::kRoleReader
                ? km::kStReader
                : role == km::kRoleWriter
                    ? km::kStWriter
                    : km::kStCompute
        );
        o.check.expect(
            s.state(1, word) == (km::kGaveDeadline | where << 8 | r.slot << 16) &&
                s.state(1, word + 1) == r.sequence,
            fmt("rank 1 recorded the deadline give-up in where %u", where)
        );
        o.check.expect(
            s.word(1, KM_CONTROL_LIVE + 4 * km::kLvGivenUp) == 1,
            "rank 1's writer counts one command given up"
        );
        finish(s, o);
    }

    // Rank 1 of a 2-core STALL waits for the stop flag alone: after the abort it does
    // not leave the command, and past the grace the deadline ends the run.
    void give_up_stall_ends_the_run(Outcome& o) {
        Config cfg = fault_config(2);
        cfg.give_up = true;
        cfg.idle_limit = 1000000;  // the stall outlasts the deadline and the grace
        cfg.deadline = 40000;
        Job stall;
        stall.handler = kHandlerStall;
        stall.count = 2;
        stall.a[0] = 1;
        cfg.jobs = {stall, map_job(0, 2, 20)};
        cfg.jobs[1].gap = 60000;
        Sim s(cfg);
        s.run();
        common(s, o.check);
        o.check.expect(s.timed_out, "the deadline ended the run");
        o.check.expect(s.jobs[0].status == kTimedOut, "the STALL was not given up");
        o.check.expect(s.jobs[1].status != kDone, "the MAP behind it never ran");
        finish(s, o);
    }

    void give_up_reader_port(Outcome& o) {
        give_up_case(o, 1, km::kRoleReader, km::kWherePort);
    }

    void give_up_compute_input(Outcome& o) {
        give_up_case(o, 2, km::kRoleUnpack, km::kWhereCInput);
    }

    void give_up_writer_output(Outcome& o) {
        give_up_case(o, 3, km::kRoleWriter, km::kWhereData);
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
                s.all_returned, "every thread of every core returned within the timeout"
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
    void mb_stop_between_ok_and_sequence(Outcome& o) {
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

    // Math is held at its read of the next token's "ok" while the stop flag makes
    // unpack's token wait give up: math reads "not ok" later and leaves.
    void mb_math_held_while_unpack_aborts(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check&, bool after) {
            if (after) {
                return;
            }
            cfg.jobs = {map_job(1, 1, 4), nop_job(0, 1)};
            auto reads = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, reads](const Probe& p) {
                // Command 0 takes 2 reads (ok, sequence) and 1 per input tile.
                if (is(p, 1, kMath, "mailbox_read") && ++*reads == 7 && !fired) {
                    fired = true;
                    sp->set_stop(1);
                    sp->freeze(1, kMath, 4 * 8 * 1024);
                }
            });
        });
    }

    // The reader is held before its first input lands; pack is held at its first input
    // "ok"; the stop flag ends unpack's input wait and later hands an EXIT.
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
                // The reader's first read barrier is input (start-up's is the timed
                // flush of the runtime).
                if (
                    !*held && is(p, 1, kReader, "noc_async_read_barrier") &&
                    ++*barriers == 1
                ) {
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

    // The input wait of command 0 times out (reader held, no stop); command
    // 1's token is good. By the rule, the tile then acknowledges nothing.
    void mb_abort_then_good_token(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check& c, bool after) {
            if (after) {
                c.expect(!sp->jobs.empty(), "the grid started");
                if (!sp->jobs.empty()) {
                    silent_from(*sp, c, 1, sp->jobs[0].sequence);
                }
                return;
            }
            cfg.jobs = {map_job(1, 1, 8), map_job(1, 1, 4)};
            auto held = std::make_shared<bool>(false);
            auto barriers = std::make_shared<uint32_t>(0);
            cfg.triggers.push_back([&, held, barriers](const Probe& p) {
                if (
                    !*held && is(p, 1, kReader, "noc_async_read_barrier") &&
                    ++*barriers == 1
                ) {
                    *held = fired = true;
                    sp->freeze(1, kReader, 12 * 1024);
                }
            });
        });
    }

    // Unpack is held at its first input tile of a 24-tile command: the reader's room
    // wait times out, the EXIT token comes while compute is still inside.
    void mb_exit_during_input_wait(Outcome& o) {
        mailbox_case(o, [](Config& cfg, Sim*& sp, bool& fired, Check& c, bool after) {
            static bool inside = false;
            if (after) {
                c.expect(
                    inside, "the EXIT token came while compute was in the command"
                );
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
                if (
                    !*held && p.core == 1 && p.role == kUnpack &&
                    (
                        !std::strcmp(p.name, "reg_read") ||
                        (!std::strcmp(p.name, "cb_wait_front") && p.cb == kCbIn)
                    ) &&
                    sp->popped(1, km::kCbTokenCompute) == 1
                ) {
                    *held = true;
                    sp->freeze(1, kUnpack, 12 * 1024);
                }
                if (
                    is(p, 1, kReader, "cb_push_back") && p.cb == km::kCbTokenCompute &&
                    sp->popped(1, km::kCbTokenCompute) == 1 && *held
                ) {
                    fired = true;
                    inside = inside || sp->popped(1, kCbIn) < 24;
                }
            });
        });
    }

    // ---- a buffer's items against its end ----

    // Per command: n items of 5 pages, then one of `last` pages, through a 15-page
    // buffer from a compute producer to a data-movement consumer.
    constexpr uint32_t kCbItems = 24;
    constexpr uint32_t kItemCommands = 20;
    uint32_t items_n = 5, items_last = 1;

    template <bool Producer>
    void items_side() {
        for (uint32_t rec = 1; rec <= kItemCommands; rec++) {
            note_command(rec);
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

    // The 1-page item leaves the next command's first item straddling the end at
    // command 2 (n = 5 and 6); a 5-page last item keeps every item whole.
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
                    s.popped(0, kCbItems) == kItemCommands * (5 * n + last),
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
                        (
                            v.find(first ? "command 2:" : "command 1:") !=
                            std::string::npos
                        ) == first,
                        at +
                            (
                                first
                                    ? "the first violation is at command 2: "
                                    : "no violation at command 1: "
                            ) +
                            v
                    );
                    first = false;
                }
                o.check.expect(!s.violations.empty(), at + "the straddle is reported");
            }
        }
    }

}  // namespace

int main(int argc, char** argv) {
    std::vector<Case> cases = pinned();
    cases.insert(
        cases.end(),
        {
            {"stop_during_long_compute", stop_during_long_compute},
            {"writer_gives_up", writer_gives_up},
            {"leader_gives_up", leader_gives_up},
            {"reader_gives_up", reader_gives_up},
            {"token_wait_after_stop", token_wait_after_stop},
            {"slot_out_of_range", slot_out_of_range},
            {"slot_not_ring_slot", slot_not_ring_slot},
            {"scan_across_wrap", scan_across_wrap},
            {"smallest_unless_part_landed", smallest_unless_part_landed},
            {"dropped_chunk_times_out", dropped_chunk_times_out},
            {"idle_zero_means_no_limit", idle_zero_means_no_limit},
            {"idle_expiry_is_a_give_up", idle_expiry_is_a_give_up},
            {"mb_stop_between_ok_and_sequence", mb_stop_between_ok_and_sequence},
            {"mb_math_held_while_unpack_aborts", mb_math_held_while_unpack_aborts},
            {
                "mb_pack_held_while_input_wait_aborts",
                mb_pack_held_while_input_wait_aborts
            },
            {"mb_abort_then_good_token", mb_abort_then_good_token},
            {"mb_exit_during_input_wait", mb_exit_during_input_wait},
            {"items_straddle_the_end", items_straddle_the_end},
            {"give_up_stall_ends_the_run", give_up_stall_ends_the_run},
            {"give_up_reader_port", give_up_reader_port},
            {"give_up_compute_input", give_up_compute_input},
            {"give_up_writer_output", give_up_writer_output},
            {"completion_every_group", completion_every_group},
            {"random_jobs", random_jobs},
        }
    );
    cases.insert(cases.end(), messages_cases().begin(), messages_cases().end());
    return run_cases(cases, argc, argv);
}
