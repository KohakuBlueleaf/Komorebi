// The scenarios that pin known runtime behaviour: run by the scenario program and, on
// their own, by tests/core/test_runtime_harness.py.
#include <stdint.h>

#include <cstring>
#include <string>
#include <vector>

#include "api/dataflow/dataflow_api.h"
#include "checks.hpp"
#include "komorebi/device/km_core.hpp"

namespace scen {

    namespace {

        // A few MAP jobs on a 4-core grid, no fault: everything completes.
        void smoke(Outcome& o) {
            Config cfg = fault_config(4);
            cfg.jobs = {
                map_job(0, 4, 20), map_job(1, 2, 5), nop_job(3, 1), map_job(0, 1, 18)
            };
            Sim s(cfg);
            s.run();
            common(s, o.check);
            o.check.expect(all_done(s), "every command and the EXIT acknowledged");
            finish(s, o);
        }

        // EXIT with group (0, 2) on four cores; the increments of cores 2 and 3 are
        // late.
        void exit_partial_group(Outcome& o) {
            Config cfg = fault_config(4);
            cfg.jobs = {nop_job(0, 4)};
            cfg.exit_group = km::make_group(0, 2);
            cfg.delay = [](const Chunk& c) -> int64_t {
                return c.atomic && c.src >= 2 ? 3000 : 40;
            };
            Sim s(cfg);
            s.run();
            common(s, o.check);
            o.check.expect(all_done(s), "every command and the EXIT acknowledged");
            for (uint32_t core = 0; core < cfg.cores; core++) {
                o.check.expect(
                    s.state(core, km::kStWriter) == 0,
                    fmt("core %u's writer did not give up", core)
                );
            }
            finish(s, o);
        }

        // A command with no output on two cores; core 1's pack is held as it takes it:
        // the command is acknowledged only after core 1's compute is released.
        void no_output_waits_for_compute(Outcome& o) {
            constexpr uint64_t kHold = 4000;
            Config cfg = fault_config(2);
            cfg.jobs = {nop_job(0, 2)};
            Sim* sp = nullptr;
            uint64_t released = 0;
            cfg.triggers.push_back([&](const Probe& p) {
                if (released == 0 && p.core == 1 && p.role == kPack &&
                    !std::strcmp(p.name, "mailbox_write")) {
                    released = p.t + kHold;
                    sp->freeze(1, kPack, kHold);
                }
            });
            Sim s(cfg);
            sp = &s;
            s.run();
            common(s, o.check);
            o.check.expect(all_done(s), "every command and the EXIT acknowledged");
            o.check.expect(released != 0, "core 1's pack was held");
            o.check.expect(
                s.jobs[0].acked >= released,
                fmt("acknowledged at %u, before compute was released at %u",
                    static_cast<unsigned>(s.jobs[0].acked),
                    static_cast<unsigned>(released))
            );
            finish(s, o);
        }

        // A clock that moves on every register read, started so that the carry of the
        // low word falls before, between or after the reads: no reading leaves them.
        void wall_clock_across_a_carry(Outcome& o) {
            for (uint64_t start = (1ull << 32) - 8; start < (1ull << 32) + 4; start++) {
                uint64_t t = start;
                const auto read = [&](bool high) {
                    t += 3;
                    return static_cast<uint32_t>(high ? t >> 32 : t);
                };
                const uint64_t got = km::wall64_of(
                    [&] {
                        return read(true);
                    },
                    [&] {
                        return read(false);
                    }
                );
                o.check.expect(
                    got > start && got <= t,
                    fmt("start 2^32 - 8 + %u: a reading outside the reads",
                        static_cast<unsigned>(start - ((1ull << 32) - 8)))
                );
            }
        }

        // One slot, three commands on cores 1 to 3: a flagged RESULT is acked with the
        // leader's result, an unflagged one leaves word 1 alone, a flagged NOP sends 0.
        void result_word_from_the_leader(Outcome& o) {
            Config cfg = fault_config(4);
            cfg.depth = 1;
            Job res = nop_job(1, 3);
            res.handler = kHandlerResult;
            res.a[0] = 0x5EED;
            res.flags = KM_F_RESULT;
            Job plain = res;
            plain.a[0] = 0x1234;
            plain.flags = 0;
            Job nop = nop_job(1, 3);
            nop.flags = KM_F_RESULT;
            cfg.jobs = {res, plain, nop};
            Sim s(cfg);
            s.run();
            common(s, o.check);
            o.check.expect(all_done(s), "every command and the EXIT acknowledged");
            const auto& j = s.jobs;
            o.check.expect(
                j[0].result == 0x5EED, fmt("result %x, not the leader's", j[0].result)
            );
            o.check.expect(
                j[1].result == 0x5EED, "an unflagged ack left the result word"
            );
            o.check.expect(j[2].result == 0, "a result never given is acked as 0");
            finish(s, o);
        }

        // ---- the timed buffer wrappers against an item past the end ----

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
            c.expect(
                s.word(core, KM_CTL_STOP) != 0, std::string(w) + " set the stop flag"
            );
            c.expect(s.all_returned, std::string(w) + ": every thread returned");
        }

        // Each of the four wrappers is asked for an item that starts one page before
        // the end: it gives up with the bad-item code and sets the stop flag.
        void wrappers_reject_straddling_items(Outcome& o) {
            for (uint32_t a1 : {1u, 0u}) {  // reader (after 5 input pages), writer
                                            // (after 1)
                Config cfg = fault_config(2);
                Job bad;
                bad.handler = kHandlerBadItem;
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
                    std::string(w) + ": the bad command is not acked"
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

        // An EXCL command on a group disjoint from the one in flight is submitted at
        // once; one that overlaps either waits until both are acknowledged.
        void excl_waits_for_overlapping_group(Outcome& o) {
            Config cfg = fault_config(4);
            cfg.jobs = {map_job(0, 2, 24), map_job(2, 2, 4), map_job(1, 2, 4)};
            cfg.jobs[1].flags = cfg.jobs[2].flags = KM_F_EXCL;
            Sim s(cfg);
            s.run();
            common(s, o.check);
            o.check.expect(all_done(s), "every command and the EXIT acknowledged");
            const auto& j = s.jobs;
            o.check.expect(
                j[1].submitted < j[0].acked,
                "the disjoint EXCL command ran beside the first"
            );
            o.check.expect(
                j[2].submitted >= j[0].acked && j[2].submitted >= j[1].acked,
                "the overlapping EXCL command waited for both groups"
            );
            finish(s, o);
        }

    }  // namespace

    const std::vector<Case>& pinned() {
        static const std::vector<Case> cases = {
            {"smoke", smoke},
            {"exit_partial_group", exit_partial_group},
            {"no_output_waits_for_compute", no_output_waits_for_compute},
            {"wall_clock_across_a_carry", wall_clock_across_a_carry},
            {"result_word_from_the_leader", result_word_from_the_leader},
            {"wrappers_reject_straddling_items", wrappers_reject_straddling_items},
            {"excl_waits_for_overlapping_group", excl_waits_for_overlapping_group},
        };
        return cases;
    }

}  // namespace scen
