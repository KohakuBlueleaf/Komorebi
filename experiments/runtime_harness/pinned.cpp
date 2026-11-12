// The scenarios that pin known runtime behaviour: run by the scenario program and, on
// their own, by tests/core/test_runtime_harness.py.
#include <stdint.h>

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
            o.check.expect(all_done(s), "every record and the EXIT acknowledged");
            finish(s, o);
        }

        // EXIT with team (0, 2) on four cores; the increments of cores 2 and 3 are
        // late.
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
            c.expect(
                s.word(core, KM_CTL_STOP) != 0, std::string(w) + " set the stop word"
            );
            c.expect(s.all_returned, std::string(w) + ": every thread returned");
        }

        // Each of the four wrappers is asked for an item that starts one page before
        // the end: it gives up with the bad-item code and sets the stop word.
        void wrappers_refuse_straddling_items(Outcome& o) {
            for (uint32_t a1 : {1u, 0u}) {  // reader (after 5 input pages), writer
                                            // (after 1)
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

        // An EXCL record on a team disjoint from the one in flight is issued at once;
        // one that overlaps either waits until both are acknowledged.
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
                j[1].issued < j[0].acked,
                "the disjoint EXCL record ran beside the first"
            );
            o.check.expect(
                j[2].issued >= j[0].acked && j[2].issued >= j[1].acked,
                "the overlapping EXCL record waited for both teams"
            );
            finish(s, o);
        }

    }  // namespace

    const std::vector<Case>& pinned() {
        static const std::vector<Case> cases = {
            {"smoke", smoke},
            {"exit_partial_team", exit_partial_team},
            {"wrappers_refuse_straddling_items", wrappers_refuse_straddling_items},
            {"excl_waits_for_overlapping_team", excl_waits_for_overlapping_team},
        };
        return cases;
    }

}  // namespace scen
