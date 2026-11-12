// Shared by the scenario programs: segment numbers, the checks every scenario ends
// with, job and grid helpers, and the runner that prints one JSON line per scenario.
#pragma once

#include <stdint.h>

#include <string>
#include <vector>

#include "harness.hpp"

// A case's compute kernels (item_kernels.cpp), one per thread.
void kmsim_item_in_unpack_main();
void kmsim_item_in_math_main();
void kmsim_item_in_pack_main();
void kmsim_item_out_unpack_main();
void kmsim_item_out_math_main();
void kmsim_item_out_pack_main();

namespace scen {

    using namespace kmsim;

    constexpr uint32_t kSegMap = 1;  // the example's segments
    constexpr uint32_t kSegStall = 2;
    constexpr uint32_t kSegBadItem = 5;  // a0: the member; a1 = 0: writer, 1: reader
    constexpr uint32_t kSegResult = 6;   // each member's result: a0 + its index
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

    struct Outcome {
        Check check;
        uint64_t cycles = 0;
        double host_ms = 0;
        std::vector<std::string> stuck, violations;
        size_t jobs = 0, done = 0;
        uint32_t mailbox_peak = 0;
    };

    struct Case {
        const char* name;
        void (*fn)(Outcome&);
    };

    std::string fmt(const char* f, unsigned a = 0, unsigned b = 0, unsigned c = 0);
    Job map_job(uint32_t first, uint32_t count, uint32_t tiles);
    Job nop_job(uint32_t first, uint32_t count);
    Config fault_config(uint32_t cores);  // short bounds: a stranded kernel shows fast
    bool all_done(const Sim& s);
    void common(const Sim& s, Check& c);  // what every scenario ends with
    void silent_from(const Sim& s, Check& c, uint32_t core, uint32_t seq);
    void finish(const Sim& s, Outcome& o);
    void items_none();  // a kernel that does nothing

    // The scenarios that pin known behaviour (pinned.cpp), and the runner of a table.
    const std::vector<Case>& pinned();
    const std::vector<Case>& dyn_cases();  // dyn_scenarios.cpp
    int run_cases(const std::vector<Case>& cases, int argc, char** argv);

}  // namespace scen
