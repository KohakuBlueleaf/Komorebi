// The pinned scenarios alone: a small program for the tests. Usage: pinned <name> |
// list
#include <cstring>

#include "checks.hpp"

int main(int argc, char** argv) {
    // and the dynamic crossing's end, gap, overflow and idle cases
    std::vector<scen::Case> cases = scen::pinned();
    for (const scen::Case& c : scen::dyn_cases()) {
        for (
            const char* name :
            {"dyn_smoke", "dyn_gap", "dyn_overflow", "dyn_idle_flapping"}
        ) {
            if (std::strcmp(c.name, name) == 0) {
                cases.push_back(c);
            }
        }
    }
    return scen::run_cases(cases, argc, argv);
}
