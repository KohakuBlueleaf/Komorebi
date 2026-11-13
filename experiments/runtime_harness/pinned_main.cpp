// The pinned scenarios alone: a small program for the tests. Usage: pinned <name> |
// list
#include <cstring>

#include "checks.hpp"

int main(int argc, char** argv) {
    // and the routed messages' end, gap, overflow and idle cases
    std::vector<scen::Case> cases = scen::pinned();
    for (const scen::Case& c : scen::messages_cases()) {
        for (
            const char* name :
            {
                "messages_smoke",
                "messages_gap",
                "messages_overflow",
                "messages_idle_flapping"
            }
        ) {
            if (std::strcmp(c.name, name) == 0) {
                cases.push_back(c);
            }
        }
    }
    return scen::run_cases(cases, argc, argv);
}
