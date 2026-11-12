// The pinned scenarios alone: a small program for the tests. Usage: pinned <name> |
// list
#include "checks.hpp"

int main(int argc, char** argv) {
    return scen::run_cases(scen::pinned(), argc, argv);
}
