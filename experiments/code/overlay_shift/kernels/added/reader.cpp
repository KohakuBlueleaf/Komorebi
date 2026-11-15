// added to the live run (run.addon): one word of arithmetic the host repeats.
#include "komorebi/device/km_worker.hpp"
#include "../shift.h"

uint32_t km_overlay(km::Reader& reader, const TagArgument& argument) {
    return argument.value ^ 0x00FF00FFu;
}
