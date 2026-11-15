// second's overlay 1 (3 in the run): one word of arithmetic the host repeats.
#include "komorebi/device/km_worker.hpp"
#include "../shift.h"

uint32_t km_overlay(km::Reader& reader, const TagArgument& argument) {
    return argument.value + 1000u;
}
