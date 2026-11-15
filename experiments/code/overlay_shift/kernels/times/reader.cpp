// first's overlay 1: one word of arithmetic the host repeats.
#include "komorebi/device/km_worker.hpp"
#include "../shift.h"

uint32_t km_overlay(km::Reader& reader, const TagArgument& argument) {
    return argument.value * 3u + 1u;
}
