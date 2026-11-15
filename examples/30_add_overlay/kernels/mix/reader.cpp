// The overlay added first, its reader part: a multiply and shift loop on a word; the
// host computes the same loop.
#include "komorebi/device/km_worker.hpp"
#include "../add.h"

uint32_t km_overlay(km::Reader& reader, const MixArgument& argument) {
    uint32_t value = argument.value;
    for (uint32_t step = 0; step < argument.steps; step++) {
        value = value * 2654435761u + step;
        value ^= value >> 13;
    }
    return value;
}
