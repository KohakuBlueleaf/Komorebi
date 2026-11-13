// Overlay mix, its reader part: each word of the core's data times 33, plus the
// round's index and its position.
#include "komorebi/device/km_worker.hpp"
#include "../ops.h"

// --8<-- [start:entry]
uint32_t km_overlay(km::Reader& reader, const DataArgument& argument) {
    return op_mix(km::l1(argument.data), argument.index);
}

// --8<-- [end:entry]
