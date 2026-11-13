// Overlay turn, its reader part: the core's data turned by 1 to 16 words, each word
// rotated and xored with the round's index.
#include "komorebi/device/km_worker.hpp"
#include "../ops.h"

uint32_t km_overlay(km::Reader& reader, const DataArgument& argument) {
    return op_turn(km::l1(argument.data), argument.index);
}
