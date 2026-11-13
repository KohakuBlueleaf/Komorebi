// Overlay shift, its reader part: fills a tile from the core's data for compute and
// folds the tile compute made back into the data.
#include "komorebi/device/km_worker.hpp"
#include "../ops.h"

uint32_t km_overlay(km::Reader& reader, const DataArgument& argument) {
    return op_tile(reader, km::l1(argument.data));
}
