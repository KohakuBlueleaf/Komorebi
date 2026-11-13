// Overlay shift, its compute part (one build per compute core): the reader's tile
// plus the round's scalar on the vector unit.
#include "komorebi/device/km_worker.hpp"
#include "../tile_ops.h"

uint32_t km_overlay(km::Compute& compute, const TileArgument& argument) {
    return compute_tile(compute, argument.scalar, false);
}
