// Overlay multiply, its writer part.
#include "komorebi/device/km_worker.hpp"
#include "../handlers.h"

uint32_t km_overlay(km::Writer& writer, const Pair& pair) {
    return pair.left * pair.right;
}
