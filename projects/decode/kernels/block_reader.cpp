// The reader part of add and gate: the block of the other operand (a7) that this root
// adds to or multiplies with its sum.
#include "ops.hpp"

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    return decode::read_block(reader, block, block.command->a7) ? 1 : 0;
}
