// Add, reader: the residual's block (a7) that this root adds to its sum, and the
// constants (a8) for the squares of the rows of Y.
#include "../ops.hpp"

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    return (
        (
            decode::read_block(reader, block, command.a7) &&
            decode::read_item(reader, command, DC_CB_CONSTANT, command.a8, 0, DC_CONSTANTS)
        )
            ? 1
            : 0
    );
}
