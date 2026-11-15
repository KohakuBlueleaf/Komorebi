// The reader part of scale and gate: the row's sums of squares, a tile a group (a7),
// and the constants (a8).
#include "../ops.hpp"

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    return (
        (
            decode::read_item(reader, command, DC_CB_PARTS, command.a7, 0, DC_PARTS) &&
            decode::read_item(reader, command, DC_CB_CONSTANT, command.a8, 0, DC_CONSTANTS)
        )
            ? 1
            : 0
    );
}
