// Cross, writer (the card's leader): waits for its reader's crossing (port 0), then
// returns, and the split's writer marks the next record's cores.
#include "api/dataflow/dataflow_api.h"
#include "cross.hpp"

uint32_t km_overlay(km::Writer& writer, const SplitBlock& block) {
    return km::taken_off(writer, *block.command, cross_done(writer), 1) ? 1 : 0;
}
