// Attend, reader: the group's query tiles and constants, this core's cached keys (the
// root appends the token's key to the last tile), then the merge's arrivals.
#include "attend.hpp"

// The root: the masks of the tokens' tiles (row i keeps the columns up to its own
// position); the first's again as the second when the tokens are in one tile.
static bool read_masks(
    km::Reader& reader, const km_command_t& command, const Attend& at
) {
    const uint32_t first = DC_MASK_ZERO + at.row;
    const uint32_t second = at.span != 0 ? first - kTileRows : first;
    return (
        decode::read_item(reader, command, DC_CB_MASK, command.a8, first, 1) &&
        decode::read_item(reader, command, DC_CB_MASK, command.a8, second, 1)
    );
}

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const Attend at = attend_of(reader, block);
    const bool root = at.rank == 0;
    if (!decode::read_item(
        reader,
        command,
        DC_CB_QUERY,
        command.a0,
        query_page(at),
        DC_HEAD_GROUP * DC_HEAD_TILES
    ) ||
        !decode::read_item(
            reader, command, DC_CB_ATTEND_CONSTANT, command.a8, DC_ATTEND_ONES, 3
        ) ||
        (root && !read_masks(reader, command, at)) ||
        !attend::read_cache(
            reader, command, at, DC_CB_KEYS, command.a1, km::kStReader
        ) ||
        (
            root &&
            !attend::append(
                reader, command, at, DC_CB_KEYS, command.a1, false, km::kStReader
            )
        )) {
        return 0;
    }
    attend::pad(at, DC_CB_KEYS);
    if (!merges(at)) {
        return 1;
    }
    const km::Arrival arrival = attend_arrival(reader.arg(block.plan + kSpArrive));
    const uint32_t form = split_form(command);
    if (km::reduce_slots(at.group, form) > km::arrival_slots(arrival)) {
        reader.gave_up(km::kStReader, km::kGaveBadItem, km::kWherePort, 0, 0);
        return 0;
    }
    for (uint32_t port = 0; port < KM_PORTS / 2; port++) {
        const uint32_t wanted = km::reduce_wants(at.group, form, port);
        if (wanted != 0 && !km::arrive(reader, command, arrival, port, wanted)) {
            return 0;
        }
    }
    return 1;
}
