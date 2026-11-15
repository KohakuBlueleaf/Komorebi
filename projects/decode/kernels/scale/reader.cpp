// The reader part of scale and gate: the row's sums of squares, a tile a group (a7),
// and the constants (a8); on qkv's q and k blocks the head's norm and rotation's too.
#include "../ops.hpp"

// The table's rows from position a9's, the live rows' bands of 4, into scratch
// (DC_CB_ROPE_ROWS), then row i into row i of DC_CB_ROPE's 8 tiles, 32 values each.
static bool read_rotary(km::Reader& reader, const km_command_t& command) {
    const km::Tensor<0> table(command.a8);
    const uint32_t first = command.a9 & 0xFFFFu;
    const uint32_t rows = (decode::live_rows(command) + 3u) & ~3u;
    constexpr uint32_t kRowsAPage = kTileBytes / DC_ROPE_ROW;
    constexpr uint32_t kTiles = 2 * DC_HEAD_TILES;
    if (
        !reader.reserve(DC_CB_ROPE_ROWS, kTiles) || !reader.reserve(DC_CB_ROPE, kTiles)
    ) {
        return false;
    }
    const uint32_t from = get_write_ptr(DC_CB_ROPE_ROWS);
    const uint32_t to = get_write_ptr(DC_CB_ROPE);
    uint32_t asked = 0;
    for (uint32_t row = first; row < first + rows;) {
        const uint32_t in_page = row % kRowsAPage;
        const uint32_t count = (
            first + rows - row < kRowsAPage - in_page
                ? first + rows - row
                : kRowsAPage - in_page
        );
        if (!km::ask_bytes(
            reader,
            command,
            table,
            DC_QK_ROTARY + row / kRowsAPage,
            in_page * DC_ROPE_ROW,
            count * DC_ROPE_ROW,
            from + (row - first) * DC_ROPE_ROW,
            asked
        )) {
            return false;
        }
        row += count;
    }
    if (!km::bytes_wait(reader, command, asked)) {
        return false;
    }
    invalidate_l1_cache();  // the core reads what the NoC wrote
    // row i's 16 values of a face half: face (i / 16) x 2 + half, its row i % 16
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t at = ((row >> 4) * 2) * 512 + (row & 15) * 32;
        for (uint32_t piece = 0; piece < 2 * kTiles; piece++) {
            volatile tt_l1_ptr uint32_t* out = (
                km::l1(to + (piece >> 1) * kTileBytes + (piece & 1) * 512 + at)
            );
            const volatile tt_l1_ptr uint32_t* in = (
                km::l1(from + row * DC_ROPE_ROW + piece * 32)
            );
            for (uint32_t word = 0; word < 8; word++) {
                out[word] = in[word];
            }
        }
    }
    cb_push_back(DC_CB_ROPE, kTiles);
    return true;
}

// The head's mean tile and the layer's q (k) norm weight's tiles into DC_CB_HEAD.
static bool read_head(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    const km::Tensor<0> constants(command.a8);
    const uint32_t layer = command.a9 >> 16;
    const uint32_t key = decode::qk_key(reader, block) ? 1u : 0u;
    if (!reader.reserve(DC_CB_HEAD, DC_HEAD_ITEM)) {
        return false;
    }
    const uint32_t to = get_write_ptr(DC_CB_HEAD);
    const uint32_t weight = DC_QK_WEIGHTS + (2u * layer + key) * DC_HEAD_TILES;
    if (
        !km::read_pages(reader, command, constants, DC_QK_HEAD_MEAN, 1, to) ||
        !km::read_pages(
            reader, command, constants, weight, DC_HEAD_TILES, to + kTileBytes
        ) ||
        !flushed(reader, km::kStReader, km::kFlushReads)
    ) {
        return false;
    }
    cb_push_back(DC_CB_HEAD, DC_HEAD_ITEM);
    return true;
}

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    if (
        !decode::read_item(reader, command, DC_CB_PARTS, command.a7, 0, DC_PARTS) ||
        !decode::read_item(reader, command, DC_CB_CONSTANT, command.a8, 0, DC_CONSTANTS)
    ) {
        return 0;
    }
    if (decode::qk_block(reader, block)) {
        return read_head(reader, block) && read_rotary(reader, command) ? 1 : 0;
    }
    return 1;
}
