// The four operations on a core's data, as the reader runs them: in an overlay's
// reader part, or compiled into the reader. The host replays the same steps.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_worker.hpp"
#include "leader_overlay.h"

// The core's data at the start of a command: word k of core `core` from `first` on.
inline uint32_t data_start(uint32_t core, uint32_t first, uint32_t k) {
    return first * 2654435761u + core * 40503u + k * 97u;
}

// mix: each word times 33, plus the round's index and its position.
inline uint32_t op_mix(volatile tt_l1_ptr uint32_t* data, uint32_t index) {
    for (uint32_t k = 0; k < kDataWords; k++) {
        data[k] = data[k] * 33u + index + k;
    }
    return 1;
}

// turn: word k takes word k + 1 + (index % 16), rotated left by 5, xor the index.
inline uint32_t op_turn(volatile tt_l1_ptr uint32_t* data, uint32_t index) {
    uint32_t old[kDataWords];
    for (uint32_t k = 0; k < kDataWords; k++) {
        old[k] = data[k];
    }
    const uint32_t by = 1 + (index & 15u);
    for (uint32_t k = 0; k < kDataWords; k++) {
        const uint32_t word = old[(k + by) & (kDataWords - 1)];
        data[k] = (word << 5 | word >> 27) ^ index;
    }
    return 1;
}

// The tile's value j: 3 bits of data word j % 32 (from bit 4 * ((j / 32) % 8)), as
// bfloat16 bits of 0 to 7. Two values a 32-bit store.
inline bool tile_fill(km::Reader& reader, const uint32_t* words) {
    constexpr uint16_t kSmall[8] = {
        0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0
    };
    if (!reader.reserve(kCbIn, 1)) {
        return false;
    }
    volatile tt_l1_ptr uint32_t* tile = km::l1(get_write_ptr(kCbIn));
    for (uint32_t row = 0; row < kTileValues / kDataWords; row++) {
        const uint32_t shift = 4 * (row & 7u);
        for (uint32_t k = 0; k < kDataWords; k += 2) {
            const uint32_t low = kSmall[(words[k] >> shift) & 7u];
            const uint32_t high = kSmall[(words[k + 1] >> shift) & 7u];
            tile[(row * kDataWords + k) / 2] = low | high << 16;
        }
    }
    cb_push_back(kCbIn, 1);
    return true;
}

// Folds compute's tile into the data: word j % 32 = word * 3 + bits of value j, for
// j from 0 to 1023 in order. False: the tile did not come (recorded).
inline bool tile_fold(km::Reader& reader, uint32_t* words) {
    const uint32_t code = reader.poll(
        [] {
            return cb_pages_available_at_front(kCbOut, 1);
        },
        reader.args().wait_limit
    );
    if (code != km::kGaveNone) {
        reader.gave_up(km::kStReader, code, km::kWhereData, 0, 0);
        return false;
    }
    cb_wait_front(kCbOut, 1);
    volatile tt_l1_ptr uint32_t* tile = km::l1(get_read_ptr(kCbOut));
    for (uint32_t pair = 0; pair < kTileValues / 2; pair++) {
        const uint32_t bits = tile[pair];
        const uint32_t k = (2 * pair) & (kDataWords - 1);
        words[k] = words[k] * 3u + (bits & 0xFFFFu);
        words[k + 1] = words[k + 1] * 3u + (bits >> 16);
    }
    cb_pop_front(kCbOut, 1);
    return true;
}

// scale and shift on the reader: the tile out, compute's tile back. 0: gave up.
inline uint32_t op_tile(km::Reader& reader, volatile tt_l1_ptr uint32_t* data) {
    uint32_t words[kDataWords];
    for (uint32_t k = 0; k < kDataWords; k++) {
        words[k] = data[k];
    }
    if (!tile_fill(reader, words) || !tile_fold(reader, words)) {
        return 0;
    }
    for (uint32_t k = 0; k < kDataWords; k++) {
        data[k] = words[k];
    }
    return 1;
}
