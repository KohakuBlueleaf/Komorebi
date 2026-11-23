// The upscaler's mover step: each band row copied into the three versions the products
// read, as it is and shifted by one pixel either way (README.md, "The shift").
#pragma once

#include "komorebi/device/core/km_core.hpp"

namespace upscale {

    constexpr uint32_t kTile = 2048;    // a bfloat16 tile's bytes
    constexpr uint32_t kFace = 512;     // a 16 x 16 face's bytes
    constexpr uint32_t kFaceRow = 32;   // a face row: 16 channels of one pixel
    constexpr uint32_t kRowTiles = 16;  // an image row: 512 pixels
    constexpr uint32_t kRowBytes = kRowTiles * kTile;
    constexpr uint32_t kImageRows = 512;

    // A local copy of `bytes` from L1 `from` to L1 `to` by this core's NoC.
    inline void copy(uint32_t to, uint32_t from, uint32_t bytes) {
        noc_async_read(get_noc_addr(from), to, bytes);
    }

    // A row shifted a pixel: one copy 32 B (a face row) later or earlier, then the four
    // face rows a tile it took from the wrong face. `zero`: 512 B of zeros.
    inline void shifted(uint32_t to, uint32_t row, uint32_t zero, bool later) {
        copy(to, later ? row + kFaceRow : row - kFaceRow, kRowBytes);
        noc_async_read_barrier();  // the fixes land over the copy
        for (uint32_t t = 0; t < kRowTiles; t++) {
            const uint32_t d = to + t * kTile;
            const uint32_t s = row + t * kTile;
            const bool edge = later ? t + 1 == kRowTiles : t == 0;
            if (later) {  // row 15 of each face: the pixel 16 or 32 on
                const uint32_t last = 15 * kFaceRow;
                copy(d + last, s + 2 * kFace, kFaceRow);
                copy(d + kFace + last, s + 3 * kFace, kFaceRow);
                copy(d + 2 * kFace + last, edge ? zero : s + kTile, kFaceRow);
                copy(d + 3 * kFace + last, edge ? zero : s + kTile + kFace, kFaceRow);
            } else {  // row 0 of each face: the pixel 1 or 17 before
                const uint32_t last = 15 * kFaceRow;
                copy(d, edge ? zero : s - kTile + 2 * kFace + last, kFaceRow);
                copy(d + kFace, edge ? zero : s - kTile + 3 * kFace + last, kFaceRow);
                copy(d + 2 * kFace, s + last, kFaceRow);
                copy(d + 3 * kFace, s + kFace + last, kFaceRow);
            }
        }
    }

    // Band row k (image row first + k - 1) into `taps` as versions 3 k + dx, zeros
    // outside the image; then a page of `ready` pushed: compute may read `taps`.
    inline void shift(
        uint32_t band,
        uint32_t taps,
        uint32_t ones,
        uint32_t ready,
        uint32_t first,
        uint32_t rows
    ) {
        const uint32_t to = get_write_ptr(taps);
        const uint32_t zero = ones + kFace;  // face 1 of the ones tile: zeros
        for (uint32_t k = 0; k < rows; k++) {
            const uint32_t row = band + k * kRowBytes;
            const uint32_t at = to + 3 * k * kRowBytes;
            if (first + k == 0 || first + k > kImageRows) {  // the zero padding
                for (uint32_t part = 0; part < 3 * kRowBytes; part += kFace) {
                    copy(at + part, zero, kFace);
                }
                continue;
            }
            copy(at + kRowBytes, row, kRowBytes);
            shifted(at, row, zero, false);
            shifted(at + 2 * kRowBytes, row, zero, true);
        }
        noc_async_read_barrier();
        cb_reserve_back(ready, 1);
        cb_push_back(ready, 1);
    }

}  // namespace upscale
