// The page-of-tiles probe's handler, command words, buffers and profile words, shared
// by the host and the kernels.
#pragma once

#include <stdint.h>

// READ: a0 tensor, a1 page bytes, a2 pages a core, a3 pages a block, a4 format,
// a5 repetitions, a6 profile tensor, a7 tiles a block, a8 pattern (block_on_b).
constexpr uint32_t kHandlerRead = 1;

// A buffer of one-tile pages a format and a mover: bfp8_b's 1,088 B, bfp4_b's 576 B.
constexpr uint32_t kBufferBfp8A = 0;
constexpr uint32_t kBufferBfp4A = 1;
constexpr uint32_t kBufferProfile = 2;
constexpr uint32_t kBufferBfp8B = 3;
constexpr uint32_t kBufferBfp4B = 4;
constexpr uint32_t kProfileBytes = 64;

// Profile words of the last READ, a page a mover (mover B's 64 B after mover A's):
// its start and end (wall-clock cycles), the tensor pages it read.
constexpr uint32_t kProfileStart = 0;
constexpr uint32_t kProfileEnd = 1;
constexpr uint32_t kProfilePages = 2;
constexpr uint32_t kProfileWords = 16;

// Whether block `block` is mover B's: bit (block mod length) of `pattern`, the length
// in bits 8 and up.
inline bool block_on_b(uint32_t pattern, uint32_t block) {
    const uint32_t length = pattern >> 8 ? pattern >> 8 : 1;
    return (pattern >> (block % length)) & 1;
}

// The buffer of a format and a mover.
inline uint32_t block_buffer(uint32_t format, bool on_b) {
    return (
        format == 0
            ? (on_b ? kBufferBfp8B : kBufferBfp8A)
            : (on_b ? kBufferBfp4B : kBufferBfp4A)
    );
}
