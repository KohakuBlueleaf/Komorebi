// The page-of-tiles probe's handler, command words, buffers and profile words, shared
// by the host and the kernels.
#pragma once

#include <stdint.h>

// READ: a0 the tensor, a1 its page bytes (k tiles), a2 pages a core, a3 pages a
// block, a4 the buffer, a5 repetitions, a6 the profile tensor, a7 tiles a block.
constexpr uint32_t kHandlerRead = 1;

// One buffer of one-tile pages a tile size: bfp8_b's 1,088 B, bfp4_b's 576 B.
constexpr uint32_t kBufferBfp8 = 0;
constexpr uint32_t kBufferBfp4 = 1;
constexpr uint32_t kBufferProfile = 2;
constexpr uint32_t kProfileBytes = 64;

// Profile words of the last READ: mover A's start and end (wall-clock cycles), the
// tensor pages it read.
constexpr uint32_t kProfileStart = 0;
constexpr uint32_t kProfileEnd = 1;
constexpr uint32_t kProfilePages = 2;
constexpr uint32_t kProfileWords = 16;
