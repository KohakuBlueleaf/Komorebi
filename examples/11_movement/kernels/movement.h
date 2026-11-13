// Movement example: the handlers, buffer pairs and command arguments its kernels share.
#pragma once

#include <stdint.h>

// READ: a0 input_address, a2 tiles (per core), a3 tiles_per_block, a4 pattern word,
// a5 repetitions: a core reads its tiles a5 times; compute drops them.
constexpr uint32_t kHandlerRead = 1;
// COPY: as READ with a1 output_address, once: compute copies each block from the input
// pair into the output pair, the movers write it to the same pages of the output.
constexpr uint32_t kHandlerCopy = 2;

// The two pairs, each mover's half of the input and of the output (bfloat16 pages);
// the kernels name them as km::BufferPair kInput and kOutput.
constexpr uint32_t kInputMoverA = 0;
constexpr uint32_t kInputMoverB = 1;
constexpr uint32_t kOutputMoverA = 16;
constexpr uint32_t kOutputMoverB = 17;
