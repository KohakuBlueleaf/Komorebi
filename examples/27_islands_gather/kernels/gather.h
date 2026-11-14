// All-gather across islands: the handler its kernels share.
#pragma once

#include <stdint.h>

// GATHER: a0 the tensor's reference word (a copy), a1 first page, a2 pages.
constexpr uint32_t kHandlerGather = 1;
