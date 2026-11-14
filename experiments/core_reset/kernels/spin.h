// The map/collect example's handlers, and one that never returns (core reset ladder).
#pragma once

#include "../../../examples/01_mapcollect/kernels/mapcollect.h"

// SPIN: rank a0 spinning_rank loops for ever, ignoring the stop and abort words: a1
// spin_where 1 in the reader, 2 in compute's math thread, 3 in all compute threads.
constexpr uint32_t kHandlerSpin = 9;
constexpr uint32_t kSpinReader = 1;
constexpr uint32_t kSpinMath = 2;
constexpr uint32_t kSpinUnits = 3;
