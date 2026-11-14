// The map/collect example's handlers, and one that never returns (core reset ladder).
#pragma once

#include "../../../examples/01_mapcollect/kernels/mapcollect.h"

// SPIN: rank a0 spinning_rank loops for ever, never looking at the stop flag or the
// abort word: a1 spin_where 1 in the reader, 2 in compute's math thread, 3 in all
// three compute threads driving the units (unpack, vector unit, pack; no buffer wait).
constexpr uint32_t kHandlerSpin = 9;
constexpr uint32_t kSpinReader = 1;
constexpr uint32_t kSpinMath = 2;
constexpr uint32_t kSpinUnits = 3;
