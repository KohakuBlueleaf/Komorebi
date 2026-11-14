// Pages across islands: the program's handler.
#pragma once

#include <stdint.h>

// WHERE: the writer's result is km::island_of(tensor a0, page a1) on this core (a0:
// the tensor's reference word).
constexpr uint32_t kHandlerWhere = 1;
