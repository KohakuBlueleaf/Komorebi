// Pages across islands: the program's handler and the tensors its writer knows.
#pragma once

#include <stdint.h>

// WHERE: the writer's result is km::island_of(tensor a0, page a1) on this core.
constexpr uint32_t kHandlerWhere = 1;
