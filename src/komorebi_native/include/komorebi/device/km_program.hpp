// A program in the runtime's form: the runtime owns kernel_main and the command loop,
// and calls the program's km_program of this kernel for each command that names it.
#pragma once

#include <stdint.h>

#include "km_command.hpp"
#include "km_worker.hpp"

// A source defines km_program(km::Reader&, ...), (km::Compute&, ...) or (km::Writer&,
// ...) with a const km_command_t&; the writer's may return bool (false: no completion).
