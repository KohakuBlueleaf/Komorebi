// The spin program's writer: nothing (SPIN is the reader's).
#include "komorebi/device/core/km_program.hpp"

bool km_program(km::Writer& writer, const km_command_t& command) {
    return true;
}
