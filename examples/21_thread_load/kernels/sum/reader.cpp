// The sum program's reader, as in example 16: SUM reads the core's partition of the
// input tiles.
#include <stdint.h>

#include "../read_tiles.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) == kHandlerSum) {
        read_tiles(reader, command);
    }
}
