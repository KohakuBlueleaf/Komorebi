// The widen program's reader: WIDEN reads the core's partition of the input tiles into
// its own input buffer (another index than sum's and scale's).
#include <stdint.h>

#include "../read_tiles.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) == kHandlerWiden) {
        read_tiles<kCbWideIn>(reader, command);
    }
}
