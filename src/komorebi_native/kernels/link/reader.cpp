// A link core's reader (the session loads it on the link cores of an island run built
// with page links): the runtime's loop, its services between commands, no program.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#if !defined(KM_LINK_SERVICES)
#error "a link core's kernel is built with its services (KM_LINK_SERVICES)"
#endif

void kernel_main() {
    km::Reader reader;
    reader.run([](const km_command_t&) {});
}
