// A link core's writer: the runtime's loop and its services between commands; every
// command a link core takes that is not one of its services completes as a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#if !defined(KM_LINK_SERVICES)
#error "a link core's kernel is built with its services (KM_LINK_SERVICES)"
#endif

void kernel_main() {
    km::Writer writer;
    writer.run([](const km_command_t&) {
        return true;
    });
}
