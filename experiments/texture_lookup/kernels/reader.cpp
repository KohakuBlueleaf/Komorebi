// The texture lookup probe's reader (NoC 0): LOOP, MISS and COLOUR when the command
// names this CPU, the colour planes into kCbColour for the vector unit, and REPLAY.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "fetch.hpp"
#include "lookup.h"
#include "table.hpp"

void kernel_main() {
    km::Reader reader;
    constexpr auto pool_args = TensorAccessorArgs<0>();
    const texture::Tensors at = {
        reader.arg(kArgPages),
        reader.arg(kArgSlots),
        reader.arg(kArgStreams),
        reader.arg(kArgPlanes),
        reader.arg(kArgOut),
        reader.arg(kArgResults),
        reader.arg(kArgReplay),
    };
    const auto pool = TensorAccessor(pool_args, reader.arg(kArgPool));
    const uint32_t pool_pages = reader.arg(kArgPoolPages);
    reader.run([&](const km_command_t& command) {
        switch (km::handler(command)) {
            case kHandlerLoop:
                if (command.a0 == kCpuReader) {
                    texture::loop(at, command);
                }
                break;
            case kHandlerMiss:
                if (command.a0 == kCpuReader) {
                    texture::miss(at, pool, pool_pages, reader.me(), command);
                }
                break;
            case kHandlerColour:
                if (command.a0 == kColourInteger) {
                    if (command.a1 == kCpuReader) {
                        texture::colour(at, command);
                    }
                } else if (reader.reserve(kCbColour, 2)) {
                    // The two planes, as two pages, for the unpacker.
                    const uint32_t* from = texture::words(at.planes);
                    uint32_t* to = texture::words(get_write_ptr(kCbColour));
                    for (uint32_t word = 0; word < 2 * kPixels; word++) {
                        to[word] = from[word];
                    }
                    texture::fence();
                    cb_push_back(kCbColour, 2);
                }
                break;
            case kHandlerReplay:
                texture::replay(at, pool, pool_pages, command);
                break;
            default:
                break;
        }
    });
}
