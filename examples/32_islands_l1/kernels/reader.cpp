// An L1 tensor moved across islands, reader: each rank's tile of the tensor held in
// its L1 goes to another rank's L1 by move_l1 (km::write_l1), on any island.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_far.hpp"
#include "komorebi/device/km_port.hpp"
#include "l1.h"

namespace {

    // `bytes` at L1 `from` of rank from_rank into L1 `to` of rank to_rank: the sender
    // writes them and counts 1 at `port` there, the receiver waits for that count.
    bool move_l1(
        km::Reader& reader,
        const km_command_t& command,
        uint32_t from_rank,
        uint32_t to_rank,
        uint32_t from,
        uint32_t to,
        uint32_t bytes,
        uint32_t port
    ) {
        const uint32_t me = reader.me();
        if (
            me == from_rank &&
            !km::write_l1(
                reader,
                command,
                to_rank,
                to,
                reinterpret_cast<const void*>(from),
                bytes,
                static_cast<int>(port)
            )
        ) {
            return false;
        }
        return me != to_rank || km::port_wait(reader, command, port, 1);
    }

}  // namespace

void kernel_main() {
    using Pages = km::Tensor<0>;
    km::Reader reader;
    const uint32_t held = get_write_ptr(kCbHeld);  // never pushed: plain L1
    const uint32_t landing = get_write_ptr(kCbLanding);
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMove) {
            return;
        }
        const km::Group group = km::group_of(command, reader.me());
        const uint32_t shift = command.a2 % group.size, port = command.a3;
        const uint32_t me = reader.me();
        const uint32_t to = group.base + (group.rank + shift) % group.size;
        const uint32_t from = (
            group.base + (group.rank + group.size - shift) % group.size
        );
        // the L1 tensor: this rank's tile, read from its island's copy of the input
        if (!km::read_pages(reader, command, Pages(command.a0), me, 1, held)) {
            return;
        }
        noc_async_read_barrier();
        if (
            !move_l1(reader, command, me, to, held, landing, kTileBytes, port) ||
            !move_l1(reader, command, from, me, held, landing, kTileBytes, port)
        ) {
            return;
        }
        if (km::write_pages(reader, command, Pages(command.a1), me, 1, landing)) {
            noc_async_writes_flushed();
        }
    });
}
