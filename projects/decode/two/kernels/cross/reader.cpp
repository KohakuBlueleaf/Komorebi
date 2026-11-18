// Cross, reader (the card's leader): its half to the other card as a put line (squares
// a second), the other half waited for, then 1 on port 0 for the writer to mark.
#include "api/dataflow/dataflow_api.h"
#include "cross.hpp"

// The crossing itself (a2 pages; none: local, the mark alone). False: gave up.
static bool cross(km::Reader& reader, const km_command_t& command) {
    if (command.a2 == 0) {
        return true;
    }
    const km::Islands view = km::islands_at(reader.args().control);
    const uint32_t here = view.here(), far = 1 - here;
    // the far leader counts at its port toward this card (km::all_gather's ports)
    const uint32_t port_out = here == 0 ? KM_PORT_LEFT : KM_PORT_RIGHT;
    const uint32_t port_in = here == 0 ? KM_PORT_RIGHT : KM_PORT_LEFT;
    // a9's rank is the far leader's place among its island's ranks; km::keeper takes
    // flat ranks
    const uint32_t keeper = (
        km::keeper(view.entry_first(far) + (command.a9 & 0xFFu), port_out)
    );
    const uint32_t pages = command.a2 + command.a8;
    // serial (a9's DC2_CROSS_SERIAL): island 1 posts once island 0's half is here
    const bool after = here == 1 && (command.a9 & DC2_CROSS_SERIAL) != 0;
    km::Gather posted = {};
    return (
        (!after || km::gathered(reader, command, port_in, pages)) &&
        km::gather_pages(
            reader, command, command.a0, command.a1, command.a2, far, posted, keeper
        ) &&
        (
            command.a7 == 0 ||
            km::gather_run(
                reader,
                command,
                command.a7,
                0,
                command.a8,
                far,
                posted,
                keeper,
                0,
                command.a8
            )
        ) &&
        (after || km::gathered(reader, command, port_in, pages)) &&
        km::gather_wait(reader, command, posted)
    );
}

uint32_t km_overlay(km::Reader& reader, const SplitBlock& block) {
    const km_command_t& command = *block.command;
    if (!cross(reader, command)) {
        return 0;
    }
    noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(cross_done(reader))), 1);
    return km::coll_flushed(reader, command) ? 1 : 0;
}
