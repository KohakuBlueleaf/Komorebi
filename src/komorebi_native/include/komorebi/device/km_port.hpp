// A message to one core: emit (writer), arrive (reader), take (compute), and
// reduce by owner and by tree over them. The program supplies the combine step.
#pragma once

#include <stdint.h>

#include "km_group.hpp"
#include "km_relay.hpp"
#include "km_worker.hpp"

namespace km {

    // Timeout of a wait that lies behind a port's wait (for a part that needs
    // arrivals), in wait limits: the port's give-up comes first and names the port.
    constexpr uint32_t kBehindPort = 2;

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    inline volatile tt_l1_ptr uint32_t* port_words(const Core& core, uint32_t port) {
        return core.control(KM_CONTROL_PORT + port * KM_PORT_STRIDE);
    }

    // L1 address of `arrival_slot`: the buffer's base from its configuration, not from
    // its pointers, so any kernel of any core computes the same address.
    inline uint32_t slot_addr(const Arrival& arrival, uint32_t arrival_slot) {
        const LocalCBInterface& circular_buffer = get_local_cb_interface(arrival.cb);
        return (
            circular_buffer.fifo_limit - circular_buffer.fifo_size +
            slot_page(arrival, arrival_slot) * circular_buffer.fifo_page_size
        );
    }

    // Writer: `pages` pages (at most P) at the front of `cb` into `arrival_slot` of
    // rank `destination`, bump its port (each step timed), pop. False: gave up.
    inline bool emit(
        const Writer& writer,
        const km_command_t& command,
        const Arrival& arrival,
        uint32_t cb,
        uint32_t pages,
        uint32_t destination,
        uint32_t port,
        uint32_t arrival_slot
    ) {
        const uint32_t named = slot(command) | port << 8;
        const uint32_t bytes = pages * get_local_cb_interface(cb).fifo_page_size;
        const uint32_t room = (
            arrival.per * get_local_cb_interface(arrival.cb).fifo_page_size
        );
        if (
            port >= KM_PORTS || arrival_slot >= arrival_slots(arrival) || bytes > room
        ) {
            writer.gave_up(
                kStWriter, kGaveBadItem, kWhereEmit, named, command.sequence
            );
            return false;
        }
        if (!writer.wait_front(cb, pages, kBehindPort)) {
            return false;
        }
        if (crosses(destination)) {  // another island's rank: through the relay
            const uint32_t code = (
                relay_put(
                    writer,
                    command,
                    get_read_ptr(cb),
                    bytes,
                    destination,
                    slot_addr(arrival, arrival_slot),
                    port
                )
            );
            if (code != kGaveNone) {
                writer.gave_up(kStWriter, code, kWhereEmit, named, command.sequence);
                return false;
            }
            cb_pop_front(cb, pages);
            return true;
        }
        const Islands view(island_area());
        if (
            writer.core_of(destination) >= KM_CORES_MAX ||
            (view.ranks() != 0 && destination - view.first() >= view.ranks())
        ) {  // a rank no island holds: not this island's either
            writer.gave_up(
                kStWriter, kGaveBadItem, kWhereEmit, named, command.sequence
            );
            return false;
        }
        const uint32_t limit = writer.args().wait_limit;
        noc_async_write(
            get_read_ptr(cb),
            writer.core_noc(
                writer.core_of(destination), slot_addr(arrival, arrival_slot)
            ),
            bytes
        );
        uint32_t code = writer.poll_at(
            kWhereEmit,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        if (code == kGaveNone) {
            const uint32_t port_address = (
                writer.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE
            );
            noc_semaphore_inc(
                writer.core_noc(writer.core_of(destination), port_address), 1
            );
            code = writer.poll_at(
                kWhereEmit,
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                limit
            );
        }
        if (code != kGaveNone) {
            writer.gave_up(kStWriter, code, kWhereEmit, named, command.sequence);
            return false;
        }
        cb_pop_front(cb, pages);
        return true;
    }

    // Reader or writer: waits until the port counted `wanted`, then clears it; a
    // give-up at `where` (`named`: its top half). False: gave up, the count kept.
    template <typename Kernel>
    inline bool port_wait_at(
        const Kernel& core,
        const km_command_t& command,
        uint32_t port,
        uint32_t wanted,
        uint32_t where,
        uint32_t named
    ) {
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        if (port >= KM_PORTS) {
            core.gave_up(word, kGaveBadItem, where, named, command.sequence);
            return false;
        }
        volatile tt_l1_ptr uint32_t* words = port_words(core, port);
        words[KM_PORT_WANT] = wanted;
        const uint32_t code = core.poll_at(
            where,
            [&] {
                return static_cast<int32_t>(words[KM_PORT_COUNT] - wanted) >= 0;
            },
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(word, code, where, named, command.sequence);
            return false;
        }
        words[KM_PORT_COUNT] = 0;
        words[KM_PORT_WANT] = 0;
        return true;
    }

    // port_wait_at in the port's own words: a give-up names the port.
    inline bool port_wait(
        const Reader& reader,
        const km_command_t& command,
        uint32_t port,
        uint32_t wanted
    ) {
        return (
            port_wait_at(
                reader, command, port, wanted, kWherePort, slot(command) | port << 8
            )
        );
    }

    // Reader: waits until the port has counted `wanted` arrivals, clears it, hands all
    // S pages to compute. False: gave up; the port keeps `wanted`: the missing count.
    inline bool arrive(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        uint32_t port,
        uint32_t wanted
    ) {
        const uint32_t named = slot(command) | port << 8;
        if (port >= KM_PORTS) {
            reader.gave_up(
                kStReader, kGaveBadItem, kWherePort, named, command.sequence
            );
            return false;
        }
        volatile tt_l1_ptr uint32_t* words = port_words(reader, port);
        words[KM_PORT_WANT] = wanted;
        const uint32_t code = reader.poll_at(
            kWherePort,
            [&] {
                return static_cast<int32_t>(words[KM_PORT_COUNT] - wanted) >= 0;
            },
            reader.args().wait_limit
        );
        if (code != kGaveNone) {
            reader.gave_up(kStReader, code, kWherePort, named, command.sequence);
            return false;
        }
        words[KM_PORT_COUNT] = 0;  // every sender of this command has bumped it
        words[KM_PORT_WANT] = 0;
        if (!reader.reserve(arrival.cb, arrival.pages)) {
            return false;
        }
        cb_push_back(arrival.cb, arrival.pages);
        return true;
    }

    // Reader side of a reduce: each arrival this core waits for, in round order.
    inline bool reduce_arrive(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form
    ) {
        // a slot per sender and a port per round: else the root gives up as well
        const uint32_t ports = reduce_ports(group, form);
        if (reduce_slots(group, form) > arrival_slots(arrival) || ports > KM_PORTS) {
            const uint32_t named = (
                slot(command) | (ports > KM_PORTS ? ports - 1 : 0) << 8
            );
            reader.gave_up(
                kStReader, kGaveBadItem, kWherePort, named, command.sequence
            );
            return false;
        }
        for (uint32_t port = 0; port < KM_PORTS; port++) {
            const uint32_t wanted = reduce_wants(group, form, port);
            if (wanted != 0 && !arrive(reader, command, arrival, port, wanted)) {
                return false;
            }
        }
        return true;
    }

    // The same inside this core's island (scope km::local).
    inline bool reduce_arrive(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        Local
    ) {
        return reduce_arrive(reader, command, arrival, group.local(), form);
    }

    // Writer side of a reduce: a core other than the root emits its `pages` pages
    // of `cb` to its receiver. The root sends nothing.
    inline bool reduce_emit(
        const Writer& writer,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        uint32_t cb,
        uint32_t pages
    ) {
        const ReduceStep step = reduce_send(group, form);
        return (
            step.root ||
            emit(writer, command, arrival, cb, pages, step.to, step.port, step.slot)
        );
    }

    // The same inside this core's island (scope km::local).
    inline bool reduce_emit(
        const Writer& writer,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        uint32_t cb,
        uint32_t pages,
        Local
    ) {
        return reduce_emit(writer, command, arrival, group.local(), form, cb, pages);
    }

    // A send to a rank of island `island`: put() is emit() after a check that the rank
    // is there (on this island a local emit, on another the relay).
    struct ToIsland {
        uint32_t island;

        bool put(
            const Writer& writer,
            const km_command_t& command,
            const Arrival& arrival,
            uint32_t cb,
            uint32_t pages,
            uint32_t rank,
            uint32_t port,
            uint32_t arrival_slot
        ) const {
            if (island_of(rank) != island) {
                writer.gave_up(
                    kStWriter, kGaveBadItem, kWhereEmit, slot(command), command.sequence
                );
                return false;
            }
            return emit(writer, command, arrival, cb, pages, rank, port, arrival_slot);
        }
    };

    inline ToIsland to_island(uint32_t island) {
        return {island};
    }

    // Writer at the root of a reduce: waits for the result's `pages` pages of `cb`
    // (timed behind the ports). False: gave up (recorded).
    inline bool reduce_result(const Writer& writer, uint32_t cb, uint32_t pages) {
        return writer.wait_front(cb, pages, kBehindPort);
    }

}  // namespace km

#else
namespace km {

    // Compute: waits for the S pages arrive() pushes; pages are then read by index
    // (copy_tile(arrival.cb, slot_page(arrival, slot) + i, ...)). False: stopping.
    inline bool take(const Compute& compute, const Arrival& arrival) {
        return compute.wait_front(arrival.cb, arrival.pages, kBehindPort);
    }

    // Releases an arrival: all S pages, so the slot addresses never drift.
    inline void untake(const Arrival& arrival) {
        cb_pop_front(arrival.cb, arrival.pages);
    }

    // Compute side of a reduce: each arrival in round order; combine(slot) for its
    // slots in rank order (owner: 0 to size - 2; tree: from the port's, one each).
    template <typename Combine>
    inline bool reduce(
        const Compute& compute,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        Combine combine
    ) {
        for (uint32_t port = 0; port < KM_PORTS; port++) {
            const uint32_t wanted = reduce_wants(group, form, port);
            if (wanted == 0) {
                continue;
            }
            if (!take(compute, arrival)) {
                return false;
            }
            if (form == kReduceOwner) {
                for (uint32_t arrival_slot = 0; arrival_slot < wanted; arrival_slot++) {
                    combine(arrival_slot);
                }
            } else {
                for (
                    uint32_t arrival_slot = port; arrival_slot < port + wanted;
                    arrival_slot++
                ) {
                    combine(arrival_slot);
                }
            }
            untake(arrival);
        }
        return true;
    }

    // The same inside this core's island (scope km::local).
    template <typename Combine>
    inline bool reduce(
        const Compute& compute,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        Local,
        Combine combine
    ) {
        return reduce(compute, arrival, group.local(), form, combine);
    }

}  // namespace km
#endif
