// A message to one core: emit (writer), arrive (reader), take (compute), and
// reduce by owner and by tree over them. The program supplies the combine step.
#pragma once

#include <stdint.h>

#include "km_far.hpp"
#include "km_group.hpp"
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
        if (crosses(destination)) {  // another island's rank: by its link cores
            uint32_t island = 0;
            const uint32_t code = (
                write_l1_at(
                    writer,
                    destination,
                    slot_addr(arrival, arrival_slot),
                    get_read_ptr(cb),
                    bytes,
                    static_cast<int>(port),
                    island
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

    // arrive's steps; kFirst: the buffer reserved before the wait (a reduce's), then
    // before() (a repeated reduce's credits). False: gave up.
    template <bool kFirst, typename Before>
    inline bool arrive_at(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        uint32_t port,
        uint32_t wanted,
        Before before
    ) {
        const uint32_t named = slot(command) | port << 8;
        if (port >= KM_PORTS) {
            reader.gave_up(
                kStReader, kGaveBadItem, kWherePort, named, command.sequence
            );
            return false;
        }
        if ((kFirst && !reader.reserve(arrival.cb, arrival.pages)) || !before()) {
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
        if (!kFirst && !reader.reserve(arrival.cb, arrival.pages)) {
            return false;
        }
        cb_push_back(arrival.cb, arrival.pages);
        return true;
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
        return arrive_at<false>(reader, command, arrival, port, wanted, [] {
            return true;
        });
    }

    // A reduce's turn over one group in one command (1: the first; a chain's record
    // starts over): from turn 2 a sender waits for its receiver's credit.
    struct ReduceTurn {
        uint32_t sequence;
        uint32_t key;
        uint32_t turn;
    };

#if defined(KM_OVERLAY_PART)
    // a part's own: its .bss is zeroed at each call, and a command is one call
    inline ReduceTurn& reduce_turn_now() {
        static ReduceTurn own = {};
        return own;
    }
#else
    inline ReduceTurn reduce_turn = {};

    inline ReduceTurn& reduce_turn_now() {
        return reduce_turn;
    }
#endif

    inline uint32_t reduce_turn_next(
        const km_command_t& command, const Group& group, uint32_t form
    ) {
        const uint32_t key = (
            group.base ^ group.size << 10 ^ group.stride << 20 ^ form << 30
        );
        ReduceTurn& now = reduce_turn_now();
        if (now.turn == 0 || now.sequence != command.sequence || now.key != key) {
            now = {command.sequence, key, 0};
        }
        return ++now.turn;
    }

    inline void reduce_turns_restart() {
        reduce_turn_now().turn = 0;
    }

#if defined(KM_PAGE_LINKS)
    // km_far.hpp's (included at the end): a credit to a rank of any island.
    template <typename Kernel>
    inline bool add_word_at(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        uint32_t value,
        bool confirmed
    );
#endif

    // Turn 2 on, over islands with no page links: refused (a credit back would cross
    // the link). True: one island. With page links the credit goes by km::add_word.
    template <typename Kernel>
    inline bool reduce_turn_here(
        const Kernel& core,
        const km_command_t& command,
        const Group& group,
        uint32_t port,
        uint32_t where
    ) {
        if (reduce_parts(group).parts == 1) {
            return true;
        }
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        core.gave_up(
            word, kGaveBadItem, where, slot(command) | port << 8, command.sequence
        );
        return false;
    }

    // Reader, turn 2 on, the buffer reserved (compute has taken the turn before): a
    // credit to each sender of `port`. False: gave up (recorded).
    __attribute__((noinline)) inline bool reduce_credit_senders(
        const Reader& reader,
        const km_command_t& command,
        const Group& group,
        uint32_t form,
        uint32_t port
    ) {
#if !defined(KM_PAGE_LINKS)
        if (!reduce_turn_here(reader, command, group, port, kWherePort)) {
            return false;
        }
#endif
        const uint32_t credit = (
            reader.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
            4 * KM_PORT_CREDIT
        );
        const ReduceSenders senders = reduce_senders_flat(group, form, port);
        for (uint32_t rank = senders.first; rank < senders.end; rank++) {
#if defined(KM_PAGE_LINKS)
            // posted: its sender waits for it before it emits, so it has landed
            // before the command ends (no fence); this island's: noc_semaphore_inc
            if (!add_word_at(reader, command, core_of(group, rank), credit, 1, false)) {
                return false;
            }
#else
            noc_semaphore_inc(
                reader.core_noc(reader.core_of(core_of(group, rank)), credit), 1
            );
#endif
        }
        return reader.flushed(kFlushAtomics, kStReader);
    }

    // Reader, a reduce given up: the ports' counts cleared (a bump counted early must
    // not meet the next command's wait).
    __attribute__((noinline)) inline bool reduce_clear(const Reader& reader) {
        for (uint32_t port = 0; port < KM_PORTS; port++) {
            port_words(reader, port)[KM_PORT_COUNT] = 0;
        }
        return false;
    }

    // Reader side of a reduce: each arrival this core waits for, in round order.
    inline bool reduce_arrive(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form
    ) {
        const uint32_t turn = reduce_turn_next(command, group, form);
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
            if (wanted != 0 &&
                !arrive_at<true>(reader, command, arrival, port, wanted, [&] {
                    return (
                        turn == 1 ||
                        reduce_credit_senders(reader, command, group, form, port)
                    );
                })) {
                return reduce_clear(reader);
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

    // Writer, turn `turn` > 1: waits until its receiver's credits (word `credit`) count
    // turn - 1. False: gave up (recorded).
    __attribute__((noinline)) inline bool reduce_credit(
        const Writer& writer,
        const km_command_t& command,
        const Group& group,
        uint32_t port,
        volatile tt_l1_ptr uint32_t* credit,
        uint32_t turn
    ) {
        // over islands with no page links: refused (reduce_turn_here), one give-up
#if defined(KM_PAGE_LINKS)
        (void)group;
        const uint32_t code = writer.poll_at(
            kWhereEmit,
            [&] {
                invalidate_l1_cache();
                return (static_cast<int32_t>(*credit - (turn - 1)) >= 0);
            },
            writer.args().wait_limit
        );
#else
        const uint32_t code = reduce_parts(group).parts != 1
            ? kGaveBadItem
            : writer.poll_at(
                kWhereEmit,
                [&] {
                    invalidate_l1_cache();
                    return (static_cast<int32_t>(*credit - (turn - 1)) >= 0);
                },
                writer.args().wait_limit
            );
#endif
        if (code != kGaveNone) {
            writer.gave_up(
                kStWriter, code, kWhereEmit, slot(command) | port << 8, command.sequence
            );
            return false;
        }
        return true;
    }

    // Writer side of a reduce: a core other than the root emits its `pages` pages of
    // `cb` to its receiver (from turn 2, after its credit). The root sends nothing.
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
        const uint32_t turn = reduce_turn_next(command, group, form);
        if (step.root) {
            return true;
        }
        if (step.port < KM_PORTS) {
            volatile tt_l1_ptr uint32_t* credit = (
                &port_words(writer, step.port)[KM_PORT_CREDIT]
            );
            if (turn == 1) {
                *credit = 0;  // the receiver credits only from its turn 2
            } else if (!reduce_credit(
                writer, command, group, step.port, credit, turn
            )) {
                return false;
            }
        }
        return emit(writer, command, arrival, cb, pages, step.to, step.port, step.slot);
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
    // is there (on this island a local emit, on another by its link cores).
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
#if defined(KM_PAGE_LINKS)
#include "km_far.hpp"
#endif

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
