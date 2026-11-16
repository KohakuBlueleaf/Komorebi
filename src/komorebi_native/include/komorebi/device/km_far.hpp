// A write into any rank's L1 (a count at its port after) and an add to a word of any
// rank's L1: this island's ranks by the NoC; another island's gives up (where `far`).
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_island.hpp"

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // The give-up of a far call: the gave-up word's top byte is the rank's island.
    template <typename Kernel>
    inline bool far_gave_up(
        const Kernel& core, const km_command_t& command, uint32_t code, uint32_t island
    ) {
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        core.gave_up(
            word,
            code,
            kWhereFar,
            slot(command) | (island & 0xFFu) << 8,
            command.sequence
        );
        return false;
    }

    // The core index of `rank` on this island, KM_CORES_MAX when it is another
    // island's (`island` set) or no island's.
    template <typename Kernel>
    inline uint32_t far_core(const Kernel& core, uint32_t rank, uint32_t& island) {
        const Islands view = islands_at(core.args().control);
        const Resolved where = resolve(view, rank);
        island = where.island;
        if (!where.here) {
            return KM_CORES_MAX;
        }
        if (view.ranks() != 0 && rank - view.first() >= view.ranks()) {
            return KM_CORES_MAX;  // in no entry: not this island's either
        }
        const uint32_t index = core.core_of(rank);
        return index < KM_CORES_MAX ? index : KM_CORES_MAX;
    }

    // The one rule of a write on any island: bytes in whole 4s, both addresses
    // 16-aligned (a far rank's pieces need it; KM_GUARD refuses it on this island too).
    constexpr bool far_fits(uint32_t from, uint32_t address, uint32_t bytes) {
#if KM_GUARD
        return bytes % 4 == 0 && ((from | address) & 15u) == 0;
#else
        return ((from ^ address) & 15u) == 0;  // the NoC's own rule
#endif
    }

    // The code of a far call to another island in this build: nothing was sent.
    constexpr uint32_t far_refused() {
#if KM_GUARD && defined(KM_PAGE_LINKS)
        return kGaveRefused;  // the command alone, as a refused page line
#else
        return kGaveBadItem;
#endif
    }

    // `bytes` (4s) from L1 `from` into L1 `address` of `rank` (both 16-aligned), then 1
    // at its port (port >= 0). Done at return. False: gave up (`far`).
    template <typename Kernel>
    inline bool write_l1(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        const void* from,
        uint32_t bytes,
        int port = -1
    ) {
        const uint32_t source = reinterpret_cast<uint32_t>(from);
        uint32_t island = 0;
        const uint32_t index = far_core(core, rank, island);
        if (index == KM_CORES_MAX && island != islands_at(core.args().control).here()) {
            return far_gave_up(core, command, far_refused(), island);
        }
        if (
            index == KM_CORES_MAX ||
            (port >= 0 && static_cast<uint32_t>(port) >= KM_PORTS) ||
            !far_fits(source, address, bytes)
        ) {
            return far_gave_up(core, command, kGaveBadItem, island);
        }
        const uint32_t limit = core.args().wait_limit;
        uint32_t code = kGaveNone;
        if (bytes != 0) {
            noc_async_write(source, core.core_noc(index, address), bytes);
            code = core.poll_at(
                kWhereFar,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                limit
            );
        }
        if (code == kGaveNone && port >= 0) {
            const uint32_t count = (
                core.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
                KM_PORT_COUNT * 4
            );
            noc_semaphore_inc(core.core_noc(index, count), 1);
            code = core.poll_at(
                kWhereFar,
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                limit
            );
        }
        return code == kGaveNone || far_gave_up(core, command, code, island);
    }

    // `value` added to the word at `address` (4-byte aligned) of rank `rank`'s L1.
    // Done when it returns. False: gave up (where `far`).
    template <typename Kernel>
    inline bool add_word(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        uint32_t value
    ) {
        uint32_t island = 0;
        const uint32_t index = far_core(core, rank, island);
        if (index == KM_CORES_MAX && island != islands_at(core.args().control).here()) {
            return far_gave_up(core, command, far_refused(), island);
        }
        if (index == KM_CORES_MAX || address % 4 != 0) {
            return far_gave_up(core, command, kGaveBadItem, island);
        }
        noc_semaphore_inc(core.core_noc(index, address), value);
        const uint32_t code = core.poll_at(
            kWhereFar,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            core.args().wait_limit
        );
        return code == kGaveNone || far_gave_up(core, command, code, island);
    }

}  // namespace km
#endif
