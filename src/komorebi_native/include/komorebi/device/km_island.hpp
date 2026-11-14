// Islands on a core: the island table the host writes into the control block, rank
// resolve, and the primitives that only ask. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "../abi/km_abi.h"
#include "km_command.hpp"

namespace km {

    // Entry fields, byte offsets inside an entry (the host's islands.ENTRY).
    constexpr uint32_t kIslandEntryFirst = 0;  // u16: first rank
    constexpr uint32_t kIslandEntryCount = 2;  // u16: ranks
    constexpr uint32_t kIslandEntryLink = 4;   // u8: next link, KM_ISLAND_NO_LINK own
    constexpr uint32_t kIslandEntryHops = 5;   // u8: links crossed toward it
    // km::rate of this island: no link limits it (the NoC's own); 0: not known.
    constexpr uint32_t kRateLocal = 0xFFFFFFFFu;

    // The island area of this core's control block (defined where the kernel's
    // arguments are known: km_core.hpp; a template, so a host build may leave it out).
    template <int = 0>
    const volatile uint32_t* island_area();

    // An island area viewed through its header and its table; a card of its own
    // (KM_ONE_ISLAND) knows them at build time: island 0 of 1, no link area or tables.
    class Islands {
    public:
        explicit Islands(const volatile uint32_t* area) : area_(area) {
        }

        uint32_t here() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_HERE];
#endif
        }

        // Islands in the table; a block the host wrote no table into is one island.
        uint32_t count() const {
#if defined(KM_ONE_ISLAND)
            return 1;
#else
            const uint32_t count = area_[KM_ISLAND_COUNT];
            return count == 0 ? 1 : count;
#endif
        }

        uint32_t first() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_FIRST];
#endif
        }

        // This island's ranks; 0 when no table was written (every rank is here).
        uint32_t ranks() const {
            return area_[KM_ISLAND_RANKS];
        }

        // Its first core of the card (a card split into islands); 0 on a whole card.
        uint32_t base() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_BASE];
#endif
        }

        // The payload bytes of a packet on this island's links; 0: no links.
        uint32_t packet() const {
            return area_[KM_ISLAND_PACKET];
        }

        // The flat rank of core 0 (first - base, modulo 2^32): rank = this + core.
        uint32_t offset() const {
            return first() - base();
        }

        // The link area's L1 address on this island's cores; 0: none (no links).
        uint32_t link_area() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_LINK_AREA / 4];
#endif
        }

        // The own words' L1 address on this island's cores; 0: none (no links).
        uint32_t own_words() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_OWN / 4];
#endif
        }

        // The tensor table's L1 address on this island's cores; 0: none (one island).
        uint32_t tensors() const {
#if defined(KM_ONE_ISLAND)
            return 0;
#else
            return area_[KM_ISLAND_TENSORS / 4];
#endif
        }

        uint32_t entry_first(uint32_t island) const {
            return half(island, kIslandEntryFirst);
        }

        uint32_t entry_count(uint32_t island) const {
            return half(island, kIslandEntryCount);
        }

        uint32_t link(uint32_t island) const {
            return byte(island, kIslandEntryLink);
        }

        uint32_t hops(uint32_t island) const {
#if defined(KM_ONE_ISLAND)
            return static_cast<void>(island), 0;
#else
            return (
                island < count() && area_[KM_ISLAND_COUNT] != 0
                    ? byte(island, kIslandEntryHops)
                    : 0
            );
#endif
        }

        // The island that holds `rank`: its table entry; this island when the rank is
        // in no entry (one island: always 0).
        uint32_t island_of(uint32_t rank) const {
#if defined(KM_ONE_ISLAND)
            return static_cast<void>(rank), 0;
#else
            const uint32_t islands = area_[KM_ISLAND_COUNT];
            for (
                uint32_t island = 0; island < islands && island < KM_ISLANDS_MAX;
                island++
            ) {
                if (rank - entry_first(island) < entry_count(island)) {
                    return island;
                }
            }
            return here();
#endif
        }

    private:
        const volatile uint8_t* entry(uint32_t island) const {
            return (
                reinterpret_cast<const volatile uint8_t*>(area_) +
                KM_ISLAND_HEADER_BYTES + KM_ISLAND_ENTRY_BYTES * island
            );
        }

        uint32_t half(uint32_t island, uint32_t at) const {
            return (
                entry(island)[at] | static_cast<uint32_t>(entry(island)[at + 1]) << 8
            );
        }

        uint32_t byte(uint32_t island, uint32_t at) const {
            return entry(island)[at];
        }

        const volatile uint32_t* area_;
    };

    // The island area of the control block at `control`.
    inline Islands islands_at(uint32_t control) {
        return (
            Islands(
                reinterpret_cast<const volatile uint32_t*>(control + KM_CONTROL_ISLANDS)
            )
        );
    }

    // The part of the command's group on this island, in its cores (the core table's
    // indices, from its base): what this island's cores run and complete.
    struct Part {
        uint32_t first;
        uint32_t count;
    };

    inline Part local_part(const Islands& view, const km_command_t& command) {
        if (KM_GROUP_IS_LINK(
            first(command)
        )) {  // link cores: the cores after the ranks
            return {
                view.base() + view.ranks() + first(command) - KM_GROUP_LINK_BASE,
                count(command)
            };
        }
        const uint32_t low = view.first();
        const uint32_t high = view.ranks() == 0 ? 0xFFFFFFFFu : low + view.ranks();
        const uint32_t start = first(command) > low ? first(command) : low;
        uint32_t end = first(command) + count(command);
        end = end < high ? end : high;
        return {start - low + view.base(), end > start ? end - start : 0};
    }

    // Whether core `me` is a link core: a core after the island's ranks.
    inline bool link_core(const Islands& view, uint32_t me) {
#if defined(KM_ONE_ISLAND)
        return static_cast<void>(view), static_cast<void>(me), false;
#else
        return view.ranks() != 0 && me - view.base() >= view.ranks();
#endif
    }

    // The name core `me` has in `command`'s group: a link core's is always
    // KM_GROUP_LINK_BASE + its link-core index (in no group of ranks), a rank core's
    // its flat rank (in no group of link cores). Its rank in the group: this less the
    // group's first.
    inline uint32_t name_in(
        const Islands& view, uint32_t me, const km_command_t& command
    ) {
        return (
            link_core(view, me) || KM_GROUP_IS_LINK(first(command))
                ? KM_GROUP_LINK_BASE + (me - view.base() - view.ranks())
                : view.offset() + me
        );
    }

    // Where rank `rank` runs: its island; on this island its local core (the core
    // table's index), else this island's link toward it (its table entry's next link).
    struct Resolved {
        uint32_t island;
        bool here;
        uint32_t core;  // here: its base + rank - this island's first rank
        uint32_t link;  // not here: the next link
    };

    inline Resolved resolve(const Islands& islands, uint32_t rank) {
        const uint32_t island = islands.island_of(rank);
        if (island == islands.here()) {
            return {island, true, rank - islands.offset(), KM_ISLAND_NO_LINK};
        }
        return {island, false, 0, islands.link(island)};
    }

    // The primitives that only ask, about this core's island; one island: island 0 of
    // 1, every rank on it, no neighbours.
    inline uint32_t island() {
        return Islands(island_area()).here();
    }

    inline uint32_t islands() {
        return Islands(island_area()).count();
    }

    inline uint32_t island_of(uint32_t rank) {
        return Islands(island_area()).island_of(rank);
    }

    inline Resolved resolve(uint32_t rank) {
        return resolve(Islands(island_area()), rank);
    }

    // Whether a send to `rank` crosses a link.
    inline bool crosses(uint32_t rank) {
        const Islands view(island_area());
        return view.island_of(rank) != view.here();
    }

    // Links crossed from this island to island `i`.
    inline uint32_t hops(uint32_t i) {
        return Islands(island_area()).hops(i);
    }

    // The islands one link away, a bit each (KM_ISLANDS_MAX <= 32).
    inline uint32_t neighbours() {
        const Islands view(island_area());
        uint32_t mask = 0;
        for (uint32_t i = 0; i < view.count() && i < KM_ISLANDS_MAX; i++) {
            mask |= (view.hops(i) == 1 ? 1u : 0u) << i;
        }
        return mask;
    }

    // Bytes a second toward island `i`: kRateLocal for this island; the table holds
    // no link rate, so 0 (not known) for another.
    inline uint32_t rate(uint32_t i) {
        return i == island() ? kRateLocal : 0;
    }

}  // namespace km
