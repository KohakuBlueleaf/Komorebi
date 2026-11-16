// Groups of a group and the reduce schedules over them: who sends to whom, on which
// port and into which arrival slot. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "km_command.hpp"
#include "km_island.hpp"

namespace km {

    // A slice of the group: rank j (0 <= j < size) is core base + j * stride; this
    // core is rank `rank` (size: not in it).
    struct Group {
        uint32_t base;
        uint32_t stride;
        uint32_t size;
        uint32_t rank;

        // The part on this core's island (one island: the group itself).
        Group local() const {
            const Islands view(island_area());
            if (view.ranks() == 0 || stride == 0) {
                return *this;
            }
            const uint32_t low = view.first(), high = low + view.ranks();
            const uint32_t from = base >= low ? 0 : (low - base + stride - 1) / stride;
            uint32_t to = base >= high ? 0 : (high - base + stride - 1) / stride;
            to = to < size ? to : size;
            if (from >= to) {
                return {base, stride, 0, 0};
            }
            const uint32_t mine = rank - from < to - from ? rank - from : to - from;
            return {base + from * stride, stride, to - from, mine};
        }

        // The first rank of island part `part` (parts in island order).
        uint32_t leader(uint32_t part) const {
            const Islands view(island_area());
            const uint32_t island = view.island_of(base) + part;
            const uint32_t first = view.entry_first(island);
            return part == 0 || first < base ? base : first;
        }

        // The leaders of its island parts, one each: on one island the group's first
        // rank alone; across islands stride 0, the members by leader(j).
        Group leaders() const {
            const Islands view(island_area());
            const uint32_t last = base + (size - 1) * stride;
            const uint32_t parts = (
                size == 0 ? 0 : view.island_of(last) - view.island_of(base) + 1
            );
            const uint32_t me = base + rank * stride;
            uint32_t mine = parts;
            for (uint32_t part = 0; part < parts; part++) {
                mine = leader(part) == me ? part : mine;
            }
            return {base, parts == 1 ? 1u : 0u, parts, rank < size ? mine : parts};
        }
    };

    inline uint32_t core_of(const Group& group, uint32_t rank) {
        return group.base + rank * group.stride;
    }

    // The whole group, in core order. `me` is in it.
    inline Group group_of(const km_command_t& command, uint32_t me) {
        return {first(command), 1, count(command), me - first(command)};
    }

    // The cores whose group index differs from this core's only in the digit of weight
    // `stride`, base `radix` (index = high * stride * radix + digit * stride + low).
    inline Group digit_group(
        const km_command_t& command, uint32_t me, uint32_t stride, uint32_t radix
    ) {
        const uint32_t index = me - first(command);
        const uint32_t digit = index / stride % radix;
        const uint32_t low = index - digit * stride;
        const uint32_t fit = (count(command) - low + stride - 1) / stride;
        return {first(command) + low, stride, fit < radix ? fit : radix, digit};
    }

    // Reduce forms.
    constexpr uint32_t kReduceOwner = 0;
    constexpr uint32_t kReduceTree = 1;
    constexpr uint32_t kTreeRounds = 7;  // rounds of a group of up to 128 cores

    // Tree, `round`: an odd multiple of 2^round sends to rank - 2^round (port and slot
    // `round`); a multiple of 2^(round+1) receives. Owner: all send to rank 0.
    inline bool tree_receives(uint32_t rank, uint32_t size, uint32_t round) {
        return (rank & ((2u << round) - 1)) == 0 && rank + (1u << round) < size;
    }

    // The round in which a rank other than 0 sends: its lowest set bit.
    inline uint32_t tree_send_round(uint32_t rank) {
        uint32_t round = 0;
        while (((rank >> round) & 1u) == 0 && round < 31) {
            round++;
        }
        return round;
    }

    // A core's send in a reduce (to, port, slot), or root: it sends nothing.
    struct ReduceStep {
        bool root;
        uint32_t to;    // core index of the receiver
        uint32_t port;  // receiver's port and arrival slot
        uint32_t slot;
    };

    inline ReduceStep reduce_send_flat(const Group& group, uint32_t form) {
        if (group.rank == 0) {
            return {true, 0, 0, 0};
        }
        if (form == kReduceOwner) {
            return {false, core_of(group, 0), 0, group.rank - 1};
        }
        const uint32_t round = tree_send_round(group.rank);
        return {false, core_of(group, group.rank - (1u << round)), round, round};
    }

    // Arrival slots a reduce needs in each core's buffer (owner: size - 1 at the
    // root; tree: one per round).
    inline uint32_t reduce_slots_flat(const Group& group, uint32_t form) {
        if (form == kReduceOwner) {
            return group.size > 1 ? group.size - 1 : 0;
        }
        uint32_t rounds = 0;
        while ((1u << rounds) < group.size && rounds < 31) {
            rounds++;
        }
        return rounds;
    }

    // The program's arrival buffer: circular buffer `cb` of `pages` pages, `per` (P) to
    // a slot: slot k is pages k * P to k * P + P - 1, one L1 address on all cores.
    struct Arrival {
        uint32_t cb;
        uint32_t pages;
        uint32_t per = 1;
    };

    inline uint32_t arrival_slots(const Arrival& arrival) {
        return arrival.per == 0 ? 0 : arrival.pages / arrival.per;
    }

    // First page of `arrival_slot`; a part's page i is page
    // slot_page(arrival, arrival_slot) + i.
    inline uint32_t slot_page(const Arrival& arrival, uint32_t arrival_slot) {
        return arrival_slot * arrival.per;
    }

    // Arrivals this rank waits for on port `port` (by owner: port 0, size - 1 at the
    // root; by tree: 1 in each round it receives before it sends).
    inline uint32_t reduce_wants_flat(
        const Group& group, uint32_t form, uint32_t port
    ) {
        if (form == kReduceOwner) {
            return group.rank == 0 && port == 0 && group.size > 1 ? group.size - 1 : 0;
        }
        return (
            port < kTreeRounds && tree_receives(group.rank, group.size, port) ? 1 : 0
        );
    }

    // The ranks that send to this rank on `port` in one island's reduce: first to end -
    // 1 (owner: every rank but 0 on port 0; tree: the one rank 2^port after it).
    struct ReduceSenders {
        uint32_t first;
        uint32_t end;
    };

    inline ReduceSenders reduce_senders_flat(
        const Group& group, uint32_t form, uint32_t port
    ) {
        if (reduce_wants_flat(group, form, port) == 0) {
            return {0, 0};
        }
        if (form == kReduceOwner) {
            return {1, group.size};
        }
        return {group.rank + (1u << port), group.rank + (1u << port) + 1};
    }

    // A reduce over islands (naive hop): each island's part reduces as a group of its
    // own, then each part's first rank sends once to the group's rank 0.
    struct ReduceParts {
        Group part;      // this core's island part (one island: the group)
        uint32_t parts;  // islands the group touches
        uint32_t index;  // this part's, in island order
        uint32_t root;   // the size of the part of rank 0's island
    };

    inline ReduceParts reduce_parts(const Group& group) {
        const Group part = group.local();
        if (part.size == group.size || group.stride == 0) {
            return {group, 1, 0, group.size};
        }
        const Islands view(island_area());
        const uint32_t low = view.island_of(group.base);
        const uint32_t high = view.island_of(core_of(group, group.size - 1));
        const uint32_t end = view.entry_first(low) + view.entry_count(low);
        const uint32_t root = (end - group.base + group.stride - 1) / group.stride;
        return {part, high - low + 1, view.here() - low, root};
    }

    // The slot at rank 0 of the part of index j > 0: after the root part's.
    inline uint32_t reduce_cross(const ReduceParts& p, uint32_t form, uint32_t j) {
        const Group root = {0, 1, p.root, 0};
        return reduce_slots_flat(root, form) + j - 1;
    }

    // The one port all parts j > 0 send on to rank 0 (tree: the first after the root
    // part's rounds; owner: port 0).
    inline uint32_t reduce_cross_port(const ReduceParts& p, uint32_t form) {
        const Group root = {0, 1, p.root, 0};
        return form == kReduceOwner ? 0 : reduce_slots_flat(root, form);
    }

    // A core's send in the reduce of `group` over its islands (one island: the flat
    // schedule); `to` is a flat rank.
    inline ReduceStep reduce_send(const Group& group, uint32_t form) {
        const ReduceParts p = reduce_parts(group);
        if (p.parts == 1 || p.part.rank != 0) {
            return reduce_send_flat(p.part, form);
        }
        if (p.index == 0) {
            return {true, 0, 0, 0};
        }
        const uint32_t cross = reduce_cross(p, form, p.index);
        return {false, group.base, reduce_cross_port(p, form), cross};
    }

    inline uint32_t reduce_slots(const Group& group, uint32_t form) {
        const ReduceParts p = reduce_parts(group);
        return (
            p.parts == 1
                ? reduce_slots_flat(group, form)
                : reduce_cross(p, form, p.parts - 1) + 1
        );
    }

    inline uint32_t reduce_wants(const Group& group, uint32_t form, uint32_t port) {
        const ReduceParts p = reduce_parts(group);
        uint32_t wants = reduce_wants_flat(p.part, form, port);
        if (p.parts > 1 && p.index == 0 && p.part.rank == 0) {
            wants += port == reduce_cross_port(p, form) ? p.parts - 1 : 0;
        }
        return wants;
    }

    // Ports a reduce names at its root (0 to reduce_ports - 1).
    inline uint32_t reduce_ports(const Group& group, uint32_t form) {
        const ReduceParts p = reduce_parts(group);
        if (p.parts == 1) {
            return form == kReduceOwner ? 1 : reduce_slots_flat(group, form);
        }
        return reduce_cross_port(p, form) + 1;
    }

    // Scope km::local: a reduce, collect or broadcast stays inside this core's island
    // (one island: the whole group).
    struct Local {};

    inline const Local local{};

    inline ReduceStep reduce_send(const Group& group, uint32_t form, Local) {
        return reduce_send_flat(group.local(), form);
    }

    inline uint32_t reduce_slots(const Group& group, uint32_t form, Local) {
        return reduce_slots_flat(group.local(), form);
    }

    inline uint32_t reduce_wants(
        const Group& group, uint32_t form, uint32_t port, Local
    ) {
        return reduce_wants_flat(group.local(), form, port);
    }

}  // namespace km
