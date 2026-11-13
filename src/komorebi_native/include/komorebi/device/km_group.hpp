// Groups of a group and the reduce schedules over them: who sends to whom, on which
// port and into which arrival slot. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "km_command.hpp"

namespace km {

    // A slice of the group: rank j (0 <= j < size) is core base + j * stride; this
    // core is rank `rank`.
    struct Group {
        uint32_t base;
        uint32_t stride;
        uint32_t size;
        uint32_t rank;
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

    inline ReduceStep reduce_send(const Group& group, uint32_t form) {
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
    inline uint32_t reduce_slots(const Group& group, uint32_t form) {
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
    inline uint32_t reduce_wants(const Group& group, uint32_t form, uint32_t port) {
        if (form == kReduceOwner) {
            return group.rank == 0 && port == 0 && group.size > 1 ? group.size - 1 : 0;
        }
        return (
            port < kTreeRounds && tree_receives(group.rank, group.size, port) ? 1 : 0
        );
    }

}  // namespace km
