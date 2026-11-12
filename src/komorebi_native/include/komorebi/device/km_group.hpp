// Groups of a team and the reduce schedules over them: who sends to whom, on which
// port and into which arrival slot. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "km_record.hpp"

namespace km {

    // A slice of the team: member j (0 <= j < size) is core base + j * stride; this
    // core is member `rank`.
    struct Group {
        uint32_t base;
        uint32_t stride;
        uint32_t size;
        uint32_t rank;
    };

    inline uint32_t member(const Group& g, uint32_t j) {
        return g.base + j * g.stride;
    }

    // The whole team, in core order. `me` is a member.
    inline Group team_group(const km_record_t& r, uint32_t me) {
        return {first(r), 1, count(r), me - first(r)};
    }

    // The members whose team index differs from this core's only in the digit of
    // weight `stride` and base `radix` (index = hi * stride * radix + d * stride + lo).
    inline Group digit_group(
        const km_record_t& r, uint32_t me, uint32_t stride, uint32_t radix
    ) {
        const uint32_t t = me - first(r), d = t / stride % radix, lo = t - d * stride;
        const uint32_t fit = (count(r) - lo + stride - 1) / stride;
        return {first(r) + lo, stride, fit < radix ? fit : radix, d};
    }

    // Reduce forms.
    constexpr uint32_t kReduceOwner = 0;
    constexpr uint32_t kReduceTree = 1;
    constexpr uint32_t kTreeRounds = 7;  // rounds of a group of up to 128 members

    // Tree, round r: an odd multiple of 2^r sends to rank - 2^r (port r, slot r); a
    // multiple of 2^(r+1) receives if rank + 2^r < size. Owner: all send to rank 0.
    inline bool tree_receives(uint32_t rank, uint32_t size, uint32_t r) {
        return (rank & ((2u << r) - 1)) == 0 && rank + (1u << r) < size;
    }

    // The round in which a member other than 0 sends: its lowest set bit.
    inline uint32_t tree_send_round(uint32_t rank) {
        uint32_t r = 0;
        while (((rank >> r) & 1u) == 0 && r < 31) {
            r++;
        }
        return r;
    }

    // A member's send in a reduce (to, port, slot), or root: it sends nothing.
    struct ReduceStep {
        bool root;
        uint32_t to;    // core index of the receiver
        uint32_t port;  // receiver's port and arrival slot
        uint32_t slot;
    };

    inline ReduceStep reduce_send(const Group& g, uint32_t form) {
        if (g.rank == 0) {
            return {true, 0, 0, 0};
        }
        if (form == kReduceOwner) {
            return {false, member(g, 0), 0, g.rank - 1};
        }
        const uint32_t r = tree_send_round(g.rank);
        return {false, member(g, g.rank - (1u << r)), r, r};
    }

    // Arrivals member g waits for on port `port` (by owner: port 0, size - 1 at the
    // root; by tree: 1 in each round it receives before it sends).
    inline uint32_t reduce_wants(const Group& g, uint32_t form, uint32_t port) {
        if (form == kReduceOwner) {
            return g.rank == 0 && port == 0 && g.size > 1 ? g.size - 1 : 0;
        }
        return port < kTreeRounds && tree_receives(g.rank, g.size, port) ? 1 : 0;
    }

}  // namespace km
