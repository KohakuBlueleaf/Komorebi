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

    inline uint32_t core_of(const Group& g, uint32_t j) {
        return g.base + j * g.stride;
    }

    // The whole group, in core order. `me` is in it.
    inline Group group_of(const km_command_t& r, uint32_t me) {
        return {first(r), 1, count(r), me - first(r)};
    }

    // The cores whose group index differs from this core's only in the digit of
    // weight `stride` and base `radix` (index = hi * stride * radix + d * stride + lo).
    inline Group digit_group(
        const km_command_t& r, uint32_t me, uint32_t stride, uint32_t radix
    ) {
        const uint32_t t = me - first(r), d = t / stride % radix, lo = t - d * stride;
        const uint32_t fit = (count(r) - lo + stride - 1) / stride;
        return {first(r) + lo, stride, fit < radix ? fit : radix, d};
    }

    // Reduce forms.
    constexpr uint32_t kReduceOwner = 0;
    constexpr uint32_t kReduceTree = 1;
    constexpr uint32_t kTreeRounds = 7;  // rounds of a group of up to 128 cores

    // Tree, round r: an odd multiple of 2^r sends to rank - 2^r (port r, slot r); a
    // multiple of 2^(r+1) receives if rank + 2^r < size. Owner: all send to rank 0.
    inline bool tree_receives(uint32_t rank, uint32_t size, uint32_t r) {
        return (rank & ((2u << r) - 1)) == 0 && rank + (1u << r) < size;
    }

    // The round in which a rank other than 0 sends: its lowest set bit.
    inline uint32_t tree_send_round(uint32_t rank) {
        uint32_t r = 0;
        while (((rank >> r) & 1u) == 0 && r < 31) {
            r++;
        }
        return r;
    }

    // A core's send in a reduce (to, port, slot), or root: it sends nothing.
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
            return {false, core_of(g, 0), 0, g.rank - 1};
        }
        const uint32_t r = tree_send_round(g.rank);
        return {false, core_of(g, g.rank - (1u << r)), r, r};
    }

    // Arrival slots a reduce needs in each core's buffer (owner: size - 1 at the
    // root; tree: one per round).
    inline uint32_t reduce_slots(const Group& g, uint32_t form) {
        if (form == kReduceOwner) {
            return g.size > 1 ? g.size - 1 : 0;
        }
        uint32_t r = 0;
        while ((1u << r) < g.size && r < 31) {
            r++;
        }
        return r;
    }

    // The program's arrival buffer: circular buffer `cb` of `pages` pages, `per` (P) to
    // a slot: slot k is pages k * P to k * P + P - 1, one L1 address on all cores.
    struct Arrival {
        uint32_t cb;
        uint32_t pages;
        uint32_t per = 1;
    };

    inline uint32_t arrival_slots(const Arrival& a) {
        return a.per == 0 ? 0 : a.pages / a.per;
    }

    // First page of slot k (a part's page i is page slot_page(a, k) + i).
    inline uint32_t slot_page(const Arrival& a, uint32_t k) {
        return k * a.per;
    }

    // Arrivals rank g waits for on port `port` (by owner: port 0, size - 1 at the
    // root; by tree: 1 in each round it receives before it sends).
    inline uint32_t reduce_wants(const Group& g, uint32_t form, uint32_t port) {
        if (form == kReduceOwner) {
            return g.rank == 0 && port == 0 && g.size > 1 ? g.size - 1 : 0;
        }
        return port < kTreeRounds && tree_receives(g.rank, g.size, port) ? 1 : 0;
    }

}  // namespace km
