// The pack thread's walk of one packet of 32 rays through a tree in L1, node first:
// each box is tested against every ray of the packet still in it, depth first.
#pragma once

#include <stdint.h>

#include "raytrace.h"
#include "sfpu.hpp"

namespace walk {

    using namespace rt;

    constexpr uint32_t kMaxDepth = 64;

    // Cycles and counts of the pack thread (profile page words 0 to 31).
    struct Profile {
        uint32_t wait_in = 0;   // waiting for input pages
        uint32_t load = 0;      // a packet into Dst
        uint32_t pop = 0;       // pop: the entry test, tensix_sync and the lane check
        uint32_t box = 0;       // box tests pushed (two per inner node)
        uint32_t tri = 0;       // triangle tests pushed
        uint32_t drain = 0;     // vector unit finishing a packet
        uint32_t out = 0;       // results to the output page
        uint32_t wait_out = 0;  // waiting for a free output page
        uint32_t packets = 0;
        uint32_t pops = 0;
        uint32_t empty = 0;  // pops with no lane left
        uint32_t inner = 0;  // inner nodes visited
        uint32_t leaves = 0;
        uint32_t error = 0;  // 1: the stack bound was hit; 2: a packet's list never fit
        uint32_t total = 0;
        uint32_t emits = 0;      // treelet list entries (top walk)
        uint32_t visits = 0;     // packet walks of a treelet
        uint32_t items = 0;      // work pages
        uint32_t wait_work = 0;  // waiting for a work page
        uint32_t st_load = 0;    // packet state into Dst
        uint32_t st_store = 0;   // packet state back to L1
        uint32_t phase1 = 0;     // the top walk of every packet
        uint32_t phase2 = 0;     // the treelet visits
        uint32_t phase3 = 0;     // results out
        uint32_t batches = 0;    // top walks with visits after them (1 + rewalks)
        uint32_t deferred = 0;   // packet top walks put off for lack of list room
    };

    constexpr uint32_t kProfileWords = 26;

    struct NoEmit {
        void operator()(uint32_t) const {
        }
    };

    inline uint32_t any_lane() {
        tensix_sync();
        uint32_t status;
        asm volatile("csrr %0, 0xBC0" : "=r"(status));
        return (status >> 11) & 1;
    }

    // Box word `index` of the node record at `box`, as immediates.
    inline auto box_immediate(const uint32_t* box) {
        return [box](uint32_t lreg, uint32_t index) {
            immediate(lreg, box[index]);
        };
    }

    // The packet's lanes whose entry distance in slot s is within their best, every
    // lane enabled first; slot s is copied to s + 1 for the second child.
    inline void pop_lanes(uint32_t slot) {
        all_lanes();
        load(L0, slot);
        store(L0, slot + 1);
        load(L1, kSlotBest);
        TTI_SFPLE(0, L1, L0, kSetCC);
    }

    // The walk of the packet in Dst through a tree image (nodes, order words, triangle
    // records), nearest or any; emit(treelet) at each treelet reached, nearer first.
    template <typename Emit = NoEmit>
    inline void packet(
        const uint32_t* nodes,
        const uint32_t* order,
        const uint32_t* tris,
        uint32_t signs,
        bool any,
        Profile& profile,
        Emit emit = Emit()
    ) {
        uint32_t stack[kMaxDepth];
        uint32_t depth = 0;
        uint32_t before = wall();
        // Root: every lane; slot kSlotStack holds its entry distance or +inf.
        all_lanes();
        box_test(box_immediate(nodes), kSlotOrigin, kSlotInverse, kSlotBest);
        store(L7, kSlotStack);
        stack[depth++] = 0;
        uint32_t after = wall();
        profile.box += after - before;
        while (depth > 0) {
            --depth;
            const uint32_t node_index = stack[depth];
            const uint32_t slot = kSlotStack + depth;
            before = wall();
            pop_lanes(slot);
            const uint32_t live = any_lane();
            after = wall();
            profile.pop += after - before;
            profile.pops++;
            if (!live) {
                profile.empty++;
                continue;
            }
            const uint32_t* node = nodes + kNodeWords * node_index;
            const uint32_t first_child = node[6];
            if (first_child & kLeafBit) {
                count(kSlotLeaves, 1);
                const uint32_t* triangle = tris + kTriWords * (first_child & ~kLeafBit);
                tri_test(
                    [triangle](uint32_t lreg, uint32_t index) {
                        immediate(lreg, triangle[index]);
                    },
                    kSlotOrigin,
                    kSlotDirection,
                    kSlotBest,
                    kSlotTriangle,
                    kSlotU,
                    kSlotV,
                    any,
                    !any
                );
                profile.tri += wall() - after;
                profile.leaves++;
                continue;
            }
            if (first_child & kTreeletBit) {
                emit(first_child & ~kTreeletBit);
                profile.emits++;
                continue;
            }
            if (depth + 2 > kMaxDepth - 1) {
                profile.error = 1;
                return;
            }
            // The nearer child for the packet's direction takes the higher slot.
            const uint32_t order_word = order[node_index];
            const bool second = ((signs >> (order_word & 3)) ^ (order_word >> 2)) & 1;
            const uint32_t near = second ? node[7] : first_child;
            const uint32_t far = second ? first_child : node[7];
            count(kSlotNodes, 2);
            box_test(
                box_immediate(nodes + kNodeWords * far),
                kSlotOrigin,
                kSlotInverse,
                kSlotBest
            );
            store(L7, slot);
            box_test(
                box_immediate(nodes + kNodeWords * near),
                kSlotOrigin,
                kSlotInverse,
                kSlotBest
            );
            store(L7, slot + 1);
            stack[depth++] = far;
            stack[depth++] = near;
            profile.box += wall() - after;
            profile.inner++;
        }
    }

    // A packet page into Dst: the ray, best from the page, index -1, u = v = 0, one box
    // test (the root) and no triangle test per ray; the hit attributes zero.
    inline void load_packet(const uint32_t* page) {
        for (uint32_t slot = 0; slot < 10; ++slot) {
            to_slot(kSlotOrigin + slot, page + kVec * (kPkO + slot));
        }
        fill_slot(kSlotTriangle, 0xFFFFFFFFu);
        fill_slot(kSlotU, 0);
        fill_slot(kSlotV, 0);
        fill_slot(kSlotNodes, 1);
        fill_slot(kSlotLeaves, 0);
        order(word(kSlotLeaves) + 62);
        all_lanes();
        TTI_SFPLOADI(L5, kFloatB, 0);
        for (uint32_t slot = 0; slot < kTriAttrWords; ++slot) {
            store(L5, kSlotAttributes + slot);
        }
    }

    // A packet state image (Dst tile 0 in window order) into Dst through the window,
    // and back: the window words of slots 0 to kStSlots - 1.
    constexpr uint32_t kStateImageWords = word(kStSlots - 1) + 2 * kVec;

    inline void load_image(const uint32_t* image) {
        copy_words(win(), image, kStateImageWords);
        order(kStateImageWords - 2);
    }

    inline void store_image(uint32_t* image) {
        copy_words(image, win(), kStateImageWords);
    }

    // Broadcast `lane` of `slot` in a state image.
    inline uint32_t image_lane(const uint32_t* image, uint32_t slot, uint32_t lane) {
        return image[word(slot) + 2 * lane];
    }

    // The packet's cycles as the hit page's last vector.
    inline void store_cycles(uint32_t* hit_page, uint32_t cycles) {
        uint32_t* last_vector = hit_page + kVec * kHitCycles;
        last_vector[0] = cycles;
        for (uint32_t lane = 1; lane < kVec; ++lane) {
            last_vector[lane] = 0;
        }
    }

    // The results of the packet in Dst to a hit page (after tensix_sync); a nearest
    // hit's page with the attributes. Returns the page's words.
    inline uint32_t store_hits(uint32_t* hit_page, uint32_t cycles, bool any) {
        for (uint32_t vector = 0; vector < kHitCycles; ++vector) {
            from_slot(kSlotBest + vector, hit_page + kVec * vector);
        }
        store_cycles(hit_page, cycles);
        if (any) {
            return kHitAnyVectors * kVec;
        }
        for (uint32_t attribute = 0; attribute < kTriAttrWords; ++attribute) {
            from_slot(
                kSlotAttributes + attribute, hit_page + kVec * (kHitN + attribute)
            );
        }
        return kHitVectors * kVec;
    }

}  // namespace walk
