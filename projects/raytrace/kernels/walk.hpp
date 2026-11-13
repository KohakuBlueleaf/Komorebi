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
    struct Prof {
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

    constexpr uint32_t kProfWords = 26;

    struct NoEmit {
        void operator()(uint32_t) const {
        }
    };

    inline uint32_t any_lane() {
        tensix_sync();
        uint32_t v;
        asm volatile("csrr %0, 0xBC0" : "=r"(v));
        return (v >> 11) & 1;
    }

    // Box word k of the node record at b, as immediates.
    inline auto box_imm(const uint32_t* b) {
        return [b](uint32_t r, uint32_t k) {
            imm(r, b[k]);
        };
    }

    // The packet's lanes whose entry distance in slot s is within their best, every
    // lane enabled first; slot s is copied to s + 1 for the second child.
    inline void pop_lanes(uint32_t s) {
        all_lanes();
        load(L0, s);
        store(L0, s + 1);
        load(L1, kSBest);
        TTI_SFPLE(0, L1, L0, kSetCC);
    }

    // The walk of the packet in Dst through a tree image (nodes, order words, triangle
    // records), nearest or any; emit(t) at each treelet reached, nearer first.
    template <typename Emit = NoEmit>
    inline void packet(
        const uint32_t* nodes,
        const uint32_t* order,
        const uint32_t* tris,
        uint32_t signs,
        bool any,
        Prof& pf,
        Emit emit = Emit()
    ) {
        uint32_t stack[kMaxDepth];
        uint32_t sp = 0;
        uint32_t a = wall();
        // Root: every lane; slot kSStack holds its entry distance or +inf.
        all_lanes();
        box_test(box_imm(nodes), kSO, kSInv, kSBest);
        store(L7, kSStack);
        stack[sp++] = 0;
        uint32_t b = wall();
        pf.box += b - a;
        while (sp > 0) {
            --sp;
            const uint32_t n = stack[sp];
            const uint32_t s = kSStack + sp;
            a = wall();
            pop_lanes(s);
            const uint32_t live = any_lane();
            b = wall();
            pf.pop += b - a;
            pf.pops++;
            if (!live) {
                pf.empty++;
                continue;
            }
            const uint32_t* nd = nodes + kNodeWords * n;
            const uint32_t c0 = nd[6];
            if (c0 & kLeafBit) {
                count(kSLeaves, 1);
                const uint32_t* w = tris + kTriWords * (c0 & ~kLeafBit);
                tri_test(
                    [w](uint32_t r, uint32_t k) {
                        imm(r, w[k]);
                    },
                    kSO,
                    kSD,
                    kSBest,
                    kSTri,
                    kSU,
                    kSV,
                    any,
                    !any
                );
                pf.tri += wall() - b;
                pf.leaves++;
                continue;
            }
            if (c0 & kTreeletBit) {
                emit(c0 & ~kTreeletBit);
                pf.emits++;
                continue;
            }
            if (sp + 2 > kMaxDepth - 1) {
                pf.error = 1;
                return;
            }
            // The nearer child for the packet's direction takes the higher slot.
            const uint32_t o = order[n];
            const bool second = ((signs >> (o & 3)) ^ (o >> 2)) & 1;
            const uint32_t near = second ? nd[7] : c0;
            const uint32_t far = second ? c0 : nd[7];
            count(kSNodes, 2);
            box_test(box_imm(nodes + kNodeWords * far), kSO, kSInv, kSBest);
            store(L7, s);
            box_test(box_imm(nodes + kNodeWords * near), kSO, kSInv, kSBest);
            store(L7, s + 1);
            stack[sp++] = far;
            stack[sp++] = near;
            pf.box += wall() - b;
            pf.inner++;
        }
    }

    // A packet page into Dst: the ray, best from the page, index -1, u = v = 0, one box
    // test (the root) and no triangle test per ray; the hit attributes zero.
    inline void load_packet(const uint32_t* pk) {
        for (uint32_t k = 0; k < 10; ++k) {
            to_slot(kSO + k, pk + kVec * (kPkO + k));
        }
        fill_slot(kSTri, 0xFFFFFFFFu);
        fill_slot(kSU, 0);
        fill_slot(kSV, 0);
        fill_slot(kSNodes, 1);
        fill_slot(kSLeaves, 0);
        order(word(kSLeaves) + 62);
        all_lanes();
        TTI_SFPLOADI(L5, kFloatB, 0);
        for (uint32_t k = 0; k < kTriAttrWords; ++k) {
            store(L5, kSAttr + k);
        }
    }

    // A packet state image (Dst tile 0 in window order) into Dst through the window,
    // and back: the window words of slots 0 to kStSlots - 1.
    constexpr uint32_t kStImageWords = word(kStSlots - 1) + 2 * kVec;

    inline void load_image(const uint32_t* img) {
        copy_words(win(), img, kStImageWords);
        order(kStImageWords - 2);
    }

    inline void store_image(uint32_t* img) {
        copy_words(img, win(), kStImageWords);
    }

    // Lane l of slot s in a state image.
    inline uint32_t image_lane(const uint32_t* img, uint32_t s, uint32_t l) {
        return img[word(s) + 2 * l];
    }

    // The packet's cycles as the hit page's last vector.
    inline void store_cycles(uint32_t* out, uint32_t cycles) {
        uint32_t* v = out + kVec * kHitCycles;
        v[0] = cycles;
        for (uint32_t l = 1; l < kVec; ++l) {
            v[l] = 0;
        }
    }

    // The results of the packet in Dst to a hit page (after tensix_sync); a nearest
    // hit's page with the attributes. Returns the page's words.
    inline uint32_t store_hits(uint32_t* out, uint32_t cycles, bool any) {
        for (uint32_t k = 0; k < kHitCycles; ++k) {
            from_slot(kSBest + k, out + kVec * k);
        }
        store_cycles(out, cycles);
        if (any) {
            return kHitAnyVectors * kVec;
        }
        for (uint32_t k = 0; k < kTriAttrWords; ++k) {
            from_slot(kSAttr + k, out + kVec * (kHitN + k));
        }
        return kHitVectors * kVec;
    }

}  // namespace walk
