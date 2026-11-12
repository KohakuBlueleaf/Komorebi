// Probe 6: one full walk of a binary box tree (255 nodes, 128 leaf boxes) for up to
// 1024 rays, in two forms, on the pack thread with the vector extension.
//   Form A, ray-first: one packet of 32 rays at a time, each lane at its own node; the
//   core keeps one stack per lane in L1, copies the 32 node records into Dst, the
//   lanes test, the core reads the hit mask and pushes children.
//   Form B, node-first: up to 16 packets resident in Dst; one sweep over the nodes in
//   index order (children have larger indices); per node, every packet with rays at
//   it is tested against the node's box (pushed as immediates) under a lane mask; the
//   hit mask splits the rays into the children's bins; at a leaf the lanes update
//   their best in the vector unit.
// Node record (8 words): box min x y z, box max x y z (fp32), child 0, child 1; a leaf
// has kLeaf | leaf index in both child words. Result per ray: the entry distance of
// the closest leaf box hit (start: the best given) and that leaf's child word.
#pragma once

#include <stdint.h>

#include "probe.hpp"
#include "vec.hpp"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace walk {

    using namespace vec;

    constexpr uint32_t kLeaf = 0x80000000u;
    constexpr uint32_t kNodes = 255, kRecordWords = 8;
    constexpr uint32_t kRays = 2048;  // input: ray fields of packet p, field f, lane l
                                      // at kRays + 32 (6 p + f) + l
    constexpr uint32_t kStackDepth = 32;
    constexpr uint32_t kStepLimit = 4096;  // per packet (form A)
    constexpr uint32_t kBatch = 16;        // packets resident in Dst (form B)

    // Output data: best of ray r at r, its leaf word at 1024 + r.
    constexpr uint32_t kOutBest = 0, kOutLeaf = 1024;

    // Timing categories (cycles, summed over the run) and counters.
    enum Stat : uint32_t {
        kTotal = 0,
        kSetup,      // ray data into Dst, results out, stack / bin initialisation
        kStack,      // core: pops, pushes, bin updates
        kCopy,       // core: node records into Dst (form A)
        kVector,     // test pushed, executed and waited for (push + run + tensix_sync)
        kReadout,    // core: hit mask out of Dst (vector extension)
        kCheck,      // end test: tensix_sync + CSR (form A)
        kSteps,      // form A: steps; form B: (node, packet) tests
        kLaneTests,  // ray-node tests (active lanes summed)
        kCheckMismatch,  // CSR bit disagreed with the core's own count (form A)
        kNumStats
    };

    // Dst slot s (32 lanes): SFPLOAD address and window byte offset of lane 0.
    constexpr uint32_t slot_addr(uint32_t s) {
        return 4 * (s >> 1) + 2 * (s & 1);
    }

    constexpr uint32_t slot_byte(uint32_t s) {
        return 256 * (s >> 1) + 4 * (s & 1);
    }

    inline volatile uint32_t* slot_ptr(uint32_t s) {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow + slot_byte(s));
    }

    // Ray slots of a packet: inv x y z, ob x y z, best, leaf word.
    constexpr uint32_t kInv = 0, kOb = 3, kBest = 6, kLeafWord = 7;

    inline void write_slot(uint32_t s, const volatile uint32_t* src) {
        volatile uint32_t* p = slot_ptr(s);
        for (uint32_t l = 0; l < 32; ++l) {
            p[2 * l] = src[l];
        }
    }

    inline void fill_slot(uint32_t s, uint32_t v) {
        volatile uint32_t* p = slot_ptr(s);
        for (uint32_t l = 0; l < 32; ++l) {
            p[2 * l] = v;
        }
    }

    inline void read_slot(uint32_t s, volatile uint32_t* dst) {
        volatile uint32_t* p = slot_ptr(s);
        for (uint32_t l = 0; l < 32; ++l) {
            dst[l] = p[2 * l];
        }
    }

    // Ray slots base .. base + 7 of packet p from the input; best = best0, leaf = ~0.
    inline void load_packet(
        const volatile uint32_t* in, uint32_t p, uint32_t base, uint32_t best0
    ) {
        for (uint32_t f = 0; f < 6; ++f) {
            write_slot(base + f, in + kRays + 32 * (6 * p + f));
        }
        fill_slot(base + kBest, best0);
        fill_slot(base + kLeafWord, 0xffffffffu);
        order((slot_byte(base + kLeafWord) >> 2) + 62);
    }

    inline void store_packet(volatile uint32_t* d, uint32_t p, uint32_t base) {
        read_slot(base + kBest, d + kOutBest + 32 * p);
        read_slot(base + kLeafWord, d + kOutLeaf + 32 * p);
    }

    // Lane flags = bit `lane` of mask m (every lane enabled first); uses L5, L6.
    inline void lanes_from_mask(uint32_t m) {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSHFT(0xfff, LANE, L6, 5);  // L6 = lane
        TTI_SFPIADD(0, 9, L6, 2 | 4);     // L6 = 0 - lane
        TT_SFPLOADI(L5, 8, m >> 16);
        TT_SFPLOADI(L5, 10, m & 0xffff);
        TTI_SFPSHFT(0, L6, L5, 0);  // L5 = m >> lane
        TTI_SFPSHFT(31, 0, L5, 1);  // bit lane -> sign
        TTI_SFPSETCC(0, L5, 0, 0);  // flag = sign set
    }

    // One slab axis: L0 = min plane, L1 = max plane (already loaded); L2, L3 = inv,
    // ob of this axis from ray slots; updates L4 (near) and L5 (far).
    inline void axis_rays(uint32_t ray_base, uint32_t a) {
        TT_SFPLOAD(L2, kInt32, kAm, slot_addr(ray_base + kInv + a));
        TT_SFPLOAD(L3, kInt32, kAm, slot_addr(ray_base + kOb + a));
        TTI_SFPMAD(L0, L2, L3, L0, 0);
        TTI_SFPMAD(L1, L2, L3, L1, 0);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L1, L0, 1);
        TTI_SFPSWAP(0, L4, L0, 1);
        TTI_SFPSWAP(0, L1, L5, 1);
    }

    // Near = 0, far = best; at the end L6 = far - near (hit: sign clear), L4 = near.
    inline void test_begin(uint32_t ray_base) {
        TTI_SFPLOADI(L4, 0, 0);
        TT_SFPLOAD(L5, kInt32, kAm, slot_addr(ray_base + kBest));
    }

    inline void test_end() {
        TTI_SFPMAD(L4, L7, L5, L6, 0);
    }

    // Under the current flags: keep lanes that hit and are closer than their best,
    // then best = near, leaf word = L0 there. Flags back to every lane.
    inline void leaf_update(uint32_t ray_base) {
        TTI_SFPSETCC(0, L6, 0, 4);  // hit: L6 >= 0
        TT_SFPLOAD(L5, kInt32, kAm, slot_addr(ray_base + kBest));
        TTI_SFPMAD(L5, L7, L4, L5, 0);  // near - best
        TTI_SFPSETCC(0, L5, 0, 0);      // closer: < 0
        TT_SFPSTORE(L4, kInt32, kAm, slot_addr(ray_base + kBest));
        TT_SFPSTORE(L0, kInt32, kAm, slot_addr(ray_base + kLeafWord));
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // Hit mask (bit l: L6 of lane l >= 0) of Dst slot s, by the vector extension.
    inline uint32_t hit_mask(uint32_t s) {
        uint32_t mask = 0;
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        vint32m8_t v = (
            __riscv_vlse32_v_i32m8(
                reinterpret_cast<const int32_t*>(kDstWindow + slot_byte(s)), 8, vl
            )
        );
        __riscv_vsm_v_b4(
            reinterpret_cast<uint8_t*>(&mask), __riscv_vmsge_vx_i32m8_b4(v, 0, vl), vl
        );
        asm volatile("" ::: "memory");
#endif
        return mask;
    }

    // ---- Form A ------------------------------------------------------------------

    // Slots: rays 0..7, node record 8..15, hit 16.
    constexpr uint32_t kNodeSlot = 8, kHitSlot = 16;

    // Node records of lanes' nodes (node[l]) into slots 8..15.
    inline void copy_records_rvv(const uint32_t* table, const uint32_t* node) {
#if defined(__riscv_vector)
        asm volatile("fence" ::: "memory");  // node[] stores done before the load
        const size_t vl = __riscv_vsetvl_e32m8(32);
        vuint32m8_t off = __riscv_vsll_vx_u32m8(__riscv_vle32_v_u32m8(node, vl), 5, vl);
        auto* w = reinterpret_cast<uint32_t*>(kDstWindow);
#pragma GCC unroll 8
        for (uint32_t k = 0; k < kRecordWords; ++k) {
            vuint32m8_t v = __riscv_vluxei32_v_u32m8(table + k, off, vl);
            __riscv_vsse32_v_u32m8(w + slot_byte(kNodeSlot + k) / 4, 8, v, vl);
        }
        asm volatile("" ::: "memory");
#endif
        order((slot_byte(kNodeSlot + kRecordWords - 1) >> 2) + 62);
    }

    inline void copy_records_scalar(const uint32_t* table, const uint32_t* node) {
#pragma GCC unroll 1
        for (uint32_t l = 0; l < 32; l += 2) {
            const uint32_t* ra = table + kRecordWords * node[l];
            const uint32_t* rb = table + kRecordWords * node[l + 1];
            uint32_t a[kRecordWords], b[kRecordWords];
#pragma GCC unroll 8
            for (uint32_t k = 0; k < kRecordWords; ++k) {
                a[k] = ra[k];
                b[k] = rb[k];
            }
#pragma GCC unroll 8
            for (uint32_t k = 0; k < kRecordWords; ++k) {
                volatile uint32_t* p = slot_ptr(kNodeSlot + k);
                p[2 * l] = a[k];
                p[2 * l + 2] = b[k];
            }
        }
        order((slot_byte(kNodeSlot + kRecordWords - 1) >> 2) + 62);
    }

    // The lanes' test against their own node records, then the leaf update for leaf
    // lanes (leaf word in slot 14, sign set); L6 of every active lane to the hit slot.
    inline void test_own_nodes() {
        test_begin(0);
        TTI_SFPLOAD(L0, kInt32, kAm, slot_addr(kNodeSlot + 0));
        TTI_SFPLOAD(L1, kInt32, kAm, slot_addr(kNodeSlot + 3));
        axis_rays(0, 0);
        TTI_SFPLOAD(L0, kInt32, kAm, slot_addr(kNodeSlot + 1));
        TTI_SFPLOAD(L1, kInt32, kAm, slot_addr(kNodeSlot + 4));
        axis_rays(0, 1);
        TTI_SFPLOAD(L0, kInt32, kAm, slot_addr(kNodeSlot + 2));
        TTI_SFPLOAD(L1, kInt32, kAm, slot_addr(kNodeSlot + 5));
        axis_rays(0, 2);
        test_end();
        TTI_SFPSTORE(L6, kInt32, kAm, slot_addr(kHitSlot));
        TTI_SFPLOAD(L0, kInt32, kAm, slot_addr(kNodeSlot + 6));
        TTI_SFPSETCC(0, L0, 0, 0);  // leaf lanes only
        leaf_update(0);
    }

    struct Clock {
        uint32_t t;

        Clock() : t(wall()) {
        }

        uint32_t lap() {
            const uint32_t n = wall(), d = n - t;
            t = n;
            return d;
        }
    };

    // One packet: returns false if a stack or the step count hit its bound.
    [[gnu::noinline]] bool form_a_packet(
        const uint32_t* table, uint32_t* stacks, uint32_t* node, bool rvv, uint32_t* st
    ) {
        uint32_t sp[32];
        for (uint32_t l = 0; l < 32; ++l) {
            stacks[kStackDepth * l] = 0;
            sp[l] = 1;
            node[l] = 0;
        }
        uint32_t live = 0xffffffffu;  // lanes with a non-empty stack
        Clock c;
        st[kSetup] += c.lap();
        for (uint32_t step = 0; step < kStepLimit; ++step) {
            const uint32_t active = live;
            for (uint32_t m = active; m != 0; m &= m - 1) {
                const uint32_t l = __builtin_ctz(m);
                node[l] = stacks[kStackDepth * l + --sp[l]];
                if (sp[l] == 0) {
                    live &= ~(1u << l);
                }
            }
            const uint32_t n_active = __builtin_popcount(active);
            st[kStack] += c.lap();
            // End test: lanes enabled from the active mask, then tensix_sync + CSR.
            lanes_from_mask(active);
            tensix_sync();
            const uint32_t any = (qstatus_raw() >> 11) & 1;
            st[kCheck] += c.lap();
            if (any != (active != 0 ? 1u : 0u)) {
                ++st[kCheckMismatch];
            }
            if (!any) {
                return true;
            }
            if (rvv) {
                copy_records_rvv(table, node);
            } else {
                copy_records_scalar(table, node);
            }
            st[kCopy] += c.lap();
            test_own_nodes();
            tensix_sync();
            st[kVector] += c.lap();
            const uint32_t hit = hit_mask(kHitSlot) & active;
            st[kReadout] += c.lap();
            for (uint32_t m = hit; m != 0; m &= m - 1) {
                const uint32_t l = __builtin_ctz(m);
                const uint32_t* r = table + kRecordWords * node[l];
                const uint32_t c0 = r[6], c1 = r[7];
                if (c0 & kLeaf) {
                    continue;
                }
                if (sp[l] + 2 > kStackDepth) {
                    return false;
                }
                uint32_t* top = stacks + kStackDepth * l + sp[l];
                top[0] = c1;
                top[1] = c0;
                sp[l] += 2;
                live |= 1u << l;
            }
            st[kStack] += c.lap();
            ++st[kSteps];
            st[kLaneTests] += n_active;
        }
        return false;
    }

    // ---- Form B ------------------------------------------------------------------

    // Packet q of the batch: ray slots 8 q .. 8 q + 7, hit slot 128 + q.
    // The test of node fields nf (from L1, fp32) for packet q under mask m.
    inline void test_node_packet(
        const volatile uint32_t* nf, uint32_t q, uint32_t m, bool leaf
    ) {
        const uint32_t rb = 8 * q;
        lanes_from_mask(m);
        test_begin(rb);
        load_f32(L0, nf[0]);
        load_f32(L1, nf[3]);
        axis_rays(rb, 0);
        load_f32(L0, nf[1]);
        load_f32(L1, nf[4]);
        axis_rays(rb, 1);
        load_f32(L0, nf[2]);
        load_f32(L1, nf[5]);
        axis_rays(rb, 2);
        test_end();
        if (leaf) {
            load_f32(L0, nf[6]);
            leaf_update(rb);
        } else {
            TT_SFPSTORE(L6, kInt32, kAm, slot_addr(128 + q));
            TTI_SFPENCC(3, 0, 0, 10);
        }
    }

    // One batch of n_packets (<= 16) starting at packet p0.
    [[gnu::noinline]] void form_b_batch(
        const volatile uint32_t* in,
        volatile uint32_t* d,
        uint32_t* bins,
        uint32_t p0,
        uint32_t n_packets,
        uint32_t best0,
        uint32_t* st
    ) {
        Clock c;
        for (uint32_t q = 0; q < n_packets; ++q) {
            load_packet(in, p0 + q, 8 * q, best0);
        }
        for (uint32_t i = 0; i < kNodes * kBatch; ++i) {
            bins[i] = 0;
        }
        for (uint32_t q = 0; q < n_packets; ++q) {
            bins[q] = 0xffffffffu;
        }
        st[kSetup] += c.lap();
        for (uint32_t n = 0; n < kNodes; ++n) {
            const volatile uint32_t* nf = in + kRecordWords * n;
            const bool leaf = (nf[6] & kLeaf) != 0;
            uint32_t* b = bins + kBatch * n;
            uint32_t tested = 0;
            for (uint32_t q = 0; q < n_packets; ++q) {
                if (b[q] == 0) {
                    continue;
                }
                st[kStack] += c.lap();
                test_node_packet(nf, q, b[q], leaf);
                st[kVector] += c.lap();
                tested |= 1u << q;
                ++st[kSteps];
                st[kLaneTests] += __builtin_popcount(b[q]);
            }
            if (leaf || tested == 0) {
                st[kStack] += c.lap();
                continue;
            }
            tensix_sync();
            st[kVector] += c.lap();
            uint32_t hit[kBatch];
            for (uint32_t q = 0; q < n_packets; ++q) {
                hit[q] = (tested >> q) & 1 ? hit_mask(128 + q) & b[q] : 0;
            }
            st[kReadout] += c.lap();
            uint32_t* b0 = bins + kBatch * nf[6];
            uint32_t* b1 = bins + kBatch * nf[7];
            for (uint32_t q = 0; q < n_packets; ++q) {
                b0[q] |= hit[q];
                b1[q] |= hit[q];
            }
            st[kStack] += c.lap();
        }
        tensix_sync();
        st[kVector] += c.lap();
        for (uint32_t q = 0; q < n_packets; ++q) {
            store_packet(d, p0 + q, 8 * q);
        }
        st[kSetup] += c.lap();
    }

    // p0 rays, p1 form (0 A, 1 B), p2 copy (A: 0 vector extension, 1 scalar), p3
    // best0 bits. Stats to out[kTimes + i].
    inline void run(
        const volatile uint32_t* in,
        volatile uint32_t* out,
        uint32_t* scratch,
        const uint32_t* p,
        uint32_t& err
    ) {
        uint32_t st[kNumStats] = {};
        volatile uint32_t* d = out + probe::kData;
        const uint32_t n_packets = p[0] / 32, best0 = p[3];
        const uint32_t* table = const_cast<const uint32_t*>(in);
        asm volatile("fence" ::: "memory");
        TTI_SFPLOADI(L7, 0, 0xBF80);  // -1.0
        const uint32_t t0 = wall();
        if (p[1] == 0) {
            uint32_t* node = scratch + 32 * kStackDepth;
            for (uint32_t pk = 0; pk < n_packets; ++pk) {
                Clock c;
                load_packet(in, pk, 0, best0);
                st[kSetup] += c.lap();
                if (!form_a_packet(table, scratch, node, p[2] == 0, st)) {
                    err |= probe::kErrLoop;
                }
                tensix_sync();
                Clock c2;
                store_packet(d, pk, 0);
                st[kSetup] += c2.lap();
            }
        } else {
            for (uint32_t p0 = 0; p0 < n_packets; p0 += kBatch) {
                const uint32_t n = n_packets - p0 < kBatch ? n_packets - p0 : kBatch;
                form_b_batch(in, d, scratch, p0, n, best0, st);
            }
        }
        st[kTotal] = wall() - t0;
        for (uint32_t i = 0; i < kNumStats; ++i) {
            out[probe::kTimes + i] = st[i];
        }
        out[probe::kNumTimes] = kNumStats;
    }

}  // namespace walk
