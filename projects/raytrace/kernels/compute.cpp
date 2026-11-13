// Ray tracing compute: the unpack thread hands each input page to the pack thread,
// which walks packets (or runs the arithmetic check) on the vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "bounded.hpp"
#include "komorebi/device/km_worker.hpp"
#include "raytrace.h"
#ifdef TRISC_PACK
#include "divsqrt.hpp"
#include "walk.hpp"
#endif

#ifdef TRISC_PACK
namespace {

    using namespace rt;

    const uint32_t* g_tree = nullptr;  // the tree image, after LOAD

    // Profile error bits (besides the walk's 1 and 2): what gave up.
    constexpr uint32_t kErrUnpack = 4;   // unpack gave up and said so
    constexpr uint32_t kErrMail = 8;     // no word from unpack in time
    constexpr uint32_t kErrOut = 16;     // no output page in time
    constexpr uint32_t kErrSignal = 32;  // no signal page in time
    constexpr uint32_t kErrRoom = 64;    // a raw buffer too small
    constexpr uint32_t kErrTree = 128;   // no tree loaded

    // A word from unpack, kAbort if none came in time.
    uint32_t mbox_read() {
        uint32_t v = bounded::kAbort;
        return (
            bounded::mail(ckernel::ThreadId::UnpackThreadId, v) ? v : bounded::kAbort
        );
    }

    void mbox_release() {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
    }

    uint32_t out_ptr() {
        return get_local_cb_interface(kCbOut).fifo_wr_ptr << 4;
    }

    // Stores to L1 finished (a load of the last word written, used), then pushed.
    void push_out(uint32_t addr, uint32_t words) {
        asm volatile("fence" ::: "memory");
        const uint32_t v = reinterpret_cast<volatile uint32_t*>(addr)[words - 1];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
        cb_push_back(kCbOut, 1);
    }

    // A word from unpack; on kAbort the reason goes to the profile.
    bool mail(uint32_t& v, walk::Prof& pf) {
        if (!bounded::mail(ckernel::ThreadId::UnpackThreadId, v)) {
            pf.error |= kErrMail;
            return false;
        }
        pf.error |= v == bounded::kAbort ? kErrUnpack : 0;
        return v != bounded::kAbort;
    }

    // The next input page from the unpack thread (nullptr: gave up), its wait counted.
    const uint32_t* next_page(walk::Prof& pf) {
        const uint32_t a = wall();
        uint32_t p = 0;
        const bool ok = mail(p, pf);
        asm volatile("fence" ::: "memory");
        pf.wait_in += wall() - a;
        return ok ? reinterpret_cast<const uint32_t*>(p) : nullptr;
    }

    // A free output page (0: gave up).
    uint32_t reserve_out(walk::Prof& pf) {
        const uint32_t a = wall();
        const bool ok = bounded::back(kCbOut, 1);
        pf.wait_out += wall() - a;
        pf.error |= ok ? 0 : kErrOut;
        return ok ? out_ptr() : 0;
    }

    bool trace_list(uint32_t count, bool any, walk::Prof& pf) {
        const uint32_t* t = g_tree;
        const uint32_t* nodes = t + t[kTreeNodeOff];
        const uint32_t* order = t + t[kTreeOrderOff];
        const uint32_t* tris = t + t[kTreeTriOff];
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* pk = next_page(pf);
            if (pk == nullptr) {
                return false;
            }
            uint32_t a = wall();
            const uint32_t start = a;
            walk::load_packet(pk);
            const uint32_t signs = pk[kVec * kPkHead + kHeadSigns];
            mbox_release();
            uint32_t b = wall();
            pf.load += b - a;
            walk::packet(nodes, order, tris, signs, any, pf);
            a = wall();
            tensix_sync();
            b = wall();
            pf.drain += b - a;
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            a = wall();
            walk::store_hits(reinterpret_cast<uint32_t*>(dst), b - start);
            push_out(dst, kHitVectors * kVec);
            pf.out += wall() - a;
            pf.packets++;
        }
        return true;
    }

    uint32_t* raw(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4);
    }

    // A word other cores read: the store drained (a load of it, used), then a fence.
    void publish(volatile uint32_t* w, uint32_t v) {
        *w = v;
        const uint32_t x = *w;
        asm volatile("and x0, x0, %0" : : "r"(x) : "memory");
        asm volatile("fence" ::: "memory");
    }

    constexpr uint32_t kErrList = 256;  // a packet's list did not fit an empty pool

    // Top walks to the lists, states to L1, then the reader is told: first every packet
    // from its page, then those deferred (list did not fit) from their state.
    bool top_walks(uint32_t count, bool any, uint32_t cap, bool first, walk::Prof& pf) {
        const uint32_t list_bytes = 4 * (kLsHead + 2 * count) + 2 * cap;
        if (bounded::room(kCbState) < count * kStWords * 4 ||
            bounded::room(kCbLists) < list_bytes) {
            pf.error |= kErrRoom;
            return false;
        }
        const uint32_t* t = g_tree;
        const uint32_t* nodes = t + t[kTreeNodeOff];
        const uint32_t* order = t + t[kTreeOrderOff];
        uint32_t* ls = raw(kCbLists);
        uint32_t* st = raw(kCbState);
        uint16_t* pool = reinterpret_cast<uint16_t*>(ls + kLsHead + 2 * count);
        uint32_t used = 0, deferred = 0, admitted = 0;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t* me = st + kStWords * i;
            ls[kLsHead + 2 * i] = used;
            ls[kLsHead + 2 * i + 1] = 0;
            if (!first && me[kVec * kStHead + kStDeferred] == 0) {
                continue;
            }
            uint32_t a = wall();
            const uint32_t begin = a;
            uint32_t signs = 0;
            const uint32_t* pk = nullptr;
            if (first) {
                pk = next_page(pf);
                if (pk == nullptr) {
                    return false;
                }
                walk::load_packet(pk);
                signs = pk[kVec * kPkHead + kHeadSigns];
            } else {
                signs = me[kVec * kStHead];
                walk::load_state(me);
            }
            pf.load += wall() - a;
            const uint32_t start = used;
            walk::packet(nodes, order, nullptr, signs, any, pf, [&](uint32_t id) {
                if (used < cap) {
                    pool[used] = id;
                }
                used++;
            });
            const bool fits = used <= cap;
            used = fits ? used : start;
            if (!fits) {
                // Back to the state before the walk, so its counts are not taken twice.
                if (first) {
                    walk::load_packet(pk);
                } else {
                    walk::load_state(me);
                }
            }
            if (first) {
                mbox_release();
            }
            ls[kLsHead + 2 * i + 1] = used - start;
            deferred += fits ? 0 : 1;
            admitted += fits ? 1 : 0;
            pf.deferred += fits ? 0 : 1;
            a = wall();
            tensix_sync();
            walk::store_state(me, 0);
            me[kVec * kStHead] = signs;
            me[kVec * kStHead + kStDeferred] = fits ? 0 : 1;
            const uint32_t end = wall();
            me[kVec * kStHead + kStCycles] =
                ((first ? 0 : me[kVec * kStHead + kStCycles]) + end - begin);
            pf.st_store += end - a;
            pf.packets += first ? 1 : 0;
        }
        if (deferred > 0 && admitted == 0) {
            // Cannot happen when the lists hold every treelet (the host sees to it).
            pf.error |= kErrList;
            deferred = 0;
        }
        ls[kLsMore] = deferred;
        if (first) {
            publish(ls + kLsDone, 0);
        } else {
            publish(ls + kLsMore, deferred);
        }
        pf.batches++;
        if (!bounded::back(kCbSignal, 1)) {
            pf.error |= kErrSignal;
            return false;
        }
        cb_push_back(kCbSignal, 1);
        return true;
    }

    // Work pages up to a batch's end page (returns its word, 0: gave up): each listed
    // packet walks the page's treelet; finished pages (over batches) go to the lists.
    uint32_t treelet_visits(bool any, uint32_t& done, walk::Prof& pf) {
        uint32_t* ls = raw(kCbLists);
        uint32_t* st = raw(kCbState);
        for (;;) {
            const uint32_t a = wall();
            const uint32_t* w = next_page(pf);
            pf.wait_work += wall() - a;
            if (w == nullptr) {
                return 0;
            }
            const uint32_t n = w[kWorkCount];
            if (n == kWorkEnd || n == kWorkNext) {
                mbox_release();
                return n;
            }
            const uint32_t* t = reinterpret_cast<const uint32_t*>(w[kWorkSlot]);
            const uint32_t* nodes = t + t[kTreeNodeOff];
            const uint32_t* order = t + t[kTreeOrderOff];
            const uint32_t* tris = t + t[kTreeTriOff];
            const uint16_t* ids = reinterpret_cast<const uint16_t*>(w + kWorkIds);
            for (uint32_t j = 0; j < n; ++j) {
                uint32_t* me = st + kStWords * ids[j];
                const uint32_t v0 = wall();
                uint32_t b = v0;
                walk::load_state(me);
                pf.st_load += wall() - b;
                walk::packet(nodes, order, tris, me[kVec * kStHead], any, pf);
                b = wall();
                tensix_sync();
                walk::store_state(me, kSBest);
                const uint32_t e = wall();
                pf.st_store += e - b;
                me[kVec * kStHead + kStCycles] += e - v0;
                pf.visits++;
            }
            publish(ls + kLsDone, ++done);
            pf.items++;
            mbox_release();
        }
    }

    // The results of every packet (state vectors best to leaves) to hit pages.
    bool results_out(uint32_t count, walk::Prof& pf) {
        const uint32_t* st = raw(kCbState);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            const uint32_t a = wall();
            uint32_t* o = reinterpret_cast<uint32_t*>(dst);
            const uint32_t* me = st + kStWords * i;
            for (uint32_t k = 0; k < kHitCycles * kVec; ++k) {
                o[k] = me[kVec * kSBest + k];
            }
            walk::store_cycles(o, me[kVec * kStHead + kStCycles]);
            push_out(dst, kHitVectors * kVec);
            pf.out += wall() - a;
        }
        return true;
    }

    // Slab test with one fused multiply-add per plane (b inv - o inv, o inv in slots
    // oinv) and no exit scale: not the reference's arithmetic, for sizing only.
    template <typename Box>
    inline void fast_box(Box box, uint32_t oinv, uint32_t inv, uint32_t best) {
        TTI_SFPLOADI(L4, kFloatB, 0);
        TTI_SFPLOADI(L5, kFloatB, 0x7F80);
#pragma GCC unroll 3
        for (uint32_t k = 0; k < 3; ++k) {
            box(L0, k);
            box(L1, k + 3);
            load(L2, oinv + k);
            load(L3, inv + k);
            TT_SFPMAD(L0, L3, L2, L0, kNegC);
            TT_SFPMAD(L1, L3, L2, L1, kNegC);
            TTI_SFPNOP;
            TTI_SFPSWAP(0, L1, L0, 1);
            TTI_SFPSWAP(0, L4, L0, 1);
            TTI_SFPSWAP(0, L1, L5, 1);
        }
        load(L6, best);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L6, L5, 1);
        TTI_SFPMOV(0, INF, L7, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPLE(0, L5, L4, kSetCC);
        TTI_SFPMOV(0, L4, L7, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // Probe slots: 24 boxes broadcast from slot 100 (form data).
    constexpr uint32_t kPrBox = 100, kPrBoxes = 24;

    // One node visit of the probe in the given form (children c0, c1 of node n).
    inline void probe_visit(
        uint32_t form,
        const uint32_t* nodes,
        const uint32_t* order,
        uint32_t n,
        uint32_t i,
        uint32_t signs,
        volatile uint32_t* stack
    ) {
        const uint32_t* nd = nodes + kNodeWords * n;
        const uint32_t c0 = nd[6], c1 = nd[7];
        if (form == kProbeData) {
            const uint32_t b = (2 * i) % kPrBoxes;
            for (uint32_t c = 0; c < 2; ++c) {
                box_test(
                    [b, c](uint32_t r, uint32_t k) {
                        load(r, kPrBox + 6 * (b + c) + k);
                    },
                    kSO,
                    kSInv,
                    kSBest
                );
                store(L7, kSStack + c);
            }
            return;
        }
        if (form == kProbeFast) {
            fast_box(walk::box_imm(nodes + kNodeWords * c0), kSPv, kSInv, kSBest);
            store(L7, kSStack);
            fast_box(walk::box_imm(nodes + kNodeWords * c1), kSPv, kSInv, kSBest);
            store(L7, kSStack + 1);
            return;
        }
        uint32_t s = kSStack, near = c0, far = c1;
        if (form != kProbeImm) {
            s = kSStack + 2 * (i & 15);
            walk::pop_lanes(s);
            stack[i & 31] = walk::any_lane();
            if (form == kProbeSync) {
                return;
            }
        }
        if (form == kProbeStack) {
            const uint32_t o = order[n];
            const bool second = ((signs >> (o & 3)) ^ (o >> 2)) & 1;
            near = second ? c1 : c0;
            far = second ? c0 : c1;
            count(kSNodes, 2);
            stack[(i + 1) & 31] = far;
            stack[(i + 2) & 31] = near;
        }
        box_test(walk::box_imm(nodes + kNodeWords * far), kSO, kSInv, kSBest);
        store(L7, s);
        box_test(walk::box_imm(nodes + kNodeWords * near), kSO, kSInv, kSBest);
        store(L7, s + 1);
    }

    // Each packet against both children of every inner node, `passes` times; the
    // loop's cycles in pf.box, node visits in pf.inner; the hits as for a trace.
    bool probe(uint32_t count, uint32_t form, uint32_t passes, walk::Prof& pf) {
        const uint32_t* t = g_tree;
        const uint32_t* nodes = t + t[kTreeNodeOff];
        const uint32_t* order = t + t[kTreeOrderOff];
        const uint32_t n_nodes = t[kTreeNodes];
        volatile uint32_t stack[32];
        for (uint32_t p = 0; p < count; ++p) {
            const uint32_t* pk = next_page(pf);
            if (pk == nullptr) {
                return false;
            }
            walk::load_packet(pk);
            const uint32_t signs = pk[kVec * kPkHead + kHeadSigns];
            mbox_release();
            all_lanes();
            for (uint32_t k = 0; k < 3; ++k) {
                load(L0, kSO + k);
                load(L1, kSInv + k);
                mul(L0, L1, L0);
                store(L0, kSPv + k);
            }
            tensix_sync();
            const uint32_t* inner = nodes + kNodeWords * 0;
            for (uint32_t b = 0; b < kPrBoxes; ++b) {
                const uint32_t* box = nodes + kNodeWords * inner[6] + kNodeWords * b;
                for (uint32_t k = 0; k < 6; ++k) {
                    fill_slot(kPrBox + 6 * b + k, box[k]);
                }
            }
            rt::order(word(kPrBox + 6 * kPrBoxes - 1) + 62);
            const uint32_t a = wall();
            uint32_t i = 0;
            for (uint32_t pass = 0; pass < passes; ++pass) {
                for (uint32_t n = 0; n < n_nodes; ++n) {
                    const uint32_t w = nodes[kNodeWords * n + 6];
                    if ((w & (kLeafBit | kTreeletBit)) == 0) {
                        probe_visit(form, nodes, order, n, i++, signs, stack);
                    }
                }
            }
            tensix_sync();
            pf.box += wall() - a;
            pf.inner += i;
            pf.pops += passes;
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            walk::store_hits(reinterpret_cast<uint32_t*>(dst), 0);
            push_out(dst, kHitVectors * kVec);
            pf.packets++;
        }
        return true;
    }

    // Arithmetic check slots: inputs from 40, outputs from 70.
    constexpr uint32_t kArIn = 40, kArOut = 70;

    bool arith_list(uint32_t count, walk::Prof& pf) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* p = next_page(pf);
            if (p == nullptr) {
                return false;
            }
            for (uint32_t k = 0; k < kArInVectors; ++k) {
                to_slot(kArIn + k, p + kVec * k);
            }
            to_slot(kSBest, p + kVec * kArBest);
            to_slot(kSTri, p + kVec * kArCur);
            fill_slot(kSU, 0);
            fill_slot(kSV, 0);
            order(word(kSV) + 62);
            mbox_release();
            all_lanes();
            box_test(
                [](uint32_t r, uint32_t k) {
                    load(r, kArIn + kArBoxMin + k);
                },
                kArIn + kArO,
                kArIn + kArInv,
                kArIn + kArBest
            );
            store(L4, kArOut + kArEntry);
            store(L5, kArOut + kArExit);
            store(L7, kArOut + kArStack);
            tri_test(
                [](uint32_t r, uint32_t k) {
                    load(r, kArIn + kArV0 + k);
                },
                kArIn + kArO,
                kArIn + kArD,
                kSBest,
                kSTri,
                kSU,
                kSV,
                false
            );
            store(L3, kArOut + kArU);
            store(L0, kArOut + kArV);
            store(L1, kArOut + kArT);
            constexpr uint32_t from[6] = {kSDet, kSDetInv, kSBest, kSTri, kSU, kSV};
            constexpr uint32_t to[6] = {
                kArDet, kArDetInv, kArNewBest, kArNewTri, kArNewU, kArNewV
            };
            for (uint32_t k = 0; k < 6; ++k) {
                load(L0, from[k]);
                store(L0, kArOut + to[k]);
            }
            tensix_sync();
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            uint32_t* w = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t k = 0; k < kArOutVectors; ++k) {
                from_slot(kArOut + k, w + kVec * k);
            }
            push_out(dst, kArOutVectors * kVec);
            pf.packets++;
        }
        return true;
    }

    // Division and square root of each page's lanes: a / b and sqrt(x).
    bool math_list(uint32_t count, walk::Prof& pf) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* p = next_page(pf);
            if (p == nullptr) {
                return false;
            }
            for (uint32_t k = 0; k < kMaInVectors; ++k) {
                to_slot(kArIn + k, p + kVec * k);
            }
            order(word(kArIn + kMaInVectors - 1) + 62);
            mbox_release();
            all_lanes();
            load(L0, kArIn + kMaA);
            load(L1, kArIn + kMaB);
            divide();
            store(L2, kArOut + kMaQ);
            load(L0, kArIn + kMaX);
            square_root();
            store(L1, kArOut + kMaS);
            tensix_sync();
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            uint32_t* w = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t k = 0; k < kMaOutVectors; ++k) {
                from_slot(kArOut + k, w + kVec * k);
            }
            push_out(dst, kMaOutVectors * kVec);
            pf.packets++;
        }
        return true;
    }

    // Reciprocals of a = exponent | mantissa, mantissa = 1024 page + 32 j + lane.
    bool recip_list(uint32_t first, uint32_t count, uint32_t exponent, walk::Prof& pf) {
        constexpr uint32_t kLaneSlot = 100;
        all_lanes();
        TTI_SFPSHFT(0xFFF, LANE, L0, 5);  // lane = (2 lane) >> 1
        store(L0, kLaneSlot);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t base = (
                    (exponent << 23) + kRecipPerPage * (first + i) + 32 * j
                );
                imm(L1, base);
                load(L0, kLaneSlot);
                TT_SFPIADD(0, L1, L0, kCcNone);
                recip(L0, L3, L1, L2, L4, L5);
                store(L3, j);
            }
            tensix_sync();
            const uint32_t dst = reserve_out(pf);
            if (dst == 0) {
                return false;
            }
            uint32_t* w = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t j = 0; j < 32; ++j) {
                from_slot(j, w + kVec * j);
            }
            push_out(dst, kRecipPerPage);
            pf.packets++;
        }
        return true;
    }

    // The profile page: the pack thread's counters at words 0 to kProfWords - 1.
    void profile(const walk::Prof& pf) {
        if (!bounded::back(kCbOut, 1)) {
            return;
        }
        const uint32_t dst = out_ptr();
        volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(dst);
        const uint32_t* v = reinterpret_cast<const uint32_t*>(&pf);
        for (uint32_t k = 0; k < 32; ++k) {
            w[k] = k < walk::kProfWords ? v[k] : 0;
        }
        push_out(dst, 32);
    }

}  // namespace
#endif

#ifdef TRISC_UNPACK
namespace {

    // A page to the pack thread, then its release; false: pack did not release it.
    bool hand(uint32_t addr) {
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, addr);
        uint32_t v = 0;
        return bounded::mail(ckernel::ThreadId::PackThreadId, v);
    }

    // The reader's header page (first page, count) and every page after it to pack,
    // then (treelets) the work pages up to the end page. On a give-up pack gets kAbort.
    void unpack_pages(uint32_t seg) {
        constexpr uint32_t kPack = ckernel::ThreadId::PackThreadId;
        if (!bounded::front(kCbIn, 1)) {
            ckernel::mailbox_write(kPack, bounded::kAbort);
            return;
        }
        const volatile uint32_t* h = (
            reinterpret_cast<const volatile uint32_t*>(
                get_tile_l1_byte_address(kCbIn, 0)
            )
        );
        asm volatile("fence" ::: "memory");
        const uint32_t first = h[kDirFirst], count = h[kDirCount];
        ckernel::mailbox_write(kPack, first);
        ckernel::mailbox_write(kPack, count);
        cb_pop_front(kCbIn, 1);
        const uint32_t pages = seg == kSegRecip ? 0 : count;
        for (uint32_t i = 0; i < pages; ++i) {
            if (!bounded::front(kCbIn, 1)) {
                ckernel::mailbox_write(kPack, bounded::kAbort);
                return;
            }
            const bool ok = hand(get_tile_l1_byte_address(kCbIn, 0));
            cb_pop_front(kCbIn, 1);
            if (!ok) {
                return;
            }
        }
        for (bool more = seg == kSegTreelets; more;) {
            if (!bounded::front(kCbWork, 1)) {
                ckernel::mailbox_write(kPack, bounded::kAbort);
                return;
            }
            const uint32_t w = get_tile_l1_byte_address(kCbWork, 0);
            asm volatile("fence" ::: "memory");
            more =
                (reinterpret_cast<const volatile uint32_t*>(w)[kWorkCount] != kWorkEnd);
            const bool ok = hand(w);
            cb_pop_front(kCbWork, 1);
            more = more && ok;
        }
    }

}  // namespace
#endif

void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    cp.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg == kSegLoad) {
            // The tree stays in its buffer; the pack thread keeps its address.
            const uint32_t pages = (r.a1 + kTreePageBytes - 1) / kTreePageBytes;
#ifdef TRISC_UNPACK
            const bool ok = bounded::front(kCbTree, pages);
            ckernel::mailbox_write(
                ckernel::ThreadId::PackThreadId,
                ok ? get_tile_l1_byte_address(kCbTree, 0) : bounded::kAbort
            );
#endif
#ifdef TRISC_PACK
            const uint32_t t = mbox_read();
            g_tree =
                (t == bounded::kAbort ? nullptr : reinterpret_cast<const uint32_t*>(t));
#endif
            return;
        }
        if (seg != kSegTrace && seg != kSegArith && seg != kSegRecip &&
            seg != kSegTreelets && seg != kSegProbe && seg != kSegMath) {
            return;
        }
#ifdef TRISC_UNPACK
        unpack_pages(seg);
#endif
#ifdef TRISC_PACK
        walk::Prof pf;
        const uint32_t t0 = wall();
        uint32_t first = 0, count = 0;
        if (!mail(first, pf) || !mail(count, pf)) {
            pf.total = wall() - t0;
            profile(pf);
            return;
        }
        setup();
        const bool trees = seg == kSegTrace || seg == kSegTreelets || seg == kSegProbe;
        if (trees && g_tree == nullptr) {
            pf.error |= kErrTree;
        } else if (seg == kSegTrace) {
            trace_list(count, r.a3 == kModeAny, pf);
        } else if (seg == kSegTreelets) {
            // Batches until the reader's end page says no packet is left deferred.
            const bool any = r.a3 == kModeAny;
            uint32_t done = 0, end = kWorkNext;
            bool ok = true;
            for (uint32_t k = 0; ok && end == kWorkNext && k <= count; ++k) {
                uint32_t a = wall();
                ok = top_walks(count, any, r.a6, k == 0, pf);
                uint32_t b = wall();
                pf.phase1 += b - a;
                end = ok ? treelet_visits(any, done, pf) : 0;
                ok = end != 0;
                pf.phase2 += wall() - b;
            }
            const uint32_t a = wall();
            ok = ok && end == kWorkEnd && results_out(count, pf);
            pf.phase3 = wall() - a;
            (void)ok;
        } else if (seg == kSegProbe) {
            probe(count, r.a3, r.a5, pf);
        } else if (seg == kSegArith) {
            arith_list(count, pf);
        } else if (seg == kSegMath) {
            math_list(count, pf);
        } else {
            recip_list(first, count, r.a3, pf);
        }
        finish();
        pf.total = wall() - t0;
        profile(pf);
#endif
    });
}
