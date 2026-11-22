// Ray tracing compute: unpack hands input pages to pack, which walks packets on the
// vector unit; at treelet visits unpack and math load states, the packer stores them.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "timed.hpp"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/movement/km_chain.hpp"
#include "raytrace.h"
#ifdef TRISC_PACK
#include "divsqrt.hpp"
#include "shade.hpp"
#include "walk.hpp"
#endif

#ifdef TRISC_PACK
namespace {

    using namespace rt;

    const uint32_t* g_tree = nullptr;  // the tree image, after LOAD

    // Profile error bits (besides the walk's 1 and 2): what gave up.
    constexpr uint32_t kErrorUnpack = 4;   // unpack gave up and said so
    constexpr uint32_t kErrorMail = 8;     // no word from unpack in time
    constexpr uint32_t kErrorOut = 16;     // no output page in time
    constexpr uint32_t kErrorSignal = 32;  // no signal page in time
    constexpr uint32_t kErrorRoom = 64;    // a raw buffer too small
    constexpr uint32_t kErrorTree = 128;   // no tree loaded

    // A word from unpack, kAbort if none came in time.
    uint32_t mbox_read() {
        uint32_t value = timed::kAbort;
        return (
            timed::mail(ckernel::ThreadId::UnpackThreadId, value)
                ? value
                : timed::kAbort
        );
    }

    void mbox_release() {
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
    }

    uint32_t out_ptr() {
        return get_local_cb_interface(kCbOut).fifo_wr_ptr << 4;
    }

    // Stores to L1 finished (a load of the last word written, used), then pushed.
    void push_out(uint32_t address, uint32_t words) {
        asm volatile("fence" ::: "memory");
        const uint32_t last = reinterpret_cast<volatile uint32_t*>(address)[words - 1];
        asm volatile("and x0, x0, %0" : : "r"(last) : "memory");
        cb_push_back(kCbOut, 1);
    }

    // A word from unpack; on kAbort the reason goes to the profile.
    bool mail(uint32_t& value, walk::Profile& profile) {
        if (!timed::mail(ckernel::ThreadId::UnpackThreadId, value)) {
            profile.error |= kErrorMail;
            return false;
        }
        profile.error |= value == timed::kAbort ? kErrorUnpack : 0;
        return value != timed::kAbort;
    }

    // The next input page from the unpack thread (nullptr: gave up), its wait counted.
    const uint32_t* next_page(walk::Profile& profile) {
        const uint32_t started = wall();
        uint32_t address = 0;
        const bool arrived = mail(address, profile);
        asm volatile("fence" ::: "memory");
        profile.wait_in += wall() - started;
        return arrived ? reinterpret_cast<const uint32_t*>(address) : nullptr;
    }

    // A free output page (0: gave up).
    uint32_t reserve_out(walk::Profile& profile) {
        const uint32_t started = wall();
        const bool reserved = timed::back(kCbOut, 1);
        profile.wait_out += wall() - started;
        profile.error |= reserved ? 0 : kErrorOut;
        return reserved ? out_ptr() : 0;
    }

    bool trace_list(uint32_t packets, bool any, walk::Profile& profile) {
        const uint32_t* tree = g_tree;
        const uint32_t* nodes = tree + tree[kTreeNodeOff];
        const uint32_t* order = tree + tree[kTreeOrderOff];
        const uint32_t* tris = tree + tree[kTreeTriOff];
        for (uint32_t packet = 0; packet < packets; ++packet) {
            const uint32_t* page = next_page(profile);
            if (page == nullptr) {
                return false;
            }
            uint32_t phase_started = wall();
            const uint32_t packet_started = phase_started;
            walk::load_packet(page);
            const uint32_t signs = page[kVec * kPkHead + kHeadSigns];
            mbox_release();
            uint32_t phase_ended = wall();
            profile.load += phase_ended - phase_started;
            walk::packet(nodes, order, tris, signs, any, profile);
            phase_started = wall();
            tensix_sync();
            phase_ended = wall();
            profile.drain += phase_ended - phase_started;
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            phase_started = wall();
            push_out(
                destination,
                walk::store_hits(
                    reinterpret_cast<uint32_t*>(destination),
                    phase_ended - packet_started,
                    any
                )
            );
            profile.out += wall() - phase_started;
            profile.packets++;
        }
        return true;
    }

    uint32_t* raw(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4);
    }

    // A word other cores read: the store drained (a load of it, used), then a fence.
    void publish(volatile uint32_t* target, uint32_t value) {
        *target = value;
        const uint32_t readback = *target;
        asm volatile("and x0, x0, %0" : : "r"(readback) : "memory");
        asm volatile("fence" ::: "memory");
    }

    constexpr uint32_t kErrorList = 256;  // a packet's list did not fit an empty pool
    constexpr uint32_t kErrorDst = 512;   // no state in Dst in time

    // Packet i's state image and its header (after every image of the command).
    uint32_t* image_of(uint32_t* states, uint32_t packet) {
        return states + kStTileBytes / 4 * packet;
    }

    uint32_t* head_of(uint32_t* states, uint32_t packets, uint32_t packet) {
        return states + kStTileBytes / 4 * packets + kStHeadWords * packet;
    }

    // Math has put a state into Dst (its section committed), within the timeout.
    bool dst_ready(walk::Profile& profile) {
        const bool ready = timed::poll([] {
            return ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) > 0;
        });
        profile.error |= ready ? 0 : kErrorDst;
        if (ready) {
            tile_regs_wait();
        }
        return ready;
    }

    // Packet id's state from Dst tile 0 to its image; Dst back to math, not cleared
    // (the walk's other slots stay).
    void state_out(uint32_t id) {
        tensix_sync();
        pack_tile<true>(0, kCbState, id);
        tensix_sync();
        ::_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>();
    }

    // Top walks to the lists, states to L1, then the reader is told: first every packet
    // from its page, then those deferred (list did not fit) from their state.
    bool top_walks(
        uint32_t packets,
        bool any,
        uint32_t capacity,
        bool first_batch,
        walk::Profile& profile
    ) {
        const uint32_t list_bytes = 4 * (kLsHead + 2 * packets) + 2 * capacity;
        if (
            timed::room(kCbState) < packets * (kStTileBytes + 4 * kStHeadWords) ||
            timed::room(kCbLists) < list_bytes
        ) {
            profile.error |= kErrorRoom;
            return false;
        }
        const uint32_t* tree = g_tree;
        const uint32_t* nodes = tree + tree[kTreeNodeOff];
        const uint32_t* order = tree + tree[kTreeOrderOff];
        uint32_t* lists = raw(kCbLists);
        uint32_t* states = raw(kCbState);
        uint16_t* pool = reinterpret_cast<uint16_t*>(lists + kLsHead + 2 * packets);
        uint32_t used = 0, deferred = 0, admitted = 0;
        for (uint32_t packet = 0; packet < packets; ++packet) {
            uint32_t* image = image_of(states, packet);
            uint32_t* head = head_of(states, packets, packet);
            lists[kLsHead + 2 * packet] = used;
            lists[kLsHead + 2 * packet + 1] = 0;
            if (!first_batch && head[kStDeferred] == 0) {
                continue;
            }
            uint32_t phase_started = wall();
            const uint32_t packet_started = phase_started;
            uint32_t signs = 0;
            const uint32_t* page = nullptr;
            if (first_batch) {
                page = next_page(profile);
                if (page == nullptr) {
                    return false;
                }
                walk::load_packet(page);
                signs = page[kVec * kPkHead + kHeadSigns];
            } else {
                signs = head[kStSigns];
                walk::load_image(image);
            }
            profile.load += wall() - phase_started;
            const uint32_t list_start = used;
            walk::packet(nodes, order, nullptr, signs, any, profile, [&](uint32_t id) {
                if (used < capacity) {
                    pool[used] = id;
                }
                used++;
            });
            const bool fits = used <= capacity;
            used = fits ? used : list_start;
            if (!fits) {
                // Back to the state before the walk, so its counts are not taken twice.
                if (first_batch) {
                    walk::load_packet(page);
                } else {
                    walk::load_image(image);
                }
            }
            if (first_batch) {
                mbox_release();
            }
            lists[kLsHead + 2 * packet + 1] = used - list_start;
            deferred += fits ? 0 : 1;
            admitted += fits ? 1 : 0;
            profile.deferred += fits ? 0 : 1;
            phase_started = wall();
            tensix_sync();
            walk::store_image(image);
            head[kStSigns] = signs;
            head[kStDeferred] = fits ? 0 : 1;
            const uint32_t ended = wall();
            head[kStCycles] = (
                (first_batch ? 0 : head[kStCycles]) + ended - packet_started
            );
            profile.st_store += ended - phase_started;
            profile.packets += first_batch ? 1 : 0;
        }
        if (deferred > 0 && admitted == 0) {
            // Cannot happen when the lists hold every treelet (the host sees to it).
            profile.error |= kErrorList;
            deferred = 0;
        }
        lists[kLsMore] = deferred;
        if (first_batch) {
            publish(lists + kLsDone, 0);
        } else {
            publish(lists + kLsMore, deferred);
        }
        profile.batches++;
        if (!timed::back(kCbSignal, 1)) {
            profile.error |= kErrorSignal;
            return false;
        }
        cb_push_back(kCbSignal, 1);
        return true;
    }

    // Work pages up to a batch's end page (returns its word, 0: gave up): each listed
    // packet walks the page's treelet; finished pages (over batches) go to the lists.
    uint32_t treelet_visits(
        uint32_t packets, bool any, uint32_t& done, walk::Profile& profile
    ) {
        uint32_t* lists = raw(kCbLists);
        uint32_t* states = raw(kCbState);
        for (;;) {
            const uint32_t started = wall();
            const uint32_t* work = next_page(profile);
            profile.wait_work += wall() - started;
            if (work == nullptr) {
                return 0;
            }
            const uint32_t work_packets = work[kWorkCount];
            if (work_packets == kWorkEnd || work_packets == kWorkNext) {
                mbox_release();
                return work_packets;
            }
            const uint32_t* tree = reinterpret_cast<const uint32_t*>(work[kWorkSlot]);
            const uint32_t* nodes = tree + tree[kTreeNodeOff];
            const uint32_t* order = tree + tree[kTreeOrderOff];
            const uint32_t* tris = tree + tree[kTreeTriOff];
            const uint16_t* ids = reinterpret_cast<const uint16_t*>(work + kWorkIds);
            // Unpack and math put each listed packet's state into Dst tile 0.
            for (uint32_t entry = 0; entry < work_packets; ++entry) {
                uint32_t* head = head_of(states, packets, ids[entry]);
                const uint32_t visit_started = wall();
                if (!dst_ready(profile)) {
                    return 0;
                }
                uint32_t mark = wall();
                profile.st_load += mark - visit_started;
                walk::packet(nodes, order, tris, head[kStSigns], any, profile);
                mark = wall();
                state_out(ids[entry]);
                const uint32_t stored = wall();
                profile.st_store += stored - mark;
                head[kStCycles] += stored - visit_started;
                profile.visits++;
            }
            publish(lists + kLsDone, ++done);
            profile.items++;
            mbox_release();
        }
    }

    // The results of every packet (state vectors best to leaves, then the attributes of
    // a nearest hit) to hit pages.
    bool results_out(uint32_t packets, bool any, walk::Profile& profile) {
        const uint32_t* states = raw(kCbState);
        for (uint32_t packet = 0; packet < packets; ++packet) {
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            const uint32_t started = wall();
            uint32_t* hit_page = reinterpret_cast<uint32_t*>(destination);
            const uint32_t* image = image_of(const_cast<uint32_t*>(states), packet);
            for (uint32_t index = 0; index < kHitCycles * kVec; ++index) {
                hit_page[index] = (
                    walk::image_lane(image, kSlotBest + index / kVec, index % kVec)
                );
            }
            walk::store_cycles(
                hit_page,
                head_of(const_cast<uint32_t*>(states), packets, packet)[kStCycles]
            );
            for (uint32_t index = 0; !any && index < kTriAttrWords * kVec; ++index) {
                hit_page[kVec * kHitN + index] = (
                    walk::image_lane(
                        image, kSlotAttributes + index / kVec, index % kVec
                    )
                );
            }
            push_out(destination, (any ? kHitAnyVectors : kHitVectors) * kVec);
            profile.out += wall() - started;
        }
        return true;
    }

    // Slab test with one fused multiply-add per plane (b inv - o inv, o inv in slots
    // oinv) and no exit scale: not the reference's arithmetic, for sizing only.
    template <typename Box>
    inline void fast_box(
        Box box, uint32_t origin_inverse, uint32_t inverse, uint32_t best
    ) {
        TTI_SFPLOADI(L4, kFloatB, 0);
        TTI_SFPLOADI(L5, kFloatB, 0x7F80);
#pragma GCC unroll 3
        for (uint32_t axis = 0; axis < 3; ++axis) {
            box(L0, axis);
            box(L1, axis + 3);
            load(L2, origin_inverse + axis);
            load(L3, inverse + axis);
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
    constexpr uint32_t kProbeBox = 100, kProbeBoxes = 24;

    // One node visit of the probe in the given form (children c0, c1 of node n).
    inline void probe_visit(
        uint32_t form,
        const uint32_t* nodes,
        const uint32_t* order,
        uint32_t node_index,
        uint32_t visit,
        uint32_t signs,
        volatile uint32_t* stack
    ) {
        const uint32_t* node = nodes + kNodeWords * node_index;
        const uint32_t first_child = node[6], second_child = node[7];
        if (form == kProbeData) {
            const uint32_t first_box = (2 * visit) % kProbeBoxes;
            for (uint32_t child = 0; child < 2; ++child) {
                box_test(
                    [first_box, child](uint32_t lreg, uint32_t index) {
                        load(lreg, kProbeBox + 6 * (first_box + child) + index);
                    },
                    kSlotOrigin,
                    kSlotInverse,
                    kSlotBest
                );
                store(L7, kSlotStack + child);
            }
            return;
        }
        if (form == kProbeFast) {
            fast_box(
                walk::box_immediate(nodes + kNodeWords * first_child),
                kSlotPVector,
                kSlotInverse,
                kSlotBest
            );
            store(L7, kSlotStack);
            fast_box(
                walk::box_immediate(nodes + kNodeWords * second_child),
                kSlotPVector,
                kSlotInverse,
                kSlotBest
            );
            store(L7, kSlotStack + 1);
            return;
        }
        uint32_t slot = kSlotStack, near = first_child, far = second_child;
        if (form != kProbeImmediate) {
            slot = kSlotStack + 2 * (visit & 15);
            walk::pop_lanes(slot);
            stack[visit & 31] = walk::any_lane();
            if (form == kProbeSync) {
                return;
            }
        }
        if (form == kProbeStack) {
            const uint32_t order_word = order[node_index];
            const bool second = ((signs >> (order_word & 3)) ^ (order_word >> 2)) & 1;
            near = second ? second_child : first_child;
            far = second ? first_child : second_child;
            count(kSlotNodes, 2);
            stack[(visit + 1) & 31] = far;
            stack[(visit + 2) & 31] = near;
        }
        box_test(
            walk::box_immediate(nodes + kNodeWords * far),
            kSlotOrigin,
            kSlotInverse,
            kSlotBest
        );
        store(L7, slot);
        box_test(
            walk::box_immediate(nodes + kNodeWords * near),
            kSlotOrigin,
            kSlotInverse,
            kSlotBest
        );
        store(L7, slot + 1);
    }

    // Each packet against both children of every inner node, `passes` times; the
    // loop's cycles in pf.box, node visits in pf.inner; the hits as for a trace.
    bool probe(
        uint32_t packets, uint32_t form, uint32_t passes, walk::Profile& profile
    ) {
        const uint32_t* tree = g_tree;
        const uint32_t* nodes = tree + tree[kTreeNodeOff];
        const uint32_t* order = tree + tree[kTreeOrderOff];
        const uint32_t node_count = tree[kTreeNodes];
        volatile uint32_t stack[32];
        for (uint32_t packet = 0; packet < packets; ++packet) {
            const uint32_t* page = next_page(profile);
            if (page == nullptr) {
                return false;
            }
            walk::load_packet(page);
            const uint32_t signs = page[kVec * kPkHead + kHeadSigns];
            mbox_release();
            all_lanes();
            for (uint32_t axis = 0; axis < 3; ++axis) {
                load(L0, kSlotOrigin + axis);
                load(L1, kSlotInverse + axis);
                mul(L0, L1, L0);
                store(L0, kSlotPVector + axis);
            }
            tensix_sync();
            const uint32_t* inner = nodes + kNodeWords * 0;
            for (uint32_t box_index = 0; box_index < kProbeBoxes; ++box_index) {
                const uint32_t* box = (
                    nodes + kNodeWords * inner[6] + kNodeWords * box_index
                );
                for (uint32_t box_word = 0; box_word < 6; ++box_word) {
                    fill_slot(kProbeBox + 6 * box_index + box_word, box[box_word]);
                }
            }
            rt::order(word(kProbeBox + 6 * kProbeBoxes - 1) + 62);
            const uint32_t started = wall();
            uint32_t visits = 0;
            for (uint32_t pass = 0; pass < passes; ++pass) {
                for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
                    const uint32_t first_child_word = (
                        nodes[kNodeWords * node_index + 6]
                    );
                    if ((first_child_word & (kLeafBit | kTreeletBit)) == 0) {
                        probe_visit(
                            form, nodes, order, node_index, visits++, signs, stack
                        );
                    }
                }
            }
            tensix_sync();
            profile.box += wall() - started;
            profile.inner += visits;
            profile.pops += passes;
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            push_out(
                destination,
                walk::store_hits(reinterpret_cast<uint32_t*>(destination), 0, true)
            );
            profile.packets++;
        }
        return true;
    }

    // Arithmetic check slots: inputs from 40, outputs from 70.
    constexpr uint32_t kArithInput = 40, kArithOutput = 70;

    bool arith_list(uint32_t packets, walk::Profile& profile) {
        for (uint32_t page_index = 0; page_index < packets; ++page_index) {
            const uint32_t* page = next_page(profile);
            if (page == nullptr) {
                return false;
            }
            for (uint32_t vector = 0; vector < kArInVectors; ++vector) {
                to_slot(kArithInput + vector, page + kVec * vector);
            }
            to_slot(kSlotBest, page + kVec * kArBest);
            to_slot(kSlotTriangle, page + kVec * kArCur);
            fill_slot(kSlotU, 0);
            fill_slot(kSlotV, 0);
            order(word(kSlotV) + 62);
            mbox_release();
            all_lanes();
            box_test(
                [](uint32_t lreg, uint32_t index) {
                    load(lreg, kArithInput + kArBoxMin + index);
                },
                kArithInput + kArO,
                kArithInput + kArInv,
                kArithInput + kArBest
            );
            store(L4, kArithOutput + kArEntry);
            store(L5, kArithOutput + kArExit);
            store(L7, kArithOutput + kArStack);
            tri_test(
                [](uint32_t lreg, uint32_t index) {
                    load(lreg, kArithInput + kArV0 + index);
                },
                kArithInput + kArO,
                kArithInput + kArD,
                kSlotBest,
                kSlotTriangle,
                kSlotU,
                kSlotV,
                false
            );
            store(L3, kArithOutput + kArU);
            store(L0, kArithOutput + kArV);
            store(L1, kArithOutput + kArT);
            constexpr uint32_t sources[6] = {
                kSlotDeterminant,
                kSlotInverseDeterminant,
                kSlotBest,
                kSlotTriangle,
                kSlotU,
                kSlotV
            };
            constexpr uint32_t destinations[6] = {
                kArDet, kArDetInv, kArNewBest, kArNewTri, kArNewU, kArNewV
            };
            for (uint32_t vector = 0; vector < 6; ++vector) {
                load(L0, sources[vector]);
                store(L0, kArithOutput + destinations[vector]);
            }
            tensix_sync();
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            uint32_t* output = reinterpret_cast<uint32_t*>(destination);
            for (uint32_t vector = 0; vector < kArOutVectors; ++vector) {
                from_slot(kArithOutput + vector, output + kVec * vector);
            }
            push_out(destination, kArOutVectors * kVec);
            profile.packets++;
        }
        return true;
    }

    // Division and square root of each page's lanes: a / b and sqrt(x).
    bool math_list(uint32_t packets, walk::Profile& profile) {
        for (uint32_t page_index = 0; page_index < packets; ++page_index) {
            const uint32_t* page = next_page(profile);
            if (page == nullptr) {
                return false;
            }
            for (uint32_t vector = 0; vector < kMaInVectors; ++vector) {
                to_slot(kArithInput + vector, page + kVec * vector);
            }
            order(word(kArithInput + kMaInVectors - 1) + 62);
            mbox_release();
            all_lanes();
            load(L0, kArithInput + kMaA);
            load(L1, kArithInput + kMaB);
            divide();
            store(L2, kArithOutput + kMaQ);
            load(L0, kArithInput + kMaX);
            square_root();
            store(L1, kArithOutput + kMaS);
            tensix_sync();
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            uint32_t* output = reinterpret_cast<uint32_t*>(destination);
            for (uint32_t vector = 0; vector < kMaOutVectors; ++vector) {
                from_slot(kArithOutput + vector, output + kVec * vector);
            }
            push_out(destination, kMaOutVectors * kVec);
            profile.packets++;
        }
        return true;
    }

    // Reciprocals of value = exponent | mantissa, mantissa = 1024 * page + 32 * row +
    // lane.
    bool recip_list(
        uint32_t first_page, uint32_t pages, uint32_t exponent, walk::Profile& profile
    ) {
        constexpr uint32_t kLaneSlot = 100;
        all_lanes();
        TTI_SFPSHFT(0xFFF, LANE, L0, 5);  // lane = (2 lane) >> 1
        store(L0, kLaneSlot);
        for (uint32_t page = 0; page < pages; ++page) {
            for (uint32_t row = 0; row < 32; ++row) {
                const uint32_t first_input = (
                    (exponent << 23) + kRecipPerPage * (first_page + page) + 32 * row
                );
                immediate(L1, first_input);
                load(L0, kLaneSlot);
                TT_SFPIADD(0, L1, L0, kCcNone);
                recip(L0, L3, L1, L2, L4, L5);
                store(L3, row);
            }
            tensix_sync();
            const uint32_t destination = reserve_out(profile);
            if (destination == 0) {
                return false;
            }
            uint32_t* output = reinterpret_cast<uint32_t*>(destination);
            for (uint32_t row = 0; row < 32; ++row) {
                from_slot(row, output + kVec * row);
            }
            push_out(destination, kRecipPerPage);
            profile.packets++;
        }
        return true;
    }

    // Header of a packet page in L1: direction signs over the lanes with a ray (best
    // not -1), the first pixel.
    void packet_head(uint32_t* page, uint32_t pixel) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(kVec);
        const vbool4_t ray = (
            __riscv_vmsne_vx_u32m8_b4(
                __riscv_vle32_v_u32m8(page + kVec * kPkBest, vl), shade::kMinusOne, vl
            )
        );
        const uint32_t ray_count = __riscv_vcpop_m_b4(ray, vl);
        uint32_t signs = 0;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const vint32m8_t direction = (
                __riscv_vle32_v_i32m8(
                    reinterpret_cast<const int32_t*>(page + kVec * (kPkD + axis)), vl
                )
            );
            const vbool4_t negative = (
                __riscv_vmand_mm_b4(
                    ray, __riscv_vmslt_vx_i32m8_b4(direction, 0, vl), vl
                )
            );
            signs |= (
                (2 * __riscv_vcpop_m_b4(negative, vl) > ray_count ? 1u : 0u) << axis
            );
        }
        uint32_t* header = page + kVec * kPkHead;
        __riscv_vse32_v_u32m8(header, __riscv_vmv_v_x_u32m8(0, vl), vl);
        header[kHeadSigns] = signs;
        header[kHeadPixel] = pixel;
#else
        uint32_t ray_count = 0, negative[3] = {0, 0, 0};
        for (uint32_t lane = 0; lane < kVec; ++lane) {
            if (page[kVec * kPkBest + lane] == shade::kMinusOne) {
                continue;
            }
            ray_count++;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                negative[axis] += page[kVec * (kPkD + axis) + lane] >> 31;
            }
        }
        uint32_t* header = page + kVec * kPkHead;
        header[kHeadSigns] = 0;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            header[kHeadSigns] |= (2 * negative[axis] > ray_count ? 1u : 0u) << axis;
        }
        header[kHeadPixel] = pixel;
        for (uint32_t lane = 2; lane < kVec; ++lane) {
            header[lane] = 0;
        }
#endif
    }

    // Vectors of a stage page that go through Dst: a packet page's header stays in L1.
    constexpr uint32_t body(uint32_t vectors) {
        return vectors == kPkVectors ? kPkHead : vectors;
    }

    // A stage over the core's packets: the constant page first, then per packet its
    // input pages into Dst from shade::kInput, the stage, its output pages from kOut.
    bool stage_list(uint32_t handler, uint32_t packets, walk::Profile& profile) {
        const Stage stage = stage_of(handler);
        const uint32_t* constants_page = next_page(profile);
        if (constants_page == nullptr) {
            return false;
        }
        uint32_t constants[kCnWords];
        for (uint32_t index = 0; index < kCnWords; ++index) {
            constants[index] = constants_page[index];
        }
        for (uint32_t ramp = 0; ramp < 3; ++ramp) {
            to_slot(shade::kRamp + ramp, constants_page + kVec * (kCnDx + ramp));
        }
        order(word(shade::kRamp + 2) + 62);
        mbox_release();
        for (uint32_t packet = 0; packet < packets; ++packet) {
            uint32_t slot = shade::kInput, pixel = 0;
            for (uint32_t stream = 0; stream < stage.inputs; ++stream) {
                const uint32_t* page = next_page(profile);
                if (page == nullptr) {
                    return false;
                }
                const uint32_t load_started = wall();
                uint32_t last = slot;
                for (
                    uint32_t vector = 0; vector < body(stage.vectors[stream]); ++vector
                ) {
                    if ((stage.used[stream] >> vector) & 1) {
                        to_slot(slot + vector, page + kVec * vector);
                        last = slot + vector;
                    }
                }
                if (stream == 0) {
                    pixel = (
                        handler == kHandlerGen
                            ? page[0]
                            : page[kVec * kPkHead + kHeadPixel]
                    );
                }
                order(word(last) + 62);
                slot += stage.vectors[stream];
                mbox_release();
                profile.load += wall() - load_started;
            }
            uint32_t stage_started = wall();
            all_lanes();
            if (handler == kHandlerGen) {
                shade::gen(
                    constants, pixel % constants[kCnWidth], pixel / constants[kCnWidth]
                );
            } else if (handler == kHandlerShade1) {
                shade::shade1(constants, pixel);
            } else if (handler == kHandlerShade2) {
                shade::shade2(constants);
            } else {
                shade::image();
            }
            tensix_sync();
            uint32_t stage_ended = wall();
            profile.box += stage_ended - stage_started;
            slot = shade::kOut;
            for (
                uint32_t stream = stage.inputs; stream < stage.inputs + stage.outputs;
                ++stream
            ) {
                const uint32_t destination = reserve_out(profile);
                if (destination == 0) {
                    return false;
                }
                stage_started = wall();
                uint32_t* output = reinterpret_cast<uint32_t*>(destination);
                const uint32_t vectors = stage.vectors[stream];
                for (uint32_t vector = 0; vector < body(vectors); ++vector) {
                    from_slot(slot + vector, output + kVec * vector);
                }
                if (vectors == kPkVectors) {
                    packet_head(output, pixel);
                }
                slot += vectors;
                push_out(destination, vectors * kVec);
                profile.out += wall() - stage_started;
            }
            profile.packets++;
        }
        return true;
    }

    // A TREELETS chunk (or pulled range) of `packets`: its batches, then the hit pages.
    // Inlined: its own frame would not fit the pack thread's stack.
    __attribute__((always_inline)) inline bool treelet_chunk(
        uint32_t packets, bool any, uint32_t list_entries, walk::Profile& profile
    ) {
        uint32_t done = 0, end = kWorkNext;
        bool ok = true;
        for (uint32_t batch = 0; ok && end == kWorkNext && batch <= packets; ++batch) {
            const uint32_t top_started = wall();
            ok = top_walks(packets, any, list_entries, batch == 0, profile);
            const uint32_t top_ended = wall();
            profile.phase1 += top_ended - top_started;
            end = ok ? treelet_visits(packets, any, done, profile) : 0;
            ok = end != 0;
            profile.phase2 += wall() - top_ended;
        }
        const uint32_t results_started = wall();
        ok = ok && end == kWorkEnd && results_out(packets, any, profile);
        profile.phase3 += wall() - results_started;
        return ok;
    }

    // A pulled range's page to the writer: its first packet and count (0: the end).
    bool range_out(uint32_t first_packet, uint32_t packets, walk::Profile& profile) {
        const uint32_t destination = reserve_out(profile);
        if (destination == 0) {
            return false;
        }
        volatile uint32_t* page = reinterpret_cast<volatile uint32_t*>(destination);
        page[kRangeFirst] = first_packet;
        page[kRangeCount] = packets;
        push_out(destination, 2);
        return true;
    }

    // The profile page: the pack thread's counters at words 0 to kProfileWords - 1;
    // with `block` (TREELETS) the cost block of its `costs` packets, even of none.
    void write_profile(
        const walk::Profile& profile,
        uint32_t first_packet = 0,
        uint32_t costs = 0,
        bool block = false
    ) {
        if (!timed::back(kCbOut, 1)) {
            return;
        }
        const uint32_t destination = out_ptr();
        volatile uint32_t* page = reinterpret_cast<volatile uint32_t*>(destination);
        const uint32_t* counters = reinterpret_cast<const uint32_t*>(&profile);
        for (uint32_t index = 0; index < 32; ++index) {
            page[index] = index < walk::kProfileWords ? counters[index] : 0;
        }
        if (!block) {
            push_out(destination, 32);
            return;
        }
        uint32_t* states = raw(kCbState);
        volatile uint32_t* cost = page + kProfCost;
        for (uint32_t index = 0; index < kCostHead; ++index) {
            cost[index] = 0;
        }
        const uint32_t listed = costs < kCostMax ? costs : kCostMax;
        cost[0] = first_packet;
        cost[1] = listed;
        for (uint32_t packet = 0; packet < listed; ++packet) {
            const uint32_t* image = image_of(states, packet);
            uint32_t hits = 0;
            for (uint32_t lane = 0; lane < kVec; ++lane) {
                hits += (
                    walk::image_lane(image, kSlotTriangle, lane) != 0xFFFFFFFFu ? 1 : 0
                );
            }
            cost[kCostHead + packet] = head_of(states, costs, packet)[kStCycles];
            cost[kCostHead + kCostMax + packet] = hits;
        }
        push_out(destination, kProfCost + kCostHead + kCostMax + listed);
    }

}  // namespace
#endif

#ifdef TRISC_UNPACK
namespace {

    // A page to the pack thread, then its release; false: pack did not release it.
    bool hand(uint32_t address) {
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, address);
        uint32_t released = 0;
        return timed::mail(ckernel::ThreadId::PackThreadId, released);
    }

    void treelet_states();

    // The reader's header page and the pages after it to pack, then the work pages;
    // pulled: the same for each range's header, until a header of count 0.
    void unpack_pages(uint32_t handler, bool pulled) {
        constexpr uint32_t kPack = ckernel::ThreadId::PackThreadId;
        constexpr uint32_t kMath = ckernel::ThreadId::MathThreadId;
        for (;;) {
            if (!timed::front(kCbIn, 1)) {
                ckernel::mailbox_write(kPack, timed::kAbort);
                return;
            }
            const volatile uint32_t*
                header = (
                    reinterpret_cast<const volatile uint32_t*>(
                        get_tile_l1_byte_address(kCbIn, 0)
                    )
                );
            asm volatile("fence" ::: "memory");
            const uint32_t first_packet = header[kDirFirst];
            const uint32_t packets = header[kDirCount];
            ckernel::mailbox_write(kPack, first_packet);
            ckernel::mailbox_write(kPack, packets);
            cb_pop_front(kCbIn, 1);
            if (pulled && packets == 0) {
                if (handler == kHandlerTreelets) {
                    ckernel::mailbox_write(kMath, timed::kAbort);
                }
                return;
            }
            const Stage stage = stage_of(handler);
            const uint32_t pages = (
                handler == kHandlerRecip
                    ? 0
                    : is_stage(handler)
                        ? 1 + packets * stage.inputs
                        : packets
            );
            for (uint32_t page = 0; page < pages; ++page) {
                if (!timed::front(kCbIn, 1)) {
                    ckernel::mailbox_write(kPack, timed::kAbort);
                    return;
                }
                const bool handed = hand(get_tile_l1_byte_address(kCbIn, 0));
                cb_pop_front(kCbIn, 1);
                if (!handed) {
                    return;
                }
            }
            if (handler == kHandlerTreelets) {
                if (pulled) {
                    ckernel::mailbox_write(kMath, 1);
                }
                copy_init(kCbState);
                treelet_states();
            }
            if (!pulled) {
                return;
            }
        }
    }

    // TREELETS: each work page to pack; with it, math the count and each listed
    // packet's state into Dst tile 0; math gets kAbort at the end.
    void treelet_states() {
        constexpr uint32_t kPack = ckernel::ThreadId::PackThreadId;
        constexpr uint32_t kMath = ckernel::ThreadId::MathThreadId;
        for (bool more = true; more;) {
            if (!timed::front(kCbWork, 1)) {
                ckernel::mailbox_write(kPack, timed::kAbort);
                break;
            }
            const uint32_t work_address = get_tile_l1_byte_address(kCbWork, 0);
            asm volatile("fence" ::: "memory");
            const volatile uint32_t* page = (
                reinterpret_cast<const volatile uint32_t*>(work_address)
            );
            const uint32_t work_packets = page[kWorkCount];
            more = work_packets != kWorkEnd;
            bool handed = true;
            if (
                work_packets == kWorkEnd || work_packets == kWorkNext ||
                work_packets > kWorkMax
            ) {
                handed = hand(work_address);
            } else {
                ckernel::mailbox_write(kPack, work_address);
                ckernel::mailbox_write(kMath, work_packets);
                const volatile uint16_t*
                    ids = reinterpret_cast<const volatile uint16_t*>(page + kWorkIds);
                for (uint32_t entry = 0; entry < work_packets; ++entry) {
                    tile_regs_acquire();
                    copy_tile(kCbState, ids[entry], 0);
                    tile_regs_commit();
                }
                uint32_t released = 0;
                handed = timed::mail(kPack, released);
            }
            cb_pop_front(kCbWork, 1);
            more = more && handed;
        }
        ckernel::mailbox_write(kMath, timed::kAbort);
    }

}  // namespace
#endif

#ifdef TRISC_MATH
namespace {

    // TREELETS on math: per count from unpack, each state's copy into Dst tile 0, the
    // section given to pack once it is free (with a timeout); kAbort ends it.
    void math_states() {
        copy_init(kCbState);
        for (uint32_t pages = 0; pages < (1u << 20); ++pages) {
            uint32_t work_packets = 0;
            if (
                !timed::mail(ckernel::ThreadId::UnpackThreadId, work_packets) ||
                work_packets == timed::kAbort
            ) {
                return;
            }
            for (uint32_t entry = 0; entry < work_packets; ++entry) {
                if (!timed::poll([] {
                    return ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) == 0;
                })) {
                    return;
                }
                tile_regs_acquire();
                copy_tile(kCbState, 0, 0);
                tile_regs_commit();
            }
        }
    }

}  // namespace
#endif

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbState, kCbOut);
    const auto step = [&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerLoad) {
            // The tree stays in its buffer; the pack thread keeps its address.
            const uint32_t tree_bytes = command.a1;
            const uint32_t pages = (tree_bytes + kTreePageBytes - 1) / kTreePageBytes;
#ifdef TRISC_UNPACK
            const bool arrived = timed::front(kCbTree, pages);
            ckernel::mailbox_write(
                ckernel::ThreadId::PackThreadId,
                arrived ? get_tile_l1_byte_address(kCbTree, 0) : timed::kAbort
            );
#endif
#ifdef TRISC_PACK
            const uint32_t tree_address = mbox_read();
            g_tree = (
                tree_address == timed::kAbort
                    ? nullptr
                    : reinterpret_cast<const uint32_t*>(tree_address)
            );
#endif
            return;
        }
        if (
            handler != kHandlerTrace && handler != kHandlerArith &&
            handler != kHandlerRecip && handler != kHandlerTreelets &&
            handler != kHandlerProbe && handler != kHandlerMath && !is_stage(handler)
        ) {
            return;
        }
        const bool pulled_command = (
            (handler == kHandlerTrace || handler == kHandlerTreelets) &&
            (command.a3 & kModePulled) != 0
        );
#ifdef TRISC_UNPACK
        unpack_pages(handler, pulled_command);
#endif
#ifdef TRISC_MATH
        if (handler == kHandlerTreelets && pulled_command) {
            // A range's states after each 1 from unpack; kAbort ends the command.
            for (uint32_t range = 0; range < (1u << 20); ++range) {
                uint32_t word = 0;
                if (
                    !timed::mail(ckernel::ThreadId::UnpackThreadId, word) || word != 1
                ) {
                    break;
                }
                math_states();
            }
        } else if (handler == kHandlerTreelets) {
            math_states();
        }
#endif
#ifdef TRISC_PACK
        walk::Profile profile;
        const uint32_t started = wall();
        uint32_t first_packet = 0, packets = 0;
        if (!mail(first_packet, profile) || !mail(packets, profile)) {
            profile.total = wall() - started;
            write_profile(profile);
            return;
        }
        setup();
        const bool trees = (
            handler == kHandlerTrace || handler == kHandlerTreelets ||
            handler == kHandlerProbe
        );
        const bool pulled = pulled_command;
        const bool any = (command.a3 & kModeAny) != 0;
        if (trees && g_tree == nullptr) {
            profile.error |= kErrorTree;
        } else if (pulled) {
            // Each range as a chunk, its range page to the writer first; then the end.
            const uint32_t list_entries = command.a6;
            bool ok = true;
            while (ok && packets != 0) {
                ok = (
                    range_out(first_packet, packets, profile) &&
                    (
                        handler == kHandlerTrace
                            ? trace_list(packets, any, profile)
                            : treelet_chunk(packets, any, list_entries, profile)
                    ) &&
                    mail(first_packet, profile) && mail(packets, profile)
                );
            }
            ok = ok && range_out(first_packet, 0, profile);
            (void)ok;
        } else if (handler == kHandlerTrace) {
            trace_list(packets, any, profile);
        } else if (handler == kHandlerTreelets) {
            const uint32_t list_entries = command.a6;
            treelet_chunk(packets, any, list_entries, profile);
        } else if (handler == kHandlerProbe) {
            const uint32_t form = command.a3;
            const uint32_t passes = command.a5;
            probe(packets, form, passes, profile);
        } else if (handler == kHandlerArith) {
            arith_list(packets, profile);
        } else if (handler == kHandlerMath) {
            math_list(packets, profile);
        } else if (is_stage(handler)) {
            stage_list(handler, packets, profile);
        } else {
            const uint32_t exponent_field = command.a3;
            recip_list(first_packet, packets, exponent_field, profile);
        }
        finish();
        profile.total = wall() - started;
        write_profile(
            profile, first_packet, packets, handler == kHandlerTreelets && !pulled
        );
#endif
    };
    compute.run([&](const km_command_t& command) {
        if (is_chained(km::handler(command)) && command.a4 == kChainMode) {
            km::chain_compute(compute, command, step);
            return;
        }
        step(command);
    });
}
