// The wide-node probe: a copy of the ray tracer's PROBE handler for one ray (or a few)
// against many child boxes; the pack thread drives the vector unit, as the walk does.
#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "probe.h"
#ifdef TRISC_PACK
#include "../../../projects/raytrace/kernels/sfpu.hpp"
#endif

#ifdef TRISC_PACK
namespace {

    using namespace rt;

    // Dst slots of the probe: the child boxes (Dst tile 0 for the unpacker), the rays,
    // the entry distances.
    constexpr uint32_t kSlotBox = 0;   // six slots
    constexpr uint32_t kSlotRay = 40;  // origin, reciprocal direction, best
    constexpr uint32_t kSlotEntry = 48;

    // `count` words of L1 at `source` into lanes first to first + count - 1 of `slot`.
    inline void to_lanes(
        uint32_t slot, uint32_t first, const uint32_t* source, uint32_t count
    ) {
        const size_t vl = __riscv_vsetvl_e32m8(count);
        __riscv_vsse32_v_u32m8(
            const_cast<uint32_t*>(win() + word(slot) + 2 * first),
            8,
            __riscv_vle32_v_u32m8(source, vl),
            vl
        );
    }

    // `value` into lanes first to first + count - 1 of `slot`.
    inline void fill_lanes(
        uint32_t slot, uint32_t first, uint32_t value, uint32_t count
    ) {
        const size_t vl = __riscv_vsetvl_e32m8(count);
        __riscv_vsse32_v_u32m8(
            const_cast<uint32_t*>(win() + word(slot) + 2 * first),
            8,
            __riscv_vmv_v_x_u32m8(value, vl),
            vl
        );
    }

    // A ray's seven words into the lanes of its group, from the core.
    inline void ray_to_lanes(const uint32_t* ray, uint32_t group, uint32_t children) {
        for (uint32_t index = 0; index < 7; ++index) {
            fill_lanes(kSlotRay + index, group * children, ray[index], children);
        }
    }

    // One ray in every lane, from the vector unit's immediates.
    inline void ray_immediate(const uint32_t* ray) {
        for (uint32_t index = 0; index < 7; ++index) {
            immediate(L0, ray[index]);
            store(L0, kSlotRay + index);
        }
    }

    // Each group's node records into its lanes: one copy per group and box word.
    inline void boxes_by_group(
        const uint32_t* nodes, const uint32_t* entry, uint32_t groups, uint32_t children
    ) {
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t* node = nodes + entry[3 * group];
            for (uint32_t index = 0; index < 6; ++index) {
                to_lanes(
                    kSlotBox + index,
                    group * children,
                    node + index * children,
                    children
                );
            }
        }
        rt::order(word(kSlotBox + 5) + 2 * (kLanes - 1));
    }

    // The same with one gathered load of every lane per box word.
    inline void boxes_gathered(
        const uint32_t* nodes, const uint32_t* entry, uint32_t groups, uint32_t children
    ) {
        uint32_t offsets[kLanes];
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint32_t group = lane / children;
            offsets[lane] = 4 * (entry[3 * group] + lane - group * children);
        }
        const size_t vl = __riscv_vsetvl_e32m8(kLanes);
        const vuint32m8_t lanes = __riscv_vle32_v_u32m8(offsets, vl);
        for (uint32_t index = 0; index < 6; ++index) {
            const vuint32m8_t at = (
                __riscv_vadd_vx_u32m8(lanes, 4 * index * children, vl)
            );
            __riscv_vsse32_v_u32m8(
                const_cast<uint32_t*>(win() + word(kSlotBox + index)),
                8,
                __riscv_vluxei32_v_u32m8(nodes, at, vl),
                vl
            );
        }
        rt::order(word(kSlotBox + 5) + 2 * (kLanes - 1));
    }

    // Math has put the next node tile into Dst (its section committed), or gave up.
    inline bool tile_ready() {
        uint32_t waited = 0;
        while (ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) == 0) {
            if (++waited > (1u << 24)) {
                return false;
            }
        }
        tile_regs_wait();
        return true;
    }

    // Dst back to math, not cleared.
    inline void tile_done() {
        ::_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>();
    }

    struct Ray {
        uint32_t stack[kStackDepth];
        float entry[kStackDepth];
        uint32_t depth;
    };

    // The core's stack step for one group: its entered children (entry not +inf,
    // within `children`) pushed nearest on top, then one popped. Returns the pushes.
    inline uint32_t stack_step(
        Ray& ray, const uint32_t* entries, uint32_t first, uint32_t children
    ) {
        uint32_t order[kLanes];
        float key[kLanes];
        uint32_t count = 0;
        for (uint32_t child = 0; child < children; ++child) {
            const uint32_t bits = entries[first + child];
            if (bits == kInfBits) {
                continue;
            }
            float value;
            __builtin_memcpy(&value, &bits, 4);
            uint32_t at = count++;
            while (at > 0 && key[at - 1] < value) {
                key[at] = key[at - 1];
                order[at] = order[at - 1];
                --at;
            }
            key[at] = value;
            order[at] = child;
        }
        if (ray.depth + count >= kStackDepth) {
            ray.depth = 0;
        }
        for (uint32_t index = 0; index < count; ++index) {
            ray.stack[ray.depth] = order[index];
            ray.entry[ray.depth] = key[index];
            ++ray.depth;
        }
        if (ray.depth > 0) {
            --ray.depth;
        }
        return count;
    }

    // The probe: `ops` operations of `groups` rays, `passes` times, in `form`, boxes
    // loaded by `method`.
    void probe(
        uint32_t form,
        uint32_t method,
        uint32_t groups,
        uint32_t ops,
        uint32_t passes,
        const uint32_t* op_words,
        const uint32_t* nodes,
        const uint32_t* rays,
        uint32_t* entries_out,
        volatile uint32_t* results
    ) {
        const uint32_t children = kLanes / groups;
        uint32_t entries[kLanes];
        Ray stacks[kGroupsMax];
        uint32_t current[kGroupsMax];
        for (uint32_t group = 0; group < kGroupsMax; ++group) {
            stacks[group].depth = 0;
            current[group] = 0xFFFFFFFFu;
        }
        uint32_t visits = 0, entered = 0, ray_loads = 0, error = 0;
        tensix_sync();
        const uint32_t started = wall();
        for (uint32_t pass = 0; pass < passes && error == 0; ++pass) {
            for (uint32_t op = 0; op < ops; ++op) {
                const uint32_t* entry = op_words + kOpWords * op;
                if (form == kFormRay) {
                    for (uint32_t group = 0; group < groups; ++group) {
                        ray_to_lanes(
                            rays + kRayWords * entry[3 * group + 1], group, children
                        );
                        ++ray_loads;
                    }
                    continue;
                }
                if (groups > 1) {
                    for (uint32_t group = 0; group < groups; ++group) {
                        const uint32_t ray = entry[3 * group + 1];
                        if (ray != current[group]) {
                            ray_to_lanes(rays + kRayWords * ray, group, children);
                            current[group] = ray;
                            ++ray_loads;
                        }
                    }
                }
                if (method == kMethodUnpack) {
                    if (!tile_ready()) {
                        error = 2;
                        break;
                    }
                } else if (method == kMethodGather) {
                    boxes_gathered(nodes, entry, groups, children);
                } else {
                    boxes_by_group(nodes, entry, groups, children);
                }
                visits += groups;
                if (form != kFormLoad) {
                    if (groups == 1) {
                        ray_immediate(rays + kRayWords * entry[1]);
                    }
                    box_test(
                        [](uint32_t lreg, uint32_t index) {
                            load(lreg, kSlotBox + index);
                        },
                        kSlotRay,
                        kSlotRay + 3,
                        kSlotRay + 6
                    );
                    store(L7, kSlotEntry);
                    tensix_sync();
                }
                if (form == kFormOut || form == kFormStack || form == kFormCheck) {
                    uint32_t* destination = (
                        form == kFormCheck ? entries_out + kLanes * op : entries
                    );
                    from_slot(kSlotEntry, destination);
                }
                if (form == kFormStack) {
                    for (uint32_t group = 0; group < groups; ++group) {
                        entered += (
                            stack_step(
                                stacks[group],
                                entries,
                                group * children,
                                entry[3 * group + 2]
                            )
                        );
                    }
                }
                if (method == kMethodUnpack) {
                    tile_done();
                }
            }
        }
        tensix_sync();
        results[kResultCycles] = wall() - started;
        results[kResultOps] = ops * passes;
        results[kResultVisits] = visits;
        results[kResultEntered] = entered;
        results[kResultPushes] = entered;
        results[kResultRayLoads] = ray_loads;
        results[kResultError] = error;
    }

}  // namespace
#endif

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbNodes, kCbNodes);
    const uint32_t ops_address = compute.arg(0);
    const uint32_t nodes_address = compute.arg(1);
    const uint32_t rays_address = compute.arg(2);
    const uint32_t entries_address = compute.arg(3);
    const uint32_t results_address = compute.arg(4);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerWideProbe) {
            return;
        }
        const uint32_t form = command.a0;
        const uint32_t groups = command.a1;
        const uint32_t ops = command.a2;
        const uint32_t passes = command.a3;
        const uint32_t method = command.a4;
        const bool unpack = (
            method == kMethodUnpack && form != kFormRay && groups == 1 &&
            ops <= kTilesMax
        );
#ifdef TRISC_UNPACK
        // The node tiles the reader put in kCbNodes, in operation order, every pass.
        if (unpack) {
            cb_wait_front(kCbNodes, ops);
            copy_init(kCbNodes);
            for (uint32_t pass = 0; pass < passes; ++pass) {
                for (uint32_t op = 0; op < ops; ++op) {
                    copy_tile(kCbNodes, op, 0);
                }
            }
            cb_pop_front(kCbNodes, ops);
        }
#endif
#ifdef TRISC_MATH
        // Each node tile into Dst tile 0 once pack has given Dst back.
        if (unpack) {
            copy_init(kCbNodes);
            for (uint32_t pass = 0; pass < passes; ++pass) {
                for (uint32_t op = 0; op < ops; ++op) {
                    uint32_t waited = 0;
                    while (
                        ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) != 0
                    ) {
                        if (++waited > (1u << 24)) {
                            return;
                        }
                    }
                    tile_regs_acquire();
                    copy_tile(kCbNodes, op, 0);
                    tile_regs_commit();
                }
            }
        }
#endif
#ifdef TRISC_PACK
        volatile uint32_t* results = (
            reinterpret_cast<volatile uint32_t*>(results_address)
        );
        for (uint32_t index = 0; index < kResultWords; ++index) {
            results[index] = 0;
        }
        const bool fits = groups > 0 && groups <= kGroupsMax && kLanes % groups == 0;
        if (form >= kForms || !fits || (method == kMethodUnpack && !unpack)) {
            results[kResultError] = 1;
        } else {
            setup();
            probe(
                form,
                method,
                groups,
                ops,
                passes,
                reinterpret_cast<const uint32_t*>(ops_address),
                reinterpret_cast<const uint32_t*>(nodes_address),
                reinterpret_cast<const uint32_t*>(rays_address),
                reinterpret_cast<uint32_t*>(entries_address),
                results
            );
            finish();
        }
        results[kResultRun] = command.sequence;
        asm volatile("fence" ::: "memory");
#endif
    });
}
