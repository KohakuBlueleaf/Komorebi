// The probe's persistent compute kernel: configures the units once, parks in a wait
// loop outside every slot, and on each go word runs its part (slot k), then done.
#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ckernel.h"
#include "overlay.h"

// The compute state the overlays share with this kernel (the runtime's compute build
// defines it): kept and exported, so that an overlay links to these, not to copies.
extern uint32_t unp_cfg_context __attribute__((externally_visible));
extern uint32_t pack_sync_tile_dst_ptr __attribute__((externally_visible));
extern uint32_t math_sync_tile_dst_index __attribute__((externally_visible));
extern uint32_t gl_alu_format_spec_reg __attribute__((externally_visible));

namespace ckernel {
    extern volatile tt_reg_ptr uint* pc_buf_base __attribute__((externally_visible));
    extern volatile tt_reg_ptr uint* regfile __attribute__((externally_visible));
}  // namespace ckernel

__attribute__((used)) volatile void* const ov_exports[] = {
    &unp_cfg_context,
    &pack_sync_tile_dst_ptr,
    &math_sync_tile_dst_index,
    &gl_alu_format_spec_reg,
    &ckernel::pc_buf_base,
    &ckernel::regfile,
};

namespace {

    constexpr uint32_t kTrisc = 48;  // loader.cpp: go, done x 3, argument
    constexpr uint32_t kCbIn = 0, kCbOut = 16;
    constexpr uint32_t kLeave = 0xFFFFFFFFu;

    inline uint32_t word(volatile tt_l1_ptr uint32_t* w) {
        return ckernel::reg_read(reinterpret_cast<uint32_t>(w));
    }

    inline uint64_t wall64() {
        for (;;) {
            const uint32_t h = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_1);
            const uint32_t l = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            if (ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_1) == h) {
                return static_cast<uint64_t>(h) << 32 | l;
            }
        }
    }

}  // namespace

void kernel_main() {
    const uint32_t ctl = get_arg_val<uint32_t>(0), tslots = get_arg_val<uint32_t>(1);
    const uint32_t tslot_bytes = get_arg_val<uint32_t>(2);
    const uint64_t idle = static_cast<uint64_t>(get_arg_val<uint32_t>(3)) << 20;
    uint32_t k = 0;
    UNPACK((k = 0));
    MATH((k = 1));
    PACK((k = 2));
    compute_kernel_hw_startup(kCbIn, kCbOut);
    volatile tt_l1_ptr uint32_t* t = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + 4 * kTrisc)
    );
    const ov::Entry entry = reinterpret_cast<ov::Entry>(tslots + k * tslot_bytes);
    ov::Args args{};
    uint32_t done = 0;
    uint64_t seen_at = wall64();
    for (;;) {
        const uint32_t go = word(t);
        if (go == kLeave) {
            return;
        }
        if (go == done) {
            if (wall64() - seen_at > idle) {
                return;
            }
            continue;
        }
        seen_at = wall64();
        args.arg = word(t + 4);
        entry(&args);
        done = go;
        t[1 + k] = done;
    }
}
