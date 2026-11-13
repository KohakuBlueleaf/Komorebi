// A compute overlay (one part per compute core): the input tile to DST, times or plus
// the argument (fp32 bits) on the vector unit, packed; the loader moves the tiles.
#include "compute_part.h"
// clang-format off: the part header comes first
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "overlay_entry.h"
#include "overlay_gen.h"

extern "C" uint32_t ov_body(ov::Args* a) {
    constexpr uint32_t kCbIn = 0, kCbOut = 16;
    // this part's copy of the math library's cached flag value: unknown at entry
    MATH((ckernel::math::_invalidate_src_zero_flag_state_()));
    copy_init(kCbIn);
    tile_regs_acquire();
    copy_tile(kCbIn, 0, 0);
    binop_with_scalar_tile_init();
#if OV_ADD
    add_unary_tile(0, a->arg);
#else
    mul_unary_tile(0, a->arg);
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, kCbOut);
    tile_regs_release();
    cb_pop_front(kCbIn, 1);
    cb_push_back(kCbOut, 1);
    return 0;
}
