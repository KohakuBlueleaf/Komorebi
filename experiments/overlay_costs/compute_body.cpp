// A compute overlay (one part per compute core): api.h's compute loop, its runtime
// calls through the table (OV_TABLE) or compiled in.
#include "compute_part.h"
// clang-format off: the part header comes first
#include "api.h"
#include "entry.h"
#include "overlay_gen.h"

extern "C" uint32_t ov_body(const cs::Call* call) {
#if OV_TABLE
    return cs::compute_loop(*call, *static_cast<const cs::ComputeApi*>(call->api));
#else
    // this part's copy of the math library's cached flag value: unknown at entry
    MATH((ckernel::math::_invalidate_src_zero_flag_state_()));
    return cs::compute_loop(*call, cs::Inline{});
#endif
}
