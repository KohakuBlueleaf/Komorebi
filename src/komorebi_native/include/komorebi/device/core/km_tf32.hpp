// A class product with a float32 operand (compute): the unpacker reads a float32 buffer
// into its source register as tf32, which the library's tables do not ask for.
#pragma once

#include <stdint.h>

namespace km {

    // A product's sources, `a` (SrcA) and `b` (SrcB): a float32 side (kA, kB) as tf32,
    // the other its buffer's own formats; the math's the same.
    template <bool kA, bool kB>
    ALWI void tf32_formats([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b) {
        [[maybe_unused]] constexpr uint32_t f32 = (
            static_cast<uint32_t>(DataFormat::Float32)
        );
        [[maybe_unused]] constexpr uint32_t tf32 = (
            static_cast<uint32_t>(DataFormat::Tf32)
        );
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srca_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        kA ? f32 : unpack_src_format[a],
                        kA ? tf32 : unpack_dst_format[a],
                        get_local_cb_interface(a).fifo_page_size
                    )
            )
        );
        UNPACK(
            (
                _llk_unpack_reconfig_data_format_srcb_impl_<
                    DST_ACCUM_MODE,
                    p_dim_stride_target::IGNORE>(
                        kB ? f32 : unpack_src_format[b],
                        kB ? tf32 : unpack_dst_format[b],
                        get_local_cb_interface(b).fifo_page_size
                    )
            )
        );
        MATH((_llk_math_reconfig_data_format_<DST_ACCUM_MODE>(
            kA ? tf32 : unpack_dst_format[a], kB ? tf32 : unpack_dst_format[b]
        )));
    }

}  // namespace km
