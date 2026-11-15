// The vector unit on the rows that hold a token only (a step takes 4 rows of a face);
// rows past the live ones' bands are left as they were.
#pragma once

#include <stdint.h>

#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"

// `function`<iterations>(arguments..., vector mode) for `rows` live rows: two
// iterations a band of 4 rows in faces 0 and 1, the whole tile past 16 rows.
#define DC_ROWS(rows, function, ...)                      \
    do {                                                  \
        switch (((rows) + 3u) / 4u) {                     \
            case 1:                                       \
                function<2>(__VA_ARGS__, VectorMode::R);  \
                break;                                    \
            case 2:                                       \
                function<4>(__VA_ARGS__, VectorMode::R);  \
                break;                                    \
            case 3:                                       \
                function<6>(__VA_ARGS__, VectorMode::R);  \
                break;                                    \
            case 4:                                       \
                function<8>(__VA_ARGS__, VectorMode::R);  \
                break;                                    \
            default:                                      \
                function<8>(__VA_ARGS__, VectorMode::RC); \
        }                                                 \
    } while (0)

namespace rows {

    template <int I>
    inline void sub(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::SUB, I, DST_ACCUM_MODE),
            a,
            b,
            out,
            mode
        )));
    }

    template <int I>
    inline void mul(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, I, DST_ACCUM_MODE),
            a,
            b,
            out,
            mode
        )));
    }

    template <int I>
    inline void div(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_div,
            (APPROX, ckernel::BinaryOp::DIV, I, DST_ACCUM_MODE),
            a,
            b,
            out,
            mode
        )));
    }

    template <int I>
    inline void max(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_max_min,
            (true, I),
            a,
            b,
            out,
            mode
        )));
    }

    // DEST `dst` times (add: plus) the fp32 scalar `bits`.
    template <int I>
    inline void times(uint32_t dst, uint32_t bits, VectorMode mode) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, MUL_UNARY, I, DST_ACCUM_MODE),
            dst,
            mode,
            bits
        ));
    }

    template <int I>
    inline void plus(uint32_t dst, uint32_t bits, VectorMode mode) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, I, DST_ACCUM_MODE),
            dst,
            mode,
            bits
        ));
    }

    template <int I>
    inline void exp(uint32_t dst, VectorMode mode) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_exponential,
            (false, DST_ACCUM_MODE, false, I, true),
            dst,
            mode,
            p_sfpu::kCONST_1_FP16B
        ));
    }

    template <int I>
    inline void rsqrt(uint32_t dst, VectorMode mode) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX, I, DST_ACCUM_MODE, false),
            dst,
            mode
        ));
    }

    template <int I>
    inline void silu(uint32_t dst, VectorMode mode) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_silu,
            (DST_ACCUM_MODE, I),
            dst,
            mode
        ));
    }

}  // namespace rows
