// Flash-attention vector steps, issued by the pack thread on the current Dst half
// (fp32, tiles 0-3). Addresses are 32-bit Dst rows: tile t at 64 t, a 4-row block's
// even columns at +0 and odd columns at +2, the right face (columns 16-31) at +16.
//
// Row state tile (Dst 3, ST): per 4-row band, b = 639 - A m at even columns of both
// faces, alpha at left odd, the update flag of the tile at right odd of rows 0-3.
#pragma once

#include <stdint.h>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#ifdef ARCH_WORMHOLE
#include "lltt.h"
#endif

namespace flash {

    using namespace ckernel;

    constexpr uint32_t L0 = p_sfpu::LREG0, L1 = p_sfpu::LREG1, L2 = p_sfpu::LREG2,
                       L3 = p_sfpu::LREG3;
    constexpr uint32_t L4 = p_sfpu::LREG4, L5 = p_sfpu::LREG5, L6 = p_sfpu::LREG6,
                       L7 = p_sfpu::LREG7;
    constexpr uint32_t C512 = 11, CA = 12, CB2 = 13, CB1 = 14;
    constexpr uint32_t ONE = p_sfpu::LCONST_1, ZERO = p_sfpu::LCONST_0;
    constexpr uint32_t ST = 192;

#ifdef ARCH_WORMHOLE
    // Wormhole: a vector load's or store's address mode is 2 bits; modes 6 and 7 are
    // 2 and 3 with ADDR_MOD_SET_Base set (Based, around each step), as tt-llk's own
    // vector unit start and done set it.
    constexpr uint32_t AM_STAY = ADDR_MOD_3, AM_STEP = ADDR_MOD_2;

    struct Based {
        Based() {
            math::set_addr_mod_base();
        }
        ~Based() {
            math::clear_addr_mod_base();
        }
    };

    // A word Wormhole does not hold back: the next one reads the MAD-class result
    // (or the lane mode's) of the one before.
#define FLASH_HOLD TTI_SFPNOP

    using lltt::Exec;
    using lltt::NoExec;

    // Records `len` words of `body` at `start` of the replay buffer (with Exec: and
    // runs them), tt-llk's Blackhole load_replay_buf.
    template <lltt::ExecBool E = lltt::NoExec, typename Body>
    [[gnu::always_inline]] inline void load_replay_buf(
        uint32_t start, uint32_t len, Body&& body
    ) {
        lltt::record<E>(start, len);
        body();
    }
#else
    constexpr uint32_t AM_STAY = ADDR_MOD_7, AM_STEP = ADDR_MOD_6;

    struct Based {};

#define FLASH_HOLD
#endif

#ifdef ARCH_WORMHOLE
    // d = Blackhole's SFPARECIP of c (sign and a 7-bit estimate of 1 / |c|) in
    // Wormhole's words, t scratch: p(m) 2^(128 - e) with m c's mantissa in [1, 2) and
    // p ~ -1 / 2m a cubic in bf16 coefficients (within 2.5e-3; the tile compiler's
    // form, hazards/wormhole.py); 2^(128 - e) is ~c with its mantissa cleared.
    template <uint32_t c, uint32_t d, uint32_t t>
    inline void recip_estimate() {
        TTI_SFPSETEXP(127, c, d, 1);
        TTI_SFPSETSGN(0, d, d, 1);
        TTI_SFPMOV(0, d, t, 0);
        TTI_SFPMULI(0x3DE2, t, 0);
        FLASH_HOLD;
        TTI_SFPADDI(0xBF2A, t, 0);
        FLASH_HOLD;
        TTI_SFPMUL(t, d, ZERO, t, 0);
        FLASH_HOLD;
        TTI_SFPADDI(0x3FBC, t, 0);
        FLASH_HOLD;
        TTI_SFPMUL(t, d, ZERO, t, 0);
        TTI_SFPNOT(0, c, d, 0);
        TTI_SFPADDI(0xBFB5, t, 0);
        TTI_SFPSETMAN(0, d, d, 1);
        TTI_SFPMUL(t, d, ZERO, d, 0);
    }
#endif

#ifdef ARCH_WORMHOLE
#define FLASH_WORDS(n) (n)  // words Wormhole's forms add to a recorded body
#else
#define FLASH_WORDS(n) 0
#endif

#ifdef ARCH_WORMHOLE
#define FLASH_SHIFT9(r) TTI_SFPSHFT(9, 0, r, 1)  // r <<= 9 (Wormhole shifts VD)
#else
#define FLASH_SHIFT9(r) TTI_SFPSHFT(9, r, r, 5)  // r = r << 9 (VC by the immediate)
#endif

    // p(m) ~ 2^(m - 1) on [1, 2) (max rel err 1.75e-3).
    constexpr float kB2 = 0.33718854f, kB1 = -0.01674134f, kB0 = 0.68127801f;
    // Accurate exp: u = A x + v in [512, 768) gives 2^(u - 639). Approximate exp: i =
    // 256 A x + v read as uint16 << 15 gives about 2^(i / 256 - 127). kU0: the exponent
    // argument of P = 1; kScale: argument units per power of two.
    constexpr float kOffset = 639.0f;
#ifdef FLASH_APPROX
    constexpr float kU0 = 32500.818359375f, kScale = 256.0f;
#else
    constexpr float kU0 = kOffset, kScale = 1.0f;
#endif

    constexpr uint32_t bits_of(float x) {
        return __builtin_bit_cast(uint32_t, x);
    }

    inline void load_bits(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
        TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xffff);
    }

    inline void load_const(uint32_t lreg, float x) {
        load_bits(lreg, bits_of(x));
    }

    inline void set_const(uint32_t c, float x) {
        load_const(L0, x);
        TT_SFPCONFIG(0, c, 0);
        TTI_SFPNOP;
    }

    // Macro sequence idx = simple | mad << 8 | round << 16 | store << 24.
    inline void set_sequence(uint32_t idx, uint32_t word) {
        load_bits(L0, word);
        TT_SFPCONFIG(0, 4 + idx, 0);
        TTI_SFPNOP;
    }

    // LReg 11 = 512, 12 = A kScale (A = scale log2 e), 13 = B2, 14 = B1.
    inline void init(float a) {
        set_const(C512, 512.0f);
        set_const(CA, a * kScale);
        set_const(CB2, kB2);
        set_const(CB1, kB1);
    }

    // LReg 11 back to -1.0 and lane predication on with every flag set, the state other
    // vector-unit kernels assume; issued at kernel end by the thread that used it last.
    inline void restore_constants() {
#ifdef ARCH_WORMHOLE
        TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80);  // -1.0
        TTI_SFPCONFIG(0x5555, 11, 8);  // LReg 11, every lane (lane mask mode)
#else
        ckernel::sfpu::_restore_lconst_neg1_();
#endif
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
    }

    // Issuing-thread address modes: ADDR_MOD_7 no step, ADDR_MOD_6 steps 4 rows.
    inline void addr_init() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7
        );
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(
            ADDR_MOD_6
        );
    }

    // Approximate exp: y = A x + (v + kMagic) holds round(A x + v) in its mantissa
    // while A x + v is in [0, 2^22).
    constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23 (bf16 0x4B40)

    // Approximate exp template 3: y = max(y, LReg 11), LReg 11 = kMagic during the exp
    // step.
    inline void install_clamp_template() {
        TTI_SFPSWAP(0, C512, 15, 9);
        TTI_SFPNOP;
    }

    // LReg c = bf16 value with bits b.
    inline void set_const_bf16(uint32_t c, uint32_t b) {
        TT_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_FLOATB, b);
        TT_SFPCONFIG(0, c, 0);
        TTI_SFPNOP;
    }

    // Tile row block of band j (tile rows 4j..4j+3).
    constexpr uint32_t band_row(uint32_t j) {
        return j < 4 ? 4 * j : 32 + 4 * (j - 4);
    }

    // Moves the Dst row counter from band j - 1 to band j.
    inline void next_band(uint32_t j) {
        math::inc_dst_addr<4>();
        if (j == 4) {
            math::inc_dst_addr<8>();
            math::inc_dst_addr<8>();
        }
    }

    // L0 = max of L0 over each 8-lane row segment (L1 scratch).
    inline void row_segment_max() {
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPSWAP(0, L0, L1, 1);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSWAP(0, L0, L1, 1);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSWAP(0, L0, L1, 1);
    }

    // Max macro: template kMaxSlot = SFPSWAP(VC = L0, VD = loaded), sequence 2 = that
    // swap at +1.
#ifdef FLASH_APPROX
    constexpr uint32_t kMaxSlot = 3;
#else
    constexpr uint32_t kMaxSlot = 0;
#endif

    inline void install_max_template() {
        TTI_SFPSWAP(0, L0, 12 + kMaxSlot, 1);
        TTI_SFPNOP;
    }

    // Macro templates and sequences (delays in cycles); the max template is installed.
    inline void macro_init() {
#ifdef FLASH_APPROX
        TTI_SFPMAD(CA, 0, L6, 12, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(CA, 0, L7, 13, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L5, 14, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPNOP;
        // MAD at +0, clamp (simple) at +3, << L5 into LReg16 (round) at +5, store
        // LReg16 at +6; delays count issued instructions.
        set_sequence(0, 0x73EE849Fu);
        set_sequence(1, 0x73EE859Fu);
#endif
        install_max_template();
        set_sequence(2, 0x80u | (4 + kMaxSlot));
#ifdef FLASH_APPROX
        TTI_SFPCONFIG(0x0F00, 8, 1);
#else
        TTI_SFPCONFIG(0, 8, 1);
#endif
        TTI_SFPNOP;
    }

    // L0 = lane-wise max of the band over score tiles 0..nt-1: one load-and-swap macro
    // per group into L1 / L2.
    template <uint32_t nt>
    inline void band_lane_max() {
        constexpr uint32_t offs[4] = {0, 2, 16, 18};
        TTI_SFPLOAD(L0, 0, AM_STAY, 0);
#pragma GCC unroll 12
        for (uint32_t i = 1; i < 4 * nt; ++i) {
            TT_SFPLOADMACRO(
                (2 << 2) | (1 + (i & 1)), 0, AM_STAY, 64 * (i / 4) + offs[i % 4]
            );
            TTI_SFPNOP;
        }
        TTI_SFPNOP;
    }

    // First chunk: v = U0 - A c (fp32). L0 = c, L7 = U0.
    inline void band_state_first() {
#ifdef ARCH_WORMHOLE
        TTI_SFPMOV(0, L0, L1, 1);  // -c: Wormhole has no NEGATE_VA
        TTI_SFPMAD(L1, CA, L7, L1, 0);
#else
        TTI_SFPMAD(L0, CA, L7, L1, 1);
#endif
        TTI_SFPNOP;
        TTI_SFPSTORE(L1, 0, AM_STAY, ST + 0);
        TTI_SFPSTORE(L1, 0, AM_STAY, ST + 16);
    }

    // Later chunks: t = A c + v_old; where t > thr (L5) v becomes U0 - A c (fp32), else
    // stays. L6 |= update mask. L0 = c, L7 = U0.
    inline void band_state_next() {
        TTI_SFPLOAD(L1, 0, AM_STAY, ST + 0);
        TTI_SFPMAD(L0, CA, L1, L2, 0);
#ifdef ARCH_WORMHOLE
        // L3 = t - U0, then L4 = -1 where t > thr (t - thr > 0, t and thr finite)
        TTI_SFPMOV(0, L7, L3, 1);
        TTI_SFPMAD(L2, ONE, L3, L3, 0);
        TTI_SFPMOV(0, L5, L4, 1);
        TTI_SFPMAD(L2, ONE, L4, L2, 0);
        TTI_SFPMOV(0, ZERO, L4, 0);
        TTI_SFPSETCC(0, L2, 0, 4);  // >= 0
        TTI_SFPSETCC(0, L2, 0, 2);  // and != 0
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_SHORT, 0xFFFF);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPOR(0, L4, L6, 0);
        TTI_SFPAND(0, L4, L3, 0);
        TTI_SFPMOV(0, L3, L3, 1);
        TTI_SFPMAD(L3, ONE, L1, L1, 0);
#else
        TTI_SFPMAD(L2, ONE, L7, L3, 2);
        TTI_SFPMOV(0, L2, L4, 0);
        TTI_SFPGT(0, L5, L4, 8);
        TTI_SFPOR(0, L4, L6, 0);
        TTI_SFPAND(0, L4, L3, 0);
        TTI_SFPMAD(L3, ONE, L1, L1, 1);
#endif
        TTI_SFPNOP;
        TTI_SFPSTORE(L1, 0, AM_STAY, ST + 0);
        TTI_SFPSTORE(L1, 0, AM_STAY, ST + 16);
    }

    // Tile update flag (1.0 if any row moved, else 0) from the masks in L6 to right odd
    // of rows 0-3; the row counter is at band 7.
    inline void store_any_flag() {
        TTI_SFPMOV(0, L6, L0, 0);
        TTI_SFPMOV(0, ZERO, L1, 0);
        TTI_SFPMOV(0, ZERO, L2, 0);
        TTI_SFPMOV(0, ZERO, L3, 0);
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPOR(0, L1, L0, 0);
        TTI_SFPOR(0, L2, L0, 0);
        TTI_SFPOR(0, L3, L0, 0);
        TTI_SFPLOADI(L1, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
        TTI_SFPAND(0, L1, L0, 0);
        TTI_SFPSTORE(L0, 0, AM_STAY, ST + 18 - band_row(7));
    }

    // Two groups of the 4-row block at the row counter, in place: y = 2^(x A + L7 -
    // 639).
    inline void exp_pair_body() {
        TTI_SFPLOAD(L0, 0, AM_STAY, 0);
        TTI_SFPLOAD(L3, 0, AM_STAY, 2);
        TTI_SFPMAD(L0, CA, L7, L0, 0);
        TTI_SFPMAD(L3, CA, L7, L3, 0);
        TTI_SFPSWAP(0, L0, C512, 1);
        TTI_SFPSWAP(0, L3, C512, 1);
        FLASH_SHIFT9(L0);
        FLASH_SHIFT9(L3);
        TTI_SFPSETEXP(127, L0, L1, 1);
        TTI_SFPSETEXP(127, L3, L4, 1);
        TTI_SFPMAD(L1, CB2, CB1, L2, 0);
        TTI_SFPMAD(L4, CB2, CB1, L5, 0);
        TTI_SFPMAD(L2, L1, L6, L2, 0);
        TTI_SFPMAD(L5, L4, L6, L5, 0);
        TTI_SFPSETEXP(0, L2, L0, 2);
        TTI_SFPSETEXP(0, L5, L3, 2);
        TTI_SFPSTORE(L0, 0, AM_STAY, 0);
        TTI_SFPSTORE(L3, 0, AM_STEP, 2);
    }

    // P = exp of score tile t in place; each block's b from the state tile, 64 (3 - t)
    // rows on.
    template <uint32_t t>
    inline void exp_tile() {
        constexpr uint32_t body = 19;
        load_replay_buf<Exec>(0, body, [] {
            TTI_SFPLOAD(L7, 0, AM_STAY, 64 * (3 - t));
            exp_pair_body();
        });
#pragma GCC unroll 1
        for (uint32_t i = 1; i < 16; ++i) {
            TTI_REPLAY(0, body, 0, 0);
        }
    }

#ifdef FLASH_APPROX
    // Two 4-row blocks of one face: biases v + kMagic from `bias` (L6, block 1) and
    // bias + 4 (L7, block 2), then one macro per group, macros on every other
    // instruction.
    template <uint32_t bias>
    inline void exp_body_approx() {
        TTI_SFPLOAD(L6, 0, AM_STAY, bias);
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, AM_STAY, 0);
        TTI_SFPLOAD(L7, 0, AM_STAY, bias + 4);
        TTI_SFPLOADMACRO((0 << 2) | 1, 0, AM_STEP, 2);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO((1 << 2) | 2, 0, AM_STAY, 0);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO((1 << 2) | 3, 0, AM_STEP, 2);
    }

    // P = exp of score tile t in place (approximate): bodies over faces left, left,
    // right, right, left, left, right, right; both faces take the bias from the state
    // tile's left odd columns.
    template <uint32_t t>
    inline void exp_tile_approx() {
        constexpr uint32_t body = 8, left = 64 * (3 - t) + 2, right = left - 16;
        load_replay_buf<Exec>(0, body, [] {
            exp_body_approx<left>();
        });
        load_replay_buf<NoExec>(body, body, [] {
            exp_body_approx<right>();
        });
        TTI_REPLAY(0, body, 0, 0);
        TTI_REPLAY(body, body, 0, 0);
        TTI_REPLAY(body, body, 0, 0);
        TTI_REPLAY(0, body, 0, 0);
        TTI_REPLAY(0, body, 0, 0);
        TTI_REPLAY(body, body, 0, 0);
        TTI_REPLAY(body, body, 0, 0);
    }
#endif

    template <uint32_t t>
    inline void exp_one() {
#ifdef FLASH_APPROX
        exp_tile_approx<t>();
#else
        exp_tile<t>();
#endif
    }

    // Per band, the lane-wise max over score tiles 0..nt-1 into left odd of the state
    // tile.
    template <uint32_t nt>
    inline void state_max() {
        const Based based;
#ifdef FLASH_APPROX
        install_max_template();
#endif
        math::clear_dst_reg_addr();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            band_lane_max<nt>();
            TTI_SFPSTORE(L0, 0, AM_STAY, ST + 2);
        }
        math::clear_dst_reg_addr();
    }

    // Row state update from the lane-wise maxima in the state tile, and the tile's
    // flag.
    template <bool first>
    inline void state_update(uint32_t thr_bits) {
        const Based based;
        math::clear_dst_reg_addr();
        load_const(L7, kU0);
        if constexpr (!first) {
            TTI_SFPMOV(0, ZERO, L6, 0);
            load_bits(L5, thr_bits);
        }
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L0, 0, AM_STAY, ST + 2);
            row_segment_max();
            if constexpr (first) {
                band_state_first();
            } else {
                band_state_next();
            }
#ifdef FLASH_APPROX
            // Exp bias v + kMagic (LReg 11) at left odd.
            TTI_SFPMAD(L1, ONE, C512, L2, 0);
            TTI_SFPSTORE(L2, 0, AM_STAY, ST + 2);
#endif
        }
        if constexpr (!first) {
            store_any_flag();
        }
        math::clear_dst_reg_addr();
    }

    // P = exp of score tiles 0..nt-1 in place, v from the state tile.
    template <uint32_t nt>
    inline void exp_step() {
        const Based based;
#ifdef FLASH_APPROX
        install_clamp_template();
        TTI_SFPLOADI(L5, sfpi::SFPLOADI_MOD0_USHORT, 15);
#else
        load_const(L6, kB0);
#endif
        math::clear_dst_reg_addr();
        exp_one<0>();
        if constexpr (nt > 1) {
            exp_one<1>();
        }
        if constexpr (nt > 2) {
            exp_one<2>();
        }
#ifdef FLASH_APPROX
        // Drain the macros (their delays count issued instructions).
#pragma GCC unroll 8
        for (uint32_t i = 0; i < 8; ++i) {
            TTI_SFPNOP;
        }
#endif
        math::clear_dst_reg_addr();
    }

    // Pack-thread vector section of a step (state update and exp): LReg 11 = kMagic
    // inside (approximate exp), 512 outside.
    inline void exp_section_begin() {
#ifdef FLASH_APPROX
        set_const_bf16(C512, 0x4B40);
#endif
    }

    inline void exp_section_end() {
#ifdef FLASH_APPROX
        set_const_bf16(C512, 0x4400);
#endif
    }

    // L7 = alpha = 2^((v_new - v_old) / kScale) of the band, 1 where v did not move;
    // v_new from the state tile in Dst 2, v_old from Dst 3.
    inline void band_alpha() {
        TTI_SFPLOAD(L1, 0, AM_STAY, 128);
        TTI_SFPLOAD(L2, 0, AM_STAY, ST);
#ifdef ARCH_WORMHOLE
        // L4 = v_new - v_old, L3 = -1 where it is below +0 (its sign bit)
        TTI_SFPMOV(0, L2, L4, 1);
        TTI_SFPMAD(L1, ONE, L4, L4, 0);
        TTI_SFPMOV(0, ZERO, L3, 0);
        FLASH_HOLD;
        TTI_SFPSETCC(0, L4, 0, 0);  // < 0
        TTI_SFPLOADI(L3, sfpi::SFPLOADI_MOD0_SHORT, 0xFFFF);
        TTI_SFPENCC(3, 0, 0, 10);
#else
        TTI_SFPMAD(L1, ONE, L2, L4, 2);
        TTI_SFPMOV(0, ZERO, L3, 0);
        TTI_SFPGT(0, L4, L3, 8);
#endif
#ifdef FLASH_APPROX
        TTI_SFPMULI(0x3B80, L4, 0);
#endif
        load_const(L5, kOffset);
        TTI_SFPMAD(L4, ONE, L5, L5, 0);
        // SFPSWAP's compare reads L5 on its first cycle, which the MAD stall logic
        // misses.
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L5, C512, 1);
        FLASH_SHIFT9(L5);
        TTI_SFPSETEXP(127, L5, L2, 1);
        TTI_SFPMAD(L2, CB2, CB1, L4, 0);
        load_const(L0, kB0);
        TTI_SFPMAD(L4, L2, L0, L4, 0);
        FLASH_HOLD;
        TTI_SFPSETEXP(0, L4, L5, 2);
        TTI_SFPAND(0, L3, L5, 0);
        TTI_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
        TTI_SFPNOT(0, L3, L4, 0);
        TTI_SFPAND(0, L4, L7, 0);
        TTI_SFPOR(0, L5, L7, 0);
    }

    // Dst 0 += alpha * Dst 1 (alpha from the state tiles in Dst 2 and 3).
    inline void rescale_step() {
        const Based based;
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            band_alpha();
            TTI_SFPLOAD(L0, 0, AM_STAY, 0);
            TTI_SFPLOAD(L1, 0, AM_STAY, 64);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            FLASH_HOLD;
            TTI_SFPSTORE(L0, 0, AM_STAY, 0);
            TTI_SFPLOAD(L0, 0, AM_STAY, 2);
            TTI_SFPLOAD(L1, 0, AM_STAY, 66);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            FLASH_HOLD;
            TTI_SFPSTORE(L0, 0, AM_STAY, 2);
            TTI_SFPLOAD(L0, 0, AM_STAY, 16);
            TTI_SFPLOAD(L1, 0, AM_STAY, 80);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            FLASH_HOLD;
            TTI_SFPSTORE(L0, 0, AM_STAY, 16);
            TTI_SFPLOAD(L0, 0, AM_STAY, 18);
            TTI_SFPLOAD(L1, 0, AM_STAY, 82);
            TTI_SFPMAD(L1, L7, L0, L0, 0);
            FLASH_HOLD;
            TTI_SFPSTORE(L0, 0, AM_STAY, 18);
        }
        math::clear_dst_reg_addr();
    }

    // Dst t *= 1 / Dst 3 for t < w (Dst 3: row sums in every column).
    inline void normalize_step(uint32_t w) {
        const Based based;
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L7, 0, AM_STAY, ST + 0);
#ifdef ARCH_WORMHOLE
            recip_estimate<L7, L1, L2>();
            FLASH_HOLD;
            TTI_SFPMOV(0, L1, L2, 1);
            TTI_SFPMAD(L7, L2, ONE, L2, 0);
            FLASH_HOLD;
            TTI_SFPMAD(L1, L2, L1, L1, 0);
            FLASH_HOLD;
            TTI_SFPMOV(0, L1, L2, 1);
            TTI_SFPMAD(L7, L2, ONE, L2, 0);
            FLASH_HOLD;
            TTI_SFPMAD(L1, L2, L1, L7, 0);
#else
            TTI_SFPARECIP(0, L7, L1, 0);
            TTI_SFPMAD(L7, L1, ONE, L2, 1);
            TTI_SFPMAD(L1, L2, L1, L1, 0);
            TTI_SFPMAD(L7, L1, ONE, L2, 1);
            TTI_SFPMAD(L1, L2, L1, L7, 0);
#endif
#pragma GCC unroll 1
            for (uint32_t t = 0; t < w; ++t) {
                const uint32_t base = 64 * t;
                TT_SFPLOAD(L0, 0, AM_STAY, base);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                FLASH_HOLD;
                TT_SFPSTORE(L0, 0, AM_STAY, base);
                TT_SFPLOAD(L0, 0, AM_STAY, base + 2);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                FLASH_HOLD;
                TT_SFPSTORE(L0, 0, AM_STAY, base + 2);
                TT_SFPLOAD(L0, 0, AM_STAY, base + 16);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                FLASH_HOLD;
                TT_SFPSTORE(L0, 0, AM_STAY, base + 16);
                TT_SFPLOAD(L0, 0, AM_STAY, base + 18);
                TTI_SFPMUL(L0, L7, ZERO, L0, 0);
                FLASH_HOLD;
                TT_SFPSTORE(L0, 0, AM_STAY, base + 18);
            }
        }
        math::clear_dst_reg_addr();
    }

    // L0 = sum of L0 over each 8-lane row segment, in every lane (L1 scratch).
    inline void row_segment_sum() {
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPMAD(L0, ONE, L1, L0, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPMAD(L0, ONE, L1, L0, 0);
        TTI_SFPNOP;
        TTI_SFPSHFT2(0, L0, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPSHFT2(0, L1, L1, 3);
        FLASH_HOLD;
        TTI_SFPMAD(L0, ONE, L1, L0, 0);
        TTI_SFPNOP;
    }

    // Lane sums of score tiles 0..nt-1 (fp32 P) per band into tile 0's left even
    // columns: lane (row, c) of a band sums the row's columns 2c, 2c + 1, 2c + 16, 2c +
    // 17 of every tile.
    inline void lane_sum_step(uint32_t nt) {
        const Based based;
        math::clear_dst_reg_addr();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L0, 0, AM_STAY, 0);
            TTI_SFPLOAD(L1, 0, AM_STAY, 2);
            TTI_SFPLOAD(L2, 0, AM_STAY, 16);
            TTI_SFPLOAD(L3, 0, AM_STAY, 18);
            TTI_SFPMAD(L0, ONE, L1, L0, 0);
            TTI_SFPMAD(L2, ONE, L3, L2, 0);
#pragma GCC unroll 1
            for (uint32_t t = 1; t < nt; ++t) {
                TT_SFPLOAD(L1, 0, AM_STAY, 64 * t);
                TT_SFPLOAD(L3, 0, AM_STAY, 64 * t + 2);
                TT_SFPLOAD(L4, 0, AM_STAY, 64 * t + 16);
                TT_SFPLOAD(L5, 0, AM_STAY, 64 * t + 18);
                TTI_SFPMAD(L1, ONE, L0, L0, 0);
                TTI_SFPMAD(L3, ONE, L2, L2, 0);
                TTI_SFPMAD(L4, ONE, L0, L0, 0);
                TTI_SFPMAD(L5, ONE, L2, L2, 0);
            }
            TTI_SFPNOP;
            TTI_SFPMAD(L0, ONE, L2, L0, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, AM_STAY, 0);
        }
        math::clear_dst_reg_addr();
    }

    // log2(1 + t) on [0, 1): degree-5 fit, max error 2.9e-5.
    constexpr float kL1 = 1.4415921f, kL2 = -0.707253456f, kL3 = 0.411561489f,
                    kL4 = -0.189832449f;
    constexpr float kL5 = 0.0439286269f;

    // Row statistic per band: lse = ln sum_j exp(scale s_j) = ln 2 ((kU0 - v) / kScale
    // + log2(l)), with l the sum of the row's lane sums (Dst 3, left even columns) and
    // v the row reference (Dst 2, even columns); stored to every column of Dst 2.
    inline void band_lse2() {
        TTI_SFPLOAD(L0, 0, AM_STAY, ST + 0);
        row_segment_sum();
        TTI_SFPEXEXP(0, L0, L1, sfpi::SFPEXEXP_MOD1_DEBIAS);
        TTI_SFPSETEXP(127, L0, L2, 1);
#ifdef ARCH_WORMHOLE
        TTI_SFPABS(0, L1, L3, 0);  // the integer magnitude, then L1's sign on it
        TTI_SFPSETSGN(0, L3, L1, 0);
#else
        TTI_SFPCAST(L1, L1, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
#endif
        TTI_SFPADDI(0xBF80, L2, 0);
        TTI_SFPCAST(L1, L1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        load_const(L3, kL5);
        load_const(L4, kL4);
        TTI_SFPMAD(L3, L2, L4, L3, 0);
        load_const(L4, kL3);
        TTI_SFPNOP;
        TTI_SFPMAD(L3, L2, L4, L3, 0);
        load_const(L4, kL2);
        TTI_SFPNOP;
        TTI_SFPMAD(L3, L2, L4, L3, 0);
        load_const(L4, kL1);
        TTI_SFPNOP;
        TTI_SFPMAD(L3, L2, L4, L3, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPMAD(L3, L2, L1, L3, 0);
        TTI_SFPLOAD(L4, 0, AM_STAY, 128);
        load_const(L5, kU0);
#ifdef ARCH_WORMHOLE
        TTI_SFPMOV(0, L4, L4, 1);
        TTI_SFPMAD(L4, ONE, L5, L5, 0);
#else
        TTI_SFPMAD(L4, ONE, L5, L5, 1);
#endif
        TTI_SFPNOP;
#ifdef FLASH_APPROX
        TTI_SFPMULI(0x3B80, L5, 0);
        TTI_SFPNOP;
#endif
        TTI_SFPMAD(L5, ONE, L3, L3, 0);
        load_const(L4, 0.69314718f);
        TTI_SFPMUL(L3, L4, ZERO, L3, 0);
        TTI_SFPNOP;
        TTI_SFPSTORE(L3, 0, AM_STAY, 128);
        TTI_SFPSTORE(L3, 0, AM_STAY, 130);
        TTI_SFPSTORE(L3, 0, AM_STAY, 144);
        TTI_SFPSTORE(L3, 0, AM_STAY, 146);
    }

    // Row statistic L2 of the tile's 32 rows into Dst 2 (pack thread).
    inline void stats_step() {
        const Based based;
        math::clear_dst_reg_addr();
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            band_lse2();
        }
        math::clear_dst_reg_addr();
    }

    // Column layout of a row-layout tile (pack thread): Dst tile `dst` row r, column c
    // = Dst tile `src` row c (its even columns), for every r. Exact lane moves: each
    // band of `src` is loaded into L0-L3, SFPTRANSP turns row q of the band into all
    // lanes of L_q, and a store predicated on the lane's column slot puts it into rows
    // 0-3 of `dst` at that column; rows 0-3 are then copied to every band. L4-L7 hold 2
    // x the lane's column slot (row-independent, so SFPTRANSP leaves them as they are).
    inline void cols_step(uint32_t src, uint32_t dst) {
        const Based based;
        math::clear_dst_reg_addr();
        TTI_SFPLOADI(L4, sfpi::SFPLOADI_MOD0_USHORT, 14);
        TTI_SFPAND(0, 15, L4, 0);
        TTI_SFPMOV(0, L4, L5, 0);
        TTI_SFPMOV(0, L4, L6, 0);
        TTI_SFPMOV(0, L4, L7, 0);
        TTI_SFPENCC(1, 0, 0, 2);
#pragma GCC unroll 1
        for (uint32_t b = 0; b < 8; ++b) {
            const uint32_t a = 64 * src + band_row(b);
            TT_SFPLOAD(L0, 0, AM_STAY, a);
            TT_SFPLOAD(L1, 0, AM_STAY, a);
            TT_SFPLOAD(L2, 0, AM_STAY, a);
            TT_SFPLOAD(L3, 0, AM_STAY, a);
            TTI_SFPNOP;
            TTI_SFPTRANSP(0, 0, 0, 0);
            TTI_SFPNOP;
#pragma GCC unroll 1
            for (uint32_t q = 0; q < 4; ++q) {
                const uint32_t col = (b < 4 ? 4 * b : 16 + 4 * (b - 4)) + q;
                const uint32_t slot = (col % 16) / 2;
                const uint32_t off = 64 * dst + (col >= 16 ? 16 : 0) +
                    (col % 2 ? 2 : 0);
                TT_SFPIADD((0u - 2 * slot) & 0xfff, L4, L5, 1 | 8);
                TTI_SFPIADD(0xffe, L5, L5, 1);
                TT_SFPSTORE(L0 + q, 0, AM_STAY, off);
                TTI_SFPENCC(0, 0, 0, 0);
            }
            TTI_SFPMOV(0, L4, L5, 0);
        }
        // Lane flags back to all true with predication left on (the state every
        // kernel's vector code assumes); the copy below writes every lane.
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TT_SFPLOAD(L0, 0, AM_STAY, 64 * dst);
        TT_SFPLOAD(L1, 0, AM_STAY, 64 * dst + 2);
        TT_SFPLOAD(L2, 0, AM_STAY, 64 * dst + 16);
        TT_SFPLOAD(L3, 0, AM_STAY, 64 * dst + 18);
#pragma GCC unroll 1
        for (uint32_t b = 1; b < 8; ++b) {
            const uint32_t a = 64 * dst + band_row(b);
            TT_SFPSTORE(L0, 0, AM_STAY, a);
            TT_SFPSTORE(L1, 0, AM_STAY, a + 2);
            TT_SFPSTORE(L2, 0, AM_STAY, a + 16);
            TT_SFPSTORE(L3, 0, AM_STAY, a + 18);
        }
        math::clear_dst_reg_addr();
    }

    // Backward (pack thread), w = 639 - lse log2 e (2^(A s + w - 639) = exp(scale s -
    // lse)): the row statistics tile in Dst 3 from the row log-sum-exp (Dst 3, every
    // column) and D (Dst 0, every column): w at the even columns, D at the odd columns.
    inline void wd_step() {
        const Based based;
        math::clear_dst_reg_addr();
        load_const(L6, -1.44269504f);
        load_const(L7, 639.0f);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L0, 0, AM_STAY, ST + 0);
            TTI_SFPLOAD(L1, 0, AM_STAY, ST + 16);
            TTI_SFPLOAD(L2, 0, AM_STAY, 0);
            TTI_SFPMAD(L0, L6, L7, L0, 0);
            TTI_SFPMAD(L1, L6, L7, L1, 0);
            TTI_SFPSTORE(L2, 0, AM_STAY, ST + 2);
            TTI_SFPSTORE(L2, 0, AM_STAY, ST + 18);
            TTI_SFPSTORE(L0, 0, AM_STAY, ST + 0);
            TTI_SFPSTORE(L1, 0, AM_STAY, ST + 16);
        }
        math::clear_dst_reg_addr();
    }

    // wd_step for a query tile with rows past the last query: also adds the row mask
    // tile in Dst 1 (Const::TailRows) to w, so those rows get P = 0 against every key.
    inline void wd_step_tail() {
        const Based based;
        math::clear_dst_reg_addr();
        load_const(L6, -1.44269504f);
        load_const(L7, 639.0f);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
            TTI_SFPLOAD(L0, 0, AM_STAY, ST + 0);
            TTI_SFPLOAD(L1, 0, AM_STAY, ST + 16);
            TTI_SFPLOAD(L2, 0, AM_STAY, 0);
            TTI_SFPLOAD(L3, 0, AM_STAY, 64 + 0);
            TTI_SFPLOAD(L4, 0, AM_STAY, 64 + 16);
            TTI_SFPMAD(L0, L6, L7, L0, 0);
            TTI_SFPMAD(L1, L6, L7, L1, 0);
            TTI_SFPSTORE(L2, 0, AM_STAY, ST + 2);
            TTI_SFPSTORE(L2, 0, AM_STAY, ST + 18);
            TTI_SFPMAD(L0, ONE, L3, L0, 0);
            TTI_SFPMAD(L1, ONE, L4, L1, 0);
            TTI_SFPNOP;
            TTI_SFPSTORE(L0, 0, AM_STAY, ST + 0);
            TTI_SFPSTORE(L1, 0, AM_STAY, ST + 16);
        }
        math::clear_dst_reg_addr();
    }

    // Backward, one 4-row block: P = 2^(A s + w - 639) from s (Dst 0) and w (Dst 3,
    // even columns), then dS = P (dP - D) over dP (Dst 1) with D from Dst 3's odd
    // columns; with `keep_p` P also over s. Even (L0-L2) and odd (L3-L5) columns
    // interleave, so each multiply-add's result is read at least two instructions
    // later.
    template <bool keep_p>
    inline void p_ds_body() {
        TTI_SFPLOAD(L7, 0, AM_STAY, ST);
        TTI_SFPLOAD(L0, 0, AM_STAY, 0);
        TTI_SFPLOAD(L3, 0, AM_STAY, 2);
        TTI_SFPMAD(L0, CA, L7, L0, 0);
        TTI_SFPMAD(L3, CA, L7, L3, 0);
        TTI_SFPSWAP(0, L0, C512, 1);
        TTI_SFPSWAP(0, L3, C512, 1);
        FLASH_SHIFT9(L0);
        FLASH_SHIFT9(L3);
        TTI_SFPSETEXP(127, L0, L1, 1);
        TTI_SFPSETEXP(127, L3, L4, 1);
        TTI_SFPMAD(L1, CB2, CB1, L2, 0);
        TTI_SFPMAD(L4, CB2, CB1, L5, 0);
        TTI_SFPMAD(L2, L1, L6, L2, 0);
        TTI_SFPMAD(L5, L4, L6, L5, 0);
        TTI_SFPSETEXP(0, L2, L0, 2);
        TTI_SFPSETEXP(0, L5, L3, 2);
        if constexpr (keep_p) {
            TTI_SFPSTORE(L0, 0, AM_STAY, 0);
            TTI_SFPSTORE(L3, 0, AM_STAY, 2);
        }
        TTI_SFPLOAD(L1, 0, AM_STAY, 64);
        TTI_SFPLOAD(L4, 0, AM_STAY, 66);
        TTI_SFPLOAD(L2, 0, AM_STAY, ST + 2);
#ifdef ARCH_WORMHOLE
        TTI_SFPMOV(0, L2, L2, 1);
        TTI_SFPMAD(L2, ONE, L1, L1, 0);
        TTI_SFPMAD(L2, ONE, L4, L4, 0);
#else
        TTI_SFPMAD(L2, ONE, L1, L1, 1);
        TTI_SFPMAD(L2, ONE, L4, L4, 1);
#endif
        TTI_SFPMUL(L0, L1, ZERO, L0, 0);
        TTI_SFPMUL(L3, L4, ZERO, L3, 0);
        TTI_SFPSTORE(L0, 0, AM_STAY, 64);
        TTI_SFPSTORE(L3, 0, AM_STEP, 66);
    }

    // Backward (pack thread): P and dS of one score tile (p_ds_body over its 16
    // blocks); dS in Dst 1, with `keep_p` P in Dst 0.
    template <bool keep_p>
    inline void p_ds_step() {
        const Based based;
        // Wormhole: one more word (the negated copy of p_ds_body)
        constexpr uint32_t body = (keep_p ? 28 : 26) + FLASH_WORDS(1);
        math::clear_dst_reg_addr();
        load_const(L6, kB0);
        load_replay_buf<Exec>(0, body, [] {
            p_ds_body<keep_p>();
        });
#pragma GCC unroll 1
        for (uint32_t i = 1; i < 16; ++i) {
            TTI_REPLAY(0, body, 0, 0);
        }
        math::clear_dst_reg_addr();
    }

    // Backward, key-row orientation, one 4-row block: P^T = 2^(A s + w - 639) from s
    // (Dst 0) with w per column (Dst 3), then dS^T = P^T (dP^T - D) over dP^T (Dst 1)
    // with D per column (Dst 2); P^T also over s. Even (L0-L2, L7) and odd (L3-L5)
    // columns interleave.
    inline void p_ds_body_cols() {
        TTI_SFPLOAD(L7, 0, AM_STAY, ST);
        TTI_SFPLOAD(L5, 0, AM_STAY, ST + 2);
        TTI_SFPLOAD(L0, 0, AM_STAY, 0);
        TTI_SFPLOAD(L3, 0, AM_STAY, 2);
        TTI_SFPMAD(L0, CA, L7, L0, 0);
        TTI_SFPMAD(L3, CA, L5, L3, 0);
        TTI_SFPSWAP(0, L0, C512, 1);
        TTI_SFPSWAP(0, L3, C512, 1);
        FLASH_SHIFT9(L0);
        FLASH_SHIFT9(L3);
        TTI_SFPSETEXP(127, L0, L1, 1);
        TTI_SFPSETEXP(127, L3, L4, 1);
        TTI_SFPMAD(L1, CB2, CB1, L2, 0);
        TTI_SFPMAD(L4, CB2, CB1, L5, 0);
        TTI_SFPMAD(L2, L1, L6, L2, 0);
        TTI_SFPMAD(L5, L4, L6, L5, 0);
        TTI_SFPSETEXP(0, L2, L0, 2);
        TTI_SFPSETEXP(0, L5, L3, 2);
        TTI_SFPSTORE(L0, 0, AM_STAY, 0);
        TTI_SFPSTORE(L3, 0, AM_STAY, 2);
        TTI_SFPLOAD(L1, 0, AM_STAY, 64);
        TTI_SFPLOAD(L4, 0, AM_STAY, 66);
        TTI_SFPLOAD(L2, 0, AM_STAY, 128);
        TTI_SFPLOAD(L5, 0, AM_STAY, 130);
#ifdef ARCH_WORMHOLE
        TTI_SFPMOV(0, L2, L2, 1);
        TTI_SFPMOV(0, L5, L5, 1);
        TTI_SFPMAD(L2, ONE, L1, L1, 0);
        TTI_SFPMAD(L5, ONE, L4, L4, 0);
#else
        TTI_SFPMAD(L2, ONE, L1, L1, 1);
        TTI_SFPMAD(L5, ONE, L4, L4, 1);
#endif
        TTI_SFPMUL(L0, L1, ZERO, L0, 0);
        TTI_SFPMUL(L3, L4, ZERO, L3, 0);
        TTI_SFPSTORE(L0, 0, AM_STAY, 64);
        TTI_SFPSTORE(L3, 0, AM_STEP, 66);
    }

    // Backward (pack thread), key-row orientation: P^T in Dst 0 and dS^T in Dst 1 of
    // one score tile (p_ds_body_cols over its 16 blocks).
    inline void p_ds_step_cols() {
        const Based based;
        constexpr uint32_t body = 30 + FLASH_WORDS(2);  // Wormhole: two negated copies
        math::clear_dst_reg_addr();
        load_const(L6, kB0);
        load_replay_buf<Exec>(0, body, [] {
            p_ds_body_cols();
        });
#pragma GCC unroll 1
        for (uint32_t i = 1; i < 16; ++i) {
            TTI_REPLAY(0, body, 0, 0);
        }
        math::clear_dst_reg_addr();
    }

    // Dst tiles 0..n-1 *= the fp32 value with bits `bits` (pack thread).
    inline void scale_step(uint32_t n, uint32_t bits) {
        const Based based;
        math::clear_dst_reg_addr();
        load_bits(L5, bits);
#pragma GCC unroll 1
        for (uint32_t j = 0; j < 8; ++j) {
            if (j > 0) {
                next_band(j);
            }
#pragma GCC unroll 1
            for (uint32_t t = 0; t < n; ++t) {
                const uint32_t base = 64 * t;
                TT_SFPLOAD(L0, 0, AM_STAY, base);
                TT_SFPLOAD(L1, 0, AM_STAY, base + 2);
                TT_SFPLOAD(L2, 0, AM_STAY, base + 16);
                TT_SFPLOAD(L3, 0, AM_STAY, base + 18);
                TTI_SFPMUL(L0, L5, ZERO, L0, 0);
                TTI_SFPMUL(L1, L5, ZERO, L1, 0);
                TTI_SFPMUL(L2, L5, ZERO, L2, 0);
                TTI_SFPMUL(L3, L5, ZERO, L3, 0);
                TT_SFPSTORE(L0, 0, AM_STAY, base);
                TT_SFPSTORE(L1, 0, AM_STAY, base + 2);
                TT_SFPSTORE(L2, 0, AM_STAY, base + 16);
                TT_SFPSTORE(L3, 0, AM_STAY, base + 18);
            }
        }
        math::clear_dst_reg_addr();
    }

    // Constant tiles (math thread, Dst row counter at 0). Lane l of a load at a 4-row
    // block holds row l / 8 and column 2 (l % 8) (+1 at the odd-column address) of the
    // block [doc SFPLOAD], LReg 15 = 2 l [doc LReg]. Per lane: L4 = column, L5 = row,
    // L6 = column - row within the block, as floats.
    inline void const_lanes() {
        TTI_SFPMOV(0, 15, L4, 0);
        TTI_SFPLOADI(L3, sfpi::SFPLOADI_MOD0_USHORT, 14);
        TTI_SFPAND(0, L3, L4, 0);
        TTI_SFPMOV(0, 15, L5, 0);
        TTI_SFPSHFT(0xffc, 0, L5, 1);
        TTI_SFPCAST(L4, L4, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        TTI_SFPCAST(L5, L5, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        TTI_SFPNOP;
#ifdef ARCH_WORMHOLE
        TTI_SFPMOV(0, L5, L6, 1);
        TTI_SFPMAD(L6, ONE, L4, L6, 0);
#else
        TTI_SFPMAD(L5, ONE, L4, L6, 1);
#endif
        TTI_SFPNOP;
    }

    constexpr uint32_t bf16_bits(float x) {
        return __builtin_bit_cast(uint32_t, x) >> 16;
    }

    // L0 = min(max(L0, 0), 1).
    inline void clamp01_l0() {
        TTI_SFPSWAP(0, L0, ZERO, 1);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L0, ONE, 9);
        TTI_SFPNOP;
    }

    // Kinds of constant tile.
    enum class Const : uint32_t { Ones, Causal, CausalT, Tail, CausalTail, TailRows };

    // Constant tile `kind` into Dst tile t; `tail`: the first masked key column (Tail)
    // or row (TailRows).
    inline void const_tile(Const kind, uint32_t t, uint32_t tail) {
        const Based based;
#pragma GCC unroll 1
        for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll 1
            for (uint32_t j = 0; j < 4; ++j) {
#pragma GCC unroll 1
                for (uint32_t p = 0; p < 2; ++p) {
                    const int32_t row = 16 * static_cast<int32_t>(f >> 1) +
                        4 * static_cast<int32_t>(j);
                    const int32_t col = 16 * static_cast<int32_t>(f & 1) +
                        static_cast<int32_t>(p);
                    switch (kind) {
                        case Const::Ones:
                            TTI_SFPMOV(0, ONE, L0, 0);
                            break;
                        case Const::Causal:  // -1e30 min(max(col - row, 0), 1)
                            TTI_SFPMOV(0, L6, L0, 0);
                            TT_SFPADDI(bf16_bits(static_cast<float>(col - row)), L0, 0);
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMULI(0xF14A, L0, 0);
                            break;
                        case Const::CausalT:  // -1e30 min(max(row - col, 0), 1)
#ifdef ARCH_WORMHOLE
                            TTI_SFPMOV(0, L6, L0, 1);
                            TTI_SFPMAD(L0, ONE, ZERO, L0, 0);
#else
                            TTI_SFPMAD(L6, ONE, ZERO, L0, 1);
#endif
                            TTI_SFPNOP;
                            TT_SFPADDI(bf16_bits(static_cast<float>(row - col)), L0, 0);
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMULI(0xF14A, L0, 0);
                            break;
                        case Const::Tail:  // -1e30 min(max(col - tail + 1, 0), 1)
                            TTI_SFPMOV(0, L4, L0, 0);
                            TT_SFPADDI(
                                bf16_bits(
                                    static_cast<float>(
                                        col - static_cast<int32_t>(tail) + 1
                                    )
                                ),
                                L0,
                                0
                            );
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMULI(0xF14A, L0, 0);
                            break;
                        case Const::CausalTail:  // -1e30 where Causal or Tail is
                            TTI_SFPMOV(0, L6, L0, 0);
                            TT_SFPADDI(bf16_bits(static_cast<float>(col - row)), L0, 0);
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMOV(0, L0, L2, 0);
                            TTI_SFPMOV(0, L4, L0, 0);
                            TT_SFPADDI(
                                bf16_bits(
                                    static_cast<float>(
                                        col - static_cast<int32_t>(tail) + 1
                                    )
                                ),
                                L0,
                                0
                            );
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMAD(L0, ONE, L2, L0, 0);
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMULI(0xF14A, L0, 0);
                            break;
                        case Const::TailRows:  // -1e30 min(max(row - tail + 1, 0), 1)
                            TTI_SFPMOV(0, L5, L0, 0);
                            TT_SFPADDI(
                                bf16_bits(
                                    static_cast<float>(
                                        row - static_cast<int32_t>(tail) + 1
                                    )
                                ),
                                L0,
                                0
                            );
                            TTI_SFPNOP;
                            clamp01_l0();
                            TTI_SFPMULI(0xF14A, L0, 0);
                            break;
                    }
                    // MAD-unit results take one more instruction before they are read.
                    TTI_SFPNOP;
                    TT_SFPSTORE(L0, 0, AM_STAY, 64 * t + 16 * f + 4 * j + 2 * p);
                }
            }
        }
    }

}  // namespace flash
