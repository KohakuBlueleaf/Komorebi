// The frame's stages on the vector unit, 32 pixels at a time: primary rays from the
// camera (GEN), shading and the secondary rays (SHADE1, SHADE2), the colour (IMAGE).
#pragma once

#include <stdint.h>

#include "divsqrt.hpp"
#include "raytrace.h"
#include "sfpu.hpp"

namespace shade {

    using namespace rt;

    // Constants of the reference (reference/raytrace.py) as float32 bits.
    constexpr uint32_t kTmax = 0x7149F2CA;                // 1e30
    constexpr uint32_t kMinDir = 0x2B8CBCCC;              // 1e-12
    constexpr uint32_t kMinSquaredLength = 0x2B8CBCCC;    // 1e-12
    constexpr uint32_t kMinSquaredSum = 0x358637BD;       // 1e-6
    constexpr uint32_t kAmbient = 0x3DCCCCCD;             // 0.1
    constexpr uint32_t kGlobalIllumination = 0x3F000000;  // 0.5
    constexpr uint32_t kHalf = 0x3F000000;
    constexpr uint32_t kOne = 0x3F800000;
    constexpr uint32_t kMinusOne = 0xBF800000;
    constexpr uint32_t kTwo = 0x40000000;
    constexpr uint32_t kUnit24 = 0x33800000;  // 2^-24
    constexpr uint32_t kSign = 0x80000000;
    // The hash (reference/rng.py): multipliers, and the seed of the bounce directions.
    constexpr uint32_t kGolden = 0x9E3779B1, kMul1 = 0x7FEB352D, kMul2 = 0x846CA68B;
    constexpr uint32_t kStream = 0x85EBCA6B, kSeed = 7;

    // Slots: the stage's input pages, output pages; scratch; the lane ramp (dx, dy,
    // pixel offset); the pixel numbers; bounce directions.
    constexpr uint32_t kInput = 0;         // up to 48 vectors of input pages
    constexpr uint32_t kOut = 48;          // up to 32 vectors of output pages
    constexpr uint32_t kScratchBase = 80;  // scratch: 30 slots
    constexpr uint32_t kRamp = 110, kPixels = 113, kBounceDirections = 114;

    // Scratch slot names (kScratchBase onwards).
    constexpr uint32_t kPoint = kScratchBase;
    constexpr uint32_t kNormal = kScratchBase + 3;
    constexpr uint32_t kShadowOrigin = kScratchBase + 6;
    constexpr uint32_t kToLight = kScratchBase + 9;
    constexpr uint32_t kRandomVector = kScratchBase + 12;
    constexpr uint32_t kBounce = kScratchBase + 15;
    constexpr uint32_t kHitMask = kScratchBase + 18;
    constexpr uint32_t kScratch0 = kScratchBase + 19;
    constexpr uint32_t kScratch1 = kScratchBase + 20;
    constexpr uint32_t kScratch2 = kScratchBase + 21;
    constexpr uint32_t kScratch3 = kScratchBase + 22;
    constexpr uint32_t kAlbedo = kScratchBase + 23;
    constexpr uint32_t kCosine = kScratchBase + 26;
    constexpr uint32_t kInverse = kScratchBase + 27;

    // Two-operand steps on slots, one float32 rounding each. The long ones are calls:
    // the kernels' text must fit the program's configuration buffer.
    inline void smul(uint32_t left, uint32_t right, uint32_t destination) {
        load(L0, left);
        load(L1, right);
        mul(L0, L1, L2);
        store(L2, destination);
    }

    inline void sadd(uint32_t left, uint32_t right, uint32_t destination) {
        load(L0, left);
        load(L1, right);
        add(L0, L1, L2);
        store(L2, destination);
    }

    inline void ssub(uint32_t left, uint32_t right, uint32_t destination) {
        load(L0, left);
        load(L1, right);
        sub(L0, L1, L2);
        store(L2, destination);
    }

    // destination = source * constant, constant + source, constant - source or
    // source - constant, with the constant given as float32 bits.
    inline void smul_c(uint32_t source, uint32_t constant, uint32_t destination) {
        load(L0, source);
        immediate(L1, constant);
        mul(L0, L1, L2);
        store(L2, destination);
    }

    inline void sadd_c(uint32_t source, uint32_t constant, uint32_t destination) {
        load(L0, source);
        immediate(L1, constant);
        add(L1, L0, L2);
        store(L2, destination);
    }

    inline void ssub_from_c(uint32_t source, uint32_t constant, uint32_t destination) {
        load(L0, source);
        immediate(L1, constant);
        sub(L1, L0, L2);
        store(L2, destination);
    }

    inline void ssub_c(uint32_t source, uint32_t constant, uint32_t destination) {
        load(L0, source);
        immediate(L1, constant);
        sub(L0, L1, L2);
        store(L2, destination);
    }

    __attribute__((noinline)) void sdiv(
        uint32_t left, uint32_t right, uint32_t destination
    ) {
        load(L0, left);
        load(L1, right);
        divide();
        store(L2, destination);
    }

    __attribute__((noinline)) void ssqrt(uint32_t source, uint32_t destination) {
        load(L0, source);
        square_root();
        store(L1, destination);
    }

    __attribute__((noinline)) void srecip(uint32_t source, uint32_t destination) {
        load(L0, source);
        recip(L0, L1, L2, L3, L4, L5);
        store(L1, destination);
    }

    inline void copy(uint32_t source, uint32_t destination) {
        load(L0, source);
        store(L0, destination);
    }

    inline void fill(uint32_t destination, uint32_t bits) {
        immediate(L0, bits);
        store(L0, destination);
    }

    // destination = (left0 right0 + left1 right1) + left2 right2 over three slots each.
    inline void dot(uint32_t left, uint32_t right, uint32_t destination) {
        load(L0, left);
        load(L1, right);
        mul(L0, L1, L2);
        load(L0, left + 1);
        load(L1, right + 1);
        mul(L0, L1, L3);
        add(L2, L3, L2);
        load(L0, left + 2);
        load(L1, right + 2);
        mul(L0, L1, L3);
        add(L2, L3, L2);
        store(L2, destination);
    }

    // max(source, 0) and min(max(source, 0), 1) into destination.
    inline void max0(uint32_t source, uint32_t destination) {
        load(L0, source);
        TTI_SFPLOADI(L1, kFloatB, 0);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L0, L1, 1);  // L0 = max
        store(L0, destination);
    }

    inline void clamp01(uint32_t source, uint32_t destination) {
        max0(source, destination);
        load(L0, destination);
        immediate(L1, kOne);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L1, L0, 1);  // L0 = min
        store(L0, destination);
    }

    // Lanes enabled where `slot` has its sign bit set (a hit flag, an index of -1);
    // between push and pop.
    inline void where_negative(uint32_t slot) {
        load(L0, slot);
        TTI_SFPSETCC(0, L0, 0, kLt0);
    }

    // Sky colour of the directions in slots directions..+2 into slots colours..+2
    // (reference sky()).
    inline void sky(uint32_t directions, uint32_t colours, const uint32_t* constants) {
        dot(directions, directions, kScratch0);
        ssqrt(kScratch0, kScratch1);
        sdiv(directions + 1, kScratch1, kScratch2);
        clamp01(kScratch2, kScratch2);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul_c(kScratch2, constants[kCnSky + 3 + axis], kScratch3);
            sadd_c(kScratch3, constants[kCnSky + axis], colours + axis);
        }
    }

    // 1 / direction as the reference clamps it: |direction| below 1e-12 becomes 1e-12
    // with the direction's sign.
    __attribute__((noinline)) void inverse_dir(uint32_t directions, uint32_t inverse) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            load(L0, directions + axis);
            TTI_SFPABS(0, L0, L1, kFloatAbs);
            immediate(L2, kMinDir);
            TTI_SFPPUSHC(0, 0, 0, 0);
            TTI_SFPGT(0, L1, L2, kSetCC);  // 1e-12 > |d|
            TT_SFPMOV(0, L0, L3, 0);
            TT_SFPSETSGN(0, L2, L3, 0);
            TT_SFPMOV(0, L3, L0, 0);
            TTI_SFPPOPC(0, 0, 0, 0);
            recip(L0, L1, L2, L3, L4, L5);
            store(L1, inverse + axis);
        }
    }

    // Normal of the hit (slots hit_normals..) turned against directions, into normals;
    // vertex colours (9 slots) at (u, v) into albedo (reference surface()).
    inline void surface(
        uint32_t hit_normals,
        uint32_t colours,
        uint32_t u,
        uint32_t v,
        uint32_t directions,
        uint32_t normals,
        uint32_t albedo
    ) {
        dot(hit_normals, directions, kScratch0);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(hit_normals + axis, normals + axis);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kScratch0);
        TTI_SFPGT(0, ZERO, L0, kSetCC);  // dot(n, d) > 0
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul_c(hit_normals + axis, kMinusOne, normals + axis);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        ssub_from_c(u, kOne, kScratch1);  // (1 - u) - v
        ssub(kScratch1, v, kScratch1);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul(colours + axis, kScratch1, kScratch2);
            smul(colours + 3 + axis, u, kScratch3);
            sadd(kScratch2, kScratch3, kScratch2);
            smul(colours + 6 + axis, v, kScratch3);
            sadd(kScratch2, kScratch3, albedo + axis);
        }
    }

    // From point and normals: shadow_origin = point + normals * eps, to_light = light -
    // shadow_origin, cosine = dot(normals, to_light) / |to_light| (direct_light()).
    inline void light_ray(
        uint32_t point,
        uint32_t normals,
        uint32_t shadow_origin,
        uint32_t to_light,
        uint32_t cosine,
        const uint32_t* constants
    ) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul_c(normals + axis, constants[kCnEps], kScratch0);
            sadd(point + axis, kScratch0, shadow_origin + axis);
            ssub_from_c(
                shadow_origin + axis, constants[kCnLight + axis], to_light + axis
            );
        }
        dot(to_light, to_light, kScratch0);
        ssqrt(kScratch0, kScratch1);
        srecip(kScratch1, kScratch2);
        dot(normals, to_light, kScratch0);
        smul(kScratch0, kScratch2, cosine);
    }

    // point = origin + direction * distance.
    inline void hit_point(
        uint32_t origin, uint32_t direction, uint32_t distance, uint32_t point
    ) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul(direction + axis, distance, kScratch0);
            sadd(origin + axis, kScratch0, point + axis);
        }
    }

    // L0 = L0 * factor mod 2^32 (factor a constant), from 16-bit halves and 23-bit
    // products.
    inline void mul32(uint32_t factor) {
        immediate(L1, 0xFFFF);
        TTI_SFPAND(0, L0, L1, 0);  // x low
        TT_SFPMOV(0, L0, L2, 0);
        TT_SFPSHFT(0xFF0, 0, L2, 1);  // x high (x >> 16)
        immediate(L3, factor & 0xFFFF);
        immediate(L4, factor >> 16);
        TT_SFPMUL24(L1, L3, ZERO, L5, 0);  // low 23 bits of xl cl
        TT_SFPMUL24(L1, L3, ZERO, L6, 1);  // bits 23 on
        TT_SFPMUL24(L1, L4, ZERO, L7, 0);  // xl ch (its low 16 bits count)
        TT_SFPMUL24(L2, L3, ZERO, L3, 0);  // xh cl
        TTI_SFPNOP;
        TTI_SFPNOP;
        TT_SFPIADD(0, L3, L7, kCcNone);
        TT_SFPSHFT(16, 0, L7, 1);
        TT_SFPSHFT(23, 0, L6, 1);
        TT_SFPIADD(0, L6, L5, kCcNone);
        TT_SFPIADD(0, L7, L5, kCcNone);
        TT_SFPMOV(0, L5, L0, 0);
    }

    // L0 ^= L0 >> shift.
    inline void xorshift(uint32_t shift) {
        TT_SFPMOV(0, L0, L1, 0);
        TT_SFPSHFT((-shift) & 0xFFF, 0, L1, 1);
        TTI_SFPXOR(0, L1, L0, 0);
    }

    // Slot hash = pixel * golden of the pixels in slot `pixels`, the hash's first step.
    inline void golden(uint32_t pixels, uint32_t hash) {
        load(L0, pixels);
        mul32(kGolden);
        store(L0, hash);
    }

    // Slot destination = uniform(pixel, stream, seed) from slot hash = golden(pixel)
    // (reference/rng.py).
    inline void uniform(uint32_t hash, uint32_t stream, uint32_t destination) {
        load(L0, hash);
        immediate(L1, (stream + 1) * kStream + kSeed);
        TT_SFPIADD(0, L1, L0, kCcNone);
        xorshift(16);
        mul32(kMul1);
        xorshift(15);
        mul32(kMul2);
        xorshift(16);
        TT_SFPSHFT(0xFF8, 0, L0, 1);  // >> 8: 24 bits
        TT_SFPCAST(L0, L0, 0);
        TTI_SFPNOP;
        immediate(L1, kUnit24);
        mul(L0, L1, L2);
        store(L2, destination);
    }

    // Unit bounce directions `bounce` about the `normals` of the pixels whose golden()
    // is in slot `pixels` (reference bounce_directions()).
    inline void bounce_dir(uint32_t pixels, uint32_t normals, uint32_t bounce) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            uniform(pixels, 4 * axis, kRandomVector + axis);
            for (uint32_t sample = 1; sample < 4; ++sample) {
                uniform(pixels, 4 * axis + sample, kScratch0);
                sadd(kRandomVector + axis, kScratch0, kRandomVector + axis);
            }
            ssub_c(kRandomVector + axis, kTwo, kRandomVector + axis);
        }
        dot(kRandomVector, kRandomVector, kScratch1);
        ssqrt(kScratch1, kScratch2);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            sdiv(kRandomVector + axis, kScratch2, kRandomVector + axis);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kScratch1);
        immediate(L1, kMinSquaredLength);
        TTI_SFPGT(0, L0, L1, kSetCC);  // 1e-12 > g2
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(normals + axis, kRandomVector + axis);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            sadd(normals + axis, kRandomVector + axis, kBounce + axis);
        }
        dot(kBounce, kBounce, kScratch1);
        ssqrt(kScratch1, kScratch2);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            sdiv(kBounce + axis, kScratch2, bounce + axis);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kScratch1);
        immediate(L1, kMinSquaredSum);
        TTI_SFPGT(0, L0, L1, kSetCC);  // 1e-6 > b2
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(normals + axis, bounce + axis);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // A packet's ray slots (origin, direction, inverse, best at `packet`) for the lanes
    // of mask (sign set): origins, directions, tmax; other lanes zero, best -1.
    inline void ray_out(
        uint32_t packet,
        uint32_t mask,
        uint32_t origins,
        uint32_t directions,
        uint32_t tmax
    ) {
        inverse_dir(directions, kInverse);
        for (uint32_t slot = 0; slot < 9; ++slot) {
            fill(packet + slot, 0);
        }
        fill(packet + kPkBest, kMinusOne);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(mask);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(origins + axis, packet + kPkO + axis);
            copy(directions + axis, packet + kPkD + axis);
            copy(kInverse + axis, packet + kPkInv + axis);
        }
        fill(packet + kPkBest, tmax);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // float32 bits of an integer below 2^24 (exact).
    inline uint32_t float_bits(uint32_t integer) {
        if (integer == 0) {
            return 0;
        }
        const uint32_t exponent = 31 - __builtin_clz(integer);
        return ((127 + exponent) << 23) | ((integer << (23 - exponent)) & 0x7FFFFF);
    }

    // Slot destination = the integer constant plus slot source as integers (pixel
    // numbers).
    inline void iadd_c(uint32_t source, uint32_t constant, uint32_t destination) {
        load(L0, source);
        immediate(L1, constant);
        TT_SFPIADD(0, L1, L0, kCcNone);
        store(L0, destination);
    }

    // Slot mask = -1.0 where slot `slot` is not negative (a hit index), else 0.
    inline void flag_not_negative(uint32_t slot, uint32_t mask) {
        fill(mask, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, slot);
        TTI_SFPSETCC(0, L0, 0, kGte0);
        fill(mask, kMinusOne);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // destination = max(cosine, 0), 0 where the shadow ray's best (slot occlusion) is
    // negative: blocked.
    inline void direct(uint32_t cosine, uint32_t occlusion, uint32_t destination) {
        max0(cosine, destination);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(occlusion);
        fill(destination, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // GEN: the primary rays of the packet whose lane 0 is pixel (pixel_x, pixel_y) into
    // a packet at kOut (reference camera_rays()); `constants` is the constant page.
    inline void gen(const uint32_t* constants, uint32_t pixel_x, uint32_t pixel_y) {
        const uint32_t* camera = constants + kCnCamera;
        fill(kScratch1, camera[12]);
        sadd_c(kRamp, float_bits(pixel_x), kScratch0);  // x = float(px) + 0.5
        sadd_c(kScratch0, kHalf, kScratch0);
        ssub_c(kScratch0, camera[13], kScratch0);
        sdiv(kScratch0, kScratch1, kPoint);  // sx
        sadd_c(kRamp + 1, float_bits(pixel_y), kScratch0);
        sadd_c(kScratch0, kHalf, kScratch0);
        ssub_c(kScratch0, camera[14], kScratch0);
        sdiv(kScratch0, kScratch1, kPoint + 1);  // sy
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul_c(kPoint, camera[3 + axis], kScratch0);  // (fwd + right sx) - up sy
            sadd_c(kScratch0, camera[9 + axis], kScratch0);
            smul_c(kPoint + 1, camera[6 + axis], kScratch2);
            ssub(kScratch0, kScratch2, kOut + kPkD + axis);
            fill(kOut + kPkO + axis, camera[axis]);
        }
        inverse_dir(kOut + kPkD, kOut + kPkInv);
        fill(kOut + kPkBest, kTmax);
    }

    // SHADE1: primary packet at kInput, its hits at kInput + kPkVectors; out the shadow
    // and bounce packets and state 1 (render_pixels() up to the first traces).
    inline void shade1(const uint32_t* constants, uint32_t pixel) {
        constexpr uint32_t primary = kInput, hit = kInput + kPkVectors;
        constexpr uint32_t shadow = kOut, bounce = kOut + kPkVectors;
        constexpr uint32_t state = kOut + 2 * kPkVectors;
        flag_not_negative(hit + kHitTri, kHitMask);
        sky(primary + kPkD, state + kS1Col, constants);
        hit_point(primary + kPkO, primary + kPkD, hit + kHitT, kPoint);
        surface(
            hit + kHitN,
            hit + kHitCol,
            hit + kHitU,
            hit + kHitV,
            primary + kPkD,
            kNormal,
            kAlbedo
        );
        light_ray(kPoint, kNormal, kShadowOrigin, kToLight, state + kS1Cos, constants);
        iadd_c(kRamp + 2, pixel, kPixels);
        golden(kPixels, kPixels);
        bounce_dir(kPixels, kNormal, kBounceDirections);
        ray_out(shadow, kHitMask, kShadowOrigin, kToLight, kOne);
        ray_out(bounce, kHitMask, kShadowOrigin, kBounceDirections, kTmax);
        copy(kHitMask, state + kS1Hit);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(kHitMask);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(kAlbedo + axis, state + kS1Col + axis);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // SHADE2: state 1, the shadow hits, the bounce packet and its hits at kInput on;
    // out the shadow packet of the bounce hits and state 2.
    inline void shade2(const uint32_t* constants) {
        constexpr uint32_t state1 = kInput, occlusion = kInput + kS1Vectors;
        constexpr uint32_t bounce_packet = occlusion + kHitAnyVectors;
        constexpr uint32_t hit = bounce_packet + kPkVectors;
        constexpr uint32_t shadow = kOut, state2 = kOut + kPkVectors;
        copy(state1 + kS1Hit, state2 + kS2Hit);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(state1 + kS1Col + axis, state2 + kS2Col + axis);
        }
        direct(state1 + kS1Cos, occlusion + kHitT, kScratch3);
        sadd_c(kScratch3, kAmbient, state2 + kS2Light);
        flag_not_negative(hit + kHitTri, kHitMask);
        copy(kHitMask, state2 + kS2Hit1);
        sky(bounce_packet + kPkD, state2 + kS2Rad, constants);
        hit_point(bounce_packet + kPkO, bounce_packet + kPkD, hit + kHitT, kPoint);
        surface(
            hit + kHitN,
            hit + kHitCol,
            hit + kHitU,
            hit + kHitV,
            bounce_packet + kPkD,
            kNormal,
            state2 + kS2Alb1
        );
        light_ray(
            kPoint, kNormal, kShadowOrigin, kToLight, state2 + kS2Cos1, constants
        );
        ray_out(shadow, kHitMask, kShadowOrigin, kToLight, kOne);
    }

    // IMAGE: state 2 and the second shadow hits at kInput; the colour at kOut.
    inline void image() {
        constexpr uint32_t state2 = kInput, occlusion = kInput + kS2Vectors;
        direct(state2 + kS2Cos1, occlusion + kHitT, kScratch3);
        sadd_c(kScratch3, kAmbient, kScratch3);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            copy(state2 + kS2Rad + axis, kBounce + axis);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(state2 + kS2Hit1);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul(state2 + kS2Alb1 + axis, kScratch3, kBounce + axis);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            smul_c(kBounce + axis, kGlobalIllumination, kScratch0);
            sadd(state2 + kS2Light, kScratch0, kScratch0);
            copy(state2 + kS2Col + axis, kOut + axis);
            TTI_SFPPUSHC(0, 0, 0, 0);
            where_negative(state2 + kS2Hit);
            smul(state2 + kS2Col + axis, kScratch0, kOut + axis);
            TTI_SFPPOPC(0, 0, 0, 0);
        }
    }

}  // namespace shade
