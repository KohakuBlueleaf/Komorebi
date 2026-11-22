// LIGHT's pack thread (shade.h): each item's colour from its surface planes, a quarter
// tile at a time in Dst: texel keys, their pages (page_passes), texels, colour.
namespace shade_pass {

    // Dst for a quarter tile: its planes (row k of plane p at kLightRows p + k), each
    // row's vectors (kRv*), the lanes' column + 0.5, (px - cx) / f, 1 / f, the scratch.
    constexpr uint32_t kLightRows = 8;
    constexpr uint32_t kLvAlbedo = 0, kLvMaterial = 3, kLvNormal = 4, kLvVisibility = 7;
    constexpr uint32_t kLvSpecular = 8, kLvDepth = 11, kLvPlanes = 12;
    constexpr uint32_t kLightKeys = 4;
    // a row's: four texels (keys until gathered), the filter's fractions, 1/z, point
    constexpr uint32_t kRvValue = 0, kRvFx = 4, kRvFy = 5, kRvW = 6, kRvPoint = 7;
    constexpr uint32_t kRowVectors = 10;
    constexpr uint32_t kLvRow = kLightRows * kLvPlanes;
    constexpr uint32_t kLvLane = kLvRow + kLightRows * kRowVectors;
    constexpr uint32_t kLvXTerm = kLvLane + 1, kLvInvF = kLvLane + 2;
    constexpr uint32_t kLvScratch = kLvLane + 3;
    // A row's scratch: sky products, max(0, n.s) shadow, shadow, to the eye, lamps,
    // specular sum, a lamp's l and E, gloss, taps' corner, view direction, kind - 4.
    constexpr uint32_t kSxSky = 0;
    constexpr uint32_t kSxSun = 5;
    constexpr uint32_t kSxShadow = 6;
    constexpr uint32_t kSxEye = 7;
    constexpr uint32_t kSxPoints = 10;
    constexpr uint32_t kSxShine = 13, kSxLight = 16, kSxE = 19, kSxGloss = 20;
    constexpr uint32_t kSxCorner = 21, kSxDirection = 23, kSxKind = 26;
    constexpr uint32_t kScratchVectors = 27;
    static_assert(kLvScratch + kScratchVectors <= 256);
    // each Dst plane's page in the item (light_reader.hpp)
    constexpr uint32_t kLightSource[kLvPlanes] = {0, 1, 2, 11, 3, 4, 5, 6, 7, 8, 9, 10};
    constexpr uint32_t kLt0 = 0, kNe0 = 2;                            // SFPSETCC
    constexpr uint32_t kZeroTexel = kShadeSlots * kTexPageBytes + 4;  // after the white

    constexpr uint32_t plane_vector(uint32_t plane, uint32_t k) {
        return kLightRows * plane + k;
    }

    constexpr uint32_t row_vector(uint32_t k, uint32_t at) {
        return kLvRow + kRowVectors * k + at;
    }

    constexpr uint32_t sx_vector(uint32_t at) {
        return kLvScratch + at;
    }

    // q = a / f for `a` (kept): a times 1 / f, then one step on the remainder a - q f.
    inline void over_focal(
        uint32_t a, uint32_t q, uint32_t f, uint32_t inverse, uint32_t f_bits
    ) {
        load(inverse, kLvInvF);
        mul(a, inverse, q);
        immediate(f, f_bits);
        TT_SFPMAD(q, f, a, f, kNegA);
        TT_SFPMAD(f, inverse, q, q, 0);
    }

    // x, y, z over their length (rsqrt); length2, inverse and three work LRegs.
    inline void unit3(
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint32_t length2,
        uint32_t inverse,
        uint32_t work0,
        uint32_t work1,
        uint32_t work2
    ) {
        mul(x, x, length2);
        TT_SFPMAD(y, y, length2, length2, 0);
        TT_SFPMAD(z, z, length2, length2, 0);
        rsqrt_vector(length2, inverse, work0, work1, work2);
        mul(x, inverse, x);
        mul(y, inverse, y);
        mul(z, inverse, z);
    }

    // L5 = the key (1 + page) << 10 | texel of column L0, row L1 (floats, kept) of a
    // map in 32 x 32 pages from first - 1, `across` a row (float bits); L2 to L7 used.
    inline void tap_key(uint32_t first_bits, uint32_t across_bits) {
        mul(L0, INV32, L2);
        floor_to(L2, L3);
        TT_SFPMAD(L3, C32, L0, L2, kNegA);  // the column in its page
        mul(L1, INV32, L4);
        floor_to(L4, L5);
        TT_SFPMAD(L5, C32, L1, L4, kNegA);  // the row in its page
        TT_SFPMAD(L4, C32, L2, L4, 0);      // the texel in its page
        immediate(L6, across_bits);
        immediate(L7, first_bits);
        TT_SFPMAD(L5, L6, L7, L5, 0);
        add(L5, L3, L5);  // 1 + page
        add(L5, MAGIC, L5);
        add(L4, MAGIC, L4);
        TTI_SFPNOP;
        TT_SFPAND(0, MASK, L5, 0);
        TT_SFPAND(0, MASK, L4, 0);
        TT_SFPSHFT(10, 0, L5, kShiftImmediate);
        TT_SFPOR(0, L4, L5, 0);
    }

    // Row k (numerator cy - py): the point (world_points); its shadow taps' keys, 1/z,
    // fractions (sun_shadow; 0 outside); empty: the background's (octahedral_texel).
    inline void light_keys(
        uint32_t k, const uint32_t* table, uint32_t numerator_bits, uint32_t terms
    ) {
        const uint32_t* cam = table + kLtCamera;
        const uint32_t* sun = table + kLtShadowCamera;
        TTI_SFPENCC(3, 0, 0, 10);
        immediate(L0, numerator_bits);
        over_focal(L0, L1, L2, L3, cam[12]);  // (cy - py) / f
        // the view direction (forward + right sx) - up sy, sy = -(cy - py) / f
        load(L0, kLvXTerm);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            immediate(L2, cam[9 + axis]);
            immediate(L3, cam[3 + axis]);
            TT_SFPMAD(L3, L0, L2, L2, 0);
            immediate(L3, cam[6 + axis]);
            TT_SFPMAD(L3, L1, L2, L2, 0);
            store(L2, sx_vector(kSxDirection + axis));
        }
        // the point: z = 1 / w, ((eye + right xc) + up yc) + forward z
        load(L2, plane_vector(kLvDepth, k));
        recip(L2, L3, L4, L5, L6, L7);
        mul(L0, L3, L0);
        mul(L1, L3, L1);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            immediate(L2, cam[axis]);
            immediate(L4, cam[3 + axis]);
            TT_SFPMAD(L4, L0, L2, L2, 0);
            immediate(L4, cam[6 + axis]);
            TT_SFPMAD(L4, L1, L2, L2, 0);
            immediate(L4, cam[9 + axis]);
            TT_SFPMAD(L4, L3, L2, L2, 0);
            store(L2, row_vector(k, kRvPoint + axis));
        }
        if ((terms & kTermShadow) != 0) {
            // d = (P + n texel) - eye of the sun's camera, along its right, up, forward
            immediate(L7, table[kLtShadowTexel]);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                load(L0, row_vector(k, kRvPoint + axis));
                load(L1, plane_vector(kLvNormal + axis, k));
                TT_SFPMAD(L1, L7, L0, L0, 0);
                immediate(L1, sun[axis]);
                sub(L0, L1, L0);
                for (uint32_t along = 0; along < 3; ++along) {
                    immediate(L1, sun[3 + 3 * along + axis]);
                    if (axis == 0) {
                        mul(L0, L1, L2 + along);
                    } else {
                        TT_SFPMAD(L0, L1, L2 + along, L2 + along, 0);
                    }
                }
            }
            recip(L4, L5, L0, L1, L6, L7);
            store(L5, row_vector(k, kRvW));
            // screen x = cx + f x w, y = cy - f y w; less a half, floored
            mul(L2, L5, L2);
            mul(L3, L5, L3);
            immediate(L0, sun[12]);
            immediate(L1, sun[13]);
            TT_SFPMAD(L0, L2, L1, L2, 0);
            immediate(L1, sun[14]);
            TT_SFPMAD(L0, L3, L1, L3, kNegA);
            TT_SFPLOADI(L1, kFloatB, kHalfB);
            sub(L2, L1, L2);
            sub(L3, L1, L3);
            floor_to(L2, L4);
            floor_to(L3, L5);
            sub(L2, L4, L2);
            sub(L3, L5, L3);
            store(L2, row_vector(k, kRvFx));
            store(L3, row_vector(k, kRvFy));
            store(L4, sx_vector(kSxCorner));
            store(L5, sx_vector(kSxCorner + 1));
            for (uint32_t tap = 0; tap < kLightKeys; ++tap) {
                load(L0, sx_vector(kSxCorner));
                load(L1, sx_vector(kSxCorner + 1));
                if ((tap & 1) != 0) {
                    add(L0, ONE, L0);
                }
                if ((tap & 2) != 0) {
                    add(L1, ONE, L1);
                }
                tap_key(table[kLtShadowFirst], table[kLtShadowAcross]);
                // outside the map: 0
                immediate(L6, table[kLtShadowSide]);
                for (uint32_t axis = 0; axis < 2; ++axis) {
                    TT_SFPSETCC(0, L0 + axis, 0, kLt0);
                    add(ZERO, ZERO, L5);
                    TTI_SFPENCC(3, 0, 0, 10);
                    sub(L0 + axis, L6, L7);
                    TTI_SFPNOP;
                    TT_SFPSETCC(0, L7, 0, kGte0);
                    add(ZERO, ZERO, L5);
                    TTI_SFPENCC(3, 0, 0, 10);
                }
                store(L5, row_vector(k, kRvValue + tap));
            }
        } else {
            add(ZERO, ZERO, L0);
            for (uint32_t tap = 0; tap < kLightKeys; ++tap) {
                store(L0, row_vector(k, kRvValue + tap));
            }
        }
        add(ZERO, ZERO, L5);
        add(ZERO, ZERO, L6);
        add(ZERO, ZERO, L7);
        if ((terms & kTermBackground) != 0) {
            // the view direction turned by -turn about y (raster.py background):
            // (c x + s z, y, c z - s x), each product and sum rounded on its own
            load(L0, sx_vector(kSxDirection));
            load(L2, sx_vector(kSxDirection + 2));
            immediate(L3, table[kLtTurnCos]);
            immediate(L4, table[kLtTurnSin]);
            mul(L3, L0, L5);
            mul(L4, L2, L6);
            add(L5, L6, L5);
            mul(L3, L2, L6);
            mul(L4, L0, L7);
            sub(L6, L7, L6);
            store(L5, sx_vector(kSxDirection));
            store(L6, sx_vector(kSxDirection + 2));
            // (u, v) = (x, z) / (|x| + |y| + |z|), each by its reciprocal and a step
            load(L0, sx_vector(kSxDirection));
            load(L1, sx_vector(kSxDirection + 1));
            load(L2, sx_vector(kSxDirection + 2));
            TTI_SFPABS(0, L0, L3, 1);
            TTI_SFPABS(0, L1, L4, 1);
            add(L3, L4, L3);
            TTI_SFPABS(0, L2, L4, 1);
            add(L3, L4, L3);
            recip(L3, L4, L5, L6, L7, L1);
            mul(L0, L4, L5);
            TT_SFPMAD(L5, L3, L0, L6, kNegA);
            TT_SFPMAD(L6, L4, L5, L5, 0);
            load(L0, sx_vector(kSxDirection + 2));
            mul(L0, L4, L6);
            TT_SFPMAD(L6, L3, L0, L7, kNegA);
            TT_SFPMAD(L7, L4, L6, L6, 0);
            // where y < 0: ((1 - |v|) sign u, (1 - |u|) sign v)
            TTI_SFPNOP;
            TTI_SFPABS(0, L6, L0, 1);
            sub(ONE, L0, L0);
            TTI_SFPABS(0, L5, L2, 1);
            sub(ONE, L2, L2);
            TTI_SFPNOP;
            TT_SFPSETCC(0, L5, 0, kLt0);
            TT_SFPMAD(L0, ONE, ZERO, L0, kNegA);
            TTI_SFPENCC(3, 0, 0, 10);
            TT_SFPSETCC(0, L6, 0, kLt0);
            TT_SFPMAD(L2, ONE, ZERO, L2, kNegA);
            TTI_SFPENCC(3, 0, 0, 10);
            load(L1, sx_vector(kSxDirection + 1));
            TT_SFPSETCC(0, L1, 0, kLt0);
            add(L0, ZERO, L5);
            add(L2, ZERO, L6);
            TTI_SFPENCC(3, 0, 0, 10);
            // column, row: floor((t + 1) / 2 x side), at most side - 1, at least 0
            TT_SFPLOADI(L7, kFloatB, kHalfB);
            immediate(L4, table[kLtOctaSide]);
            add(L5, ONE, L5);
            mul(L5, L7, L5);
            mul(L5, L4, L5);
            add(L6, ONE, L6);
            mul(L6, L7, L6);
            mul(L6, L4, L6);
            floor_to(L5, L0);
            floor_to(L6, L1);
            for (uint32_t axis = 0; axis < 2; ++axis) {
                immediate(L4, table[kLtOctaLast]);
                TTI_SFPNOP;
                TT_SFPSWAP(0, L4, L0 + axis, kMinMax);
                at_least_zero(L0 + axis, L4);
            }
            tap_key(table[kLtOctaFirst], table[kLtOctaAcross]);
            immediate(L4, table[kLtOctaChannel]);
            TT_SFPMOV(0, L5, L6, 0);
            TT_SFPIADD(0, L4, L6, 4);
            TT_SFPMOV(0, L6, L7, 0);
            TT_SFPIADD(0, L4, L7, 4);
        }
        // an empty pixel's keys (bit 31 of its material word)
        add(ZERO, ZERO, L4);
        load(L0, plane_vector(kLvMaterial, k));
        TT_SFPSETCC(0, L0, 0, kLt0);
        store(L5, row_vector(k, kRvValue));
        store(L6, row_vector(k, kRvValue + 1));
        store(L7, row_vector(k, kRvValue + 2));
        store(L4, row_vector(k, kRvValue + 3));
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // Row k's gloss max(0, n . unit(l + v))^exponent, l in L0 to L2, the exponent by
    // squaring (raster.py power, `bits` steps) into the gloss vector; every LReg used.
    inline void gloss(uint32_t k, uint32_t bits) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            load(L3, sx_vector(kSxEye + axis));
            add(L0 + axis, L3, L0 + axis);
        }
        unit3(L0, L1, L2, L3, L4, L5, L6, L7);
        load(L3, plane_vector(kLvNormal, k));
        mul(L3, L0, L4);
        load(L3, plane_vector(kLvNormal + 1, k));
        TT_SFPMAD(L3, L1, L4, L4, 0);
        load(L3, plane_vector(kLvNormal + 2, k));
        TT_SFPMAD(L3, L2, L4, L4, 0);
        TTI_SFPNOP;
        at_least_zero(L4, L3);
        load(L5, plane_vector(kLvMaterial, k));
        TT_SFPSHFT(
            (-static_cast<int32_t>(kMaterialExponentShift)) & 0xFFF,
            0,
            L5,
            kShiftImmediate
        );
        add(ONE, ZERO, L6);
        immediate(L7, 1);
        for (uint32_t bit = 0; bit < bits; ++bit) {
            TT_SFPMOV(0, L5, L3, 0);
            TT_SFPAND(0, L7, L3, 0);
            TT_SFPSETCC(0, L3, 0, kNe0);
            mul(L6, L4, L6);
            TTI_SFPENCC(3, 0, 0, 10);
            if (bit + 1 < bits) {
                mul(L4, L4, L4);
                TT_SFPSHFT(0xFFF, 0, L5, kShiftImmediate);
            }
        }
        store(L6, sx_vector(kSxGloss));
    }

    // Row k's specular sum += specular x gloss x (the light's colour x `irradiance` /
    // pi): the colour's channel c the immediate light[stride c].
    inline void shine(
        uint32_t k,
        const uint32_t* light,
        uint32_t stride,
        uint32_t irradiance,
        const uint32_t* table
    ) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            load(L0, plane_vector(kLvSpecular + channel, k));
            load(L1, sx_vector(kSxGloss));
            mul(L0, L1, L0);
            immediate(L2, light[stride * channel]);
            load(L3, irradiance);
            mul(L2, L3, L2);
            immediate(L3, table[kLtInvPi]);
            mul(L2, L3, L2);
            mul(L0, L2, L0);
            load(L1, sx_vector(kSxShine + channel));
            add(L1, L0, L1);
            store(L1, sx_vector(kSxShine + channel));
        }
    }

    // Row k under one lamp (reference/lights.py point_irradiance): its irradiance in
    // the lamps' sum, with the specular term its share of the specular sum.
    inline void lamp_row(
        uint32_t k,
        const uint32_t* lamp,
        const uint32_t* table,
        bool shiny,
        uint32_t bits
    ) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            immediate(L0, lamp[axis]);
            load(L1, row_vector(k, kRvPoint + axis));
            sub(L0, L1, L2 + axis);
        }
        mul(L2, L2, L5);
        TT_SFPMAD(L3, L3, L5, L5, 0);
        TT_SFPMAD(L4, L4, L5, L5, 0);  // d^2
        rsqrt_vector(L5, L6, L0, L1, L7);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            mul(L2 + axis, L6, L2 + axis);
            store(L2 + axis, sx_vector(kSxLight + axis));
        }
        mul(L5, L6, L6);  // d
        load(L0, plane_vector(kLvNormal, k));
        mul(L0, L2, L7);
        load(L0, plane_vector(kLvNormal + 1, k));
        TT_SFPMAD(L0, L3, L7, L7, 0);
        load(L0, plane_vector(kLvNormal + 2, k));
        TT_SFPMAD(L0, L4, L7, L7, 0);
        TTI_SFPNOP;
        at_least_zero(L7, L0);
        immediate(L0, lamp[kLampIntensity]);
        mul(L0, L7, L7);
        immediate(L0, lamp[kLampSize2]);
        add(L5, L0, L0);
        recip(L0, L1, L2, L3, L4, L5);
        mul(L7, L1, L7);  // I max(0, n.l) / (d^2 + s^2)
        immediate(L0, lamp[kLampInvRadius]);
        mul(L6, L0, L1);
        mul(L1, L1, L1);
        mul(L1, L1, L2);
        sub(ONE, L2, L2);
        mul(L2, L2, L2);
        mul(L7, L2, L7);  // x (1 - (d / r)^4)^2
        immediate(L0, lamp[kLampRadius]);
        sub(L6, L0, L1);
        TTI_SFPNOP;
        TT_SFPSETCC(0, L1, 0, kGte0);
        add(ZERO, ZERO, L7);
        TTI_SFPENCC(3, 0, 0, 10);
        store(L7, sx_vector(kSxE));
        for (uint32_t channel = 0; channel < 3; ++channel) {
            immediate(L0, lamp[kLampColour + channel]);
            mul(L0, L7, L0);
            load(L1, sx_vector(kSxPoints + channel));
            add(L1, L0, L1);
            store(L1, sx_vector(kSxPoints + channel));
        }
        if (shiny) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                load(L0 + axis, sx_vector(kSxLight + axis));
            }
            gloss(k, bits);
            shine(k, lamp + kLampColour, 1, sx_vector(kSxE), table);
        }
    }

    // A command's numbers from the table.
    struct LightNumbers {
        const uint32_t* table;
        uint32_t terms, lamps, bits;
    };

    // Row k's colour into a's rows: (a / pi) E + specular terms (E: sky, ambient, sun x
    // its 2 x 2 filter, lamps); emissive a; empty the background; x / (1 + x).
    inline void light_row(uint32_t k, const LightNumbers& numbers, uint32_t* values) {
        const uint32_t* table = numbers.table;
        const uint32_t terms = numbers.terms;
        uint32_t mark = km::wall();
        auto lap = [&](uint32_t slot) {
            tensix_sync();
            const uint32_t now = km::wall();
            values[slot] += now - mark;
            mark = now;
        };
        TTI_SFPENCC(3, 0, 0, 10);
        if ((terms & kTermShadow) != 0) {
            // lit at a tap unless 1/z < its texel's - bias; then bilinear weights
            load(L0, row_vector(k, kRvW));
            immediate(L1, table[kLtShadowBias]);
            for (uint32_t tap = 0; tap < kLightKeys; ++tap) {
                load(L2, row_vector(k, kRvValue + tap));
                sub(L2, L1, L2);
                sub(L0, L2, L3);
                add(ONE, ZERO, L4 + tap);
                TTI_SFPNOP;
                TT_SFPSETCC(0, L3, 0, kLt0);
                add(ZERO, ZERO, L4 + tap);
                TTI_SFPENCC(3, 0, 0, 10);
            }
            load(L0, row_vector(k, kRvFx));
            sub(ONE, L0, L1);
            mul(L4, L1, L2);
            TT_SFPMAD(L5, L0, L2, L2, 0);
            mul(L6, L1, L3);
            TT_SFPMAD(L7, L0, L3, L3, 0);
            load(L0, row_vector(k, kRvFy));
            sub(ONE, L0, L1);
            mul(L2, L1, L2);
            TT_SFPMAD(L3, L0, L2, L2, 0);
            store(L2, sx_vector(kSxShadow));
            lap(kLightProfShadow);
        }
        load(L0, plane_vector(kLvNormal, k));
        load(L1, plane_vector(kLvNormal + 1, k));
        load(L2, plane_vector(kLvNormal + 2, k));
        mul(L0, L0, L3);
        TT_SFPMAD(L1, L1, L3, L3, kNegA);
        store(L3, sx_vector(kSxSky));
        mul(L2, L2, L3);
        store(L3, sx_vector(kSxSky + 1));
        mul(L0, L1, L3);
        store(L3, sx_vector(kSxSky + 2));
        mul(L0, L2, L3);
        store(L3, sx_vector(kSxSky + 3));
        mul(L1, L2, L3);
        store(L3, sx_vector(kSxSky + 4));
        immediate(L4, table[kLtSunDirection]);
        mul(L0, L4, L3);
        immediate(L4, table[kLtSunDirection + 1]);
        TT_SFPMAD(L1, L4, L3, L3, 0);
        immediate(L4, table[kLtSunDirection + 2]);
        TT_SFPMAD(L2, L4, L3, L3, 0);
        TTI_SFPNOP;
        at_least_zero(L3, L4);
        if ((terms & kTermShadow) != 0) {
            load(L4, sx_vector(kSxShadow));
            mul(L3, L4, L3);
        }
        store(L3, sx_vector(kSxSun));
        // the kind less emissive's (0: emissive)
        load(L7, plane_vector(kLvMaterial, k));
        TT_SFPSHFT(
            (-static_cast<int32_t>(kMaterialKindShift)) & 0xFFF, 0, L7, kShiftImmediate
        );
        immediate(L6, kMaterialKindMask);
        TT_SFPAND(0, L6, L7, 0);
        TT_SFPIADD((-static_cast<int32_t>(kKindEmissive)) & 0xFFF, L7, L7, 1 | 4);
        store(L7, sx_vector(kSxKind));
        add(ZERO, ZERO, L0);
        for (uint32_t i = 0; i < 6; ++i) {
            store(L0, sx_vector(kSxPoints + i));  // and the specular sum
        }
        const bool shiny = (terms & kTermSpecular) != 0;
        if (shiny) {
            const uint32_t* cam = table + kLtCamera;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                immediate(L0, cam[axis]);
                load(L1, row_vector(k, kRvPoint + axis));
                sub(L0, L1, L2 + axis);
            }
            unit3(L2, L3, L4, L0, L1, L5, L6, L7);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                store(L2 + axis, sx_vector(kSxEye + axis));
                immediate(L0 + axis, table[kLtSunDirection + axis]);
            }
            gloss(k, numbers.bits);
            shine(k, table + kLtSun, kLtChannel, sx_vector(kSxSun), table);
            lap(kLightProfSpecular);
        }
        if ((terms & kTermLamps) != 0) {
            for (uint32_t lamp = 0; lamp < numbers.lamps; ++lamp) {
                lamp_row(
                    k, table + kLtLamp + kLampWords * lamp, table, shiny, numbers.bits
                );
            }
            lap(kLightProfLamps);
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const uint32_t* sky = table + kLtChannel * channel;
            load(L0, plane_vector(kLvNormal, k));
            load(L1, plane_vector(kLvNormal + 1, k));
            load(L2, plane_vector(kLvNormal + 2, k));
            immediate(L3, sky[kLtSky]);
            // K0 + KZ z z + KX (x x - y y) + (x y, x z, y z terms) + (x, y, z terms)
            constexpr uint32_t kScratchOf[5] = {1, 0, 2, 3, 4};
            for (uint32_t term = 0; term < 5; ++term) {
                immediate(L4, sky[kLtSky + 1 + term]);
                load(L5, sx_vector(kSxSky + kScratchOf[term]));
                TT_SFPMAD(L5, L4, L3, L3, 0);
            }
            for (uint32_t axis = 0; axis < 3; ++axis) {
                immediate(L4, sky[kLtSky + 6 + axis]);
                TT_SFPMAD(L0 + axis, L4, L3, L3, 0);
            }
            TTI_SFPNOP;
            at_least_zero(L3, L4);
            load(L5, plane_vector(kLvVisibility, k));
            mul(L3, L5, L3);
            immediate(L4, sky[kLtAmbient]);
            add(L3, L4, L3);
            immediate(L4, sky[kLtSun]);
            load(L5, sx_vector(kSxSun));
            TT_SFPMAD(L4, L5, L3, L3, 0);
            load(L4, sx_vector(kSxPoints + channel));
            add(L3, L4, L3);
            load(L5, plane_vector(kLvAlbedo + channel, k));
            immediate(L4, table[kLtInvPi]);
            mul(L5, L4, L6);
            mul(L6, L3, L3);
            load(L4, sx_vector(kSxShine + channel));
            add(L3, L4, L3);
            // emissive: a; empty (bit 31): the background
            load(L7, sx_vector(kSxKind));
            TT_SFPSETCC(0, L7, 0, kEq0);
            add(L5, ZERO, L3);
            TTI_SFPENCC(3, 0, 0, 10);
            if ((terms & kTermBackground) != 0) {
                load(L4, row_vector(k, kRvValue + channel));
            } else {
                immediate(L4, table[kLtBackground + channel]);
            }
            load(L7, plane_vector(kLvMaterial, k));
            TT_SFPSETCC(0, L7, 0, kLt0);
            add(L4, ZERO, L3);
            TTI_SFPENCC(3, 0, 0, 10);
            immediate(L4, table[kLtExposure]);
            mul(L3, L4, L3);
            add(L3, ONE, L4);
            recip(L4, L5, L6, L7, L0, L1);
            mul(L3, L5, L6);
            store(L6, plane_vector(kLvAlbedo + channel, k));
        }
        lap(kLightProfMath);
    }

    // LIGHT of one command: each item's word (kCbIn), planes a quarter tile at a time
    // into Dst, keys, pages and texels, colour, out to kCbOut's first three pages.
    inline void light_pack(const km_command_t& command) {
        (void)command;
        uint32_t values[kLightProfTotal + 1] = {};
        const uint32_t started = km::wall();
        setup();
        bool succeeded = true;
        const uint32_t table_address = front(kCbShadeTable, 1);
        const uint32_t* table = reinterpret_cast<const uint32_t*>(table_address);
        LightNumbers numbers = {table, 0, 0, 0};
        if (table_address == 0) {
            succeeded = false;
        } else {
            numbers.terms = table[kLtTerms];
            numbers.lamps = table[kLtLamps];
            numbers.bits = table[kLtExponentBits];
            if (numbers.lamps > kLampsMax || numbers.bits > 11) {
                fill_k::give_up(kGaveBadItem, kWhereShadeBad);
                succeeded = false;
            }
        }
        const bool keyed = (
            succeeded && (numbers.terms & (kTermShadow | kTermBackground)) != 0
        );
        const bool pointed = (numbers.terms & (kTermLamps | kTermSpecular)) != 0;
        const uint32_t page_limit = succeeded ? table[kLtPages] : 0;
        const uint32_t* cam = table + kLtCamera;
        const uint32_t slots_address = write_ptr(kCbShadeSlots);
        *reinterpret_cast<volatile uint32_t*>(slots_address + kZeroTexel) = 0;
        uint32_t* shade_work = reinterpret_cast<uint32_t*>(write_ptr(kCbShadeWork));
        Pages pages;
        pages.keys = shade_work + 1024;
        pages.slots = pages.keys + kHashSize;
        pages.used = pages.slots + kHashSize;
        for (uint32_t i = 0; i < kHashSize; ++i) {
            pages.keys[i] = 0;
        }
        uint32_t* work = reinterpret_cast<uint32_t*>(write_ptr(kCbLightWork));
        uint32_t* index = work + kLwIndex;  // kLightKeys x 1,024 texel addresses
        const uint32_t* slots = reinterpret_cast<const uint32_t*>(slots_address);
        const size_t vl = __riscv_vsetvl_e32m8(32);
        if (succeeded) {
            // the lanes' column + 0.5, 1 / f
            for (uint32_t lane = 0; lane < 32; ++lane) {
                window()[word(kLvLane) + 2 * lane] = as_bits(
                    static_cast<float>(lane) + 0.5f
                );
            }
            pass_k::setup_pass::order(word(kLvLane) + 62);
            TTI_SFPENCC(3, 0, 0, 10);
            immediate(L0, cam[12]);
            recip(L0, L1, L2, L3, L4, L5);
            store(L1, kLvInvF);
        }
        PassCounts counts;
        while (succeeded) {
            uint32_t mark = km::wall();
            const uint32_t answer = front(kCbIn, 1);
            if (answer == 0) {
                succeeded = false;
                break;
            }
            const uint32_t rows = reinterpret_cast<const volatile uint32_t*>(answer)[0];
            const uint32_t tile = reinterpret_cast<const volatile uint32_t*>(answer)[1];
            if (!pop(kCbIn, 1)) {
                succeeded = false;
                break;
            }
            if (rows == kShadeAskEnd) {
                break;
            }
            const uint32_t first_row = rows & 0xFF, end_row = rows >> 8 & 0xFF;
            if (!(first_row < end_row && end_row <= 32 && tile < 1024)) {
                fill_k::give_up(kGaveBadItem, kWhereShadeBad);
                succeeded = false;
                break;
            }
            const uint32_t planes = front(kCbShadePlanes, kLightItemPages);
            if (planes == 0 || !reserve(kCbOut, kShadeOutPages)) {
                succeeded = false;
                break;
            }
            const uint32_t out = write_ptr(kCbOut);
            uint32_t now = km::wall();
            values[kLightProfWait] += now - mark;
            const uint32_t limit = (
                get_local_cb_interface(kCbShadePlanes).fifo_limit << 4
            );
            const uint32_t size = get_local_cb_interface(kCbShadePlanes).fifo_size << 4;
            const uint32_t tile_x = tile % 32, tile_y = tile / 32;
            TTI_SFPENCC(3, 0, 0, 10);
            load(L0, kLvLane);
            immediate(L1, pass_k::float_bits(32 * tile_x));
            add(L0, L1, L0);
            immediate(L1, cam[13]);
            sub(L0, L1, L0);
            over_focal(L0, L2, L3, L4, cam[12]);
            store(L2, kLvXTerm);
            for (
                uint32_t quarter = 0; quarter < 32 / kLightRows && succeeded; ++quarter
            ) {
                const uint32_t row0 = kLightRows * quarter;
                const uint32_t first = first_row > row0 ? first_row : row0;
                const uint32_t end = (
                    end_row < row0 + kLightRows ? end_row : row0 + kLightRows
                );
                if (first >= end) {
                    continue;
                }
                mark = km::wall();
                // a plane's kLightRows rows are 256 words in its page and in Dst
                constexpr uint32_t kWords = 32 * kLightRows;
                for (uint32_t plane = 0; plane < kLvPlanes; ++plane) {
                    uint32_t page = planes + kLightSource[plane] * kOutPageBytes;
                    if (page >= limit) {
                        page -= size;
                    }
                    const uint32_t* from = (
                        reinterpret_cast<const uint32_t*>(page) + kWords * quarter
                    );
                    for (uint32_t i = 0; i < kWords; i += 32) {
                        __riscv_vse32_v_u32m8(
                            const_cast<uint32_t*>(window() + kWords * plane + i),
                            __riscv_vle32_v_u32m8(from + i, vl),
                            vl
                        );
                    }
                }
                asm volatile("" ::: "memory");
                pass_k::setup_pass::order(kWords * kLvPlanes - 1);
                now = km::wall();
                values[kLightProfCopyIn] += now - mark;
                mark = now;
                // cy - py of a pixel row
                auto numerator = [&](uint32_t r) -> uint32_t {
                    const float py = static_cast<float>(32 * tile_y + r) + 0.5f;
                    return as_bits(as_float(cam[14]) - py);
                };
                auto colour = [&](uint32_t pass_first, uint32_t pass_end) -> bool {
                    for (uint32_t r = pass_first; r < pass_end; ++r) {
                        light_row(r - row0, numbers, values);
                        values[kLightProfRows]++;
                    }
                    return true;
                };
                if (!keyed) {
                    if (pointed) {  // the scene point only (keys 0)
                        for (uint32_t r = first; r < end; ++r) {
                            light_keys(r - row0, table, numerator(r), numbers.terms);
                        }
                        tensix_sync();
                        now = km::wall();
                        values[kLightProfKeys] += now - mark;
                    }
                    colour(first, end);
                } else {
                    for (uint32_t r = first; r < end; ++r) {
                        light_keys(r - row0, table, numerator(r), numbers.terms);
                    }
                    tensix_sync();
                    now = km::wall();
                    values[kLightProfKeys] += now - mark;
                    // a row's texel addresses (the zero texel where a key is 0)
                    auto scan = [&](uint32_t row) -> bool {
                        for (uint32_t tap = 0; tap < kLightKeys; ++tap) {
                            if (row_addresses(
                                pages,
                                from_window(
                                    row_vector(row - row0, kRvValue + tap), vl
                                ),
                                kNoTags,
                                index + 1024 * tap + 32 * row,
                                page_limit,
                                kZeroTexel,
                                false,
                                true,
                                vl
                            ) < 0) {
                                return false;
                            }
                        }
                        return true;
                    };
                    auto gather = [&](uint32_t pass_first, uint32_t pass_end) -> bool {
                        const uint32_t gathered = km::wall();
                        for (uint32_t r = pass_first; r < pass_end; ++r) {
                            for (uint32_t tap = 0; tap < kLightKeys; ++tap) {
                                to_window(
                                    row_vector(r - row0, kRvValue + tap),
                                    __riscv_vluxei32_v_u32m8(
                                        slots,
                                        __riscv_vle32_v_u32m8(
                                            index + 1024 * tap + 32 * r, vl
                                        ),
                                        vl
                                    ),
                                    vl
                                );
                            }
                        }
                        asm volatile("" ::: "memory");
                        pass_k::setup_pass::order(
                            word(row_vector(pass_end - 1 - row0, kRvValue + 3)) + 62
                        );
                        values[kLightProfGather] += km::wall() - gathered;
                        return colour(pass_first, pass_end);
                    };
                    if (!page_passes(first, end, pages, counts, scan, gather)) {
                        succeeded = false;
                        break;
                    }
                }
                mark = km::wall();
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    uint32_t* to = (
                        reinterpret_cast<uint32_t*>(
                            out + (1 + channel) * kOutPageBytes
                        ) +
                        kWords * quarter
                    );
                    for (uint32_t i = 0; i < kWords; i += 32) {
                        __riscv_vse32_v_u32m8(
                            to + i,
                            __riscv_vle32_v_u32m8(
                                const_cast<const uint32_t*>(
                                    window() + kWords * (kLvAlbedo + channel) + i
                                ),
                                vl
                            ),
                            vl
                        );
                    }
                }
                asm volatile("" ::: "memory");
                values[kLightProfOut] += km::wall() - mark;
            }
            if (!succeeded) {
                break;
            }
            // popped five pages at a time, as pushed (light_reader.hpp)
            for (
                uint32_t popped = 0; popped < kLightItemPages && succeeded;
                popped += kPlanes
            ) {
                succeeded = pop(kCbShadePlanes, kPlanes);
            }
            if (!succeeded) {
                break;
            }
            push(kCbOut, out + 4 * kOutPageBytes - 4, kShadeOutPages);
            values[kLightProfTiles]++;
        }
        values[kLightProfScan] = counts.scan;
        values[kLightProfWaitPages] = counts.wait;
        values[kLightProfPasses] = counts.passes;
        values[kLightProfPages] = counts.pages;
        // the reader's end, the table back, the unpack thread's end
        if (reserve(kCbShadeAsk, 1)) {
            const uint32_t ask = write_ptr(kCbShadeAsk);
            *reinterpret_cast<volatile uint32_t*>(ask) = kShadeAskEnd;
            push(kCbShadeAsk, ask, 1);
        }
        if (table_address != 0) {
            pop(kCbShadeTable, 1);
        }
        mailbox_write(ThreadId::UnpackThreadId, mail(kOpEnd, 0, 0));
        fill_k::fill::finish();
        if (succeeded && reserve(kCbOut, kShadeOutPages)) {
            values[kLightProfTotal] = km::wall() - started;
            volatile uint32_t* profile = (
                reinterpret_cast<volatile uint32_t*>(write_ptr(kCbOut))
            );
            for (uint32_t i = 0; i < kShadeProfBytes / 4; ++i) {
                profile[i] = i <= kLightProfTotal ? values[i] : 0;
            }
            push(
                kCbOut,
                reinterpret_cast<uint32_t>(profile) + kShadeProfBytes - 4,
                kShadeOutPages
            );
        }
    }

}  // namespace shade_pass
