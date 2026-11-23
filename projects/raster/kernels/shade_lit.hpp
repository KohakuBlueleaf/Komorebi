// Lit SHADE's pack thread (shade.h, a7 the surface directory): each item's surface
// planes, out in three parts; merging (a6), the opaque layer's where the masked loses.
namespace shade_pass {

    using fill_k::kRecCol;
    using fill_k::kRecW;
    static_assert(kShadeNoFrame == ::pass_k::kShadeWordKept);

    // Dst after the records: kLitRows rows at a time, kGatherVectors vectors a row: the
    // texels (albedo linear, normal signed, specular unit), the slot's words 1 to 15.
    constexpr uint32_t kVecGather = 160;
    constexpr uint32_t kGatherVectors = 24;
    constexpr uint32_t kLitRows = 4;
    constexpr uint32_t kGAlbedo = 0, kGNormal = 3, kGSpecular = 6, kGSlot = 9;
    constexpr uint32_t kGBase = kGSlot, kGSpecularColour = kGSlot + 3;
    constexpr uint32_t kGTangent = kGSlot + 6, kGBitangent = kGSlot + 9;
    constexpr uint32_t kGTriangleNormal = kGSlot + 12;
    static_assert(kVecGather + kLitRows * kGatherVectors <= 256);
    static_assert(kGTriangleNormal + 3 == kGatherVectors);
    constexpr uint32_t kMagicRsqrt = 0x5F3759DF;
    constexpr uint32_t kSlotUnfilled = 0xFFFFFFFF;  // a slot's material word until filled
    constexpr uint32_t kThreeHalvesB = 0x3FC0,
                       kHalfB = 0x3F00;  // float32's upper halves

    inline float as_float(uint32_t pattern) {
        float number;
        __builtin_memcpy(&number, &pattern, 4);
        return number;
    }

    inline uint32_t as_bits(float number) {
        uint32_t pattern;
        __builtin_memcpy(&pattern, &number, 4);
        return pattern;
    }

    struct Vec3 {
        float x, y, z;
    };

    inline Vec3 v3(float x, float y, float z) {
        return Vec3{x, y, z};
    }

    inline float dot(const Vec3& a, const Vec3& b) {
        return (a.x * b.x + a.y * b.y) + a.z * b.z;
    }

    inline Vec3 cross(const Vec3& a, const Vec3& b) {
        return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    inline Vec3 scaled(const Vec3& a, float s) {
        return v3(a.x * s, a.y * s, a.z * s);
    }

    // 1 / sqrt(x), x > 0: the magic estimate and three Newton steps (light.py rsqrt).
    inline float rsqrt_scalar(float x) {
        float y = as_float(kMagicRsqrt - (as_bits(x) >> 1));
        const float half = x * 0.5f;
        for (uint32_t step = 0; step < 3; ++step) {
            y = y * (1.5f - half * y * y);
        }
        return y;
    }

    inline Vec3 unit(const Vec3& v) {
        const float length2 = dot(v, v);
        return length2 > 0.0f ? scaled(v, rsqrt_scalar(length2)) : v3(0, 0, 0);
    }

    // Camera-space direction to the scene's: right x + up y + forward z.
    inline Vec3 to_world(const float* cam, const Vec3& v) {
        return (
            v3(
                (cam[3] * v.x + cam[6] * v.y) + cam[9] * v.z,
                (cam[4] * v.x + cam[7] * v.y) + cam[10] * v.z,
                (cam[5] * v.x + cam[8] * v.y) + cam[11] * v.z
            )
        );
    }

    // A winner's slot from its first record: material word, lit base, specular, and
    // (mapped: a normal texture, the host's frame_ok) the frame of its planes.
    inline void fill_slot(
        uint32_t* slot,
        const volatile uint32_t* record,
        uint32_t material,
        const uint32_t* material_at,
        const float* cam,
        bool mapped
    ) {
        const float f = cam[12], cx = cam[13], cy = cam[14];
        Vec3 m[3];
        for (uint32_t plane = 0; plane < 3; ++plane) {
            const uint32_t at = plane == 0 ? kRecW : kRecCol + 3 * (plane - 1);
            const float a = as_float(record[at]), b = as_float(record[at + 1]);
            m[plane] = v3(a, b, (a * cx + b * cy) + as_float(record[at + 2]));
        }
        const Vec3 normal = unit(to_world(cam, v3(-(m[0].x * f), m[0].y * f, -m[0].z)));
        const float det = dot(m[0], cross(m[1], m[2]));
        const float sign = det < 0.0f ? -1.0f : 1.0f;
        const Vec3 u_side = cross(m[2], m[0]), v_side = cross(m[0], m[1]);
        const Vec3 along_u = (
            scaled(to_world(cam, v3(u_side.x, -u_side.y, u_side.z * f)), sign)
        );
        const Vec3 along_v = (
            scaled(to_world(cam, v3(v_side.x, -v_side.y, v_side.z * f)), sign)
        );
        const float along = dot(normal, along_u);
        const Vec3 tangent = (
            unit(
                v3(
                    along_u.x - normal.x * along,
                    along_u.y - normal.y * along,
                    along_u.z - normal.z * along
                )
            )
        );
        Vec3 side = cross(normal, tangent);
        if (dot(side, along_v) < 0.0f) {
            side = scaled(side, -1.0f);
        }
        const bool valid = mapped && det != 0.0f && dot(tangent, tangent) > 0.0f;
        const uint32_t kind = material_at[0] & 0xFF;
        slot[0] = (
            material | kind << kMaterialKindShift |
            (material_at[kMaterialExponent] & kExponentMax) << kMaterialExponentShift
        );
        for (uint32_t channel = 0; channel < 3; ++channel) {
            slot[1 + channel] = material_at[kMaterialLitBase + channel];
            slot[4 + channel] = material_at[kMaterialSpecularColour + channel];
        }
        const Vec3 frame[3] = {
            valid ? tangent : v3(0, 0, 0), valid ? side : v3(0, 0, 0), normal
        };
        for (uint32_t axis = 0; axis < 3; ++axis) {
            slot[7 + 3 * axis] = as_bits(frame[axis].x);
            slot[8 + 3 * axis] = as_bits(frame[axis].y);
            slot[9 + 3 * axis] = as_bits(frame[axis].z);
        }
    }

    // The winners of an item: triangle bits -> slot, open addressing (kLwHashSize).
    struct Winners {
        uint32_t* keys;
        uint32_t* slots;
        uint32_t* words;  // kLitSlotWords a slot, the empty pixels' at kLitSlotEmpty
        uint32_t count = 0;

        void clear(size_t vl) {
            const vuint32m8_t zero = __riscv_vmv_v_x_u32m8(0, vl);
            for (uint32_t i = 0; i < kLwHashSize; i += 32) {
                __riscv_vse32_v_u32m8(keys + i, zero, vl);
            }
            asm volatile("" ::: "memory");
            count = 0;
        }

        // The slot of triangle `bits` (insert: a new one, unfilled); kLitSlotEmpty when
        // not found.
        uint32_t find(uint32_t bits, bool insert) {
            const uint32_t key = bits + 1;
            uint32_t place = (key * 2654435761u) >> 21;
            for (;;) {
                const uint32_t held = keys[place];
                if (held == key) {
                    return slots[place];
                }
                if (held == 0) {
                    if (!insert || count == kLitSlotEmpty) {
                        return kLitSlotEmpty;
                    }
                    keys[place] = key;
                    slots[place] = count;
                    words[kLitSlotWords * count] = kSlotUnfilled;
                    return count++;
                }
                place = (place + 1) & (kLwHashSize - 1);
            }
        }
    };

    // A texture's level's numbers (record_numbers): false without that texture.
    inline bool texture_numbers(
        const uint32_t* entry_words,
        uint32_t entries,
        uint32_t entry_base,
        uint32_t level,
        uint32_t* numbers
    ) {
        if (entry_base == 0 || entry_base - 1 + level >= entries) {
            return false;
        }
        const uint32_t* entry = entry_words + kEntryWords * (entry_base - 1 + level);
        const uint32_t width = entry[1] & 0xFFFF, height = entry[1] >> 16;
        if (width == 0 || height == 0) {
            return false;
        }
        numbers[kConstW] = pass_k::float_bits(width);
        numbers[kConstWLast] = pass_k::float_bits(width - 1);
        numbers[kConstH] = pass_k::float_bits(height);
        numbers[kConstHLast] = pass_k::float_bits(height - 1);
        numbers[kConstAcross] = pass_k::float_bits((width + 31) / 32);
        numbers[kConstFirst] = pass_k::float_bits(entry[0] + 1);
        return true;
    }

    inline void window_row(uint32_t vector, uint32_t* to, size_t vl) {
        __riscv_vse32_v_u32m8(
            to,
            __riscv_vlse32_v_u32m8(
                const_cast<const uint32_t*>(window() + word(vector)), 8, vl
            ),
            vl
        );
    }

    inline void to_window(uint32_t vector, vuint32m8_t value, size_t vl) {
        __riscv_vsse32_v_u32m8(
            const_cast<uint32_t*>(window() + word(vector)), 8, value, vl
        );
    }

    // Dst vector `vector` to row r of the plane page at `page`.
    inline void row_out(uint32_t vector, uint32_t page, uint32_t r, size_t vl) {
        __riscv_vsse32_v_u32m8(
            reinterpret_cast<uint32_t*>(page + 4 * (64 * (r >> 1) + (r & 1))),
            8,
            __riscv_vlse32_v_u32m8(
                const_cast<const uint32_t*>(window() + word(vector)), 8, vl
            ),
            vl
        );
    }

    // Row r's gathered vectors from `at`: three texels by table, the slot's 15 words;
    // merging, the depth row 1 where the masked layer's pixel is taken (shade's rule).
    inline void gather_row(
        uint32_t r,
        uint32_t at,
        const uint32_t* index,
        const uint32_t* slot_offsets,
        const uint32_t* slot_words,
        const uint32_t* table,
        const uint32_t* texels,
        bool merging,
        size_t vl
    ) {
        const vuint32m8_t address = __riscv_vle32_v_u32m8(index + 32 * r, vl);
        const vuint32m8_t albedo_address = (
            merging
                ? __riscv_vand_vx_u32m8(address, (1u << kMaskSlotShift) - 1, vl)
                : address
        );
        if (merging) {
            const vuint32m8_t mask_slot = (
                __riscv_vsrl_vx_u32m8(address, kMaskSlotShift, vl)
            );
            const vuint32m8_t mask_address = (
                __riscv_vmerge_vvm_u32m8(
                    albedo_address,
                    __riscv_vor_vv_u32m8(
                        __riscv_vsll_vx_u32m8(
                            __riscv_vsub_vx_u32m8(mask_slot, 1, vl), 12, vl
                        ),
                        __riscv_vand_vx_u32m8(address, kTexPageBytes - 1, vl),
                        vl
                    ),
                    __riscv_vmsne_vx_u32m8_b4(mask_slot, 0, vl),
                    vl
                )
            );
            const vuint32m8_t alpha_texel = (
                __riscv_vluxei32_v_u32m8(texels, mask_address, vl)
            );
            const vuint32m8_t triangle = (
                __riscv_vlse32_v_u32m8(
                    const_cast<const uint32_t*>(window() + word(kVecTri + r)), 8, vl
                )
            );
            const vbool4_t take = (
                __riscv_vmand_mm_b4(
                    __riscv_vmsgeu_vx_u32m8_b4(alpha_texel, kAlphaOpaque << 24, vl),
                    __riscv_vmsne_vx_u32m8_b4(triangle, kMinusOne, vl),
                    vl
                )
            );
            to_window(
                kVecDepth + r,
                __riscv_vmerge_vxm_u32m8(__riscv_vmv_v_x_u32m8(0, vl), 1, take, vl),
                vl
            );
        }
        const uint32_t* tables[3] = {
            table + kTabGamma, table + kTabSigned, table + kTabUnit
        };
        for (uint32_t texture = 0; texture < 3; ++texture) {
            const vuint32m8_t texel = (
                __riscv_vluxei32_v_u32m8(
                    texels,
                    texture == 0
                        ? albedo_address
                        : __riscv_vle32_v_u32m8(index + 1024 * texture + 32 * r, vl),
                    vl
                )
            );
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const vuint32m8_t byte = (
                    __riscv_vand_vx_u32m8(
                        __riscv_vsrl_vx_u32m8(texel, 8 * channel, vl), 0xFF, vl
                    )
                );
                to_window(
                    at + 3 * texture + channel,
                    __riscv_vluxei32_v_u32m8(
                        tables[texture], __riscv_vsll_vx_u32m8(byte, 2, vl), vl
                    ),
                    vl
                );
            }
        }
        const vuint32m8_t offset = __riscv_vle32_v_u32m8(slot_offsets + 32 * r, vl);
        for (uint32_t w = 1; w < kLitSlotWords; ++w) {
            to_window(
                at + kGSlot + w - 1,
                __riscv_vluxei32_v_u32m8(slot_words + w, offset, vl),
                vl
            );
        }
    }

    // y = 1 / sqrt(x), x > 0 (kept): magic - (bits >> 1), three Newton steps; three
    // LRegs of work.
    inline void rsqrt_vector(
        uint32_t x, uint32_t y, uint32_t work0, uint32_t work1, uint32_t work2
    ) {
        TTI_SFPNOP;
        TT_SFPMOV(0, x, y, 0);
        TT_SFPSHFT(0xFFF, 0, y, kShiftImmediate);
        immediate(work0, kMagicRsqrt);
        TT_SFPIADD(0, work0, y, 2 | 4);  // y = magic - y, no condition
        TT_SFPLOADI(work0, kFloatB, kHalfB);
        mul(x, work0, work1);
        TT_SFPLOADI(work0, kFloatB, kThreeHalvesB);
        for (uint32_t step = 0; step < 3; ++step) {
            mul(work1, y, work2);
            mul(work2, y, work2);
            TT_SFPMAD(work2, ONE, work0, work2, kNegA);
            mul(y, work2, y);
        }
    }

    // A row's surface from its gathered vectors at `at`: a = linear albedo x base,
    // specular x its colour, the normal (T tx + B ty) + N tz over its length.
    inline void surface_row(uint32_t at) {
        TTI_SFPENCC(3, 0, 0, 10);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            load(L0, at + kGAlbedo + channel);
            load(L1, at + kGBase + channel);
            mul(L0, L1, L2);
            store(L2, at + kGAlbedo + channel);
            load(L0, at + kGSpecular + channel);
            load(L1, at + kGSpecularColour + channel);
            mul(L0, L1, L2);
            store(L2, at + kGSpecular + channel);
        }
        load(L0, at + kGNormal);
        load(L1, at + kGNormal + 1);
        load(L2, at + kGNormal + 2);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            load(L3, at + kGTangent + axis);
            load(L4, at + kGBitangent + axis);
            load(L5, at + kGTriangleNormal + axis);
            mul(L3, L0, L6);
            TT_SFPMAD(L4, L1, L6, L6, 0);
            TT_SFPMAD(L5, L2, L6, L6, 0);
            store(L6, at + kGNormal + axis);
        }
        load(L0, at + kGNormal);
        load(L1, at + kGNormal + 1);
        load(L2, at + kGNormal + 2);
        mul(L0, L0, L3);
        TT_SFPMAD(L1, L1, L3, L3, 0);
        TT_SFPMAD(L2, L2, L3, L3, 0);
        rsqrt_vector(L3, L4, L5, L6, L7);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            mul(axis == 0 ? L0 : axis == 1 ? L1 : L2, L4, L6);
            store(L6, at + kGNormal + axis);
        }
    }

    // The three parts' pages of an item in kCbOut (each five pages, none wraps).
    struct Parts {
        uint32_t address[kLitParts];

        void place(uint32_t first) {
            const uint32_t limit = get_local_cb_interface(kCbOut).fifo_limit << 4;
            const uint32_t size = get_local_cb_interface(kCbOut).fifo_size << 4;
            uint32_t at = first;
            for (uint32_t part = 0; part < kLitParts; ++part) {
                address[part] = at;
                at += kShadeOutPages * kOutPageBytes;
                if (at >= limit) {
                    at -= size;
                }
            }
        }

        uint32_t page(uint32_t part, uint32_t plane) const {
            return address[part] + plane * kOutPageBytes;
        }
    };

    // Row r's surface to the parts: the material word | a, normal x, y | normal z,
    // visibility, specular.
    inline void surface_out(
        uint32_t r,
        uint32_t at,
        const Parts& parts,
        const uint32_t* slot_offsets,
        const uint32_t* slot_words,
        size_t vl
    ) {
        __riscv_vsse32_v_u32m8(
            reinterpret_cast<uint32_t*>(
                parts.page(0, kPlanes - 1) + 4 * (64 * (r >> 1) + (r & 1))
            ),
            8,
            __riscv_vluxei32_v_u32m8(
                slot_words, __riscv_vle32_v_u32m8(slot_offsets + 32 * r, vl), vl
            ),
            vl
        );
        // surface plane p: part 1 + p / 5, its page p % 5
        constexpr uint32_t kSources = kSurfacePlanes - 1;  // the gathered ones
        const uint32_t vectors[kSources] = {
            at + kGAlbedo,
            at + kGAlbedo + 1,
            at + kGAlbedo + 2,
            at + kGNormal,
            at + kGNormal + 1,
            at + kGNormal + 2,
            at + kGSpecular,
            at + kGSpecular + 1,
            at + kGSpecular + 2
        };
        for (uint32_t plane = 0; plane < kSurfacePlanes; ++plane) {
            const uint32_t vector = (
                plane == kSurfaceVisibility
                    ? kVecBlue + r
                    : plane < kSurfaceVisibility
                        ? vectors[plane]
                        : vectors[plane - 1]
            );
            row_out(vector, parts.page(1 + plane / kPlanes, plane % kPlanes), r, vl);
        }
    }

    // Rows [first, end): texel addresses of the three keys (page_passes), then kLitRows
    // rows at a time gathered, surfaced and written out.
    inline bool lit_texels(
        uint32_t first,
        uint32_t end,
        uint32_t* work,
        Pages& pages,
        const uint32_t* table,
        uint32_t slots_address,
        uint32_t page_limit,
        bool merging,
        const Parts& parts,
        Counters& counters
    ) {
        uint32_t* values = counters.values;
        const uint32_t white = kShadeSlots * kTexPageBytes;
        const size_t vl = __riscv_vsetvl_e32m8(32);
        uint32_t* index = work + kLwIndex;
        const uint32_t* slot_offsets = work + kLwSlot;
        const uint32_t* slot_words = work + kLwSlots;
        const uint32_t* texels = reinterpret_cast<const uint32_t*>(slots_address);
        uint32_t* keys = work + kLwRow;
        auto scan = [&](uint32_t row) -> bool {
            window_row(kVecDepth + row, keys, vl);
            window_row(kVecRed + row, keys + 32, vl);
            window_row(kVecGreen + row, keys + 64, vl);
            asm volatile("" ::: "memory");
            uint32_t row_texels = 0;
            for (uint32_t texture = 0; texture < 3; ++texture) {
                const int32_t texels = (
                    row_addresses(
                        pages,
                        __riscv_vle32_v_u32m8(keys + 32 * texture, vl),
                        merging && texture == 0 ? kVecTri + row : kNoTags,
                        index + 1024 * texture + 32 * row,
                        page_limit,
                        white,
                        true,
                        false,
                        vl
                    )
                );
                if (texels < 0) {
                    return false;
                }
                row_texels += static_cast<uint32_t>(texels);
            }
            values[kShadeProfTexels] += row_texels;
            return true;
        };
        auto gather = [&](uint32_t pass_first, uint32_t pass_end) -> bool {
            const uint32_t started = km::wall();
            for (uint32_t block = pass_first; block < pass_end; block += kLitRows) {
                const uint32_t block_end = (
                    block + kLitRows < pass_end ? block + kLitRows : pass_end
                );
                for (uint32_t r = block; r < block_end; ++r) {
                    gather_row(
                        r,
                        kVecGather + kGatherVectors * (r - block),
                        index,
                        slot_offsets,
                        slot_words,
                        table,
                        texels,
                        merging,
                        vl
                    );
                }
                asm volatile("" ::: "memory");
                pass_k::setup_pass::order(
                    word(kVecGather + kGatherVectors * (block_end - block) - 1) + 62
                );
                for (uint32_t r = block; r < block_end; ++r) {
                    surface_row(kVecGather + kGatherVectors * (r - block));
                }
                tensix_sync();
                for (uint32_t r = block; r < block_end; ++r) {
                    surface_out(
                        r,
                        kVecGather + kGatherVectors * (r - block),
                        parts,
                        slot_offsets,
                        slot_words,
                        vl
                    );
                }
                asm volatile("" ::: "memory");
            }
            values[kShadeProfGather] += km::wall() - started;
            return true;
        };
        PassCounts counts;
        const bool done = page_passes(first, end, pages, counts, scan, gather);
        add_counts(counters, counts);
        return done;
    }

    // Rows [first_word, + words) of `planes` pages at `out` from `opaque` where the
    // depth row is 0 (the masked layer's pixel not taken).
    inline void merge_planes(
        uint32_t out,
        const uint32_t* opaque,
        uint32_t planes,
        uint32_t first_word,
        uint32_t words,
        size_t vl
    ) {
        uint32_t* to = reinterpret_cast<uint32_t*>(out);
        for (uint32_t at = first_word; at < first_word + words; at += 32) {
            const vbool4_t take = (
                __riscv_vmsne_vx_u32m8_b4(
                    __riscv_vle32_v_u32m8(const_cast<const uint32_t*>(window() + at), vl),
                    0,
                    vl
                )
            );
            for (uint32_t plane = 0; plane < planes; ++plane) {
                uint32_t* value = to + plane * kPlaneWords + at;
                __riscv_vse32_v_u32m8(
                    value,
                    __riscv_vmerge_vvm_u32m8(
                        __riscv_vle32_v_u32m8(opaque + plane * kPlaneWords + at, vl),
                        __riscv_vle32_v_u32m8(value, vl),
                        take,
                        vl
                    ),
                    vl
                );
            }
        }
        asm volatile("" ::: "memory");
    }

    // Lit SHADE of one command (shade.h): as shade_compute_command's loop, the records
    // keying three textures, each winner's slot filled once.
    inline void shade_lit(const km_command_t& command) {
        Counters counters;
        uint32_t* values = counters.values;
        const uint32_t started = km::wall();
        const bool merging = command.a6 != 0;
        const uint32_t item_pages = merging ? 2 * kPlanes : kPlanes;
        setup();
        bool succeeded = true;
        const uint32_t table_address = front(kCbShadeTable, 1);
        const uint32_t* table = reinterpret_cast<const uint32_t*>(table_address);
        const uint32_t materials = table_address ? table[kTabMaterials] : 0;
        const uint32_t entries = table_address ? table[kTabEntries] : 0;
        const uint32_t page_limit = table_address ? table[kTabPages] : 0;
        if (
            table_address == 0 ||
            kTabMaterial + kMaterialWords * materials + kEntryWords * entries >
                kShadeTableBytes / 4
        ) {
            fill_k::give_up(kGaveBadItem, kWhereShadeBad);
            succeeded = false;
        }
        const float* cam = reinterpret_cast<const float*>(table + kTabCamera);
        const uint32_t* material_words = table + kTabMaterial;
        const uint32_t* entry_words = material_words + kMaterialWords * materials;
        const uint32_t slots_address = write_ptr(kCbShadeSlots);
        *reinterpret_cast<volatile uint32_t*>(
            slots_address + kShadeSlots * kTexPageBytes
        ) = 0xFFFFFFFF;  // the white texel
        uint32_t* shade_work = reinterpret_cast<uint32_t*>(write_ptr(kCbShadeWork));
        Pages pages;
        pages.keys = shade_work + 1024;
        pages.slots = pages.keys + kHashSize;
        pages.used = pages.slots + kHashSize;
        for (uint32_t i = 0; i < kHashSize; ++i) {
            pages.keys[i] = 0;
        }
        uint32_t* work = reinterpret_cast<uint32_t*>(write_ptr(kCbLightWork));
        Winners winners;
        winners.keys = work + kLwHashKeys;
        winners.slots = work + kLwHashSlots;
        winners.words = work + kLwSlots;
        uint32_t* empty = winners.words + kLitSlotWords * kLitSlotEmpty;
        empty[0] = 0xFFFFFFFF;
        for (uint32_t w = 1; w < kLitSlotWords; ++w) {
            empty[w] = 0;
        }
        uint32_t* slot_offsets = work + kLwSlot;
        uint32_t* row_bits = work + kLwRow;
        List list;
        const size_t vl = __riscv_vsetvl_e32m8(32);
        while (succeeded) {
            const volatile uint32_t* header = list.next();
            if (header == nullptr) {
                succeeded = false;
                break;
            }
            if (header[0] == kMarkEnd) {
                break;
            }
            const uint32_t tile = header[kHdrTile], records = header[kHdrCount];
            const int32_t first_row = header[kHdrRow0], end_row = header[kHdrRow1];
            if (
                !(0 <= first_row && first_row < end_row && end_row <= 32 && tile < 1024)
            ) {
                fill_k::give_up(kGaveBadItem, kWhereShadeBad);
                succeeded = false;
                break;
            }
            uint32_t mark = km::wall();
            const uint32_t planes = front(kCbShadePlanes, item_pages);
            if (planes == 0) {
                succeeded = false;
                break;
            }
            uint32_t now = km::wall();
            values[kShadeProfWaitIn] += now - mark;
            mark = now;
            const uint32_t* plane_words = reinterpret_cast<const uint32_t*>(planes);
            uint32_t opaque_planes = planes + kPlanes * kOutPageBytes;
            if (
                opaque_planes >= get_local_cb_interface(kCbShadePlanes).fifo_limit << 4
            ) {
                opaque_planes -= get_local_cb_interface(kCbShadePlanes).fifo_size << 4;
            }
            const uint32_t* opaque_words = (
                reinterpret_cast<const uint32_t*>(opaque_planes)
            );
            for (uint32_t plane = 0; plane < kPlanes; ++plane) {
                for (int32_t pair = first_row / 2; pair < end_row / 2; ++pair) {
                    const uint32_t at = plane * kPlaneWords + 64 * pair;
                    for (uint32_t half = 0; half < 64; half += 32) {
                        vuint32m8_t value = (
                            __riscv_vle32_v_u32m8(plane_words + at + half, vl)
                        );
                        if (merging && plane == kPlanes - 1) {
                            const uint32_t depth_at = 64 * pair + half;
                            const vbool4_t nearer = (
                                __riscv_vmand_mm_b4(
                                    __riscv_vmsltu_vx_u32m8_b4(value, 0x80000000, vl),
                                    __riscv_vmsgtu_vv_u32m8_b4(
                                        __riscv_vle32_v_u32m8(plane_words + depth_at, vl),
                                        __riscv_vle32_v_u32m8(opaque_words + depth_at, vl),
                                        vl
                                    ),
                                    vl
                                )
                            );
                            value = (
                                __riscv_vmerge_vvm_u32m8(
                                    __riscv_vmv_v_x_u32m8(kMinusOne, vl), value, nearer, vl
                                )
                            );
                        }
                        __riscv_vse32_v_u32m8(
                            const_cast<uint32_t*>(window() + at + half), value, vl
                        );
                    }
                }
            }
            asm volatile("" ::: "memory");
            pass_k::setup_pass::order((kPlanes - 1) * kPlaneWords + 32 * end_row - 1);
            // the winners: each pixel's slot from its triangle (the empty pixels' slot)
            winners.clear(vl);
            uint32_t last_bits = 0xFFFFFFFF, last_slot = kLitSlotEmpty;
            for (int32_t r = first_row; r < end_row; ++r) {
                window_row(kVecTri + r, row_bits, vl);
                asm volatile("" ::: "memory");
                for (uint32_t column = 0; column < 32; ++column) {
                    const uint32_t bits = row_bits[column];
                    if (bits != last_bits) {
                        last_bits = bits;
                        last_slot = (
                            bits >> 31 ? kLitSlotEmpty : winners.find(bits, true)
                        );
                    }
                    slot_offsets[32 * r + column] = 4 * kLitSlotWords * last_slot;
                }
            }
            if (!merging && !pop(kCbShadePlanes, kPlanes)) {
                succeeded = false;
                break;
            }
            now = km::wall();
            values[kShadeProfCopyIn] += now - mark;
            mark = now;
            TTI_SFPENCC(3, 0, 0, 10);
            TT_SFPLOADI(L7, kFloatB, 0);
            for (int32_t r = first_row; r < end_row; ++r) {
                divide_row(r);
                store(L7, kVecRed + r);  // no normal, no specular key
                store(L7, kVecGreen + r);
            }
            now = km::wall();
            values[kShadeProfDivide] += now - mark;
            mark = now;
            const uint32_t tile_y = tile / grid_of(command.a8).across;
            for (uint32_t record_index = 0; record_index < records; ++record_index) {
                const volatile uint32_t* record = list.next();
                if (record == nullptr) {
                    succeeded = false;
                    break;
                }
                const uint32_t triangle = record[kRecTri];
                const uint32_t slot = winners.find(triangle, false);
                const uint32_t shade_word = record[kRecShade];
                const uint32_t material = shade_word & 0xFFFF;
                const int32_t box_first = (
                    static_cast<int32_t>(record[kRecY0]) - 32 * tile_y
                );
                const int32_t box_last = (
                    static_cast<int32_t>(record[kRecY1]) - 32 * tile_y
                );
                const int32_t row_first = box_first < first_row ? first_row : box_first;
                const int32_t row_last = (
                    box_last > end_row - 1 ? end_row - 1 : box_last
                );
                if (
                    slot == kLitSlotEmpty || material >= materials ||
                    row_first > row_last
                ) {
                    continue;  // wins no pixel of the item, or no material
                }
                const uint32_t* material_at = (
                    material_words + kMaterialWords * material
                );
                uint32_t* slot_at = winners.words + kLitSlotWords * slot;
                uint32_t numbers[kRecordNumbers] = {};
                numbers[kConstTri] = triangle;
                if (slot_at[0] == kSlotUnfilled) {
                    fill_slot(
                        slot_at,
                        record,
                        material,
                        material_at,
                        cam,
                        (shade_word & kShadeNoFrame) == 0 &&
                            texture_numbers(
                                entry_words,
                                entries,
                                material_at[kMaterialNormal],
                                shade_word >> kShadeNormalLevelShift & 0xF,
                                numbers
                            )
                    );
                }
                // the normal's and specular's keys (none: the white texel), then the
                // albedo's (and mask's), which marks the pixels
                const uint32_t kind = material_at[0] & 0xFF;
                const bool bent = (slot_at[7] | slot_at[8] | slot_at[9]) != 0;
                const bool shiny = (slot_at[4] | slot_at[5] | slot_at[6]) != 0;
                for (uint32_t texture = 0; texture < 2; ++texture) {
                    const bool used = texture == 0 ? bent : shiny;
                    if (
                        used &&
                        texture_numbers(
                            entry_words,
                            entries,
                            material_at
                                [texture == 0 ? kMaterialNormal : kMaterialSpecular],
                            shade_word >> (
                                texture == 0
                                    ? kShadeNormalLevelShift
                                    : kShadeSpecularLevelShift
                            ) &
                                0xF,
                            numbers
                        )
                    ) {
                        record_numbers(numbers);
                        for (int32_t r = row_first; r <= row_last; ++r) {
                            record_row(
                                r,
                                kLightTexture,
                                texture == 0 ? kVecRed : kVecGreen,
                                false,
                                false
                            );
                        }
                    }
                }
                const uint32_t level = shade_word >> kShadeLevelShift & 0xF;
                const uint32_t entry_base = material_at[0] >> 8;
                Mode mode = kFlat;
                if (texture_numbers(entry_words, entries, entry_base, level, numbers)) {
                    mode = kLightTexture;
                    const uint32_t* entry = (
                        entry_words + kEntryWords * (entry_base - 1 + level)
                    );
                    const uint32_t mask_base = material_at[kMaterialMask];
                    if (
                        merging && kind == kKindMasked && mask_base != 0 &&
                        mask_base - 1 + level < entries
                    ) {
                        const uint32_t* mask_entry = (
                            entry_words + kEntryWords * (mask_base - 1 + level)
                        );
                        if (mask_entry[1] == entry[1]) {
                            const int32_t pages_on = (
                                static_cast<int32_t>(mask_entry[0] - entry[0])
                            );
                            mode = kLightTextureMask;
                            numbers[kConstMask] = (
                                pass_k::float_bits(
                                    static_cast<uint32_t>(
                                        pages_on < 0 ? -pages_on : pages_on
                                    )
                                ) |
                                (pages_on < 0 ? 0x80000000u : 0u)
                            );
                        }
                    }
                }
                record_numbers(numbers);
                for (int32_t r = row_first; r <= row_last; ++r) {
                    record_row(r, mode, kVecDepth, false, true);
                    values[kShadeProfRows]++;
                }
                values[kShadeProfRecords]++;
            }
            if (!succeeded) {
                break;
            }
            tensix_sync();
            now = km::wall();
            values[kShadeProfRecord] += now - mark;
            if (!reserve(kCbOut, kLitParts * kShadeOutPages)) {
                succeeded = false;
                break;
            }
            Parts parts;
            parts.place(write_ptr(kCbOut));
            if (!lit_texels(
                first_row,
                end_row,
                work,
                pages,
                table,
                slots_address,
                page_limit,
                merging,
                parts,
                counters
            )) {
                succeeded = false;
                break;
            }
            mark = km::wall();
            const uint32_t first_word = kRowBytes / 4 * first_row;
            const uint32_t words = kRowBytes / 4 * (end_row - first_row);
            if (merging) {
                // 1/z: the masked layer's where taken; the material word: the opaque's
                // where not; then its surface's two parts, each in its own five pages
                uint32_t* depth = reinterpret_cast<uint32_t*>(parts.page(0, 0));
                for (uint32_t at = first_word; at < first_word + words; at += 32) {
                    __riscv_vse32_v_u32m8(
                        depth + at, __riscv_vle32_v_u32m8(plane_words + at, vl), vl
                    );
                }
                asm volatile("" ::: "memory");
                merge_planes(
                    parts.page(0, 0), opaque_words, kPlanes, first_word, words, vl
                );
                if (!pop(kCbShadePlanes, 2 * kPlanes)) {
                    succeeded = false;
                    break;
                }
                for (uint32_t part = 1; part < kLitParts; ++part) {
                    const uint32_t surface = front(kCbShadePlanes, kPlanes);
                    if (surface == 0) {
                        succeeded = false;
                        break;
                    }
                    merge_planes(
                        parts.page(part, 0),
                        reinterpret_cast<const uint32_t*>(surface),
                        kPlanes,
                        first_word,
                        words,
                        vl
                    );
                    if (!pop(kCbShadePlanes, kPlanes)) {
                        succeeded = false;
                        break;
                    }
                }
                if (!succeeded) {
                    break;
                }
            }
            for (uint32_t part = 0; part < kLitParts; ++part) {
                push(
                    kCbOut,
                    parts.page(part, kPlanes - 1) + 4 * (first_word + words - 1),
                    kShadeOutPages
                );
            }
            values[kShadeProfOut] += km::wall() - mark;
            values[kShadeProfTiles]++;
        }
        // the reader's end, the list's last page, the table back, the unpack thread's
        // end
        if (reserve(kCbShadeAsk, 1)) {
            const uint32_t ask = write_ptr(kCbShadeAsk);
            *reinterpret_cast<volatile uint32_t*>(ask) = kShadeAskEnd;
            push(kCbShadeAsk, ask, 1);
        }
        list.end();
        if (table_address != 0) {
            pop(kCbShadeTable, 1);
        }
        mailbox_write(ThreadId::UnpackThreadId, mail(kOpEnd, 0, 0));
        fill_k::fill::finish();
        if (succeeded && reserve(kCbOut, kShadeOutPages)) {
            values[kShadeProfTotal] = km::wall() - started;
            volatile uint32_t* profile = (
                reinterpret_cast<volatile uint32_t*>(write_ptr(kCbOut))
            );
            for (uint32_t i = 0; i < kShadeProfBytes / 4; ++i) {
                profile[i] = i < kShadeProfWords ? values[i] : 0;
            }
            push(
                kCbOut,
                reinterpret_cast<uint32_t>(profile) + kShadeProfBytes - 4,
                kShadeOutPages
            );
        }
    }

}  // namespace shade_pass
