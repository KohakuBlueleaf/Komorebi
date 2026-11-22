// LIGHT's pack thread (shade.h): each item's colour from its surface planes, half a
// tile at a time in Dst, the scene's numbers as immediates from the table.
namespace shade_pass {

    // Dst for half a tile (16 rows, row k of a plane at 16 plane + k): a rgb, material
    // word, normal xyz, visibility; then the row's scratch vectors.
    constexpr uint32_t kLightRows = 16;
    constexpr uint32_t kLvAlbedo = 0, kLvMaterial = 3, kLvNormal = 4, kLvVisibility = 7;
    constexpr uint32_t kLvPlanes = 8;
    constexpr uint32_t kLvScratch = kLightRows *
        kLvPlanes;                               // x x - y y, z z, x y, x z,
    constexpr uint32_t kLvSun = kLvScratch + 5;  // y z; max(0, n . s)
    // each Dst plane's page in the item (light_reader.hpp): a, material, normal, vis
    constexpr uint32_t kLightSource[kLvPlanes] = {0, 1, 2, 5, 3, 4, 6, 7};

    inline void table_immediate(uint32_t lreg, const uint32_t* table, uint32_t at) {
        immediate(lreg, table[at]);
    }

    // Row k: E = max(0, sky(n)) vis + ambient + sun max(0, n.s), x = exposure a E / pi
    // (emissive: exposure a; empty: exposure x background); x / (1 + x) into a's rows.
    inline void light_row(uint32_t k, const uint32_t* table) {
        TTI_SFPENCC(3, 0, 0, 10);
        load(L0, kLightRows * kLvNormal + k);
        load(L1, kLightRows * (kLvNormal + 1) + k);
        load(L2, kLightRows * (kLvNormal + 2) + k);
        mul(L0, L0, L3);
        TT_SFPMAD(L1, L1, L3, L3, kNegA);
        store(L3, kLvScratch);
        mul(L2, L2, L3);
        store(L3, kLvScratch + 1);
        mul(L0, L1, L3);
        store(L3, kLvScratch + 2);
        mul(L0, L2, L3);
        store(L3, kLvScratch + 3);
        mul(L1, L2, L3);
        store(L3, kLvScratch + 4);
        table_immediate(L4, table, kLtSunDirection);
        mul(L0, L4, L3);
        table_immediate(L4, table, kLtSunDirection + 1);
        TT_SFPMAD(L1, L4, L3, L3, 0);
        table_immediate(L4, table, kLtSunDirection + 2);
        TT_SFPMAD(L2, L4, L3, L3, 0);
        TTI_SFPNOP;
        at_least_zero(L3, L4);
        store(L3, kLvSun);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const uint32_t* numbers = table + kLtChannel * channel;
            load(L0, kLightRows * kLvNormal + k);
            load(L1, kLightRows * (kLvNormal + 1) + k);
            load(L2, kLightRows * (kLvNormal + 2) + k);
            table_immediate(L3, numbers, kLtSky);
            // K0 + KZ z z + KX (x x - y y) + (x y, x z, y z terms) + (x, y, z terms)
            constexpr uint32_t kScratchOf[5] = {1, 0, 2, 3, 4};
            for (uint32_t term = 0; term < 5; ++term) {
                table_immediate(L4, numbers, kLtSky + 1 + term);
                load(L5, kLvScratch + kScratchOf[term]);
                TT_SFPMAD(L5, L4, L3, L3, 0);
            }
            for (uint32_t axis = 0; axis < 3; ++axis) {
                table_immediate(L4, numbers, kLtSky + 6 + axis);
                TT_SFPMAD(axis == 0 ? L0 : axis == 1 ? L1 : L2, L4, L3, L3, 0);
            }
            TTI_SFPNOP;
            at_least_zero(L3, L4);
            load(L5, kLightRows * kLvVisibility + k);
            mul(L3, L5, L3);
            table_immediate(L4, numbers, kLtAmbient);
            add(L3, L4, L3);
            table_immediate(L4, numbers, kLtSun);
            load(L5, kLvSun);
            TT_SFPMAD(L5, L4, L3, L3, 0);
            load(L5, kLightRows * (kLvAlbedo + channel) + k);
            mul(L5, L3, L3);
            table_immediate(L4, table, kLtScale);
            mul(L3, L4, L3);
            // emissive (kind 4 in the material word's bits 16 on): exposure a
            table_immediate(L4, table, kLtExposure);
            mul(L5, L4, L6);
            load(L7, kLightRows * kLvMaterial + k);
            TT_SFPSHFT(
                (-static_cast<int32_t>(kMaterialKindShift)) & 0xFFF,
                0,
                L7,
                kShiftImmediate
            );
            TT_SFPIADD((-static_cast<int32_t>(kKindEmissive)) & 0xFFF, L7, L7, 1 | 4);
            TTI_SFPNOP;
            TT_SFPSETCC(0, L7, 0, kEq0);
            add(L6, ZERO, L3);
            TTI_SFPENCC(3, 0, 0, 10);
            // empty (bit 31): exposure x background
            table_immediate(L4, table, kLtBackground + channel);
            load(L7, kLightRows * kLvMaterial + k);
            TT_SFPSETCC(0, L7, 0, 0);  // < 0
            add(L4, ZERO, L3);
            TTI_SFPENCC(3, 0, 0, 10);
            add(L3, ONE, L4);
            recip(L4, L5, L6, L7, L0, L1);
            mul(L3, L5, L6);
            store(L6, kLightRows * (kLvAlbedo + channel) + k);
        }
    }

    // LIGHT of one command: rows from the reader's answer page, planes in two parts of
    // five pages, the colour to kCbOut's three first pages.
    inline void light_pack(const km_command_t& command) {
        (void)command;
        uint32_t values[kLightProfTotal + 1] = {};
        const uint32_t started = km::wall();
        setup();
        bool succeeded = true;
        const uint32_t table_address = front(kCbShadeTable, 1);
        const uint32_t* table = reinterpret_cast<const uint32_t*>(table_address);
        if (table_address == 0) {
            succeeded = false;
        }
        const size_t vl = __riscv_vsetvl_e32m8(32);
        while (succeeded) {
            uint32_t mark = km::wall();
            const uint32_t answer = front(kCbShadeDone, 1);
            if (answer == 0) {
                succeeded = false;
                break;
            }
            const uint32_t rows = *reinterpret_cast<const volatile uint32_t*>(answer);
            if (!pop(kCbShadeDone, 1)) {
                succeeded = false;
                break;
            }
            if (rows == kShadeAskEnd) {
                break;
            }
            const uint32_t first_row = rows & 0xFF, end_row = rows >> 8 & 0xFF;
            if (!(first_row < end_row && end_row <= 32)) {
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
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t row0 = kLightRows * half;
                const uint32_t first = first_row > row0 ? first_row : row0;
                const uint32_t end = (
                    end_row < row0 + kLightRows ? end_row : row0 + kLightRows
                );
                if (first >= end) {
                    continue;
                }
                mark = km::wall();
                // a plane's 16 rows are 512 words in its page and in Dst
                for (uint32_t plane = 0; plane < kLvPlanes; ++plane) {
                    uint32_t page = planes + kLightSource[plane] * kOutPageBytes;
                    if (page >= limit) {
                        page -= size;
                    }
                    const uint32_t* from = (
                        reinterpret_cast<const uint32_t*>(page) + 512 * half
                    );
                    for (uint32_t i = 0; i < 512; i += 32) {
                        __riscv_vse32_v_u32m8(
                            const_cast<uint32_t*>(window() + 512 * plane + i),
                            __riscv_vle32_v_u32m8(from + i, vl),
                            vl
                        );
                    }
                }
                asm volatile("" ::: "memory");
                pass_k::setup_pass::order(512 * kLvPlanes - 1);
                now = km::wall();
                values[kLightProfCopyIn] += now - mark;
                mark = now;
                for (uint32_t r = first; r < end; ++r) {
                    light_row(r - row0, table);
                    values[kLightProfRows]++;
                }
                tensix_sync();
                now = km::wall();
                values[kLightProfMath] += now - mark;
                mark = now;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    uint32_t* to = (
                        reinterpret_cast<uint32_t*>(
                            out + (1 + channel) * kOutPageBytes
                        ) +
                        512 * half
                    );
                    for (uint32_t i = 0; i < 512; i += 32) {
                        __riscv_vse32_v_u32m8(
                            to + i,
                            __riscv_vle32_v_u32m8(
                                const_cast<const uint32_t*>(
                                    window() + 512 * channel + i
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
            if (!pop(kCbShadePlanes, kLightItemPages)) {
                succeeded = false;
                break;
            }
            push(kCbOut, out + 4 * kOutPageBytes - 4, kShadeOutPages);
            values[kLightProfTiles]++;
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
