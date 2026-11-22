// SHADE's compute (shade.h): unpack waits on buffers for pack; pack colours each
// item: u, v, light, texel keys, pages read, texels gathered (RVV), colour, (a6) merge.
#include "shade.h"

namespace shade_pass {

    using fill_k::kCbIn;
    using fill_k::kCbOut;
    using fill_k::kGaveBadItem;
    using fill_k::kHdrCount;
    using fill_k::kHdrRow0;
    using fill_k::kHdrRow1;
    using fill_k::kHdrTile;
    using fill_k::kMarkEnd;
    using fill_k::kOutPageBytes;
    using fill_k::kPlanes;
    using fill_k::kPlaneWords;
    using fill_k::kRecTri;
    using fill_k::kRecY0;
    using fill_k::kRecY1;
    using fill_k::kRowBytes;
    using fill_k::kSlotsPerPage;
    using fill_k::kSlotWords;

    // The unpack thread's answers to the pack thread's mails: the front page of a
    // buffer (its address; 0: gave up), a pop (1), the end (no answer).
    constexpr uint32_t kOpFront = 1;
    constexpr uint32_t kOpPop = 2;
    constexpr uint32_t kOpEnd = 3;

    constexpr uint32_t mail(uint32_t op, uint32_t cb, uint32_t pages) {
        return cb | pages << 8 | op << 16;
    }

}  // namespace shade_pass

#ifdef TRISC_UNPACK
// Serves the pack thread's mails until its end (or a wait gives up).
void shade_unpack() {
    using namespace shade_pass;
    for (;;) {
        if (!fill_k::mail_from(ckernel::ThreadId::PackThreadId, kWhereShadeMail)) {
            return;
        }
        const uint32_t request = ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
        const uint32_t op = request >> 16, cb = request & 0xFF;
        const uint32_t pages = request >> 8 & 0xFF;
        if (op == kOpPop) {
            cb_pop_front(cb, pages);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 1);
            continue;
        }
        if (op != kOpFront) {
            return;
        }
        const bool ready = fill_k::timed(
            [&] {
                const uint16_t received = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb)))
                );
                const uint16_t acked = get_local_cb_interface(cb).tiles_acked;
                return static_cast<uint16_t>(received - acked) >= pages;
            },
            kWhereShadeFront
        );
        if (!ready) {
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, 0);
            return;
        }
        cb_wait_front(cb, pages);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(cb, 0)
        );
    }
}
#endif

#ifdef TRISC_PACK
namespace shade_pass {

    using namespace ckernel;
    using pass_k::setup_pass::recip;

    constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
    constexpr uint32_t ZERO = 9, ONE = 10;
    // LRegs 11 to 14, set by SFPCONFIG at the start: 1.5 x 2^23, its integer mask,
    // 32 and 1 / 32.
    constexpr uint32_t MAGIC = 11, MASK = 12, C32 = 13, INV32 = 14;
    constexpr uint32_t kInt32 = 4;
    constexpr uint32_t kAddressMode = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kLower = 10;
    constexpr uint32_t kNegA = 1, kNegC = 2;
    constexpr uint32_t kSetCC = 1;
    constexpr uint32_t kGte0 = 4, kEq0 = 6;  // SFPSETCC
    constexpr uint32_t kMinMax = 1;          // SFPSWAP: VC the larger, VD the smaller
    constexpr uint32_t kShiftImmediate = 1;

    // Dst vectors: FILL's planes (32 p + r), frac(u) rows, 1 - frac(v) rows, a record's
    // numbers; after the records the albedo's linear rgb rows from kVecAlbedo.
    constexpr uint32_t kVecDepth = 0, kVecRed = 32, kVecGreen = 64, kVecBlue = 96;
    constexpr uint32_t kVecTri = 128;
    constexpr uint32_t kVecS = 160, kVecT = 192, kVecRecord = 224, kVecAlbedo = 160;
    constexpr uint32_t kConstTri = 0, kConstW = 1, kConstWLast = 2, kConstH = 3;
    constexpr uint32_t kConstHLast = 4, kConstAcross = 5, kConstFirst = 6;
    constexpr uint32_t kConstBase = 7;   // red, green, blue
    constexpr uint32_t kConstMask = 10;  // the mask's first page less the albedo's
    constexpr uint32_t kRecordNumbers = 11;
    constexpr uint32_t kMagicBits = 0x4B400000;
    constexpr uint32_t kMinusTwo = 0xC0000000;  // a pixel's triangle once coloured
    constexpr uint32_t kMinusOne = 0xBF800000;  // FILL's triangle of an empty pixel
    // A texel address with the mask's slot: (1 + slot) at this bit (0: the albedo's
    // texel has the alpha byte); the texel's place in its page the albedo's.
    constexpr uint32_t kMaskSlotShift = 20;
    static_assert(kShadeSlots * kTexPageBytes < 1u << kMaskSlotShift);
    static_assert(kVecTri == 32 * (kPlanes - 1));

    // How a record's pixels are coloured; kLightTextureMask also keys its mask's texel.
    enum Mode : uint32_t {
        kFlat,
        kLight,
        kLightTexture,
        kEmissive,
        kEmissiveTexture,
        kLightTextureMask
    };

    constexpr uint32_t dst_address(uint32_t vector) {
        return 4 * (vector >> 1) + 2 * (vector & 1);
    }

    // The window word of vector v, lane 0 (lane l at + 2 l).
    constexpr uint32_t word(uint32_t vector) {
        return 64 * (vector >> 1) + (vector & 1);
    }

    inline volatile uint32_t* window() {
        return km::dst::window();
    }

    inline void load(uint32_t lreg, uint32_t vector) {
        TT_SFPLOAD(lreg, kInt32, kAddressMode, dst_address(vector));
    }

    inline void store(uint32_t lreg, uint32_t vector) {
        TT_SFPSTORE(lreg, kInt32, kAddressMode, dst_address(vector));
    }

    inline void immediate(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, kFloatB, bits >> 16);
        TT_SFPLOADI(lreg, kLower, bits & 0xFFFF);
    }

    inline void mul(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, b, ZERO, d, kNegC);
    }

    inline void add(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, ONE, b, d, 0);
    }

    inline void sub(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, ONE, b, d, kNegC);
    }

    // out = floor(x) for |x| < 2^22: x rounded by 1.5 x 2^23, one less where above.
    inline void floor_to(uint32_t x, uint32_t out) {
        add(x, MAGIC, out);
        sub(out, MAGIC, out);
        TTI_SFPNOP;
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPGT(0, x, out, kSetCC);
        sub(out, ONE, out);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // x = max(x, 0), with `work` free.
    inline void at_least_zero(uint32_t x, uint32_t work) {
        TT_SFPLOADI(work, kFloatB, 0);
        TTI_SFPNOP;
        TT_SFPSWAP(0, x, work, kMinMax);
    }

    // Thread state as the fill's, and LRegs 11 to 14.
    inline void setup() {
        fill_k::fill::setup();
        immediate(L0, kMagicBits);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, MAGIC, 0);
        TTI_SFPNOP;
        immediate(L0, 0x003FFFFF);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, MASK, 0);
        TTI_SFPNOP;
        immediate(L0, 0x42000000);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, C32, 0);
        TTI_SFPNOP;
        immediate(L0, 0x3D000000);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, INV32, 0);
        TTI_SFPNOP;
        tensix_sync();
    }

    // Row r: covered pixels' rgb / (1/z) (q = a r, q + (a - q w) r), frac(u), 1 -
    // frac(v) (at least 0); then the depth row to 0 (no texel).
    inline void divide_row(uint32_t r) {
        TTI_SFPENCC(3, 0, 0, 10);
        load(L0, kVecTri + r);
        TT_SFPSETCC(0, L0, 0, kGte0);
        load(L1, kVecDepth + r);
        recip(L1, L2, L3, L4, L5, L6);
        for (uint32_t plane = 1; plane <= 3; ++plane) {
            load(L3, 32 * plane + r);
            mul(L3, L2, L4);
            TT_SFPMAD(L4, L1, L3, L5, kNegA);
            TT_SFPMAD(L5, L2, L4, L4, 0);
            store(L4, 32 * plane + r);
            if (plane == 1) {
                floor_to(L4, L5);
                sub(L4, L5, L4);
                TTI_SFPNOP;
                at_least_zero(L4, L5);
                store(L4, kVecS + r);
            } else if (plane == 2) {
                floor_to(L4, L5);
                sub(L4, L5, L4);
                sub(ONE, L4, L4);
                TTI_SFPNOP;
                at_least_zero(L4, L5);
                store(L4, kVecT + r);
            }
        }
        TTI_SFPENCC(3, 0, 0, 10);
        TT_SFPLOADI(L0, kFloatB, 0);
        store(L0, kVecDepth + r);
    }

    // A record's numbers to its vectors (every lane): triangle, width, width - 1,
    // height, height - 1, blocks a row, 1 + first page, base rgb, mask's pages on.
    inline void record_numbers(const uint32_t* numbers) {
        TTI_SFPENCC(3, 0, 0, 10);
        for (uint32_t i = 0; i < kRecordNumbers; ++i) {
            immediate(L0, numbers[i]);
            store(L0, kVecRecord + i);
        }
    }

    // Row r, the record's pixels: key (1 + page) << 10 | texel to the depth row, the
    // colour the albedo multiplies; triangle -2 (a mask: -2's bits | mask's key).
    inline void record_row(uint32_t r, Mode mode) {
        load(L0, kVecTri + r);
        load(L1, kVecRecord + kConstTri);
        sub(L0, L1, L0);
        TTI_SFPNOP;
        TT_SFPSETCC(0, L0, 0, kEq0);
        const bool textured = (
            mode == kLightTexture || mode == kEmissiveTexture ||
            mode == kLightTextureMask
        );
        if (textured) {
            // x = min(floor(s width), width - 1), its block and column in the block
            load(L0, kVecS + r);
            load(L1, kVecRecord + kConstW);
            mul(L0, L1, L0);
            floor_to(L0, L2);
            load(L1, kVecRecord + kConstWLast);
            TTI_SFPNOP;
            TT_SFPSWAP(0, L1, L2, kMinMax);
            mul(L2, INV32, L3);
            floor_to(L3, L4);
            TT_SFPMAD(L4, C32, L2, L2, kNegA);
            // y likewise from 1 - t and the height
            load(L0, kVecT + r);
            load(L1, kVecRecord + kConstH);
            mul(L0, L1, L0);
            floor_to(L0, L3);
            load(L1, kVecRecord + kConstHLast);
            TTI_SFPNOP;
            TT_SFPSWAP(0, L1, L3, kMinMax);
            mul(L3, INV32, L5);
            floor_to(L5, L6);
            TT_SFPMAD(L6, C32, L3, L3, kNegA);
            // texel in the page, 1 + page; both as integers, then the key
            TT_SFPMAD(L3, C32, L2, L2, 0);
            load(L0, kVecRecord + kConstAcross);
            load(L1, kVecRecord + kConstFirst);
            TT_SFPMAD(L6, L0, L1, L6, 0);
            add(L6, L4, L6);
            if (mode == kLightTextureMask) {  // the mask's 1 + page: the same block
                load(L0, kVecRecord + kConstMask);
                add(L6, L0, L7);
            }
            add(L6, MAGIC, L6);
            add(L2, MAGIC, L2);
            TTI_SFPNOP;
            TT_SFPAND(0, MASK, L6, 0);
            TT_SFPAND(0, MASK, L2, 0);
            TT_SFPSHFT(10, 0, L6, kShiftImmediate);
            TT_SFPOR(0, L2, L6, 0);
            store(L6, kVecDepth + r);
            if (mode == kLightTextureMask) {  // its key, tagged, to the triangle row
                add(L7, MAGIC, L7);
                TTI_SFPNOP;
                TT_SFPAND(0, MASK, L7, 0);
                TT_SFPSHFT(10, 0, L7, kShiftImmediate);
                TT_SFPOR(0, L2, L7, 0);
                TT_SFPLOADI(L0, kFloatB, kMinusTwo >> 16);
                TTI_SFPNOP;
                TT_SFPOR(0, L0, L7, 0);
                store(L7, kVecTri + r);
            }
        }
        if (mode == kLight || mode == kLightTexture || mode == kLightTextureMask) {
            load(L0, kVecBlue + r);
            for (uint32_t channel = 0; channel < 3; ++channel) {
                load(L1, kVecRecord + kConstBase + channel);
                mul(L1, L0, L2);
                store(L2, kVecRed + 32 * channel + r);
            }
        } else if (mode == kEmissive || mode == kEmissiveTexture) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                load(L1, kVecRecord + kConstBase + channel);
                store(L1, kVecRed + 32 * channel + r);
            }
        }
        if (mode != kLightTextureMask) {
            TT_SFPLOADI(L0, kFloatB, kMinusTwo >> 16);
            store(L0, kVecTri + r);
        }
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // Row r's colour: each channel times its albedo.
    inline void colour_row(uint32_t r) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            load(L0, kVecAlbedo + 32 * channel + r);
            load(L1, kVecRed + 32 * channel + r);
            mul(L0, L1, L2);
            store(L2, kVecRed + 32 * channel + r);
        }
    }

    // The pack thread's mails to the unpack thread.
    inline uint32_t front(uint32_t cb, uint32_t pages) {
        mailbox_write(ThreadId::UnpackThreadId, mail(kOpFront, cb, pages));
        if (!fill_k::mail_from(ThreadId::UnpackThreadId, kWhereShadeMail)) {
            return 0;
        }
        const uint32_t address = mailbox_read(ThreadId::UnpackThreadId);
        asm volatile("fence" ::: "memory");
        return address;
    }

    inline bool pop(uint32_t cb, uint32_t pages) {
        mailbox_write(ThreadId::UnpackThreadId, mail(kOpPop, cb, pages));
        if (!fill_k::mail_from(ThreadId::UnpackThreadId, kWhereShadeMail)) {
            return false;
        }
        (void)mailbox_read(ThreadId::UnpackThreadId);
        return true;
    }

    // Timed cb_reserve_back of `pages` on a buffer the pack thread fills.
    inline bool reserve(uint32_t cb, uint32_t pages) {
        const bool reserved = fill_k::timed(
            [&] {
                const uint16_t acked = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_acked_ptr(cb)))
                );
                const uint16_t used = get_local_cb_interface(cb).tiles_received - acked;
                return get_local_cb_interface(cb).fifo_num_pages - used >= pages;
            },
            kWhereShadeFront
        );
        if (reserved) {
            cb_reserve_back(cb, pages);
        }
        return reserved;
    }

    inline uint32_t write_ptr(uint32_t cb) {
        return get_local_cb_interface(cb).fifo_wr_ptr << 4;
    }

    // Stores to L1 done (a load of the last word, used), then pushed.
    inline void push(uint32_t cb, uint32_t last_address, uint32_t pages) {
        asm volatile("fence" ::: "memory");
        const uint32_t last_word = *reinterpret_cast<volatile uint32_t*>(last_address);
        asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
        cb_push_back(cb, pages);
    }

    // The list's slots, page by page from the unpack thread.
    struct List {
        uint32_t page = 0;
        uint32_t slot = kSlotsPerPage;
        uint32_t wait = 0;

        const volatile uint32_t* next() {
            if (slot == kSlotsPerPage) {
                const uint32_t asked = km::wall();
                if (page != 0 && !pop(kCbIn, 1)) {
                    return nullptr;
                }
                page = front(kCbIn, 1);
                wait += km::wall() - asked;
                if (page == 0) {
                    return nullptr;
                }
                slot = 0;
            }
            return (
                reinterpret_cast<const volatile uint32_t*>(page) + kSlotWords * slot++
            );
        }

        bool end() {
            return page == 0 || pop(kCbIn, 1);
        }
    };

    // The pages of a pass of rows: an open-addressed table of (1 + page) -> slot, the
    // slots' pages in order, and the last page found.
    constexpr uint32_t kHashSize = 256;

    struct Pages {
        uint32_t* keys;   // kHashSize words: 1 + page, 0 empty
        uint32_t* slots;  // kHashSize words
        uint32_t* used;   // kShadeSlots words: the table places filled
        uint32_t* ask;    // the ask page being written: count, then the pages
        uint32_t count = 0, last_page = 0xFFFFFFFF, last_slot = 0;

        void clear() {
            for (uint32_t i = 0; i < count; ++i) {
                keys[used[i]] = 0;
            }
            count = 0;
            last_page = 0xFFFFFFFF;
        }

        // The slot of `page`; kShadeSlots when the slots are full.
        uint32_t slot_of(uint32_t page) {
            if (page == last_page) {
                return last_slot;
            }
            uint32_t place = (page * 2654435761u) >> 24;
            for (;;) {
                const uint32_t key = keys[place];
                if (key == page + 1) {
                    break;
                }
                if (key == 0) {
                    if (count == kShadeSlots) {
                        return kShadeSlots;
                    }
                    keys[place] = page + 1;
                    slots[place] = count;
                    used[count] = place;
                    ask[1 + count] = page;
                    ++count;
                    break;
                }
                place = (place + 1) & (kHashSize - 1);
            }
            last_page = page;
            last_slot = slots[place];
            return last_slot;
        }
    };

    struct Counters {
        uint32_t values[kShadeProfTotal + 1] = {};
    };

    // Rows [first, end): texel offsets from the keys, in passes whose pages fit the
    // slots, texels to albedo rows; merging also mask's, depth row 1 where it takes.
    inline bool texels(
        uint32_t first,
        uint32_t end,
        uint32_t* index,
        Pages& pages,
        const uint32_t* gamma,
        uint32_t slots_address,
        uint32_t page_limit,
        bool merging,
        Counters& counters
    ) {
        uint32_t* values = counters.values;
        const uint32_t white = kShadeSlots * kTexPageBytes;
        const size_t vl = __riscv_vsetvl_e32m8(32);
        uint32_t row = first;
        while (row < end) {
            uint32_t started = km::wall();
            if (!reserve(kCbShadeAsk, 1)) {
                return false;
            }
            pages.ask = reinterpret_cast<uint32_t*>(write_ptr(kCbShadeAsk));
            pages.clear();
            const uint32_t pass_first = row;
            uint32_t* keys = pages.used + kShadeSlots;  // a row's keys
            uint32_t* mask_keys = keys + 32;            // its triangle row (merging)
            for (; row < end; ++row) {
                __riscv_vse32_v_u32m8(
                    keys,
                    __riscv_vlse32_v_u32m8(
                        const_cast<const uint32_t*>(window() + word(kVecDepth + row)),
                        8,
                        vl
                    ),
                    vl
                );
                if (merging) {
                    __riscv_vse32_v_u32m8(
                        mask_keys,
                        __riscv_vlse32_v_u32m8(
                            const_cast<const uint32_t*>(window() + word(kVecTri + row)),
                            8,
                            vl
                        ),
                        vl
                    );
                }
                asm volatile("" ::: "memory");
                uint32_t* row_index = index + 32 * row;
                bool full = false;
                uint32_t row_texels = 0;
                for (uint32_t column = 0; column < 32; ++column) {
                    const uint32_t key = keys[column];
                    const uint32_t page = (key >> 10) - 1;
                    if (key < 1024 || page >= page_limit) {
                        row_index[column] = white;
                        continue;
                    }
                    const uint32_t slot = pages.slot_of(page);
                    if (slot == kShadeSlots) {
                        full = true;
                        break;
                    }
                    uint32_t address = slot * kTexPageBytes + 4 * (key & 1023);
                    const uint32_t tagged = merging ? mask_keys[column] : 0;
                    const uint32_t mask_key = tagged & 0x3FFFFFFF;
                    const uint32_t mask_page = (mask_key >> 10) - 1;
                    if (
                        tagged >> 30 == 3 && mask_key >= 1024 && mask_page < page_limit
                    ) {
                        const uint32_t mask_slot = pages.slot_of(mask_page);
                        if (mask_slot == kShadeSlots) {
                            full = true;
                            break;
                        }
                        address |= (mask_slot + 1) << kMaskSlotShift;
                    }
                    row_index[column] = address;
                    ++row_texels;
                }
                if (full) {  // this row starts the next pass
                    break;
                }
                values[kShadeProfTexels] += row_texels;
            }
            if (row == pass_first) {  // one row cannot need more pages than slots
                fill_k::give_up(kGaveBadItem, kWhereShadeBad);
                return false;
            }
            uint32_t now = km::wall();
            values[kShadeProfScan] += now - started;
            if (pages.count != 0) {
                pages.ask[0] = pages.count;
                push(kCbShadeAsk, reinterpret_cast<uint32_t>(pages.ask), 1);
                if (front(kCbShadeDone, 1) == 0 || !pop(kCbShadeDone, 1)) {
                    return false;
                }
                values[kShadeProfPages] += pages.count;
            }
            started = km::wall();
            values[kShadeProfWaitPages] += started - now;
            values[kShadeProfPasses]++;
            const uint32_t* slots = reinterpret_cast<const uint32_t*>(slots_address);
            for (uint32_t r = pass_first; r < row; ++r) {
                const vuint32m8_t address = __riscv_vle32_v_u32m8(index + 32 * r, vl);
                const vuint32m8_t albedo_address = (
                    merging
                        ? __riscv_vand_vx_u32m8(address, (1u << kMaskSlotShift) - 1, vl)
                        : address
                );
                const vuint32m8_t texel = (
                    __riscv_vluxei32_v_u32m8(slots, albedo_address, vl)
                );
                if (merging) {
                    // the alpha byte: the mask's texel at the same place, else the
                    // albedo's
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
                        __riscv_vluxei32_v_u32m8(slots, mask_address, vl)
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
                    __riscv_vsse32_v_u32m8(
                        const_cast<uint32_t*>(window() + word(kVecDepth + r)),
                        8,
                        __riscv_vmerge_vxm_u32m8(
                            __riscv_vmv_v_x_u32m8(0, vl), 1, take, vl
                        ),
                        vl
                    );
                }
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const vuint32m8_t byte = (
                        __riscv_vand_vx_u32m8(
                            __riscv_vsrl_vx_u32m8(texel, 8 * channel, vl), 0xFF, vl
                        )
                    );
                    __riscv_vsse32_v_u32m8(
                        const_cast<uint32_t*>(
                            window() + word(kVecAlbedo + 32 * channel + r)
                        ),
                        8,
                        __riscv_vluxei32_v_u32m8(
                            gamma, __riscv_vsll_vx_u32m8(byte, 2, vl), vl
                        ),
                        vl
                    );
                }
            }
            asm volatile("" ::: "memory");
            values[kShadeProfGather] += km::wall() - started;
        }
        return true;
    }

}  // namespace shade_pass
#endif

// SHADE of one command (shade.h); the frame program's compute kernel calls it.
void shade_compute_command(const km_command_t& command) {
    if (km::handler(command) != kHandlerShade) {
        return;
    }
#ifdef TRISC_UNPACK
    shade_unpack();
#endif
#ifdef TRISC_PACK
    using namespace shade_pass;
    Counters counters;
    uint32_t* values = counters.values;
    const uint32_t started = km::wall();
    // the masked layer's command: its planes, then the opaque layer's, an item each
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
    const uint32_t* gamma = table + kTabGamma;
    const uint32_t* material_words = table + kTabMaterial;
    const uint32_t* entry_words = material_words + kMaterialWords * materials;
    const uint32_t slots_address = write_ptr(kCbShadeSlots);
    *reinterpret_cast<volatile uint32_t*>(
        slots_address + kShadeSlots * kTexPageBytes
    ) = 0xFFFFFFFF;  // the white texel: albedo 1
    uint32_t* work = reinterpret_cast<uint32_t*>(write_ptr(kCbShadeWork));
    uint32_t* index = work;  // 1,024 texel addresses, row by row
    Pages pages;
    pages.keys = work + 1024;
    pages.slots = pages.keys + kHashSize;
    pages.used = pages.slots + kHashSize;
    for (uint32_t i = 0; i < kHashSize; ++i) {
        pages.keys[i] = 0;
    }
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
        if (!(0 <= first_row && first_row < end_row && end_row <= 32 && tile < 1024)) {
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
        // the item's rows of the five planes into Dst (two rows: 64 words); merging,
        // triangle -1 unless nearer than the opaque (1/z as unsigned bits, tie to it)
        const uint32_t* plane_words = reinterpret_cast<const uint32_t*>(planes);
        // the opaque layer's five pages after them in the ring (shade_reader.hpp)
        uint32_t opaque_planes = planes + kPlanes * kOutPageBytes;
        if (opaque_planes >= get_local_cb_interface(kCbShadePlanes).fifo_limit << 4) {
            opaque_planes -= get_local_cb_interface(kCbShadePlanes).fifo_size << 4;
        }
        const uint32_t* opaque_words = reinterpret_cast<const uint32_t*>(opaque_planes);
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
        if (!merging && !pop(kCbShadePlanes, kPlanes)) {
            succeeded = false;
            break;
        }
        now = km::wall();
        values[kShadeProfCopyIn] += now - mark;
        mark = now;
        for (int32_t r = first_row; r < end_row; ++r) {
            divide_row(r);
        }
        now = km::wall();
        values[kShadeProfDivide] += now - mark;
        mark = now;
        const uint32_t tile_y = tile / 32;
        for (uint32_t record_index = 0; record_index < records; ++record_index) {
            const volatile uint32_t* record = list.next();
            if (record == nullptr) {
                succeeded = false;
                break;
            }
            const uint32_t shade_word = record[kRecShade];
            const uint32_t material = shade_word & 0xFFFF;
            const uint32_t level = shade_word >> kShadeLevelShift & 0xF;
            const int32_t box_first = (
                static_cast<int32_t>(record[kRecY0]) - 32 * tile_y
            );
            const int32_t box_last = static_cast<int32_t>(record[kRecY1]) - 32 * tile_y;
            const int32_t row_first = box_first < first_row ? first_row : box_first;
            const int32_t row_last = box_last > end_row - 1 ? end_row - 1 : box_last;
            if (material >= materials || row_first > row_last) {
                continue;  // no material: FILL's colour divided by 1/z (flat)
            }
            const uint32_t* material_at = material_words + kMaterialWords * material;
            const uint32_t kind = material_at[0] & 0xFF;
            const uint32_t entry_base = material_at[0] >> 8;
            const bool emissive = kind == kKindEmissive;
            Mode mode = kind == kKindFlat ? kFlat : emissive ? kEmissive : kLight;
            uint32_t numbers[kRecordNumbers] = {};
            numbers[kConstTri] = record[kRecTri];
            if (
                kind != kKindFlat && entry_base != 0 && entry_base - 1 + level < entries
            ) {
                const uint32_t* entry = (
                    entry_words + kEntryWords * (entry_base - 1 + level)
                );
                const uint32_t width = entry[1] & 0xFFFF, height = entry[1] >> 16;
                if (width != 0 && height != 0) {
                    mode = emissive ? kEmissiveTexture : kLightTexture;
                    numbers[kConstW] = pass_k::float_bits(width);
                    numbers[kConstWLast] = pass_k::float_bits(width - 1);
                    numbers[kConstH] = pass_k::float_bits(height);
                    numbers[kConstHLast] = pass_k::float_bits(height - 1);
                    numbers[kConstAcross] = pass_k::float_bits((width + 31) / 32);
                    numbers[kConstFirst] = pass_k::float_bits(entry[0] + 1);
                }
                // merging, a masked material's mask at the same level and sizes
                const uint32_t mask_base = material_at[kMaterialMask];
                if (
                    merging && mode == kLightTexture && kind == kKindMasked &&
                    mask_base != 0 && mask_base - 1 + level < entries
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
            for (uint32_t channel = 0; channel < 3; ++channel) {
                numbers[kConstBase + channel] = material_at[1 + channel];
            }
            if (mode == kFlat) {
                continue;  // FILL's colour divided by 1/z, no texel
            }
            record_numbers(numbers);
            for (int32_t r = row_first; r <= row_last; ++r) {
                record_row(r, mode);
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
        if (!texels(
            first_row,
            end_row,
            index,
            pages,
            gamma,
            slots_address,
            page_limit,
            merging,
            counters
        )) {
            succeeded = false;
            break;
        }
        mark = km::wall();
        pass_k::setup_pass::order(word(kVecAlbedo + 64 + end_row - 1) + 62);
        for (int32_t r = first_row; r < end_row; ++r) {
            colour_row(r);
        }
        tensix_sync();
        if (!reserve(kCbOut, kShadeOutPages)) {
            succeeded = false;
            break;
        }
        const uint32_t destination = write_ptr(kCbOut);
        const uint32_t first_word = kRowBytes / 4 * first_row;
        const uint32_t words = kRowBytes / 4 * (end_row - first_row);
        for (uint32_t plane = 0; plane < kShadeColourPages; ++plane) {
            km::dst::read(
                reinterpret_cast<volatile uint32_t*>(
                    destination + 4 * (plane * kPlaneWords + first_word)
                ),
                (1 + plane) * kPlaneWords + first_word,
                words
            );
        }
        if (merging) {  // the opaque layer's colour where the depth row is not 1
            uint32_t* out = reinterpret_cast<uint32_t*>(destination);
            for (uint32_t at = first_word; at < first_word + words; at += 32) {
                const vbool4_t take = (
                    __riscv_vmsne_vx_u32m8_b4(
                        __riscv_vle32_v_u32m8(
                            const_cast<const uint32_t*>(window() + at), vl
                        ),
                        0,
                        vl
                    )
                );
                for (uint32_t plane = 0; plane < kShadeColourPages; ++plane) {
                    uint32_t* colour = out + plane * kPlaneWords + at;
                    __riscv_vse32_v_u32m8(
                        colour,
                        __riscv_vmerge_vvm_u32m8(
                            __riscv_vle32_v_u32m8(
                                opaque_words + (1 + plane) * kPlaneWords + at, vl
                            ),
                            __riscv_vle32_v_u32m8(colour, vl),
                            take,
                            vl
                        ),
                        vl
                    );
                }
            }
            asm volatile("" ::: "memory");
        }
        push(
            kCbOut,
            destination +
                4 * ((kShadeColourPages - 1) * kPlaneWords + first_word + words - 1),
            kShadeOutPages
        );
        if (
            merging && (!pop(kCbShadePlanes, kPlanes) || !pop(kCbShadePlanes, kPlanes))
        ) {
            succeeded = false;
            break;
        }
        values[kShadeProfOut] += km::wall() - mark;
        values[kShadeProfTiles]++;
    }
    // the reader's end, the list's last page, the table back, the unpack thread's end
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
            profile[i] = i <= kShadeProfTotal ? values[i] : 0;
        }
        push(
            kCbOut,
            reinterpret_cast<uint32_t>(profile) + kShadeProfBytes - 4,
            kShadeOutPages
        );
    }
#endif
}
