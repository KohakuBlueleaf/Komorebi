// COUNT and SETUP compute: unpack hands each mesh page to pack, which runs the vector
// unit, keeps or clips each triangle, and counts (COUNT) or sends (SETUP) its records.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "komorebi/device/km_worker.hpp"
#include "setup.h"
#ifdef TRISC_PACK
#include "setup.hpp"
#endif

namespace {

    // A give-up into the writer's gave-up word (the first one is kept) and the stop
    // word.
    void give_up(uint32_t code, uint32_t where) {
        const uint32_t ctl = get_arg_val<uint32_t>(0);
        volatile uint32_t* st = (
            reinterpret_cast<volatile uint32_t*>(ctl + KM_CTL_STATE)
        );
        if (st[km::kStWriter] == 0) {
            st[km::kStWriter] = code | where << 8;
        }
        *reinterpret_cast<volatile uint32_t*>(ctl + KM_CTL_STOP) = 1;
    }

    // Polls ready() for at most kWaitCycles. False: gave up (recorded at `where`).
    template <typename Ready>
    bool timed(Ready ready, uint32_t where) {
        const uint32_t t0 = km::wall();
        while (!ready()) {
            if (km::wall() - t0 > kWaitCycles) {
                give_up(kGaveWait, where);
                return false;
            }
        }
        return true;
    }

    bool mail_from(ckernel::ThreadId thread, uint32_t where) {
        return timed(
            [&] {
                return ckernel::mailbox_not_empty(thread);
            },
            where
        );
    }

#ifdef TRISC_UNPACK
    // Waits for a page of `cb` and gives its address to the pack thread.
    bool hand(uint32_t cb, uint32_t where) {
        const bool page = timed(
            [&] {
                const uint16_t received = (
                    reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb)))
                );
                const uint16_t acked = get_local_cb_interface(cb).tiles_acked;
                return static_cast<uint16_t>(received - acked) >= 1;
            },
            where
        );
        if (!page) {
            return false;
        }
        cb_wait_front(cb, 1);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(cb, 0)
        );
        return true;
    }
#endif

}  // namespace

#ifdef TRISC_PACK
namespace {

    // float32 operations one at a time, so that nothing is fused. The cores' floats
    // round to nearest and flush results below the normal range; they cannot divide.
    inline float fsub(float a, float b) {
        volatile float r = a - b;
        return r;
    }

    inline float fmul(float a, float b) {
        volatile float r = a * b;
        return r;
    }

    inline float fadd(float a, float b) {
        volatile float r = a + b;
        return r;
    }

    inline uint32_t bits(float f) {
        uint32_t u;
        __builtin_memcpy(&u, &f, 4);
        return u;
    }

    inline float value(uint32_t u) {
        float f;
        __builtin_memcpy(&f, &u, 4);
        return f;
    }

    // a / b rounded to nearest even, in integers: a normal or 0, b normal, a result
    // below the normal range flushed to a signed 0 as the cores' floats do.
    inline float fdiv(float a, float b) {
        const uint32_t ua = bits(a), ub = bits(b);
        const uint32_t sign = (ua ^ ub) & 0x80000000;
        if ((ua & 0x7FFFFFFF) == 0) {
            return value(sign);
        }
        uint32_t ma = (ua & 0x7FFFFF) | 0x800000;
        const uint32_t mb = (ub & 0x7FFFFF) | 0x800000;
        int32_t e =
            (static_cast<int32_t>(ua >> 23 & 0xFF) -
             static_cast<int32_t>(ub >> 23 & 0xFF) + 127);
        if (ma < mb) {
            ma <<= 1;
            e -= 1;
        }
        // 25 quotient bits (24 and a guard bit), the remainder as the sticky bit
        uint32_t q = 0, r = ma;
        for (uint32_t i = 0; i < 25; ++i) {
            q <<= 1;
            if (r >= mb) {
                r -= mb;
                q |= 1;
            }
            r <<= 1;
        }
        uint32_t m = q >> 1;
        if ((q & 1) && (r != 0 || (m & 1))) {
            ++m;
        }
        if (m == 0x1000000) {
            m >>= 1;
            ++e;
        }
        if (e <= 0) {
            return value(sign);
        }
        return value(sign | static_cast<uint32_t>(e) << 23 | (m & 0x7FFFFF));
    }

    // An integer below 2^24 as float32 bits (exact).
    inline uint32_t float_bits(uint32_t n) {
        if (n == 0) {
            return 0;
        }
        const uint32_t e = 31 - __builtin_clz(n);
        return (127 + e) << 23 | ((n << (23 - e)) & 0x7FFFFF);
    }

    // A corner: camera-space x, y, z, then red, green, blue.
    struct Corner {
        float v[6];
    };

    // The polygon of a triangle cut at z = near (ref/raster.py clip_polygon): each
    // corner beyond, and a point where an edge crosses, from its corner beyond.
    uint32_t clip(const Corner* c, float near, Corner* out) {
        uint32_t n = 0;
        for (uint32_t k = 0; k < 3; ++k) {
            const Corner& a = c[k];
            const Corner& b = c[(k + 1) % 3];
            // z > near on the bits (near > 0, z not NaN): no float compare, whose flag
            // save and restore (frflags, fsflags) the cores' documentation leaves out.
            const int32_t nb = static_cast<int32_t>(bits(near));
            const bool ia = static_cast<int32_t>(bits(a.v[2])) > nb;
            const bool ib = static_cast<int32_t>(bits(b.v[2])) > nb;
            if (ia) {
                out[n++] = a;
            }
            if (ia != ib) {
                const Corner& in = ia ? a : b;
                const Corner& o = ia ? b : a;
                const float t = fdiv(fsub(near, in.v[2]), fsub(o.v[2], in.v[2]));
                for (uint32_t i = 0; i < 6; ++i) {
                    out[n].v[i] = fadd(in.v[i], fmul(t, fsub(o.v[i], in.v[i])));
                }
                out[n].v[2] = near;
                ++n;
            }
        }
        return n;
    }

    // An integer in [0, 2048) as float16 bits.
    inline uint32_t half(uint32_t n) {
        if (n == 0) {
            return 0;
        }
        const uint32_t e = 31 - __builtin_clz(n);
        return (e + 15) << 10 | ((n << (10 - e)) & 0x3FF);
    }

    // The box of a raw record: columns x0, x1 and rows y0, y1.
    struct Box {
        int32_t x0, x1, y0, y1;
    };

    inline Box box_of(const uint32_t* w) {
        return {
            static_cast<int32_t>(w[kRawX0] - kMagic),
            static_cast<int32_t>(w[kRawX1] - kMagic),
            static_cast<int32_t>(w[kRawY0] - kMagic),
            static_cast<int32_t>(w[kRawY1] - kMagic)
        };
    }

    // Kept by the reference's cull: a non-zero area and a box that is not empty.
    inline bool kept(const uint32_t* w, const Box& b) {
        return (w[kRawArea] & 0x7FFFFFFF) != 0 && b.x0 <= b.x1 && b.y0 <= b.y1;
    }

    // The words the vector unit leaves raw: thresholds (top-left), box, triangle.
    void finish(uint32_t* w, const Box& b, uint32_t tri) {
        for (uint32_t k = 0; k < 3; ++k) {
            const int32_t a = static_cast<int32_t>(w[k]);
            const int32_t bb = static_cast<int32_t>(w[3 + k]);
            const bool tl = a > 0 || ((w[k] & 0x7FFFFFFF) == 0 && bb > 0);
            w[9 + k] = tl ? 0x80000001 : 0;
        }
        w[24] = half(b.x0);
        w[25] = half(b.x1 + 1);
        w[26] = b.y0;
        w[27] = b.y1;
        w[28] = float_bits(tri);
        w[29] = 0;
        w[30] = 0;
        w[31] = 0;
    }

    uint32_t* cb_ptr(uint32_t cb) {
        return reinterpret_cast<uint32_t*>(get_local_cb_interface(cb).fifo_wr_ptr << 4);
    }

    // Zeroes n words (a multiple of 32), 32 per vector store.
    void zero(uint32_t* p, uint32_t n) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        const vuint32m8_t z = __riscv_vmv_v_x_u32m8(0, vl);
        for (uint32_t i = 0; i < n; i += 32) {
            __riscv_vse32_v_u32m8(p + i, z, vl);
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t i = 0; i < n; ++i) {
            p[i] = 0;
        }
#endif
    }

    // SETUP's out pages: up to kOutEntries records, each with its target slot.
    struct Out {
        uint32_t* page = nullptr;
        uint32_t n = 0;
        uint32_t wait = 0;     // cycles waiting for a free out page
        uint32_t entries = 0;  // records sent

        bool open(const km::Compute& cp) {
            if (page == nullptr) {
                const uint32_t a = km::wall();
                const bool ok = cp.reserve_back(kCbOut, 1);
                wait += km::wall() - a;
                if (!ok) {
                    return false;
                }
                page = cb_ptr(kCbOut);
                n = 0;
            }
            return true;
        }

        bool add(const km::Compute& cp, const uint32_t* rec, uint32_t slot) {
            if (!open(cp)) {
                return false;
            }
            uint32_t* dst = page + 32 * (1 + n);
#if defined(__riscv_vector)
            const size_t vl = __riscv_vsetvl_e32m8(32);
            __riscv_vse32_v_u32m8(dst, __riscv_vle32_v_u32m8(rec, vl), vl);
            asm volatile("" ::: "memory");
#else
            for (uint32_t k = 0; k < 32; ++k) {
                dst[k] = rec[k];
            }
#endif
            page[1 + n] = slot;
            ++entries;
            if (++n == kOutEntries) {
                push(false);
            }
            return true;
        }

        void push(bool last) {
            page[0] = n | (last ? kOutLast : 0);
            asm volatile("fence" ::: "memory");
            const uint32_t v = reinterpret_cast<volatile uint32_t*>(page)[0];
            asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
            cb_push_back(kCbOut, 1);
            page = nullptr;
        }

        // The last page (possibly without entries) tells the writer to stop.
        bool end(const km::Compute& cp) {
            if (!open(cp)) {
                return false;
            }
            push(true);
            return true;
        }
    };

    // Corners beyond the near distance (z > near > 0: float32 order as integers).
    uint32_t beyond_of(const uint32_t* w, uint32_t near) {
        uint32_t n = 0;
        for (uint32_t k = 0; k < 3; ++k) {
            n += static_cast<int32_t>(w[kRawZ + k]) > static_cast<int32_t>(near);
        }
        return n;
    }

    // One pass's state: the mode, the counters (COUNT) or the tiles' next slots and
    // the out pages (SETUP).
    struct Pass {
        const km::Compute& cp;
        uint32_t handler;
        uint32_t* cnt = nullptr;   // COUNT: entries of tile t at t, rows at kTiles + t
        uint16_t* hist = nullptr;  // COUNT: tile t's start and end counts at 32 t
        uint32_t* slot = nullptr;  // SETUP: item i's next slot
        const uint32_t* items = nullptr;  // SETUP: the items page (setup.h)
        uint32_t slots = 0;               // SETUP: the lists' slots
        Out out;
        bool ok = true;
        uint32_t kept_n = 0, clipped = 0;  // the profile's counts

        // A finished record that touches box b (inside the image, or a give-up).
        void record(const uint32_t* w, const Box& b) {
            if (b.x0 < 0 || b.y0 < 0 || b.x1 >= static_cast<int32_t>(kWidth) ||
                b.y1 >= static_cast<int32_t>(kHeight)) {
                give_up(kGaveBadBox, w[28] & 0xFFFF);
                ok = false;
                return;
            }
            for (int32_t ty = b.y0 >> 5; ty <= (b.y1 >> 5); ++ty) {
                const int32_t top = 32 * ty;
                const int32_t r0 = b.y0 > top ? b.y0 : top;
                const int32_t r1 = b.y1 < top + 31 ? b.y1 : top + 31;
                for (int32_t tx = b.x0 >> 5; tx <= (b.x1 >> 5); ++tx) {
                    const uint32_t t = kTilesX * ty + tx;
                    if (handler == kHandlerCount) {
                        cnt[t] += 1;
                        cnt[kTiles + t] += r1 - r0 + 1;
                        uint16_t* h = hist + 32 * t;
                        if (++h[(r0 - top) >> 1] == 0 ||
                            ++h[16 + ((r1 - top) >> 1)] == 0) {
                            give_up(kGaveBadHist, t);
                            ok = false;
                            return;
                        }
                        continue;
                    }
                    const uint32_t first = items[t] & 0xFFFF, n = items[t] >> 16;
                    if (first + n > kSlotsBytes / 4) {
                        give_up(kGaveBadSlot, t);
                        ok = false;
                        return;
                    }
                    const uint16_t* rows = (
                        reinterpret_cast<const uint16_t*>(items + kItemRows / 4)
                    );
                    for (uint32_t i = first; i < first + n; ++i) {
                        // the item's rows [b0, b1) against the entry's 2-row groups
                        const uint32_t b0 = rows[i] & 0xFF, b1 = rows[i] >> 8;
                        if (((r0 - top) >> 1) >= static_cast<int32_t>(b1 >> 1) ||
                            ((r1 - top) >> 1) < static_cast<int32_t>(b0 >> 1)) {
                            continue;
                        }
                        const uint32_t s = slot[i]++;
                        if (s >= slots) {
                            give_up(kGaveBadSlot, t);
                            ok = false;
                            return;
                        }
                        if (!out.add(cp, w, s)) {
                            ok = false;
                            return;
                        }
                    }
                }
            }
        }

        // Clips a crossing triangle and sends its pieces (one or two) through the
        // vector unit; their records go out as the triangle's. `rec`: free space.
        void pieces(const Corner* c, float near, uint32_t* rec, uint32_t tri) {
            Corner poly[6];  // at most 4; room for 6 so the check below comes first
            const uint32_t n = clip(c, near, poly);
            if (n < 3 || n > 4) {
                give_up(kGaveBadClip, tri & 0xFFFF);
                ok = false;
                return;
            }
            constexpr uint32_t fan[2][3] = {{0, 1, 2}, {0, 2, 3}};
            const uint32_t m = n - 2;
            ++clipped;
            volatile uint32_t* win = su::win();
            for (uint32_t j = 0; j < m; ++j) {
                for (uint32_t k = 0; k < 3; ++k) {
                    const Corner& p = poly[fan[j][k]];
                    for (uint32_t i = 0; i < 3; ++i) {
                        win[su::word(su::kSCs + 3 * k + i) + 2 * j] = bits(p.v[i]);
                        win[su::word(su::kSCol + 3 * k + i) + 2 * j] = bits(p.v[3 + i]);
                    }
                }
            }
            su::order(su::word(su::kSCol + 8) + 2 * (m - 1));
            su::triangles(handler == kHandlerSetup);
            ckernel::tensix_sync();
            su::records_out(rec);
            for (uint32_t j = 0; j < m && ok; ++j) {
                uint32_t* w = rec + 32 * j;
                const Box b = box_of(w);
                if (!kept(w, b)) {
                    continue;
                }
                finish(w, b, tri);
                ++kept_n;
                record(w, b);
            }
        }

        // The block's kept triangles in order, from raw records at `rec` (Dst holds the
        // block), a crossing one's pieces at its place; `work`: 2,048 words.
        void block(uint32_t* rec, const uint32_t* mesh, uint32_t* work, uint32_t tri0) {
            const uint32_t near = su::win()[su::word(su::kSCam + 15)];
            Corner* corners = reinterpret_cast<Corner*>(work + 1024);
            uint32_t crossing = 0;
            for (uint32_t l = 0; l < 32; ++l) {
                const uint32_t beyond = beyond_of(rec + 32 * l, near);
                if (beyond != 1 && beyond != 2) {
                    continue;
                }
                crossing |= 1u << l;
                for (uint32_t k = 0; k < 3; ++k) {
                    Corner& c = corners[3 * l + k];
                    for (uint32_t i = 0; i < 3; ++i) {
                        const uint32_t cs = su::word(su::kSCs + 3 * k + i) + 2 * l;
                        const uint32_t col = su::word(su::kSCol + 3 * k + i) + 2 * l;
                        c.v[i] = value(su::win()[cs]);
                        c.v[3 + i] = value(mesh[col]);
                    }
                }
            }
            for (uint32_t l = 0; l < 32 && ok; ++l) {
                uint32_t* w = rec + 32 * l;
                if (crossing >> l & 1) {
                    pieces(corners + 3 * l, value(near), work, tri0 + l);
                    continue;
                }
                if (beyond_of(w, near) != 3) {
                    continue;
                }
                const Box b = box_of(w);
                if (!kept(w, b)) {
                    continue;
                }
                finish(w, b, tri0 + l);
                ++kept_n;
                record(w, b);
            }
        }
    };

}  // namespace
#endif

// The passes' work for one command; the frame program's compute kernel calls it as
// well.
void setup_compute_command(const km::Compute& cp, const km_command_t& r) {
    const uint32_t handler = km::handler(r);
    if ((handler != kHandlerLanes && handler != kHandlerCount &&
         handler != kHandlerSetup) ||
        r.a5 == 0 || cp.me() >= r.a5) {
        return;
    }
    // with kRanges the camera's page brings the core's range (setup_reader.cpp)
    const bool ranged = (r.a2 & kRanges) != 0;
    const uint32_t blocks = r.a2 & ~kRanges;
    uint32_t b0 = cp.me() * blocks / r.a5;
    uint32_t pages = 1 + ((cp.me() + 1) * blocks / r.a5 - b0);
#ifdef TRISC_UNPACK
    // SETUP: the slots and items pages first, held until the pass ends.
    const bool slots = (
        handler != kHandlerSetup || (hand(kCbSlots, 4) && hand(kCbItems, 4))
    );
    // Hands each page to the pack thread; frees it when the pack thread is done.
    for (uint32_t p = 0; slots && p < pages; ++p) {
        if (!hand(kCbMesh, 1)) {
            break;
        }
        if (p == 0 && ranged) {
            asm volatile("fence" ::: "memory");
            const volatile uint32_t* cam = (
                reinterpret_cast<const volatile uint32_t*>(
                    get_tile_l1_byte_address(kCbMesh, 0)
                )
            );
            pages = 1 + (cam[kCamWords + 1] - cam[kCamWords]);
        }
        if (!mail_from(ckernel::ThreadId::PackThreadId, 2)) {
            break;
        }
        (void)ckernel::mailbox_read(ckernel::ThreadId::PackThreadId);
        cb_pop_front(kCbMesh, 1);
    }
    if (handler == kHandlerSetup && slots) {
        cb_pop_front(kCbSlots, 1);
        cb_pop_front(kCbItems, 1);
    }
#endif
#ifdef TRISC_PACK
    const uint32_t t0 = km::wall();
    uint32_t mail = 0, vec = 0, scalar = 0;
    su::setup();
    Pass pass{cp, handler};
    if (handler == kHandlerSetup) {
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            return;
        }
        pass.slot = (
            reinterpret_cast<uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            return;
        }
        pass.items = (
            reinterpret_cast<const uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        pass.slots = r.a6 & ~kSetupPrefix;
    }
    if (handler == kHandlerCount) {
        if (!cp.reserve_back(kCbCount, 1)) {
            return;
        }
        pass.cnt = cb_ptr(kCbCount);
        zero(pass.cnt, 2 * kTiles);
        // never pushed: the buffer's space, kept for HIST
        uint32_t* hist = cb_ptr(kCbHist);
        zero(hist, kHistBytes / 4);
        pass.hist = reinterpret_cast<uint16_t*>(hist);
    }
    uint32_t* work = cb_ptr(kCbWork);
    for (uint32_t p = 0; p < pages; ++p) {
        const uint32_t a = km::wall();
        if (!mail_from(ckernel::ThreadId::UnpackThreadId, 3)) {
            pass.ok = false;
            break;
        }
        mail += km::wall() - a;
        const uint32_t* in = (
            reinterpret_cast<const uint32_t*>(
                ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)
            )
        );
        asm volatile("fence" ::: "memory");
        if (p == 0) {
            if (ranged) {
                b0 = in[kCamWords];
                pages = 1 + (in[kCamWords + 1] - b0);
            }
            // the camera: each number to every lane of its slot
            for (uint32_t i = 0; i < kCamWords; ++i) {
                su::fill_slot(su::kSCam + i, in[i]);
            }
            su::order(su::word(su::kSCam + kCamWords - 1) + 62);
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            continue;
        }
        const uint32_t v0 = km::wall();
        su::block_in(in);
        su::camera_space();
        su::triangles(handler != kHandlerCount);
        ckernel::tensix_sync();
        if (handler == kHandlerLanes) {
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
            if (!cp.reserve_back(kCbRec, 1)) {
                break;
            }
            uint32_t* rec = cb_ptr(kCbRec);
            su::records_out(rec);
            asm volatile("fence" ::: "memory");
            const uint32_t v = reinterpret_cast<volatile uint32_t*>(rec)[1023];
            asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
            cb_push_back(kCbRec, 1);
            continue;
        }
        su::records_out(work);
        const uint32_t s0 = km::wall();
        vec += s0 - v0;
        pass.block(work, in, work + 1024, 32 * (b0 + p - 1));
        scalar += km::wall() - s0;
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);
        if (!pass.ok) {
            break;
        }
    }
    if (r.a9 != 0) {
        volatile uint32_t* pf = cb_ptr(kCbSProf);
        pf[kSProfPack + 0] = mail;
        pf[kSProfPack + 1] = vec;
        pf[kSProfPack + 2] = scalar;
        pf[kSProfPack + 3] = km::wall() - t0;
        pf[kSProfPack + 4] = pass.kept_n;
        pf[kSProfPack + 5] = pass.clipped;
        pf[kSProfPack + 6] = pass.out.entries;
        pf[kSProfPack + 7] = pass.out.wait;
        asm volatile("fence" ::: "memory");
    }
    if (handler == kHandlerCount && pass.cnt != nullptr) {
        asm volatile("fence" ::: "memory");
        const uint32_t v = (
            reinterpret_cast<volatile uint32_t*>(pass.cnt)[2 * kTiles - 1]
        );
        const uint32_t u = (
            reinterpret_cast<volatile uint32_t*>(pass.hist)[kHistBytes / 4 - 1]
        );
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
        asm volatile("and x0, x0, %0" : : "r"(u) : "memory");
        cb_push_back(kCbCount, 1);
    }
    if (handler == kHandlerSetup && pass.ok) {
        pass.out.end(cp);
    }
    su::finish();
#endif
}

#ifndef RASTER_NO_ENTRY
void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbMesh, kCbRec);
    cp.run([&](const km_command_t& r) {
        setup_compute_command(cp, r);
    });
}
#endif
