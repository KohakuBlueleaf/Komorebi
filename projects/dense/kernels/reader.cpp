// Matrix product reader: LOAD reads all of W into the W buffer; PRODUCT reads the rows
// of A this member owns (and, in the streamed form, W's chunks), in buffer items.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

namespace {

    // Reads `rows` x `width` tiles of t from tile `first` (rows `stride` tiles apart)
    // into the reserved item of cb, then pushes it. False: the reads gave up.
    template <typename Accessor>
    bool read_item(
        const km::Reader& rd,
        uint32_t cb,
        const Accessor& t,
        uint32_t first,
        uint32_t rows,
        uint32_t width,
        uint32_t stride
    ) {
        uint32_t dst = get_write_ptr(cb);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                noc_async_read_page(first + i * stride + j, t, dst);
                dst += kTileBytes;
            }
        }
        if (!flushed(rd, km::kStReader, km::kFlushReads)) {
            return false;
        }
        cb_push_back(cb, rows * width);
        return true;
    }

    // LAYER: A's row from DRAM per owned row when A is not held (B is always held).
    template <typename Accessor>
    void layer(
        const km::Reader& rd,
        const km_record_t& r,
        const Accessor& a,
        volatile tt_l1_ptr uint32_t* prof
    ) {
        if ((r.a8 & 0xFF) != kDram) {
            return;
        }
        const km::Share s = km::share(r.a2, rd.me() - km::first(r), km::count(r));
        const uint32_t kt = r.a3;
        uint32_t wait = 0, read = 0, n = 0;
        for (uint32_t m = s.start; m < s.start + s.len; ++m) {
            const uint32_t t1 = km::wall();
            if (!rd.reserve(kCbA, kt)) {
                return;
            }
            const uint32_t t2 = km::wall();
            if (!read_item(rd, kCbA, a, m * kt, 1, kt, 0)) {
                return;
            }
            wait += t2 - t1;
            read += km::wall() - t2;
            n += kt;
        }
        km::add_sat(prof + kPrRdWait, wait);
        km::add_sat(prof + kPrRdRead, read);
        km::add_sat(prof + kPrRdTiles, n);
    }

}  // namespace

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<0>();  // every operand: bf16 tiles in DRAM
    km::Reader rd;
    volatile tt_l1_ptr uint32_t* prof = km::l1(get_write_ptr(kCbProf));
    for (uint32_t w = 0; w < kProfWords; ++w) {
        prof[w] = 0;
    }
    rd.run([&](const km_record_t& r) {
        const uint32_t t0 = km::wall();
        const uint32_t kt = r.a3, nt = r.a4;
        if (km::seg(r) == kSegLoad) {
            if (!rd.reserve(r.a1, kt * nt)) {
                return;
            }
            const auto w = TensorAccessor(tiles, r.a0);
            if (read_item(rd, r.a1, w, 0, 1, kt * nt, 0) && rd.reserve(kCbLoaded, 1)) {
                cb_push_back(kCbLoaded, 1);
            }
            return;
        }
        if (km::seg(r) == kSegLayer) {
            layer(rd, r, TensorAccessor(tiles, r.a0), prof);
            km::add_sat(prof + kPrRdBody, km::wall() - t0);
            return;
        }
        if (km::seg(r) != kSegProduct || r.a7 == kModeComputeOnly) {
            return;
        }
        const km::Share s = km::share(r.a2, rd.me() - km::first(r), km::count(r));
        const uint32_t kc = r.a5, ct = r.a6;
        const auto a = TensorAccessor(tiles, r.a0);
        const auto w = TensorAccessor(tiles, r.a9);
        const bool stream = r.a8 == kFormStream;
        // The row is one item when it is one chunk or when W is streamed.
        const bool row_item = stream || kc == kt;
        uint32_t wait = 0, read = 0, ws_wait = 0, ws_read = 0, n = 0;
        for (uint32_t m = s.start; m < s.start + s.len; ++m) {
            for (uint32_t j0 = 0; j0 < nt; j0 += ct) {
                for (uint32_t k0 = 0; k0 < kt; k0 += kc) {
                    if (!row_item || (j0 == 0 && k0 == 0)) {
                        const uint32_t len = row_item ? kt : kc;
                        const uint32_t t1 = km::wall();
                        if (!rd.reserve(kCbA, len)) {
                            return;
                        }
                        const uint32_t t2 = km::wall();
                        if (!read_item(rd, kCbA, a, m * kt + k0, 1, len, 0)) {
                            return;
                        }
                        wait += t2 - t1;
                        read += km::wall() - t2;
                        n += len;
                    }
                    if (stream) {
                        const uint32_t t1 = km::wall();
                        if (!rd.reserve(kCbWs, kc * ct)) {
                            return;
                        }
                        const uint32_t t2 = km::wall();
                        if (!read_item(rd, kCbWs, w, k0 * nt + j0, kc, ct, nt)) {
                            return;
                        }
                        ws_wait += t2 - t1;
                        ws_read += km::wall() - t2;
                        n += kc * ct;
                    }
                }
            }
        }
        km::add_sat(prof + kPrRdWait, wait);
        km::add_sat(prof + kPrRdRead, read);
        km::add_sat(prof + kPrRdWsWait, ws_wait);
        km::add_sat(prof + kPrRdWsRead, ws_read);
        km::add_sat(prof + kPrRdTiles, n);
        km::add_sat(prof + kPrRdBody, km::wall() - t0);
    });
}
