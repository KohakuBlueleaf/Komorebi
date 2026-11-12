// A case's compute kernels: after one page, a bounded wait (in) or reservation (out) of
// a 2-page item in a 2-page buffer. Built per thread; KMSIM_PART names the thread.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/km_worker.hpp"

#define KMSIM_CAT(a, b, c) a##b##c
#define KMSIM_NAME(a, b, c) KMSIM_CAT(a, b, c)

namespace {

    constexpr uint32_t kCbItemIn = 24;
    constexpr uint32_t kCbItemOut = 25;

}  // namespace

void KMSIM_NAME(kmsim_item_in_, KMSIM_PART, _main)() {
    const km::Compute c;
    UNPACK((cb_wait_front(kCbItemIn, 1), cb_pop_front(kCbItemIn, 1)));
    (void)c.wait_front(kCbItemIn, 2);
}

void KMSIM_NAME(kmsim_item_out_, KMSIM_PART, _main)() {
    const km::Compute c;
    PACK((cb_reserve_back(kCbItemOut, 1), cb_push_back(kCbItemOut, 1)));
    (void)c.reserve_back(kCbItemOut, 2);
}
