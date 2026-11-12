// DRAM visibility probe, Tensix side: writes a pattern into pages of one DRAM bank or
// copies them to an output in that bank (page j at bank address addr + j * page).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dram_probe.h"

namespace {

    constexpr uint32_t kScratchCb = 0;
    constexpr uint32_t kReportCb = 1;           // 64 bytes
    constexpr uint32_t kSpinLimit = 100000000;  // about a second of polls

    // Polls until every read (or write) of this core has landed; false past the bound.
    bool reads_done() {
        for (uint32_t i = 0; i < kSpinLimit; i++) {
            if (ncrisc_noc_reads_flushed(noc_index)) {
                invalidate_l1_cache();
                return true;
            }
        }
        return false;
    }

    bool writes_done() {
        for (uint32_t i = 0; i < kSpinLimit; i++) {
            if (ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                return true;
            }
        }
        return false;
    }

}  // namespace

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t addr = get_arg_val<uint32_t>(2);  // the region's bank address
    const uint32_t pages = get_arg_val<uint32_t>(3);
    const uint32_t page = get_arg_val<uint32_t>(4);  // bytes, at most the scratch
    const uint32_t out = get_arg_val<uint32_t>(5);   // the output's bank address
    const uint32_t seed = get_arg_val<uint32_t>(6);  // DP_WRITE: the pattern's
    const uint32_t report = get_write_ptr(kReportCb);
    volatile tt_l1_ptr uint32_t* rep = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report)
    );
    const uint32_t l1 = get_write_ptr(kScratchCb);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint64_t first = get_noc_addr_from_bank_id<true>(bank, addr);
    rep[DP_REP_NOC_LO] = static_cast<uint32_t>(first);
    rep[DP_REP_NOC_HI] = static_cast<uint32_t>(first >> 32);
    rep[DP_REP_BANK_OFFSET] = bank_to_dram_offset[bank];
    uint32_t failed = 0;
    for (uint32_t j = 0; j < pages && !failed; j++) {
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank, addr + j * page);
        if (mode == DP_WRITE) {
            for (uint32_t w = 0; w < page / 4; w++) {
                words[w] = dp_word(seed, j, w);
            }
            noc_async_write(l1, src, page);
            failed = !writes_done();
        } else {
            noc_async_read(src, l1, page);
            failed = !reads_done();
            if (!failed) {
                const uint64_t dst = (
                    get_noc_addr_from_bank_id<true>(bank, out + j * page)
                );
                noc_async_write(l1, dst, page);
                failed = !writes_done();
            }
        }
    }
    rep[DP_REP_FAILED] = failed;
    rep[DP_REP_DONE] = DP_DONE;
    // The report line goes to page `pages` of the output (DP_COPY) or the region.
    const uint32_t at = (mode == DP_WRITE ? addr : out) + pages * page;
    noc_async_write(report, get_noc_addr_from_bank_id<true>(bank, at), 64);
    writes_done();
}
