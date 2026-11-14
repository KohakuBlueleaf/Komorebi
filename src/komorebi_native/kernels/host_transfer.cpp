// A transfer between DRAM and the host channel without a run: one launch, one rank a
// core, the slice km_transfer.hpp moves; it reports in its L1 report words.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_core.hpp"
#include "komorebi/device/km_transfer.hpp"

namespace {

    constexpr uint32_t kBuffer = 0;  // the circular buffer that holds the two chunks
    constexpr uint32_t kLimit = 1u
        << 18;  // 1024-cycle units of every wait (about 0.2 s)
    // Report words: status (kTransfer* + 1 once done), wall clock at the start and
    // after the transfer (low, high), then the rank.
    constexpr uint32_t kReportStatus = 0;
    constexpr uint32_t kReportStart = 1;
    constexpr uint32_t kReportEnd = 3;
    constexpr uint32_t kReportRank = 5;
    constexpr uint32_t kReportStop = 7;  // stays 0: the stop word of every wait
    constexpr uint32_t kReportWords = 8;

}  // namespace

// Arguments: report address, 1 push or 0 pull, rank, ranks, the ten transfer words.
void kernel_main() {
    volatile tt_l1_ptr uint32_t* report = km::l1(get_arg_val<uint32_t>(0));
    const bool push = get_arg_val<uint32_t>(1) == 1;
    const uint32_t rank = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    uint32_t words[10];
    for (uint32_t i = 0; i < 10; ++i) {
        words[i] = get_arg_val<uint32_t>(4 + i);
    }
    km::Transfer t = {
        words[KM_TRANSFER_ADDRESS],
        words[KM_TRANSFER_PAGE],
        words[KM_TRANSFER_FIRST],
        words[KM_TRANSFER_PAGES],
        words[KM_TRANSFER_OFFSET],
        words[KM_TRANSFER_WORDS],
        words[KM_TRANSFER_REGION],
        words[KM_TRANSFER_NONCE],
        get_write_ptr(kBuffer),
        words[KM_TRANSFER_CHUNK]
    };
    for (uint32_t i = 0; i < kReportWords; ++i) {
        report[i] = 0;
    }
    const auto wait = [&](auto ready) {
        return km::poll_stop(report + kReportStop, ready, kLimit, 0) == km::kGaveNone;
    };
    const uint64_t start = km::wall64();
    const uint32_t status = (
        push
            ? km::push_to_host(t, rank, count, wait)
            : km::pull_from_host(t, rank, count, wait)
    );
    const uint64_t end = km::wall64();
    report[kReportStart] = static_cast<uint32_t>(start);
    report[kReportStart + 1] = static_cast<uint32_t>(start >> 32);
    report[kReportEnd] = static_cast<uint32_t>(end);
    report[kReportEnd + 1] = static_cast<uint32_t>(end >> 32);
    report[kReportRank] = rank;
    report[kReportStatus] = status + 1;
}
