// The frame program's reader: the passes' (segments from 11) and the fill's, each
// file included in a namespace of its own without its entry, chosen by segment.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#define RASTER_NO_ENTRY

namespace fill_k {
#include "reader.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_reader.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) >= pass_k::kSegLanes) {
            pass_k::setup_reader_record(rd, r);
        } else {
            fill_k::fill_reader_record(rd, r);
        }
    });
}
