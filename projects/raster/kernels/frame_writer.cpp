// The frame program's writer: the passes' (segments from 11) and the fill's, each
// file included in a namespace of its own without its entry, chosen by segment.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#define RASTER_NO_ENTRY

namespace fill_k {
#include "writer.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_writer.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) >= pass_k::kSegLanes) {
            return pass_k::setup_writer_record(wr, r);
        }
        return fill_k::fill_writer_record(wr, r);
    });
}
