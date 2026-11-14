// The writer: in a PASS, takes the reader's jobs; for each dense range it loads the
// dense overlay's writer part (the meeting with the reader's load) and calls it.
#include <stdint.h>

#include "../split.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_overlay.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != TS_HANDLER_PASS || command.a7 != 0) {
            return true;
        }
        for (;;) {
            if (!writer.wait_front(TS_CB_JOB_WRITER, 1, 64)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* job = km::l1(get_read_ptr(TS_CB_JOB_WRITER));
            const ts_drain drain = {
                job[TS_JOB_FIRST],
                job[TS_JOB_COUNT],
                command.a1,
                command.a2,
                km::slot(command),
                command.sequence
            };
            const uint32_t overlay = job[TS_JOB_OVERLAY];
            cb_pop_front(TS_CB_JOB_WRITER, 1);
            if (overlay != TS_OVERLAY_DENSE) {
                return true;
            }
            km::Overlay dense(writer, TS_OVERLAY_DENSE);
            if (!dense.load() || dense.call(drain) != drain.count) {
                return false;
            }
        }
    });
}
