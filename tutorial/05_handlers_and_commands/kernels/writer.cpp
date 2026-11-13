// The writer kernel: one handler per kind of command, each returns one result word.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer wr;
    // --8<-- [start:body]
    wr.run([&](const km_record_t& r) {
        switch (km::seg(r)) {
            case kSegAdd:
                wr.result(r.a0 + r.a1);
                break;
            case kSegDot:
                wr.result(r.a0 * r.a3 + r.a1 * r.a4 + r.a2 * r.a5);
                break;
            case kSegWho:
                wr.result(wr.me() + r.a0);
                break;
            default:
                break;  // NOP and any handler this kernel has no body for
        }
        return true;
    });
    // --8<-- [end:body]
}
