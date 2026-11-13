// The grid side: the reader owns the port. Per record: wait for the X280 core's salt,
// ask a0 values, hash them, tell the X280 core the hash and the asks' cycles.
#include <stdint.h>

#include "../x280_ports.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_x280.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    km::X280Port port(rd);
    rd.run([&](const km_record_t& r) {
        km::X280Msg m;
        if (km::seg(r) != HX_SEG_WORK || !port.wait_told(m)) {
            return;
        }
        uint32_t acc = m.w[0], sum = 0, top = 0;
        for (uint32_t k = 1; k <= r.a0; k++) {
            const uint32_t t0 = km::wall();
            if (!port.call(k, 0, 0, m)) {
                return;
            }
            const uint32_t dt = km::wall() - t0;
            sum += dt;
            top = dt > top ? dt : top;
            acc = HX_MIX(acc, m.w[0]);
        }
        port.tell(acc, sum, top);
    });
}
