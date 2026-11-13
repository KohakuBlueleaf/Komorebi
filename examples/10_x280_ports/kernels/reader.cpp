// The grid side: the reader owns the port. Per record: wait for the X280 core's salt,
// ask a0 values, hash them, tell the X280 core the hash and the asks' cycles.

// HX_HANDLER_PASS runs units (equal or pulled ranges); HX_HANDLER_REPORT tells numbers.
#include <stdint.h>

#include "../x280_ports.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_x280.hpp"
#include "komorebi/device/km_worker.hpp"

// Busy for `cycles` on the wall clock: a unit of the pass.
static void spin(uint32_t cycles) {
    const uint32_t t0 = km::wall();
    while (km::wall() - t0 < cycles) {
        ;
    }
}

void kernel_main() {
    km::Reader rd;
    km::X280Port port(rd);
    uint32_t got[HP_L_WHO] = {0, 0, 0, 0, 0, 0};  // the last pass's numbers
    rd.run([&](const km_record_t& r) {
        km::X280Msg m;
        if (km::seg(r) == HX_HANDLER_REPORT) {
            const uint32_t* w = got + (r.a0 == 0 ? HP_L_UNITS : HP_L_HASH);
            port.tell(w[0], w[1], w[2]);
            return;
        }
        if (km::seg(r) == HX_HANDLER_PASS) {
            const uint32_t t0 = km::wall();
            uint32_t units = 0, work = 0, hash = 0, pulls = 0, waited = 0;
            auto run = [&](uint32_t first, uint32_t count) {
                const uint32_t t1 = km::wall();
                for (uint32_t u = first; u < first + count; u++) {
                    spin(HX_COST(u, r.a2, r.a3, r.a4, r.a5));
                    hash += HX_MARK(u);
                }
                work += km::wall() - t1;
                units += count;
            };
            if (r.a1 == 0) {
                const km::Share s = (
                    km::share(r.a0, rd.me() - km::first(r), km::count(r))
                );
                run(s.start, s.len);
            } else {
                uint32_t first = 0;
                for (;;) {
                    const uint32_t t1 = km::wall();
                    const uint32_t count = port.next(first);
                    waited += km::wall() - t1;
                    if (count == 0) {
                        break;
                    }
                    pulls++;
                    run(first, count);
                }
            }
            const uint32_t now[HP_L_WHO] = {
                units, work, km::wall() - t0, hash, pulls, waited
            };
            for (uint32_t k = 0; k < HP_L_WHO; k++) {
                got[k] = now[k];
            }
            return;
        }
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
