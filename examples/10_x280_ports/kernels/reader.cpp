// The grid side: the reader owns the port. Per command: wait for the X280 core's salt,
// ask `asks` values, hash them, tell the X280 core the hash and the asks' cycles.

// HX_HANDLER_PASS runs units (equal or pulled ranges); HX_HANDLER_REPORT tells numbers.
#include <stdint.h>

#include "../x280_ports.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_x280.hpp"
#include "komorebi/device/km_worker.hpp"

// Busy for `cycles` on the wall clock: a unit of the pass.
static void spin(uint32_t cycles) {
    const uint32_t start = km::wall();
    while (km::wall() - start < cycles) {
        ;
    }
}

void kernel_main() {
    km::Reader reader;
    km::X280Port port(reader);
    // the last pass's numbers
    uint32_t last_pass[HP_LINE_SERVED_BY] = {0, 0, 0, 0, 0, 0};
    reader.run([&](const km_command_t& command) {
        km::X280Msg message;
        if (km::handler(command) == HX_HANDLER_REPORT) {
            const uint32_t page = command.a0;
            const uint32_t* numbers = (
                last_pass + (page == 0 ? HP_LINE_UNITS : HP_LINE_HASH)
            );
            port.tell(numbers[0], numbers[1], numbers[2]);
            return;
        }
        if (km::handler(command) == HX_HANDLER_PASS) {
            const uint32_t units = command.a0;
            const uint32_t pulled = command.a1;
            const uint32_t base = command.a2;
            const uint32_t extra = command.a3;
            const uint32_t centre = command.a4;
            const uint32_t half = command.a5;
            const uint32_t pass_start = km::wall();
            uint32_t units_run = 0, work = 0, hash = 0, pulls = 0, waited = 0;
            auto run_range = [&](uint32_t first, uint32_t count) {
                const uint32_t range_start = km::wall();
                for (uint32_t unit = first; unit < first + count; unit++) {
                    spin(HX_COST(unit, base, extra, centre, half));
                    hash += HX_MARK(unit);
                }
                work += km::wall() - range_start;
                units_run += count;
            };
            if (pulled == 0) {
                const km::Partition partition = (
                    km::partition(
                        units, reader.me() - km::first(command), km::count(command)
                    )
                );
                run_range(partition.start, partition.len);
            } else {
                uint32_t first = 0;
                for (;;) {
                    const uint32_t wait_start = km::wall();
                    const uint32_t count = port.next(first);
                    waited += km::wall() - wait_start;
                    if (count == 0) {
                        break;
                    }
                    pulls++;
                    run_range(first, count);
                }
            }
            const uint32_t numbers[HP_LINE_SERVED_BY] = {
                units_run, work, km::wall() - pass_start, hash, pulls, waited
            };
            for (uint32_t number = 0; number < HP_LINE_SERVED_BY; number++) {
                last_pass[number] = numbers[number];
            }
            return;
        }
        if (km::handler(command) != HX_HANDLER_WORK || !port.wait_told(message)) {
            return;
        }
        const uint32_t asks = command.a0;
        uint32_t hash = message.w[0], total = 0, largest = 0;
        for (uint32_t ask = 1; ask <= asks; ask++) {
            const uint32_t start = km::wall();
            if (!port.call(ask, 0, 0, message)) {
                return;
            }
            const uint32_t cycles = km::wall() - start;
            total += cycles;
            largest = cycles > largest ? cycles : largest;
            hash = HX_MIX(hash, message.w[0]);
        }
        port.tell(hash, total, largest);
    });
}
