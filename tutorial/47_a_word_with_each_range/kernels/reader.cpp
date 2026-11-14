// The reader kernel: it pulls ranges of numbers from its X280 core. The word that comes
// with each range says what to add up: 1 the numbers, 2 their squares.
#include <stdint.h>

#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_x280.hpp"

void kernel_main() {
    km::Reader reader;
    km::X280Port port(reader);
    reader.run([&](const km_command_t& command) {
        uint32_t total = 0;
        // --8<-- [start:pull]
        uint32_t first = 0, word = 0;
        while (const uint32_t count = port.next(first, word)) {
            for (uint32_t number = first; number < first + count; number++) {
                total += word == 2 ? number * number : number;
            }
            port.tell(total, 0, 0);
        }
        // --8<-- [end:pull]
    });
}
