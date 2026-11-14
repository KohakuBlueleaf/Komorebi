// The texture lookup probe's writer (NoC 1): LOOP, MISS, COLOUR for this CPU, the
// packed colour plane into `out`, REPORT (results and out to the DRAM report).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "fetch.hpp"
#include "lookup.h"
#include "table.hpp"

void kernel_main() {
    km::Writer writer;
    constexpr auto pool_args = TensorAccessorArgs<0>();
    constexpr auto report_args = (
        TensorAccessorArgs<pool_args.next_compile_time_args_offset()>()
    );
    const texture::Tensors at = {
        writer.arg(kArgPages),
        writer.arg(kArgSlots),
        writer.arg(kArgStreams),
        writer.arg(kArgPlanes),
        writer.arg(kArgOut),
        writer.arg(kArgResults),
        writer.arg(kArgReplay),
    };
    const auto pool = TensorAccessor(pool_args, writer.arg(kArgPool));
    const uint32_t pool_pages = writer.arg(kArgPoolPages);
    const auto report = TensorAccessor(report_args, writer.arg(kArgReport));
    writer.run([&](const km_command_t& command) {
        switch (km::handler(command)) {
            case kHandlerLoop:
                if (command.a0 == kCpuWriter) {
                    texture::loop(at, command);
                }
                break;
            case kHandlerMiss:
                if (command.a0 == kCpuWriter) {
                    texture::miss(at, pool, pool_pages, writer.me(), command);
                }
                break;
            case kHandlerColour:
                if (command.a0 == kColourInteger) {
                    if (command.a1 == kCpuWriter) {
                        texture::colour(at, command);
                    }
                } else if (writer.wait_front(kCbPacked, 1)) {
                    const uint32_t* from = texture::words(get_read_ptr(kCbPacked));
                    uint32_t* to = texture::words(at.out);
                    for (uint32_t word = 0; word < kPixels; word++) {
                        to[word] = from[word];
                    }
                    texture::fence();
                    cb_pop_front(kCbPacked, 1);
                }
                break;
            case kHandlerReport:
                // After the commands before it: the core's results and out words.
                if (writer.me() < kReportCores) {
                    noc_async_write(
                        at.results, report.get_noc_addr(writer.me()), kPageBytes
                    );
                    for (uint32_t half = 0; half < 2; half++) {
                        noc_async_write(
                            at.out + half * kPageBytes,
                            report.get_noc_addr(kReportOut + 2 * writer.me() + half),
                            kPageBytes
                        );
                    }
                }
                break;
            default:
                break;
        }
        return true;
    });
}
