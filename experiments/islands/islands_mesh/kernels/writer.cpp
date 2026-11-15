// Islands experiment, writer: the link core holds a fabric connection across commands
// (OPEN, SEND, CLOSE); COPY reads a core's slot back to DRAM. Every wait is bounded.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "islands.h"
#include "komorebi/device/km_worker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt::tt_fabric;

// Writer arguments: the Ethernet channel (kNoLink: none), the teardown and cursor
// addresses (the connection's), the row's address.
constexpr uint32_t kArgChannel = 0;
constexpr uint32_t kArgRow = 3;

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    km::Writer writer;
    const bool linked = writer.arg(kArgChannel) != kNoLink;
    const uint32_t row = writer.arg(kArgRow);
    WorkerToFabricEdmSender sender;
    if (linked) {
        size_t index = km::kArgs + kArgChannel;
        sender = WorkerToFabricEdmSender::
            build_from_args<ProgrammableCoreType::TENSIX, true>(index);
    }
    bool open = false;
    auto words = [&](uint32_t offset) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + offset);
    };
    auto bounded = [&](auto ready) {
        return writer.poll(
            [&] {
                invalidate_l1_cache();
                return ready();
            },
            writer.args().wait_limit
        ) == km::kGaveNone;
    };
    auto sent = [] {
        return (
            ncrisc_noc_nonposted_writes_sent(noc_index) &&
            ncrisc_noc_nonposted_writes_sent(get_fabric_worker_noc())
        );
    };
    auto flushed = [] {
        return (
            ncrisc_noc_nonposted_writes_flushed(noc_index) &&
            ncrisc_noc_nonposted_writes_flushed(get_fabric_worker_noc())
        );
    };
    writer.run([&](const km_command_t& command) {
        switch (km::handler(command)) {
            case kHandlerOpen: {
                if (!linked || open) {
                    return false;
                }
                for (uint32_t w = 0; w < kRowHeader / 4; w++) {
                    words(0)[w] = 0;
                }
                sender.open();
                open = true;
                return true;
            }
            case kHandlerSend: {
                const uint32_t x = command.a0;
                const uint32_t y = command.a1;
                const uint32_t address = command.a2;
                const uint32_t seed = command.a3;
                if (!open || address % 16 != 0) {
                    return false;
                }
                for (uint32_t w = 0; w < kSlotBytes / 4; w++) {
                    words(kRowPayload)[w] = slot_word(seed, w);
                }
                auto* header = (
                    reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                        row + kRowHeader
                    )
                );
                for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                    words(kRowHeader)[w] = 0;
                }
                header->to_chip_unicast(1);
                header->to_noc_unicast_write(
                    NocUnicastCommandHeader{
                        safe_get_noc_addr(x, y, address, edm_to_local_chip_noc)
                    },
                    kSlotBytes
                );
                if (!bounded([&] {
                    return sender.edm_has_space_for_packet<1>();
                })) {
                    return false;
                }
                sender.send_payload_without_header_non_blocking_from_address(
                    row + kRowPayload, kSlotBytes
                );
                sender.send_payload_flush_non_blocking_from_address(
                    row + kRowHeader, sizeof(PACKET_HEADER_TYPE)
                );
                return bounded(sent) && bounded(flushed);
            }
            case kHandlerCopy: {
                const uint32_t out_address = command.a0;
                const uint32_t page = command.a1;
                const uint32_t seed = command.a3;
                if (!bounded([&] {
                    return words(kRowSlot)[0] == seed;
                })) {
                    return false;
                }
                const auto out = TensorAccessor(out_args, out_address);
                noc_async_write_page(page, out, row + kRowSlot);
                return bounded([] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                });
            }
            case kHandlerClose: {
                if (!open) {
                    return false;
                }
                open = false;
                sender.close_start();
                if (!bounded(flushed) || !bounded([&] {
                    return *sender.worker_teardown_addr == 1;
                })) {
                    return false;
                }
                *sender.worker_teardown_addr = 0;
                return true;
            }
            default:
                return true;
        }
    });
}
