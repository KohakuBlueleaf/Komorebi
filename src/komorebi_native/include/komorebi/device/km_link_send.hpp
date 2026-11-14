// A run over islands: the link core's writer sends what its dispatcher staged in the
// link area (KM_HANDLER_LINK) over the fabric into the other island's dispatcher.
#pragma once

#if defined(KM_LINK_CORE) && !defined(COMPILE_FOR_TRISC)
#include <stdint.h>

#include "hostdev/fabric_edm_packet_header.hpp"
#include "km_core.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

namespace km {

    // The session's defines for a program with a link: KM_LINK_CORE the link core's
    // index, KM_LINK_AREA its link area's L1 address, KM_LINK_ARG the index of the
    // connection's runtime arguments of its writer (channel, teardown, cursor).
    class LinkSender {
    public:
        // LINK: the stage's packet, then its count line, one hop into the other
        // island; true once both left the core. Every wait is bounded.
        template <typename Kernel>
        bool send(const Kernel& kernel, const km_command_t& command) {
            if (kernel.args().me != KM_LINK_CORE) {
                return false;
            }
            if (!open_) {
                for (uint32_t w = 0; w < KM_LINK_HEADERS / 4; w++) {
                    words(0)[w] = 0;
                }
                size_t index = KM_LINK_ARG;
                sender_ = tt::tt_fabric::WorkerToFabricEdmSender::
                    build_from_args<ProgrammableCoreType::TENSIX, true>(index);
                sender_.open();
                open_ = true;
            }
            const uint32_t stage = command.a0, number = command.a1;
            const uint32_t x = command.a2 & 0xFFFF, y = command.a2 >> 16;
            if (stage >= KM_LINK_STAGES || (command.a3 | command.a4) % 16 != 0) {
                return false;
            }
            const uint32_t at = KM_LINK_STAGE + stage * KM_LINK_STAGE_BYTES;
            if (!bounded(kernel, [&] {
                return (
                    words(at)[0] == number && words(at + 64)[0] == number &&
                    words(at + KM_LINK_PACKET_BYTES)[1] == number
                );
            })) {
                return false;  // the dispatcher's stores did not land
            }
            return (
                packet(kernel, 0, x, y, command.a3, at, KM_LINK_PACKET_BYTES) &&
                packet(kernel, 1, x, y, command.a4, at + KM_LINK_PACKET_BYTES, 64) &&
                bounded(kernel, sent) && bounded(kernel, flushed)
            );
        }

        // At the writer's end: the connection closed and the router's answer waited
        // for (bounded).
        template <typename Kernel>
        void close(const Kernel& kernel) {
            if (!open_) {
                return;
            }
            open_ = false;
            sender_.close_start();
            if (bounded(kernel, flushed) && bounded(kernel, [&] {
                return *sender_.worker_teardown_addr == 1;
            })) {
                *sender_.worker_teardown_addr = 0;
            }
        }

    private:
        static volatile tt_l1_ptr uint32_t* words(uint32_t offset) {
            return (
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(KM_LINK_AREA + offset)
            );
        }

        static bool sent() {
            return (
                ncrisc_noc_nonposted_writes_sent(noc_index) &&
                ncrisc_noc_nonposted_writes_sent(tt::tt_fabric::get_fabric_worker_noc())
            );
        }

        static bool flushed() {
            return (
                ncrisc_noc_nonposted_writes_flushed(noc_index) &&
                ncrisc_noc_nonposted_writes_flushed(
                    tt::tt_fabric::get_fabric_worker_noc()
                )
            );
        }

        template <typename Kernel, typename Ready>
        static bool bounded(const Kernel& kernel, Ready ready) {
            return kernel.poll(
                [&] {
                    invalidate_l1_cache();
                    return ready();
                },
                kernel.args().wait_limit
            ) == kGaveNone;
        }

        // One packet: `bytes` from the area at `from` to `address` of tile (x, y) of
        // the next island, with header `header` (0 or 1) of the area.
        template <typename Kernel>
        bool packet(
            const Kernel& kernel,
            uint32_t header,
            uint32_t x,
            uint32_t y,
            uint32_t address,
            uint32_t from,
            uint32_t bytes
        ) {
            const uint32_t at = KM_LINK_HEADERS + header * 64;
            for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / 4; w++) {
                words(at)[w] = 0;
            }
            auto* made = (
                reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                    KM_LINK_AREA + at
                )
            );
            made->to_chip_unicast(1);
            made->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{safe_get_noc_addr(
                    x, y, address, tt::tt_fabric::edm_to_local_chip_noc
                )},
                bytes
            );
            if (!bounded(kernel, [&] {
                return sender_.edm_has_space_for_packet<1>();
            })) {
                return false;
            }
            sender_.send_payload_without_header_non_blocking_from_address(
                KM_LINK_AREA + from, bytes
            );
            sender_.send_payload_flush_non_blocking_from_address(
                KM_LINK_AREA + at, sizeof(PACKET_HEADER_TYPE)
            );
            return true;
        }

        tt::tt_fabric::WorkerToFabricEdmSender sender_;
        bool open_ = false;
    };

}  // namespace km
#endif
