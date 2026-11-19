// link_wormhole: Komorebi's link transport between Tensix cores of two chips of a 1 x N
// fabric mesh, both ends kernels/end.cpp (README.md); prints one JSON line.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"

#include "../kernels/wl.h"

namespace {

    namespace distributed = tt::tt_metal::distributed;
    using tt::tt_metal::CoreCoord;
    using Clock = std::chrono::steady_clock;

    const auto kStarted = Clock::now();
    constexpr double kOpenGiveUpS = 200.0;
    constexpr uint32_t kArgs = 17;  // kernels/end.cpp's

    double seconds() {
        return std::chrono::duration<double>(Clock::now() - kStarted).count();
    }

    void log(const std::string& message) {
        std::printf("[%7.2f] %s\n", seconds(), message.c_str());
        std::fflush(stdout);
    }

    struct Options {
        uint32_t chips = 2;  // the mesh is 1 x chips; the ends are its first and last
        uint32_t from = 0;   // the near end's column
        int32_t to = -1;     // the far end's column (-1: the other end of the line)
        uint32_t link = 0;   // the link index toward the neighbour
        uint32_t phases = kPhasePing | kPhaseLine | kPhasePages;
        uint32_t rounds = 1000;
        uint32_t pages = 16384;
        uint32_t page = 4096;
        uint32_t window = 8;
        uint32_t batch = 4;
        uint32_t check = 0;  // 1: the receiver checks every word of every page
        uint32_t router_payload = 0;
        uint32_t base = 0x100000;  // the area's L1 address on both ends
        CoreCoord near_core{0, 0}, far_core{0, 0};  // logical
        double limit_s = 5.0;
        double tick_mhz = 985.5;  // Wormhole's measured wall clock (hazards.md)
        uint32_t seed = 0x5EED1234u;
    };

    CoreCoord core_of(const std::string& text) {
        const auto comma = text.find(',');
        return CoreCoord{
            std::stoul(text.substr(0, comma)), std::stoul(text.substr(comma + 1))
        };
    }

    Options parse(int argc, char** argv) {
        Options o;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--chips") {
                o.chips = std::stoul(value);
            } else if (key == "--from") {
                o.from = std::stoul(value);
            } else if (key == "--to") {
                o.to = std::stoi(value);
            } else if (key == "--link") {
                o.link = std::stoul(value);
            } else if (key == "--phases") {
                o.phases = std::stoul(value);
            } else if (key == "--rounds") {
                o.rounds = std::stoul(value);
            } else if (key == "--pages") {
                o.pages = std::stoul(value);
            } else if (key == "--page") {
                o.page = std::stoul(value);
            } else if (key == "--window") {
                o.window = std::stoul(value);
            } else if (key == "--batch") {
                o.batch = std::stoul(value);
            } else if (key == "--check") {
                o.check = std::stoul(value);
            } else if (key == "--router-payload") {
                o.router_payload = std::stoul(value);
            } else if (key == "--base") {
                o.base = std::stoul(value, nullptr, 0);
            } else if (key == "--near-core") {
                o.near_core = core_of(value);
            } else if (key == "--far-core") {
                o.far_core = core_of(value);
            } else if (key == "--limit-s") {
                o.limit_s = std::stod(value);
            } else if (key == "--tick-mhz") {
                o.tick_mhz = std::stod(value);
            } else {
                throw std::invalid_argument("unknown option " + key);
            }
        }
        return o;
    }

    std::string status_json(const std::vector<uint32_t>& s, double mhz, bool near) {
        const auto us = [mhz](uint64_t ticks) {
            return ticks / mhz;
        };
        const uint64_t span = (
            static_cast<uint64_t>(s[kWordSpanHi]) << 32 | s[kWordSpanLo]
        );
        std::ostringstream out;
        out << "{\"magic\": " << (s[kWordMagic] == kMagic ? "true" : "false")
            << ", \"code\": " << s[kWordCode] << ", \"step\": " << s[kWordStep]
            << ", \"wrong\": " << s[kWordWrong] << ", \"early\": " << s[kWordEarly]
            << ", \"pages\": " << s[kWordPages] << ", \"lines\": " << s[kWordLines]
            << ", \"span_us\": " << us(span);
        if (near) {
            const uint32_t rounds = std::max<uint32_t>(s[kWordRounds], 1);
            const uint32_t lines = std::max<uint32_t>(s[kWordLines], 1);
            out << ", \"rounds\": " << s[kWordRounds] << ", \"ping_us\": ["
                << us(s[kWordPingMin]) << ", " << us(s[kWordPingSum]) / rounds << ", "
                << us(s[kWordPingMax]) << "], \"line_us\": [" << us(s[kWordLineMin])
                << ", " << us(s[kWordLineSum]) / lines << ", " << us(s[kWordLineMax])
                << "]";
        }
        out << "}";
        return out.str();
    }

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& error) {
        log(std::string("options: ") + error.what());
        return 2;
    }
    if (o.to < 0) {
        o.to = o.from == 0 ? o.chips - 1 : 0;
    }
    if (
        o.chips < 2 || o.from >= o.chips || static_cast<uint32_t>(o.to) >= o.chips ||
        static_cast<uint32_t>(o.to) == o.from
    ) {
        log("--chips 2 or more; --from and --to two columns of them");
        return 2;
    }
    if (o.page % 16 != 0 || o.batch == 0 || o.window < o.batch || o.base % 64 != 0) {
        log("page whole 16s, 0 < batch <= window, base 64-aligned");
        return 2;
    }
    std::atomic<bool> opening{true};
    std::thread watch([&] {
        while (opening) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (opening && seconds() > kOpenGiveUpS) {
                log("open not done after 200 s: giving up");
                std::_Exit(3);
            }
        }
    });
    tt::tt_fabric::FabricRouterConfig router;
    if (o.router_payload != 0) {
        router.max_packet_payload_size_bytes = o.router_payload;
    }
    tt::tt_fabric::SetFabricConfig(
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        tt::tt_fabric::FabricTensixConfig::DISABLED,
        tt::tt_fabric::FabricUDMMode::DISABLED,
        tt::tt_fabric::FabricManagerMode::DEFAULT,
        router
    );
    std::shared_ptr<distributed::MeshDevice> mesh;
    try {
        mesh = (
            distributed::MeshDevice::create(
                distributed::MeshDeviceConfig(distributed::MeshShape(1, o.chips))
            )
        );
    } catch (const std::exception& error) {
        opening = false;
        watch.join();
        log(std::string("open failed: ") + error.what());
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        return 1;
    }
    opening = false;
    watch.join();
    log("mesh open, 1 x " + std::to_string(o.chips));
    int rc = 0;
    try {
        const uint32_t n = o.chips;
        const uint32_t far_col = static_cast<uint32_t>(o.to);
        const uint32_t hops = far_col > o.from ? far_col - o.from : o.from - far_col;
        std::vector<tt::tt_metal::IDevice*> chips;
        std::vector<tt::tt_fabric::FabricNodeId> nodes;
        for (uint32_t c = 0; c < n; c++) {
            chips.push_back(mesh->get_device(distributed::MeshCoordinate(0, c)));
            nodes.push_back(
                mesh->get_fabric_node_id(distributed::MeshCoordinate(0, c))
            );
        }
        const uint32_t max_payload = (
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()
        );
        log("fabric max payload " + std::to_string(max_payload) + " B");
        if (o.page > max_payload) {
            throw std::runtime_error("page over the fabric's largest payload");
        }
        const uint32_t cols[2] = {o.from, far_col};
        const CoreCoord cores[2] = {o.near_core, o.far_core};
        std::vector<tt::tt_metal::Program> programs;
        programs.push_back(tt::tt_metal::CreateProgram());
        programs.push_back(tt::tt_metal::CreateProgram());
        const std::string here = KM_LINK_KERNELS;
        for (uint32_t end = 0; end < 2; end++) {
            const uint32_t col = cols[end], other = cols[1 - end];
            const uint32_t next = col < other ? col + 1 : col - 1;  // the first hop
            const auto links = (
                tt::tt_fabric::get_forwarding_link_indices(nodes[col], nodes[next])
            );
            if (o.link >= links.size()) {
                throw std::runtime_error(
                    "link " + std::to_string(o.link) + " not found"
                );
            }
            const auto peer = (
                chips[other]->worker_core_from_logical_core(cores[1 - end])
            );
            std::vector<uint32_t> connection;
            tt::tt_fabric::append_fabric_connection_rt_args(
                nodes[col],
                nodes[next],
                links[o.link],
                programs[end],
                cores[end],
                connection
            );
            const uint32_t channel = connection.at(0);
            const auto eth = (
                tt::tt_fabric::get_forwarding_eth_core(
                    *mesh, nodes[col], nodes[next], links[o.link]
                )
            );
            const auto eth_virtual = chips[col]->ethernet_core_from_logical_core(eth);
            const uint32_t limit = (
                static_cast<uint32_t>(o.limit_s * o.tick_mhz * 1e6 / (1 << 20)) + 1
            );
            std::vector<uint32_t> args = {
                o.base,
                end == 0 ? 1u : 0u,
                channel,
                hops,
                static_cast<uint32_t>(peer.x),
                static_cast<uint32_t>(peer.y),
                o.phases,
                o.rounds,
                o.pages,
                o.page,
                o.window,
                o.batch,
                o.check,
                o.seed,
                limit,
                static_cast<uint32_t>(20000 * o.tick_mhz),
                kArgs
            };
            const auto kernel = tt::tt_metal::CreateKernel(
                programs[end],
                here + "/end.cpp",
                cores[end],
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = {kArgs}
                }
            );
            tt::tt_metal::SetRuntimeArgs(programs[end], kernel, cores[end], args);
            std::vector<uint32_t> zeros(0x100 / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                chips[col], cores[end], o.base + kCounts, zeros
            );
            std::ostringstream text;
            text << (end == 0 ? "near" : "far") << ": column " << col << " (chip "
                 << chips[col]->id() << ") core (" << cores[end].x << ","
                 << cores[end].y << "), link " << links[o.link] << " channel "
                 << channel << " eth (" << eth_virtual.x << "," << eth_virtual.y
                 << "), " << hops << " hop(s) to core virtual (" << peer.x << ","
                 << peer.y << ")";
            log(text.str());
        }
        for (uint32_t end = 0; end < 2; end++) {
            programs[end].impl().compile(chips[cols[end]]);
        }
        log("built");
        distributed::MeshWorkload workload;
        for (uint32_t end = 0; end < 2; end++) {
            workload.add_program(
                distributed::MeshCoordinateRange(
                    distributed::MeshCoordinate(0, cols[end])
                ),
                std::move(programs[end])
            );
        }
        auto& queue = mesh->mesh_command_queue();
        distributed::EnqueueMeshWorkload(queue, workload, false);
        distributed::Finish(queue);
        log("ended");
        std::vector<uint32_t> status[2];
        bool ok = true;
        for (uint32_t end = 0; end < 2; end++) {
            tt::tt_metal::detail::ReadFromDeviceL1(
                chips[cols[end]],
                cores[end],
                o.base + kStatus,
                4 * kStatusWords,
                status[end]
            );
            const auto& s = status[end];
            ok &= (
                s[kWordMagic] == kMagic && s[kWordCode] == kCodeOk &&
                s[kWordWrong] == 0 && s[kWordEarly] == 0
            );
        }
        const auto& near = status[0];
        const auto& far = status[1];
        if (o.phases & kPhasePing) {
            ok &= near[kWordRounds] == o.rounds;
        }
        if (o.phases & kPhaseLine) {
            ok &= near[kWordLines] == o.rounds && far[kWordLines] == o.rounds;
        }
        if (o.phases & kPhasePages) {
            ok &= near[kWordPages] == o.pages && far[kWordPages] == o.pages;
        }
        const uint64_t span = (
            static_cast<uint64_t>(far[kWordSpanHi]) << 32 | far[kWordSpanLo]
        );
        const double rx_gbs = (
            span && o.pages > o.batch
                ? static_cast<double>(o.pages - o.batch) * o.page /
                    (span / o.tick_mhz) / 1e3
                : 0.0
        );
        std::ostringstream report;
        report << "{\"options\": {\"chips\": " << o.chips << ", \"from\": " << o.from
               << ", \"to\": " << far_col << ", \"hops\": " << hops
               << ", \"link\": " << o.link << ", \"phases\": " << o.phases
               << ", \"rounds\": " << o.rounds << ", \"pages\": " << o.pages
               << ", \"page\": " << o.page << ", \"window\": " << o.window
               << ", \"batch\": " << o.batch << ", \"check\": " << o.check
               << ", \"router_payload\": " << o.router_payload
               << ", \"tick_mhz\": " << o.tick_mhz
               << "}, \"near\": " << status_json(near, o.tick_mhz, true)
               << ", \"far\": " << status_json(far, o.tick_mhz, false)
               << ", \"pages_rx_gbs\": " << rx_gbs
               << ", \"ok\": " << (ok ? "true" : "false") << "}";
        std::printf("%s\n", report.str().c_str());
        std::fflush(stdout);
        rc = ok ? 0 : 1;
    } catch (const std::exception& error) {
        log(std::string("failed: ") + error.what());
        rc = 1;
    }
    log("closing the mesh");
    mesh->close();
    mesh.reset();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    log("mesh closed");
    return rc;
}
