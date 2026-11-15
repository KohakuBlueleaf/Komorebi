// link_packet: cores of one card write packets into the L1 of cores of another over
// tt-metal's 1-D fabric in slow dispatch, then print every core's status (README.md).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"

#include "../kernels/link.h"

namespace {

    namespace distributed = tt::tt_metal::distributed;
    using tt::tt_metal::CoreCoord;
    using Clock = std::chrono::steady_clock;

    const auto kStarted = Clock::now();
    constexpr double kOpenGiveUpS = 200.0;  // no open by then: exit before the watcher's limit
    constexpr uint32_t kStatusPages = 64;  // one page a core
    constexpr uint32_t kPageBytes = kStatusBytes + kCopyBytes;

    double seconds() {
        return std::chrono::duration<double>(Clock::now() - kStarted).count();
    }

    void log(const std::string& message) {
        std::printf("[%7.2f] %s\n", seconds(), message.c_str());
        std::fflush(stdout);
    }

    uint32_t aiclk(int card) {
        std::ifstream file(
            "/sys/class/tenstorrent/tenstorrent!" + std::to_string(card) + "/tt_aiclk"
        );
        uint32_t mhz = 0;
        file >> mhz;
        return mhz;
    }

    struct Options {
        std::string from = "0";
        int to = -1;  // the peer's column; -1: the next column, or the previous for the
                      // last
        uint32_t cards = 2;  // the mesh is 1 x cards
        uint32_t hops = 1;
        uint32_t first_link = 0;  // the first of the --links links used
        double delay_us = 0;  // the senders' wait after the open, before the first send
        uint32_t packets = 1;
        uint32_t payload = 64;
        uint32_t window = 0;
        uint32_t links = 1;
        uint32_t senders = 1;
        uint32_t seed = 0x1234567u;
        double limit_s = 10.0;
    };

    Options parse(int argc, char** argv) {
        Options o;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--from") {
                o.from = value;
            } else if (key == "--cards") {
                o.cards = std::stoul(value);
            } else if (key == "--to") {
                o.to = std::stoi(value);
            } else if (key == "--first-link") {
                o.first_link = std::stoul(value);
            } else if (key == "--delay-us") {
                o.delay_us = std::stod(value);
            } else if (key == "--hops") {
                o.hops = std::stoul(value);
            } else if (key == "--packets") {
                o.packets = std::stoul(value);
            } else if (key == "--payload") {
                o.payload = std::stoul(value);
            } else if (key == "--window") {
                o.window = std::stoul(value);
            } else if (key == "--links") {
                o.links = std::stoul(value);
            } else if (key == "--senders") {
                o.senders = std::stoul(value);
            } else if (key == "--limit-s") {
                o.limit_s = std::stod(value);
            } else if (key == "--seed") {
                o.seed = std::stoul(value, nullptr, 0);
            } else {
                throw std::invalid_argument("unknown option " + key);
            }
        }
        if (o.window == 0) {
            o.window = o.packets;
        }
        return o;
    }

    // One sender and its receiver on the peer chip.
    struct Pair {
        uint32_t from, to, link, index;  // the mesh columns, the link, the status page
        CoreCoord sender, receiver;      // logical
    };

    // The logical Tensix core in rows 0 and 1 nearest at or right of the Ethernet
    // core's x (NoC1 reaches the Ethernet row from there), not yet taken.
    CoreCoord under(
        tt::tt_metal::IDevice* device,
        CoreCoord eth,
        CoreCoord grid,
        std::set<CoreCoord>& used
    ) {
        const int64_t eth_x = device->ethernet_core_from_logical_core(eth).x;
        CoreCoord best{0, 0};
        int64_t best_cost = -1;
        for (uint32_t y = 0; y < 2; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord core{x, y};
                if (used.contains(core)) {
                    continue;
                }
                const int64_t wx = device->worker_core_from_logical_core(core).x;
                const int64_t cost = (
                    (wx >= eth_x ? wx - eth_x : 1000 + eth_x - wx) + 10000 * y
                );
                if (best_cost < 0 || cost < best_cost) {
                    best_cost = cost;
                    best = core;
                }
            }
        }
        used.insert(best);
        return best;
    }

    std::string status_json(const std::vector<uint32_t>& words, uint32_t mhz) {
        const uint64_t cycles = (
            words[kWordCyclesLo] | static_cast<uint64_t>(words[kWordCyclesHi]) << 32
        );
        std::ostringstream out;
        out << "{\"magic\": " << (words[kWordMagic] == kMagic ? "true" : "false")
            << ", \"code\": " << words[kWordCode]
            << ", \"packets\": " << words[kWordPackets]
            << ", \"wrong\": " << words[kWordWrong] << ", \"cycles\": " << cycles
            << ", \"us\": " << (mhz ? cycles / static_cast<double>(mhz) : 0.0)
            << ", \"first_kcycles\": " << words[kWordFirst] << "}";
        return out.str();
    }

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.payload == 0 || o.payload % 16 != 0) {
        log("payload must be a multiple of 16");
        return 2;
    }
    const char* beat_path = std::getenv("KTT_HEARTBEAT");
    std::atomic<bool> opening{true};
    std::thread heartbeat([&] {
        double reported = 0, beaten = 0;
        while (opening) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (!opening || seconds() - beaten < 5) {
                continue;
            }
            beaten = seconds();
            if (beat_path) {
                std::ofstream(beat_path, std::ios::app)
                    .close();  // created once, then touched
                ::utimensat(AT_FDCWD, beat_path, nullptr, 0);
            }
            if (seconds() - reported >= 20) {
                reported = seconds();
                log("open in progress");
            }
            if (seconds() > kOpenGiveUpS) {
                log("open not done after 200 s: giving up before the watcher's limit");
                std::_Exit(3);
            }
        }
    });

    log(
        "fabric 1-D, opening the 1 x " + std::to_string(o.cards) +
        " mesh (slow dispatch: " +
        std::string(std::getenv("TT_METAL_SLOW_DISPATCH_MODE") ? "yes" : "no") + ")"
    );
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
    std::shared_ptr<distributed::MeshDevice> mesh;
    try {
        mesh = (
            distributed::MeshDevice::create(
                distributed::MeshDeviceConfig(distributed::MeshShape(1, o.cards))
            )
        );
    } catch (const std::exception& error) {
        opening = false;
        heartbeat.join();
        log(std::string("open failed: ") + error.what());
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        return 1;
    }
    opening = false;
    heartbeat.join();
    log("mesh open");

    int rc = 0;
    try {
        const uint32_t n = o.cards;
        std::vector<tt::tt_metal::IDevice*> chips;
        std::vector<tt::tt_fabric::FabricNodeId> nodes;
        std::vector<uint32_t> mhz;
        for (uint32_t c = 0; c < n; c++) {
            chips.push_back(mesh->get_device(distributed::MeshCoordinate(0, c)));
            nodes.push_back(
                mesh->get_fabric_node_id(distributed::MeshCoordinate(0, c))
            );
            mhz.push_back(aiclk(chips[c]->id()));
        }
        auto peer = [&](uint32_t from) {
            return (
                o.to >= 0
                    ? static_cast<uint32_t>(o.to)
                    : (from + 1 < n ? from + 1 : from - 1)
            );
        };
        const uint32_t max_payload = (
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()
        );
        log("fabric max payload " + std::to_string(max_payload) + " B");
        if (o.payload > max_payload) {
            throw std::runtime_error("payload over the fabric's maximum");
        }
        const CoreCoord grid = mesh->compute_with_storage_grid_size();
        std::ostringstream links_text;
        for (uint32_t c = 0; c < n; c++) {
            const auto links = (
                tt::tt_fabric::get_forwarding_link_indices(nodes[c], nodes[peer(c)])
            );
            links_text << "column " << c << " = chip " << chips[c]->id() << ": "
                       << links.size() << " links (eth cores";
            for (const auto link : links) {
                const auto eth = (
                    tt::tt_fabric::get_forwarding_eth_core(
                        *mesh, nodes[c], nodes[peer(c)], link
                    )
                );
                const auto virt = chips[c]->ethernet_core_from_logical_core(eth);
                links_text << " " << link << ":(" << virt.x << "," << virt.y << ")";
            }
            links_text << "); ";
        }
        log(links_text.str());

        std::vector<uint32_t> froms;
        if (o.from == "both") {
            froms = {0, peer(0)};
        } else {
            froms = {static_cast<uint32_t>(std::stoul(o.from))};
        }
        std::vector<std::set<CoreCoord>> used(n);
        std::vector<Pair> pairs;
        std::vector<uint32_t> receivers_on(n, 0);
        for (const uint32_t from : froms) {
            const uint32_t to = (
                o.from == "both" ? froms[from == froms[0] ? 1 : 0] : peer(from)
            );
            const auto links = (
                tt::tt_fabric::get_forwarding_link_indices(nodes[from], nodes[to])
            );
            if (o.first_link + o.links > links.size()) {
                throw std::runtime_error(
                    "asked for links " + std::to_string(o.first_link) + " + " +
                    std::to_string(o.links) + ", " + std::to_string(links.size()) +
                    " found"
                );
            }
            for (uint32_t l = o.first_link; l < o.first_link + o.links; l++) {
                const auto eth = (
                    tt::tt_fabric::get_forwarding_eth_core(
                        *mesh, nodes[from], nodes[to], links[l]
                    )
                );
                for (uint32_t s = 0; s < o.senders; s++) {
                    Pair pair{
                        from, to, links[l], static_cast<uint32_t>(pairs.size()), {}, {}
                    };
                    pair.sender = under(chips[from], eth, grid, used[from]);
                    const uint32_t r = receivers_on[to]++;
                    pair.receiver = CoreCoord{r % grid.x, grid.y - 1 - r / grid.x};
                    used[to].insert(pair.receiver);
                    pairs.push_back(pair);
                }
            }
        }

        // Every core's L1: status at base, the counter at base + 64, the slots from
        // base + 128.
        const uint32_t base = (
            chips[0]->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)
        );
        const uint32_t status_l1 = (base + 63) / 64 * 64;
        const uint32_t counter_l1 = status_l1 + 64;
        const uint32_t slots_l1 = status_l1 + 128;
        const uint64_t slot_bytes = static_cast<uint64_t>(o.window) * o.payload;
        if (slots_l1 + slot_bytes > (1536u << 10) - 4096) {
            throw std::runtime_error("the window does not fit in L1");
        }
        const uint32_t limit_mcycles = (
            static_cast<uint32_t>(o.limit_s * 1350.0 * 1e6 / (1 << 20)) + 1
        );

        distributed::DeviceLocalBufferConfig local{
            .page_size = kPageBytes, .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        auto status_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = kStatusPages * kPageBytes},
            local,
            mesh.get()
        );
        auto& queue = mesh->mesh_command_queue();
        std::vector<uint32_t> zeros(kStatusPages * kPageBytes / 4, 0);
        for (uint32_t c = 0; c < n; c++) {
            distributed::WriteShard(
                queue, status_buffer, zeros, distributed::MeshCoordinate(0, c), true
            );
        }
        // The receivers' counters are zeroed before the launch: no packet can come
        // first.
        for (const auto& pair : pairs) {
            std::vector<uint32_t> counter(4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                chips[pair.to], pair.receiver, counter_l1, counter
            );
        }

        const std::string here = KM_LINK_KERNELS;
        std::vector<tt::tt_metal::Program> programs;
        for (uint32_t c = 0; c < n; c++) {
            programs.push_back(tt::tt_metal::CreateProgram());
        }
        for (const auto& pair : pairs) {
            const uint32_t to = pair.to;
            const auto rx = chips[to]->worker_core_from_logical_core(pair.receiver);
            std::vector<uint32_t> sender_args = {
                static_cast<uint32_t>(status_buffer->address()),
                pair.index,
                kPageBytes,
                status_l1,
                slots_l1,
                counter_l1,
                static_cast<uint32_t>(rx.x),
                static_cast<uint32_t>(rx.y),
                o.packets,
                o.payload,
                o.window,
                o.seed,
                pair.index,
                limit_mcycles,
                o.hops,
                static_cast<uint32_t>(o.delay_us * 1350.0)
            };
            const auto sender = tt::tt_metal::CreateKernel(
                programs[pair.from],
                here + "/sender.cpp",
                pair.sender,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = {static_cast<uint32_t>(sender_args.size())}
                }
            );
            tt::tt_fabric::append_fabric_connection_rt_args(
                nodes[pair.from],
                nodes[to],
                pair.link,
                programs[pair.from],
                pair.sender,
                sender_args
            );
            tt::tt_metal::SetRuntimeArgs(
                programs[pair.from], sender, pair.sender, sender_args
            );
            std::vector<uint32_t> receiver_args = {
                static_cast<uint32_t>(status_buffer->address()),
                kStatusPages / 2 + pair.index,
                kPageBytes,
                status_l1,
                slots_l1,
                counter_l1,
                o.packets,
                o.payload,
                o.window,
                o.seed,
                pair.index,
                limit_mcycles
            };
            const auto receiver = tt::tt_metal::CreateKernel(
                programs[to],
                here + "/receiver.cpp",
                pair.receiver,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = {static_cast<uint32_t>(receiver_args.size())}
                }
            );
            tt::tt_metal::SetRuntimeArgs(
                programs[to], receiver, pair.receiver, receiver_args
            );
            std::ostringstream text;
            text << "pair " << pair.index << ": chip column " << pair.from << " link "
                 << pair.link << " sender (" << pair.sender.x << "," << pair.sender.y
                 << ") -> column " << to << " receiver (" << pair.receiver.x << ","
                 << pair.receiver.y << ") virtual (" << rx.x << "," << rx.y << ")";
            log(text.str());
        }
        // Columns with kernels; the receivers' columns are launched first.
        std::vector<uint32_t> order;
        for (const auto& pair : pairs) {
            if (std::find(order.begin(), order.end(), pair.to) == order.end()) {
                order.push_back(pair.to);
            }
        }
        for (const auto& pair : pairs) {
            if (std::find(order.begin(), order.end(), pair.from) == order.end()) {
                order.push_back(pair.from);
            }
        }
        for (const uint32_t c : order) {
            programs[c].impl().compile(chips[c]);
        }
        log("every program built");
        // One workload: in slow dispatch a second enqueue waits for the first's cores
        // to be idle, so a sender launched apart would start after its receiver ended.
        distributed::MeshWorkload workload;
        for (const uint32_t c : order) {
            workload.add_program(
                distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, c)),
                std::move(programs[c])
            );
        }
        log("launching every column");
        distributed::EnqueueMeshWorkload(queue, workload, false);
        distributed::Finish(queue);
        log("every program ended");

        std::vector<std::vector<uint32_t>> pages(n);
        for (const uint32_t c : order) {
            distributed::ReadShard(
                queue, pages[c], status_buffer, distributed::MeshCoordinate(0, c), true
            );
        }
        std::ostringstream clocks;
        for (uint32_t c = 0; c < n; c++) {
            mhz[c] = std::max(mhz[c], aiclk(chips[c]->id()));
            clocks << (c ? ", " : "") << mhz[c];
        }
        bool ok = true;
        uint64_t bytes = 0;
        std::ostringstream report;
        report << "{\"options\": {\"from\": \"" << o.from
               << "\", \"packets\": " << o.packets << ", \"payload\": " << o.payload
               << ", \"window\": " << o.window << ", \"links\": " << o.links
               << ", \"senders\": " << o.senders << ", \"hops\": " << o.hops
               << ", \"first_link\": " << o.first_link << "}, \"aiclk_mhz\": ["
               << clocks.str() << "], \"pairs\": [";
        for (const auto& pair : pairs) {
            const uint32_t to = pair.to;
            const uint32_t words = kPageBytes / 4;
            std::vector<uint32_t> sent(
                pages[pair.from].begin() + pair.index * words,
                pages[pair.from].begin() + (pair.index + 1) * words
            );
            const uint32_t rpage = kStatusPages / 2 + pair.index;
            std::vector<uint32_t> got(
                pages[to].begin() + rpage * words,
                pages[to].begin() + (rpage + 1) * words
            );
            // The host's own check of the receiver's copy of its first slots.
            const uint32_t filled = std::min(o.packets, o.window);
            const uint32_t copied = std::min(filled * o.payload, kCopyBytes) / 4;
            uint32_t host_wrong = 0;
            for (uint32_t w = 0; w < copied; w++) {
                const uint32_t slot = w / (o.payload / 4), word = w % (o.payload / 4);
                host_wrong += (
                    got[kStatusBytes / 4 + w] != pattern(o.seed, pair.index, slot, word)
                );
            }
            const bool pair_ok = (
                sent[kWordMagic] == kMagic && got[kWordMagic] == kMagic &&
                sent[kWordCode] == kCodeOk && got[kWordCode] == kCodeOk &&
                got[kWordCount] == o.packets && got[kWordWrong] == 0 && host_wrong == 0
            );
            ok &= pair_ok;
            bytes += static_cast<uint64_t>(got[kWordCount]) * o.payload;
            report << (pair.index ? ", " : "") << "{\"from\": " << pair.from
                   << ", \"to\": " << to << ", \"link\": " << pair.link
                   << ", \"ok\": " << (pair_ok ? "true" : "false")
                   << ", \"host_checked_words\": " << copied
                   << ", \"host_wrong\": " << host_wrong
                   << ", \"sender\": " << status_json(sent, mhz[pair.from])
                   << ", \"receiver\": " << status_json(got, mhz[to]) << "}";
        }
        report << "], \"bytes\": " << bytes << ", \"ok\": " << (ok ? "true" : "false")
               << "}";
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
