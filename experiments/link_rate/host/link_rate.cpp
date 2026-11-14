// link_rate: cores of one card stream packets over tt-metal's 1-D fabric into the other
// card in the shapes README.md compares; it prints every core's status and the checks.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"

#include <llrt/tt_cluster.hpp>

#include "../kernels/link.h"

namespace {

    namespace distributed = tt::tt_metal::distributed;
    using tt::tt_metal::CoreCoord;
    using Clock = std::chrono::steady_clock;

    const auto kStarted = Clock::now();
    constexpr double kOpenGiveUpS = 200.0;  // no open by then: exit before the watcher's limit
    constexpr uint32_t kStatusPages = 64;  // one page a core, the receivers' from 32
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
        std::string from = "0";  // a mesh column or "both"
        uint32_t cards = 2;      // the mesh is 1 x cards
        uint32_t hops = 1;
        uint32_t first_link = 0;  // the first of the --links links used
        uint32_t links = 1;
        double delay_us = 20000;  // the senders' wait after the open
        uint32_t packets = 1024;
        uint32_t payload = 4352;
        uint32_t window = 64;
        uint32_t fused = 0;       // 1: every packet is a fused write and increment
        uint32_t batch = 64;      // plain mode: packets a count
        uint32_t wait_every = 0;  // the sender waits for its sends every this many
        uint32_t space_every = 1;  // packets a check for free slots in the router (1 or 8)
        std::string place = "spread";  // under, row or spread
        std::string rx_place = "top";  // the receiving cores: top rows, or as place
        uint32_t noc = 0;              // the sender kernels' NoC
        std::string target = "l1";     // l1 (a receiver kernel), bare or dram
        uint32_t dual = 0;  // 1: links pair up on one core, one link a data-movement CPU
        uint32_t send = 0;  // the sender's way a packet leaves (kernels/sender.cpp)
        uint32_t router_payload = 0;  // the routers' packet payload (0: the fabric's)
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
            } else if (key == "--first-link") {
                o.first_link = std::stoul(value);
            } else if (key == "--links") {
                o.links = std::stoul(value);
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
            } else if (key == "--fused") {
                o.fused = std::stoul(value);
            } else if (key == "--batch") {
                o.batch = std::stoul(value);
            } else if (key == "--wait-every") {
                o.wait_every = std::stoul(value);
            } else if (key == "--space-every") {
                o.space_every = std::stoul(value);
            } else if (key == "--place") {
                o.place = value;
            } else if (key == "--rx-place") {
                o.rx_place = value;
            } else if (key == "--noc") {
                o.noc = std::stoul(value);
            } else if (key == "--target") {
                o.target = value;
            } else if (key == "--dual") {
                o.dual = std::stoul(value);
            } else if (key == "--send") {
                o.send = std::stoul(value);
            } else if (key == "--router-payload") {
                o.router_payload = std::stoul(value);
            } else if (key == "--limit-s") {
                o.limit_s = std::stod(value);
            } else if (key == "--seed") {
                o.seed = std::stoul(value, nullptr, 0);
            } else {
                throw std::invalid_argument("unknown option " + key);
            }
        }
        return o;
    }

    // One sender and its destination on the peer chip.
    struct Pair {
        uint32_t from, to, link, index;  // the mesh columns, the link, the status page
        CoreCoord sender, receiver;      // logical; the receiver is a core of `to`
        bool second;                     // dual: the second link of its sender core
        CoreCoord eth;                   // logical: the link's Ethernet core on `from`
    };

    // `--place at:x,y;x,y;...`: the sending cores (logical) in pair order.
    std::vector<CoreCoord> places_at(const std::string& place) {
        std::vector<CoreCoord> out;
        std::stringstream list(place.substr(3));
        std::string item;
        while (std::getline(list, item, ';')) {
            const auto comma = item.find(',');
            out.push_back(
                CoreCoord{
                    std::stoul(item.substr(0, comma)),
                    std::stoul(item.substr(comma + 1))
                }
            );
        }
        return out;
    }

    // A core's NoC0 coordinates on `chip` from its translated ones.
    std::pair<uint32_t, uint32_t> noc0_of(
        tt::tt_metal::IDevice* chip, CoreCoord translated, tt::CoreType type
    ) {
        const auto& soc = (
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                chip->id()
            )
        );
        const auto core = (
            soc.translate_coord_to(
                tt::umd::CoreCoord(
                    translated.x, translated.y, type, tt::CoordSystem::TRANSLATED
                ),
                tt::CoordSystem::NOC0
            )
        );
        return {static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
    }

    // The logical Tensix core in rows 0 and 1 nearest at or right of the Ethernet
    // core's x, not yet taken.
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

    // The next core of a placement that is not taken: along row 0, or spread over
    // rows and columns.
    CoreCoord next_core(
        const std::string& place, CoreCoord grid, std::set<CoreCoord>& used
    ) {
        for (uint32_t k = 0; k < grid.x * grid.y * 4; k++) {
            const CoreCoord core = place == "row"
                ? CoreCoord{k % grid.x, k / grid.x}
                : CoreCoord{(3 * k) % grid.x, (2 * k + 1) % grid.y};
            if (!used.contains(core)) {
                used.insert(core);
                return core;
            }
        }
        throw std::runtime_error("no free core");
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
            << ", \"wait_cycles\": "
            << (words[kWordWaitLo] | static_cast<uint64_t>(words[kWordWaitHi]) << 32)
            << ", \"router_slots\": " << words[kWordSlots]
            << ", \"first_kcycles\": " << words[kWordFirst] << "}";
        return out.str();
    }

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.payload == 0 || o.payload % 16 != 0 || o.window == 0 || o.batch == 0) {
        log("payload must be a multiple of 16; window and batch at least 1");
        return 2;
    }
    if (o.send > 3 || (o.send >= 2 && o.dual)) {
        log("send is 0 to 3; 2 and 3 (header and payload joined) without --dual");
        return 2;
    }
    const bool kernel_receiver = o.target == "l1";
    const bool dram = o.target == "dram";
    if (!kernel_receiver && !dram && o.target != "bare") {
        log("target is l1, bare or dram");
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
    tt::tt_fabric::FabricRouterConfig router;
    if (o.router_payload != 0) {
        router.max_packet_payload_size_bytes = o.router_payload;
        log(
            "routers told: packets of " + std::to_string(o.router_payload) +
            " B payload"
        );
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
            return from + 1 < n ? from + 1 : from - 1;
        };
        const uint32_t max_payload = (
            tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()
        );
        log("fabric max payload " + std::to_string(max_payload) + " B");
        if (o.payload > max_payload) {
            throw std::runtime_error("payload over the fabric's maximum");
        }
        const CoreCoord grid = mesh->compute_with_storage_grid_size();
        log("grid " + std::to_string(grid.x) + " x " + std::to_string(grid.y));
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
        std::vector<uint32_t> at_taken(n, 0);  // --place at: the cores given out
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
                Pair pair{
                    from,
                    to,
                    links[l],
                    static_cast<uint32_t>(pairs.size()),
                    {},
                    {},
                    false,
                    eth
                };
                if (o.dual && (l - o.first_link) % 2 == 1) {
                    pair.sender = pairs.back().sender;  // the first link's core
                    pair.second = true;
                } else if (o.place.rfind("at:", 0) == 0) {
                    const auto at = places_at(o.place);
                    const uint32_t k = at_taken[from]++;
                    if (k >= at.size()) {
                        throw std::runtime_error("--place at: names too few cores");
                    }
                    pair.sender = at[k];
                    used[from].insert(at[k]);
                } else if (o.place == "under") {
                    pair.sender = under(chips[from], eth, grid, used[from]);
                } else {
                    pair.sender = next_core(o.place, grid, used[from]);
                }
                if (o.rx_place == "top") {
                    const uint32_t r = receivers_on[to]++;
                    pair.receiver = CoreCoord{r % grid.x, grid.y - 1 - r / grid.x};
                } else {
                    pair.receiver = next_core(o.place, grid, used[to]);
                }
                used[to].insert(pair.receiver);
                pairs.push_back(pair);
            }
        }

        // Every core's L1: status at base, the counter at base + 64, the packet
        // headers from base + 128, then the slots.
        const uint32_t base = (
            chips[0]->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)
        );
        const uint32_t status_l1 = (base + 63) / 64 * 64;
        const uint32_t counter_l1 = status_l1 + 64;
        const uint32_t headers_l1 = status_l1 + 128;
        const uint32_t slots_l1 = (
            (headers_l1 + 2 * o.window * kHeaderStride + 63) / 64 * 64
        );
        const uint64_t slot_bytes = static_cast<uint64_t>(o.window) * o.payload;
        // The second CPU of a dual core keeps its status, headers and slots apart.
        const uint32_t second_shift = (
            (slots_l1 - status_l1 + slot_bytes + 63) / 64 * 64
        );
        // send 2 and 3: the sender's packets (header and payload) from headers_l1
        const uint64_t joined_end = (
            headers_l1 + 2ull * o.window * (kHeaderStride + o.payload)
        );
        if (
            slots_l1 + (o.dual ? second_shift : 0) + slot_bytes >
                (1536u << 10) - 4096 ||
            (o.send >= 2 && joined_end > (1536u << 10) - 4096)
        ) {
            throw std::runtime_error("the window does not fit in L1");
        }
        const uint32_t limit_mcycles = (
            static_cast<uint32_t>(o.limit_s * 1350.0 * 1e6 / (1 << 20)) + 1
        );
        // Packets counted at the receiver: all, or one of each batch and the last.
        const uint32_t expect = (
            o.fused ? o.packets : (o.packets + o.batch - 1) / o.batch
        );

        distributed::DeviceLocalBufferConfig local{
            .page_size = kPageBytes, .buffer_type = tt::tt_metal::BufferType::DRAM
        };
        auto status_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = kStatusPages * kPageBytes},
            local,
            mesh.get()
        );
        // The DRAM target: one page, so one bank, a region of slots for each pair.
        std::shared_ptr<distributed::MeshBuffer> data_buffer;
        uint64_t dram_bank_address = 0;
        CoreCoord dram_core{0, 0};
        if (dram) {
            const uint64_t bytes = pairs.size() * slot_bytes;
            distributed::DeviceLocalBufferConfig data_local{
                .page_size = bytes, .buffer_type = tt::tt_metal::BufferType::DRAM
            };
            data_buffer = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = bytes},
                data_local,
                mesh.get()
            );
            const uint32_t to = pairs[0].to;
            dram_bank_address = (
                data_buffer->address() +
                chips[to]->allocator()->get_bank_offset(
                    tt::tt_metal::BufferType::DRAM, 0
                )
            );
            dram_core = (
                chips[to]->virtual_core_from_logical_core(
                    chips[to]->logical_core_from_dram_channel(0), tt::CoreType::DRAM
                )
            );
            log(
                "dram bank 0 of column " + std::to_string(to) + ": core (" +
                std::to_string(dram_core.x) + "," + std::to_string(dram_core.y) +
                ") address " + std::to_string(dram_bank_address)
            );
        }
        auto& queue = mesh->mesh_command_queue();
        std::vector<uint32_t> zeros(kStatusPages * kPageBytes / 4, 0);
        for (uint32_t c = 0; c < n; c++) {
            distributed::WriteShard(
                queue, status_buffer, zeros, distributed::MeshCoordinate(0, c), true
            );
            if (dram) {
                std::vector<uint32_t> empty(data_buffer->size() / 4, 0);
                distributed::WriteShard(
                    queue, data_buffer, empty, distributed::MeshCoordinate(0, c), true
                );
            }
        }
        // The receivers' counters and slots are zeroed before the launch: no packet
        // can come first, and no slot is left from an earlier run.
        for (const auto& pair : pairs) {
            std::vector<uint32_t> counter(4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(
                chips[pair.to], pair.receiver, counter_l1, counter
            );
            if (!dram) {
                std::vector<uint32_t> empty(slot_bytes / 4, 0);
                tt::tt_metal::detail::WriteToDeviceL1(
                    chips[pair.to], pair.receiver, slots_l1, empty
                );
            }
        }

        const std::string here = KM_LINK_KERNELS;
        std::vector<tt::tt_metal::Program> programs;
        for (uint32_t c = 0; c < n; c++) {
            programs.push_back(tt::tt_metal::CreateProgram());
        }
        for (const auto& pair : pairs) {
            const uint32_t to = pair.to;
            const auto rx = chips[to]->worker_core_from_logical_core(pair.receiver);
            const uint32_t dst_x = dram ? dram_core.x : rx.x;
            const uint32_t dst_y = dram ? dram_core.y : rx.y;
            const uint32_t dst_addr = (
                dram ? dram_bank_address + pair.index * slot_bytes : slots_l1
            );
            const uint32_t shift = pair.second ? second_shift : 0;
            std::vector<uint32_t> sender_args = {
                static_cast<uint32_t>(status_buffer->address()),
                pair.index,
                kPageBytes,
                status_l1 + shift,
                headers_l1 + shift,
                slots_l1 + shift,
                dst_x,
                dst_y,
                dst_addr,
                static_cast<uint32_t>(rx.x),
                static_cast<uint32_t>(rx.y),
                counter_l1,
                o.packets,
                o.payload,
                o.window,
                o.seed,
                pair.index,
                limit_mcycles,
                o.hops,
                static_cast<uint32_t>(o.delay_us * 1350.0),
                o.fused,
                o.batch,
                o.wait_every,
                o.space_every,
                o.send
            };
            // The second link of a dual core runs on the other data-movement CPU,
            // which uses the other NoC.
            const bool cpu1 = pair.second || (!o.dual && o.noc == 1);
            const auto sender = tt::tt_metal::CreateKernel(
                programs[pair.from],
                here + "/sender.cpp",
                pair.sender,
                tt::tt_metal::DataMovementConfig{
                    .processor = cpu1 ? tt::tt_metal::DataMovementProcessor::RISCV_1
                                      : tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = cpu1 ? tt::tt_metal::NOC::RISCV_1_default
                                : tt::tt_metal::NOC::RISCV_0_default,
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
            if (kernel_receiver) {
                std::vector<uint32_t> receiver_args = {
                    static_cast<uint32_t>(status_buffer->address()),
                    kStatusPages / 2 + pair.index,
                    kPageBytes,
                    status_l1,
                    slots_l1,
                    counter_l1,
                    expect,
                    std::min(o.packets, o.window),
                    o.payload,
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
            }
            std::ostringstream text;
            text << "pair " << pair.index << ": chip column " << pair.from << " link "
                 << pair.link << " sender (" << pair.sender.x << "," << pair.sender.y
                 << ")" << (cpu1 ? " cpu 1" : " cpu 0") << " -> column " << to
                 << " core (" << pair.receiver.x << "," << pair.receiver.y
                 << ") virtual (" << rx.x << "," << rx.y << ")";
            log(text.str());
        }
        // Columns with kernels; the receivers' columns are launched first.
        std::vector<uint32_t> order;
        for (const auto& pair : pairs) {
            if (
                kernel_receiver &&
                std::find(order.begin(), order.end(), pair.to) == order.end()
            ) {
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
        std::vector<std::vector<uint32_t>> data(n);
        if (dram) {
            for (uint32_t c = 0; c < n; c++) {
                distributed::ReadShard(
                    queue, data[c], data_buffer, distributed::MeshCoordinate(0, c), true
                );
            }
        }
        std::ostringstream clocks;
        for (uint32_t c = 0; c < n; c++) {
            mhz[c] = std::max(mhz[c], aiclk(chips[c]->id()));
            clocks << (c ? ", " : "") << mhz[c];
        }
        bool ok = true;
        uint64_t bytes = 0;
        const auto& soc0 = (
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                chips[0]->id()
            )
        );
        const std::pair<uint32_t, uint32_t> noc_size{
            static_cast<uint32_t>(soc0.grid_size.x),
            static_cast<uint32_t>(soc0.grid_size.y)
        };
        std::ostringstream report;
        report << "{\"options\": {\"from\": \"" << o.from
               << "\", \"packets\": " << o.packets << ", \"payload\": " << o.payload
               << ", \"window\": " << o.window << ", \"links\": " << o.links
               << ", \"hops\": " << o.hops << ", \"first_link\": " << o.first_link
               << ", \"fused\": " << o.fused << ", \"batch\": " << o.batch
               << ", \"wait_every\": " << o.wait_every
               << ", \"space_every\": " << o.space_every << ", \"place\": \"" << o.place
               << "\", \"rx_place\": \"" << o.rx_place << "\", \"noc\": " << o.noc
               << ", \"target\": \"" << o.target << "\", \"dual\": " << o.dual
               << ", \"send\": " << o.send
               << ", \"router_payload\": " << o.router_payload << "}, \"noc_size\": ["
               << noc_size.first << ", " << noc_size.second
               << "], \"expect\": " << expect << ", \"aiclk_mhz\": [" << clocks.str()
               << "], \"pairs\": [";
        for (const auto& pair : pairs) {
            const uint32_t to = pair.to;
            const uint32_t words = kPageBytes / 4;
            std::vector<uint32_t> sent(
                pages[pair.from].begin() + pair.index * words,
                pages[pair.from].begin() + (pair.index + 1) * words
            );
            std::vector<uint32_t> got(words, 0);
            if (kernel_receiver) {
                const uint32_t rpage = kStatusPages / 2 + pair.index;
                got.assign(
                    pages[to].begin() + rpage * words,
                    pages[to].begin() + (rpage + 1) * words
                );
            }
            // The host's check of what landed: the receiver's copy of its first slots,
            // or every slot read back (no receiver kernel), and the counter.
            const uint32_t filled = std::min(o.packets, o.window);
            uint32_t host_wrong = 0, host_words = 0, counted = got[kWordPackets];
            if (kernel_receiver) {
                host_words = std::min(filled * o.payload, kCopyBytes) / 4;
                for (uint32_t w = 0; w < host_words; w++) {
                    const uint32_t slot = w / (o.payload / 4);
                    const uint32_t word = w % (o.payload / 4);
                    host_wrong += (
                        got[kStatusBytes / 4 + w] !=
                        pattern(o.seed, pair.index, slot, word)
                    );
                }
            } else {
                std::vector<uint32_t> landed(slot_bytes / 4, 0);
                if (dram) {
                    const uint64_t at = pair.index * slot_bytes / 4;
                    landed.assign(
                        data[to].begin() + at, data[to].begin() + at + slot_bytes / 4
                    );
                } else {
                    tt::tt_metal::detail::ReadFromDeviceL1(
                        chips[to], pair.receiver, slots_l1, slot_bytes, landed
                    );
                }
                std::vector<uint32_t> counter(4, 0);
                tt::tt_metal::detail::ReadFromDeviceL1(
                    chips[to], pair.receiver, counter_l1, 16, counter
                );
                counted = counter[0];
                host_words = filled * o.payload / 4;
                for (uint32_t w = 0; w < host_words; w++) {
                    const uint32_t slot = w / (o.payload / 4);
                    const uint32_t word = w % (o.payload / 4);
                    host_wrong += landed[w] != pattern(o.seed, pair.index, slot, word);
                }
            }
            const bool pair_ok = (
                sent[kWordMagic] == kMagic && sent[kWordCode] == kCodeOk &&
                counted == expect && host_wrong == 0 &&
                (
                    !kernel_receiver ||
                    (
                        got[kWordMagic] == kMagic && got[kWordCode] == kCodeOk &&
                        got[kWordWrong] == 0
                    )
                )
            );
            ok &= pair_ok;
            bytes += static_cast<uint64_t>(sent[kWordPackets]) * o.payload;
            const auto sender_noc0 = (
                noc0_of(
                    chips[pair.from],
                    chips[pair.from]->worker_core_from_logical_core(pair.sender),
                    tt::CoreType::TENSIX
                )
            );
            const auto eth_noc0 = (
                noc0_of(
                    chips[pair.from],
                    chips[pair.from]->ethernet_core_from_logical_core(pair.eth),
                    tt::CoreType::ETH
                )
            );
            report << (pair.index ? ", " : "") << "{\"from\": " << pair.from
                   << ", \"to\": " << to << ", \"link\": " << pair.link
                   << ", \"sender_core\": [" << pair.sender.x << ", " << pair.sender.y
                   << "], \"sender_noc0\": [" << sender_noc0.first << ", "
                   << sender_noc0.second << "], \"eth_noc0\": [" << eth_noc0.first
                   << ", " << eth_noc0.second << "], \"sender_noc\": "
                   << ((pair.second || (!o.dual && o.noc == 1)) ? 1 : 0)
                   << ", \"ok\": " << (pair_ok ? "true" : "false")
                   << ", \"counted\": " << counted
                   << ", \"host_checked_words\": " << host_words
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
