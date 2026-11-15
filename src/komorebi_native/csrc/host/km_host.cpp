// _km_host: the X280 runtime, the dispatcher's memory layout (km_dispatch.h), its
// start and stop, batched host commands and the event ring. Every wait has a timeout.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <immintrin.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/context/context_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include <llrt/llrt.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/chip_helpers/sysmem_buffer.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/io_window_config.hpp>
#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_state.hpp"
#include "x280_address_map.h"

#include "komorebi/abi/km_abi.h"
#include "km_dispatch.h"

namespace nb = nanobind;
using tt::tt_metal::distributed::MeshDevice;
using Clock = std::chrono::steady_clock;

namespace {

    constexpr double kMtimeHz = 50e6;
    // One page: the largest buffer that maps to the NoC without an IOMMU.
    constexpr size_t kHostBufferBytes = 4096;
    constexpr uint32_t kArgsBytes = 256;  // the argument block, whole 64-byte lines
    // Ring depths are powers of two, so slot = number % depth holds across the wrap.
    constexpr uint32_t kEventDepthMax = 128;
    static_assert(KMD_HOST_RING + kEventDepthMax * 16 <= kHostBufferBytes);

    bool power_of_two(uint32_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    // The X280 idle image of the tree in use (firmware-lim-idle.bin), set by the
    // package when it loads this module: the module does not name a build directory.
    std::string idle_image_path;

    static_assert(sizeof(km_command_t) == 64);
    static_assert(sizeof(km_event_t) == 16 && sizeof(kmd_args_t) <= kArgsBytes);
    static_assert(kArgsBytes <= KMD_LIM_MAILBOX);

    using Commands = nb::
        ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
    using Events = nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>;
    using Bytes = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>;
    using Numbers = nb::
        ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    using Numbers8 = nb::
        ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    Clock::time_point after(double seconds) {
        return (
            Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(seconds)
            )
        );
    }

    std::chrono::milliseconds millis(double seconds) {
        return std::chrono::milliseconds(static_cast<int64_t>(seconds * 1000.0));
    }

    // The go message's run state of a launched program (RUN_MSG_GO, dev_msgs.h).
    constexpr int kRunMsgGo = 0x80;

    tt::tt_metal::IDevice* chip(const MeshDevice& mesh) {
        return mesh.get_devices().front();
    }

    // Writes words into one Tensix core's L1 from the host, through the device API
    // (not the X280): the stop flag when the dispatcher is not there to send it.
    void write_l1(
        const MeshDevice& mesh,
        uint32_t x,
        uint32_t y,
        uint32_t address,
        std::vector<uint32_t> words
    ) {
        tt::tt_metal::detail::WriteToDeviceL1(
            chip(mesh), tt::tt_metal::CoreCoord{x, y}, address, words
        );
    }

    // `count` words of core (x, y)'s L1 at `address`, read through the driver: no
    // command queue, so it does not wait for a launched program.
    std::vector<uint32_t> read_l1(
        const MeshDevice& mesh, uint32_t x, uint32_t y, uint32_t address, uint32_t count
    ) {
        std::vector<uint32_t> words;
        tt::tt_metal::detail::ReadFromDeviceL1(
            chip(mesh), tt::tt_metal::CoreCoord{x, y}, address, count * 4, words
        );
        return words;
    }

    // The router's connection words (the fabric's connection_interface): unused, close.
    constexpr uint32_t kConnectionUnused = 0;
    constexpr uint32_t kConnectionClose = 2;

    // Closes core `where`'s (in reset) fabric connection on each of `channels` as its
    // close would, the cursor at the router's read counter. False: no teardown in time.
    bool close_connections(
        tt::tt_metal::IDevice* dev,
        const tt_cxy_pair& where,
        const std::vector<uint32_t>& channels,
        double timeout_s
    ) {
        namespace tf = tt::tt_fabric;
        if (channels.empty()) {
            return true;
        }
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto& cluster = context.get_cluster();
        tf::tensix_fabric_connections_l1_info_t table{};
        cluster.read_core(
            &table,
            sizeof(table),
            where,
            context.hal().get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::TENSIX,
                tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS
            )
        );
        for (const uint32_t channel : channels) {
            if (
                channel >=
                    tf::tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS ||
                !(table.valid_connections_mask >> channel & 1u)
            ) {
                throw std::invalid_argument("reset_core: no fabric connection there");
            }
            const auto& connection = table.read_only[channel];
            const tt_cxy_pair router(
                dev->id(),
                tt::tt_metal::CoreCoord{connection.edm_noc_x, connection.edm_noc_y}
            );
            const uint32_t close = kConnectionClose;
            cluster.write_core_immediate(
                &close, sizeof(close), router, connection.edm_connection_handshake_addr
            );
            const auto until = Clock::now() + std::chrono::duration<double>(timeout_s);
            uint32_t state = close;
            while (state != kConnectionUnused) {
                if (Clock::now() > until) {
                    return false;
                }
                cluster.read_core(
                    &state,
                    sizeof(state),
                    router,
                    connection.edm_connection_handshake_addr
                );
            }
            uint32_t counter = 0;
            cluster.read_core(
                &counter,
                sizeof(counter),
                router,
                connection.edm_worker_location_info_addr +
                    offsetof(tf::EDMChannelWorkerLocationInfo, edm_read_counter)
            );
            const std::array<uint32_t, 4> cursor{
                counter, counter % connection.num_buffers_per_channel, 0u, 0u
            };
            cluster.write_core(
                cursor.data(),
                sizeof(cursor),
                router,
                connection.buffer_index_semaphore_id
            );
        }
        cluster.l1_barrier(dev->id());
        return true;
    }

    // Core `where` in reset: each queue of `drops` (tail, taken, lines, count) drops
    // its untaken lines: tail = taken, words 0 and 15 of its 64-byte lines cleared.
    void drop_lines(const tt_cxy_pair& where, const std::vector<uint32_t>& drops) {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        for (size_t at = 0; at < drops.size(); at += 4) {
            uint32_t taken = 0;
            cluster.read_core(&taken, sizeof(taken), where, drops[at + 1]);
            cluster.write_core(&taken, sizeof(taken), where, drops[at]);
            const uint32_t zero = 0;
            for (uint32_t line = 0; line < drops[at + 3]; line++) {
                const uint64_t base = drops[at + 2] + 64ull * line;
                cluster.write_core(&zero, sizeof(zero), where, base);
                cluster.write_core(&zero, sizeof(zero), where, base + 60);
            }
        }
    }

    // The grid (x, y), then per core in core order its translated NoC address (the
    // core table's) and its NoC0 coordinates, as x | y << 16 words; no X280 needed.
    std::tuple<uint32_t, uint32_t, std::vector<uint32_t>, std::vector<uint32_t>>
    core_nocs(MeshDevice& mesh) {
        const auto grid = mesh.compute_with_storage_grid_size();
        const auto& soc = (
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                chip(mesh)->id()
            )
        );
        std::vector<uint32_t> table, noc0;
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const auto virtual_core = x280::X280Runtime::virtual_core_from_logical_core(
                    mesh,
                    tt::tt_metal::CoreCoord{x, y},
                    tt::CoreType::WORKER,
                    tt::tt_metal::NOC::NOC_0
                );
                const auto noc0_core = (
                    soc.translate_coord_to(
                        tt::umd::CoreCoord(
                            virtual_core.x(),
                            virtual_core.y(),
                            tt::CoreType::TENSIX,
                            tt::CoordSystem::TRANSLATED
                        ),
                        tt::CoordSystem::NOC0
                    )
                );
                table.push_back(
                    static_cast<uint32_t>(virtual_core.x() | virtual_core.y() << 16)
                );
                noc0.push_back(static_cast<uint32_t>(noc0_core.x | noc0_core.y << 16));
            }
        }
        return {
            static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), table, noc0
        };
    }

    // The number of DRAM banks of interleaved buffers (bank b is DRAM view b).
    uint32_t dram_banks(const MeshDevice& mesh) {
        return chip(mesh)->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    }

    // [kernel buffer bytes, L1 bytes left for buffers, L1 size] of a Tensix core, from
    // the runtime's layout and the allocator's base.
    std::array<uint64_t, 3> kernel_buffer(const MeshDevice& mesh) {
        auto* dev = chip(mesh);
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        const auto tensix = tt::tt_metal::HalProgrammableCoreType::TENSIX;
        const uint64_t config = (
            hal.get_dev_addr(tensix, tt::tt_metal::HalL1MemAddrType::KERNEL_CONFIG)
        );
        const uint64_t l1 = (
            hal.get_dev_addr(tensix, tt::tt_metal::HalL1MemAddrType::BASE) +
            hal.get_dev_size(tensix, tt::tt_metal::HalL1MemAddrType::BASE)
        );
        const uint64_t base = (
            dev->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)
        );
        return {base - config, l1 - base, l1};
    }

    // A build: per kernel its source and ELFs; config bytes (0: it does not fit), the
    // buffer, packed text per Tensix processor, the runtime's message.
    using KernelBuild = std::tuple<
        std::vector<std::tuple<std::string, std::vector<std::string>>>,
        uint32_t,
        uint32_t,
        std::vector<uint32_t>,
        std::string>;

    // Builds `desc`'s kernels for this device as generic_op does (its own build then
    // hits the JIT cache) and lays out its kernel config, which checks its size.
    KernelBuild kernel_build(
        const MeshDevice& mesh, const tt::tt_metal::ProgramDescriptor& desc
    ) {
        tt::tt_metal::IDevice* dev = chip(mesh);
        tt::tt_metal::Program program(desc);
        program.impl().compile(dev);
        const auto& env = (
            tt::tt_metal::BuildEnvManager::get_instance(
                tt::tt_metal::extract_context_id(dev)
            )
                .get_device_build_env(dev->build_id())
        );
        const std::string root = env.build_env.get_out_kernel_root_path();
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        const uint32_t types = hal.get_programmable_core_type_count();
        std::vector<std::tuple<std::string, std::vector<std::string>>> kernels;
        for (uint32_t core_type = 0; core_type < types; core_type++) {
            for (const auto& [id, kernel] : program.impl().get_kernels(core_type)) {
                const auto& source = kernel->kernel_source();
                kernels.emplace_back(
                    source.path_.empty() ? kernel->name() : source.path_.string(),
                    kernel->elf_paths_by_processor_index(*dev, root)
                );
            }
        }
        const auto tensix = tt::tt_metal::HalProgrammableCoreType::TENSIX;
        const uint32_t index = hal.get_programmable_core_type_index(tensix);
        uint32_t config_bytes = 0;
        std::string error;
        try {
            program.impl().finalize_offsets(dev);
            config_bytes = program.impl().get_program_config_sizes()[index];
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        // The text sizes are set before the size check, so they are there either way.
        std::vector<uint32_t> text(hal.get_num_risc_processors(tensix), 0);
        for (const auto& group : program.impl().get_kernel_groups(index)) {
            const auto sizes = (
                group->launch_msg.view().kernel_config().kernel_text_size()
            );
            for (size_t processor = 0; processor < text.size(); processor++) {
                text[processor] = (
                    std::max(text[processor], static_cast<uint32_t>(sizes[processor]))
                );
            }
        }
        return {kernels, config_bytes, kernel_buffer(mesh)[0], text, error};
    }

    // Per kernel its source and, per processor, the runtime's build of it: target, ELF,
    // compiler, optimisation level, flags, includes, link flags, link level, defines.
    using KernelRecipes = std::vector<
        std::tuple<std::string, std::vector<std::vector<std::string>>>>;

    // Builds `desc`'s kernels (a JIT cache hit after kernel_build) and reads back the
    // commands the runtime compiled and linked each processor's ELF with.
    KernelRecipes kernel_recipes(
        const MeshDevice& mesh, const tt::tt_metal::ProgramDescriptor& desc
    ) {
        tt::tt_metal::IDevice* dev = chip(mesh);
        tt::tt_metal::Program program(desc);
        program.impl().compile(dev);
        auto& manager = (
            tt::tt_metal::BuildEnvManager::get_instance(
                tt::tt_metal::extract_context_id(dev)
            )
        );
        const auto& env = manager.get_device_build_env(dev->build_id());
        const std::string root = env.build_env.get_out_kernel_root_path();
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        KernelRecipes out;
        for (uint32_t type = 0; type < hal.get_programmable_core_type_count(); type++) {
            for (const auto& [id, kernel] : program.impl().get_kernels(type)) {
                // by processor index, empty before this kernel's first processor
                std::vector<std::string> elves;
                for (
                    const auto& path : kernel->elf_paths_by_processor_index(*dev, root)
                ) {
                    if (!path.empty()) {
                        elves.push_back(path);
                    }
                }
                const uint32_t core_type = (
                    hal.get_programmable_core_type_index(
                        kernel->get_kernel_programmable_core_type()
                    )
                );
                const auto processor_class = (
                    static_cast<uint32_t>(kernel->get_kernel_processor_class())
                );
                std::vector<std::vector<std::string>> builds;
                for (int index = 0; index < kernel->expected_num_binaries(); index++) {
                    const auto& state = (
                        manager.get_kernel_build_state(
                            dev->build_id(),
                            core_type,
                            processor_class,
                            kernel->get_kernel_processor_type(index)
                        )
                    );
                    const auto recipe = state.export_target_recipe(kernel.get());
                    std::vector<std::string> build = {
                        recipe.target_name,
                        elves.at(index),
                        env.build_env.get_gpp(),
                        recipe.compiler_opt_level,
                        recipe.cflags,
                        recipe.includes,
                        recipe.lflags,
                        recipe.linker_opt_level
                    };
                    build.insert(
                        build.end(), recipe.defines.begin(), recipe.defines.end()
                    );
                    builds.push_back(std::move(build));
                }
                const auto& source = kernel->kernel_source();
                out.emplace_back(
                    source.path_.empty() ? kernel->name() : source.path_.string(),
                    std::move(builds)
                );
            }
        }
        return out;
    }

    // Reads (bank, address, bytes) spans of DRAM after one DRAM barrier, one read each.
    // No wait for idle cores: the caller reads only what no kernel is still writing.
    std::vector<Bytes> read_dram(
        const MeshDevice& mesh, const std::vector<std::array<uint64_t, 3>>& spans
    ) {
        auto* dev = chip(mesh);
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        std::vector<std::unique_ptr<std::vector<uint8_t>>> data;  // until the capsules
        for (const auto& span : spans) {
            if (span[0] >= dram_banks(mesh) || span[2] > UINT32_MAX) {
                throw std::runtime_error("a DRAM span outside the banks or over 4 GB");
            }
            data.push_back(std::make_unique<std::vector<uint8_t>>(span[2]));
        }
        {
            nb::gil_scoped_release release;
            cluster.dram_barrier(dev->id());
            for (size_t span_index = 0; span_index < spans.size(); span_index++) {
                const auto& [bank, address, bytes] = spans[span_index];
                cluster.read_dram_vec(
                    data[span_index]->data(),
                    static_cast<uint32_t>(bytes),
                    dev->id(),
                    static_cast<int>(bank),
                    address
                );
            }
        }
        std::vector<Bytes> out;
        for (auto& held : data) {
            auto* owned = held.get();
            nb::capsule owner(owned, [](void* memory) noexcept {
                delete static_cast<std::vector<uint8_t>*>(memory);
            });
            held.release();  // the capsule owns it now
            const size_t shape[1] = {owned->size()};
            out.emplace_back(owned->data(), 1, shape, owner);
        }
        return out;
    }

    // Writes `data` at `address` of every DRAM bank in `banks`, with no wait for idle
    // cores (a live run's block); on the card only after dram_barrier().
    void write_dram(
        const MeshDevice& mesh,
        const std::vector<uint32_t>& banks,
        uint64_t address,
        const Numbers8& data
    ) {
        auto* dev = chip(mesh);
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        for (const uint32_t bank : banks) {
            if (bank >= dram_banks(mesh) || data.size() > UINT32_MAX) {
                throw std::runtime_error("a DRAM write outside the banks or over 4 GB");
            }
        }
        nb::gil_scoped_release release;
        for (const uint32_t bank : banks) {
            cluster.write_dram_vec(
                data.data(),
                static_cast<uint32_t>(data.size()),
                dev->id(),
                static_cast<int>(bank),
                address
            );
        }
    }

    // Returns once every earlier write_dram() of the device is in DRAM.
    void dram_barrier(const MeshDevice& mesh) {
        auto* dev = chip(mesh);
        nb::gil_scoped_release release;
        tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(dev->id());
    }

    // Waits up to timeout_s for the kernels of grid_y rows from first_row to return, in
    // slices of at most a second, signals checked between them. True: returned.
    bool wait_grid(
        const MeshDevice& mesh,
        uint32_t grid_x,
        uint32_t grid_y,
        double timeout_s,
        uint32_t first_row = 0
    ) {
        auto* dev = chip(mesh);
        std::unordered_set<tt::tt_metal::CoreCoord> left;
        for (uint32_t y = first_row; y < first_row + grid_y; y++) {
            for (uint32_t x = 0; x < grid_x; x++) {
                left.insert(dev->virtual_core_from_logical_core(
                    tt::tt_metal::CoreCoord{x, y}, tt::CoreType::WORKER
                ));
            }
        }
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto deadline = after(timeout_s);
        for (;;) {
            const auto rest = (
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - Clock::now()
                )
            );
            const int slice = (
                static_cast<int>(std::clamp<int64_t>(rest.count(), 1, 1000))
            );
            {
                nb::gil_scoped_release release;
                try {
                    tt::llrt::internal_::wait_until_cores_done(
                        context, dev->id(), kRunMsgGo, left, slice
                    );
                } catch (const std::exception&) {
                    // the slice ran out; the set holds the cores not done yet
                }
            }
            if (left.empty()) {
                return true;
            }
            if (Clock::now() >= deadline) {
                return false;
            }
            if (PyErr_CheckSignals() != 0) {
                throw nb::python_error();
            }
        }
    }

    // The X280 runtime of a mesh of several islands, shared by their Dispatches; empty
    // until Islands::open_x280.
    struct IslandRuntime {
        std::shared_ptr<x280::RuntimeOpenResult> opened;
    };

    // The tables the open writes into each Tensix core's firmware scratch, which the
    // kernels' area later covers: bank to NoC, bank offsets, logical to virtual.
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> firmware_tables(
        tt::tt_metal::IDevice* dev
    ) {
        namespace tm = tt::tt_metal;
        const auto& hal = tm::MetalContext::instance().hal();
        const auto& cluster = tm::MetalContext::instance().get_cluster();
        const auto& soc = cluster.get_soc_desc(dev->id());
        const auto* allocator = dev->allocator().get();
        const auto tensix = tm::HalProgrammableCoreType::TENSIX;
        const auto xy = [&](uint32_t noc, uint32_t x, uint32_t y) {
            return (
                static_cast<uint16_t>(
                    ((y << hal.get_noc_addr_node_id_bits()) | x)
                    << hal.get_noc_coord_reg_offset()
                )
            );
        };
        std::vector<uint16_t> noc_xy;
        std::vector<int32_t> offsets;
        const uint32_t dram_banks = allocator->get_num_banks(tm::BufferType::DRAM);
        const uint32_t l1_banks = allocator->get_num_banks(tm::BufferType::L1);
        const bool virtualized = (
            cluster.get_cluster_desc()->get_noc_translation_table_en().at(dev->id()) &&
            hal.get_virtualized_core_types().contains(
                tm::dev_msgs::AddressableCoreType::DRAM
            )
        );
        for (uint32_t noc = 0; noc < hal.get_num_nocs(); noc++) {
            for (uint32_t bank = 0; bank < dram_banks; bank++) {
                const auto core = (
                    soc.get_preferred_worker_core_for_dram_view(
                        allocator->impl().get_dram_channel_from_bank_id(bank), noc
                    )
                );
                noc_xy.push_back(
                    virtualized
                        ? xy(noc, core.x, core.y)
                        : xy(noc,
                               hal.noc_coordinate(noc, soc.grid_size.x, core.x),
                               hal.noc_coordinate(noc, soc.grid_size.y, core.y))
                );
            }
        }
        for (uint32_t noc = 0; noc < hal.get_num_nocs(); noc++) {
            for (uint32_t bank = 0; bank < l1_banks; bank++) {
                const auto core = (
                    cluster.get_virtual_coordinate_from_logical_coordinates(
                        dev->id(),
                        allocator->get_logical_core_from_bank_id(bank),
                        tt::CoreType::WORKER
                    )
                );
                noc_xy.push_back(xy(noc, core.x, core.y));  // Blackhole: as it is
            }
        }
        for (uint32_t bank = 0; bank < dram_banks; bank++) {
            offsets.push_back(allocator->get_bank_offset(tm::BufferType::DRAM, bank));
        }
        for (uint32_t bank = 0; bank < l1_banks; bank++) {
            offsets.push_back(allocator->get_bank_offset(tm::BufferType::L1, bank));
        }
        std::vector<uint8_t> banks(noc_xy.size() * 2 + offsets.size() * 4);
        std::memcpy(banks.data(), noc_xy.data(), noc_xy.size() * 2);
        std::memcpy(
            banks.data() + noc_xy.size() * 2, offsets.data(), offsets.size() * 4
        );
        if (
            banks.size() >
            hal.get_dev_size(tensix, tm::HalL1MemAddrType::BANK_TO_NOC_SCRATCH)
        ) {
            throw std::runtime_error("the bank tables do not fit the firmware scratch");
        }
        const auto grid = soc.get_grid_size(tt::CoreType::TENSIX);
        const auto translated = [&](size_t x, size_t y) {
            return soc.translate_coord_to(
                {tt_xy_pair{x, y}, tt::CoreType::TENSIX, tt::CoordSystem::LOGICAL},
                tt::CoordSystem::TRANSLATED
            );
        };
        const size_t columns = (soc.grid_size.x + 3) / 4 * 4;
        std::vector<uint8_t> logical(columns + (grid.y + 3) / 4 * 4, 0);
        for (size_t x = 0; x < grid.x; x++) {
            logical[x] = static_cast<uint8_t>(translated(x, 0).x);
        }
        for (size_t y = 0; y < grid.y; y++) {
            logical[columns + y] = static_cast<uint8_t>(translated(0, y).y);
        }
        return {
            {
                hal.get_dev_addr(tensix, tm::HalL1MemAddrType::BANK_TO_NOC_SCRATCH),
                banks
            },
            {hal.get_dev_addr(tensix, tm::HalL1MemAddrType::LOGICAL_TO_VIRTUAL_SCRATCH),
             logical},
        };
    }

    // An open card: the 1 x 1 mesh of the first visible device, opened as ttnn's
    // open_mesh_device opens it, with the program cache off; or an island of Islands.
    class Device {
    public:
        Device(
            std::shared_ptr<MeshDevice> island, std::shared_ptr<IslandRuntime> x280
        ) :
            mesh_(std::move(island)), x280_(std::move(x280)) {
        }

        explicit Device(std::optional<size_t> worker_l1_size) {
            namespace distributed = tt::tt_metal::distributed;
            mesh_ = MeshDevice::create(
                distributed::MeshDeviceConfig(distributed::MeshShape(1, 1)),
                DEFAULT_L1_SMALL_SIZE,
                DEFAULT_TRACE_REGION_SIZE,
                1,
                tt::tt_metal::DispatchCoreConfig{},
                {},
                worker_l1_size.value_or(DEFAULT_WORKER_L1_SIZE)
            );
            mesh_->disable_and_clear_program_cache();
        }

        MeshDevice& mesh() const {
            if (!mesh_) {
                throw std::runtime_error("the device is closed");
            }
            return *mesh_;
        }

        std::shared_ptr<MeshDevice> shared() const {
            mesh();
            return mesh_;
        }

        bool is_open() const {
            return mesh_ != nullptr;
        }

        void close() {
            if (mesh_) {
                mesh_->close();
                mesh_.reset();
            }
        }

        std::tuple<uint32_t, uint32_t> grid() const {
            const auto size = mesh().compute_with_storage_grid_size();
            return {static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)};
        }

        // A Tensix core's NoC coordinates (the runtime's virtual ones) from its
        // logical (x, y).
        std::tuple<uint32_t, uint32_t> worker_core(uint32_t x, uint32_t y) const {
            const auto core = mesh().worker_core_from_logical_core({x, y});
            return {static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
        }

        // The NoC coordinates of DRAM bank `bank`'s core.
        std::tuple<uint32_t, uint32_t> dram_core(uint32_t bank) const {
            const auto core = mesh().virtual_core_from_logical_core(
                {bank, 0}, tt::CoreType::DRAM
            );
            return {static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
        }

        // The host address of the driver's own mapping of the card's host channel (its
        // hugepage of host memory channel 0); 0 when it mapped none, or on a mock.
        uint64_t host_channel() const {
            auto* dev = chip(mesh());
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            if (cluster.get_target_device_type() != tt::TargetDevice::Silicon) {
                return 0;
            }
            return (
                reinterpret_cast<uintptr_t>(cluster.host_dma_address(0, dev->id(), 0))
            );
        }

        // Waits until every launched program has finished.
        void finish() const {
            tt::tt_metal::distributed::Finish(mesh().mesh_command_queue());
        }

        // The shared X280 runtime of an island; nullptr for a card opened alone.
        const std::shared_ptr<IslandRuntime>& x280() const {
            return x280_;
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
        std::shared_ptr<IslandRuntime> x280_;
    };

    // Several chips as a 1 x count fabric mesh (slow dispatch), each an island with its
    // own queue; `packet`: router payload bytes (0: default).
    class Islands {
    public:
        Islands(uint32_t count, std::optional<size_t> worker_l1_size, uint32_t packet) {
            namespace distributed = tt::tt_metal::distributed;
            if (count < 2) {
                throw std::invalid_argument(
                    "islands: at least 2 (the fabric's minimum)"
                );
            }
            if (packet % 16 != 0) {
                throw std::invalid_argument(
                    "islands: a packet of " + std::to_string(packet) +
                    " bytes (whole 16-byte words)"
                );
            }
            tt::tt_fabric::FabricRouterConfig router;
            if (packet != 0) {
                router.max_packet_payload_size_bytes = packet;
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
            try {
                mesh_ = MeshDevice::create(
                    distributed::MeshDeviceConfig(distributed::MeshShape(1, count)),
                    DEFAULT_L1_SMALL_SIZE,
                    DEFAULT_TRACE_REGION_SIZE,
                    1,
                    tt::tt_metal::DispatchCoreConfig{},
                    {},
                    worker_l1_size.value_or(DEFAULT_WORKER_L1_SIZE)
                );
                mesh_->disable_and_clear_program_cache();
                for (uint32_t island = 0; island < count; island++) {
                    auto sub = (
                        mesh_->create_submesh(
                            distributed::MeshShape(1, 1),
                            distributed::MeshCoordinate(0, island)
                        )
                    );
                    sub->disable_and_clear_program_cache();
                    islands_.push_back(std::move(sub));
                }
            } catch (...) {
                close();
                throw;
            }
            x280_ = std::make_shared<IslandRuntime>();
        }

        ~Islands() {
            close();
        }

        uint32_t count() const {
            return static_cast<uint32_t>(islands_.size());
        }

        // The chip id of each island, in mesh column order.
        std::vector<int> chips() const {
            std::vector<int> out;
            for (const auto& island : islands_) {
                out.push_back(chip(*island)->id());
            }
            return out;
        }

        // Island `island` as a Device for a Session given a device.
        Device device(uint32_t island) const {
            if (!mesh_ || island >= islands_.size()) {
                throw std::runtime_error("no such island, or the islands are closed");
            }
            return Device(islands_[island], x280_);
        }

        // Opens the X280 runtime over the whole mesh once; every island's Dispatch
        // shares it. Raises with the runtime's outcome when it is not ready.
        void open_x280() {
            if (!mesh_) {
                throw std::runtime_error("the islands are closed");
            }
            if (x280_->opened) {
                return;
            }
            auto opened = std::make_shared<x280::RuntimeOpenResult>(
                x280::X280Runtime::open(
                    mesh_,
                    x280::RuntimeOpenConfig{
                        .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
                    }
                )
            );
            if (!opened->ready()) {
                throw std::runtime_error(
                    std::string("X280 runtime not ready (") +
                    x280::to_string(opened->outcome) +
                    (opened->requires_reset() ? ", reset-required" : "") +
                    "): " + opened->diagnostic
                );
            }
            x280_->opened = std::move(opened);
        }

        // Drops the runtime (the Dispatches must be closed), the islands, the mesh,
        // then turns the fabric off.
        void close() {
            if (x280_) {
                x280_->opened.reset();
            }
            islands_.clear();
            if (mesh_) {
                mesh_->close();
                mesh_.reset();
                tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
            }
        }

        bool is_open() const {
            return mesh_ != nullptr;
        }

        // Link `link` from island `from` to `to`: (links found, the Ethernet channel a
        // Tensix core connects through, the Ethernet core's NoC x and y on `from`).
        std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> link(
            uint32_t from, uint32_t to, uint32_t link
        ) const {
            namespace distributed = tt::tt_metal::distributed;
            if (
                !mesh_ || from >= islands_.size() || to >= islands_.size() || from == to
            ) {
                throw std::runtime_error("link: two different islands of an open mesh");
            }
            const auto source = (
                mesh_->get_fabric_node_id(distributed::MeshCoordinate(0, from))
            );
            const auto target = (
                mesh_->get_fabric_node_id(distributed::MeshCoordinate(0, to))
            );
            const auto links = (
                tt::tt_fabric::get_forwarding_link_indices(source, target)
            );
            if (link >= links.size()) {
                throw std::runtime_error(
                    "link " + std::to_string(link) + ": " +
                    std::to_string(links.size()) + " found"
                );
            }
            // The connection's first argument is the channel; the descriptor is a
            // scratch.
            tt::tt_metal::ProgramDescriptor scratch;
            std::vector<uint32_t> args;
            tt::tt_fabric::append_fabric_connection_rt_args(
                source,
                target,
                links[link],
                scratch,
                tt::tt_metal::CoreCoord{0, 0},
                args
            );
            const auto eth = (
                tt::tt_fabric::get_forwarding_eth_core(
                    *mesh_, source, target, links[link]
                )
            );
            const auto core = (
                chip(*islands_[from])->ethernet_core_from_logical_core(eth)
            );
            return {
                static_cast<uint32_t>(links.size()),
                args.at(0),
                static_cast<uint32_t>(core.x),
                static_cast<uint32_t>(core.y)
            };
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
        std::vector<std::shared_ptr<MeshDevice>> islands_;
        std::shared_ptr<IslandRuntime> x280_;
    };

    using MeshBuffer = tt::tt_metal::distributed::MeshBuffer;

    // A device buffer: interleaved in DRAM, or (shard_grid set) height-sharded in L1,
    // one page a core, cores in row-major order; `address`: where an L1 buffer goes.
    class DeviceBuffer {
    public:
        DeviceBuffer(
            const Device& device,
            uint64_t size,
            uint64_t page_size,
            std::optional<std::array<uint32_t, 2>> shard_grid,
            std::optional<uint64_t> address
        ) :
            mesh_(device.shared()) {
            namespace distributed = tt::tt_metal::distributed;
            using tt::tt_metal::BufferType;
            if (page_size == 0 || size % page_size != 0) {
                throw std::invalid_argument("the size is not a whole number of pages");
            }
            distributed::DeviceLocalBufferConfig local{
                .page_size = page_size, .buffer_type = BufferType::DRAM
            };
            if (shard_grid) {
                const auto [grid_x, grid_y] = *shard_grid;
                const uint32_t cores = grid_x * grid_y;
                if (size != uint64_t{cores} * page_size || page_size % 4 != 0) {
                    throw std::invalid_argument(
                        "a sharded buffer holds one page of whole words per core"
                    );
                }
                const auto words = static_cast<uint32_t>(page_size / 4);
                const tt::tt_metal::CoreRangeSet grid(
                    tt::tt_metal::CoreRange({0, 0}, {grid_x - 1, grid_y - 1})
                );
                const auto row = tt::tt_metal::ShardOrientation::ROW_MAJOR;
                tt::tt_metal::ShardSpecBuffer spec(
                    grid, {1, words}, row, {1, words}, {cores, 1}
                );
                auto spread = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
                    tt::tt_metal::Shape({cores, words}),
                    tt::tt_metal::Shape({1, words}),
                    tt::tt_metal::Shape2D(1, words),
                    grid,
                    row,
                    tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D
                );
                local.buffer_type = BufferType::L1;
                local.sharding_args = (
                    tt::tt_metal::BufferShardingArgs(
                        std::move(spread),
                        std::move(spec),
                        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED
                    )
                );
            }
            buffer_ = MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = size},
                local,
                mesh_.get(),
                address
            );
        }

        const MeshBuffer& get() const {
            if (!buffer_) {
                throw std::runtime_error("the buffer was freed");
            }
            return *buffer_;
        }

        uint64_t address() const {
            return get().address();
        }

        uint64_t size() const {
            return get().size();
        }

        uint64_t page_size() const {
            return get().page_size();
        }

        uint64_t aligned_page_size() const {
            return get().get_backing_buffer()->aligned_page_size();
        }

        uint64_t num_pages() const {
            return get().get_backing_buffer()->num_pages();
        }

        // The kernels' TensorAccessorArgs compile-time arguments for this buffer.
        std::vector<uint32_t> accessor_args() const {
            return tt::tt_metal::TensorAccessorArgs(get()).get_compile_time_args();
        }

        // Writes the whole buffer and returns once it is on the card; waits first
        // for the cores of a launched program, as every buffer write does.
        void write(const Numbers8& data) {
            if (data.size() != size()) {
                throw std::invalid_argument("the data is not the buffer's size");
            }
            nb::gil_scoped_release release;
            mesh_->mesh_command_queue().enqueue_write_mesh_buffer(
                buffer_, data.data(), true
            );
        }

        // Reads the whole buffer, after the cores of a launched program are done.
        Bytes read() {
            auto* owned = new std::vector<uint8_t>(size());
            {
                nb::gil_scoped_release release;
                try {
                    mesh_->mesh_command_queue().enqueue_read_mesh_buffer(
                        owned->data(), buffer_, true
                    );
                } catch (...) {
                    delete owned;
                    throw;
                }
            }
            nb::capsule owner(owned, [](void* memory) noexcept {
                delete static_cast<std::vector<uint8_t>*>(memory);
            });
            const size_t shape[1] = {owned->size()};
            return Bytes(owned->data(), 1, shape, owner);
        }

        void free() {
            if (buffer_) {
                buffer_->deallocate();
                buffer_.reset();
            }
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
        std::shared_ptr<MeshBuffer> buffer_;
    };

    // Core ranges as (x0, y0, x1, y1) rectangles, ends included.
    using Rects = std::vector<std::array<uint32_t, 4>>;

    tt::tt_metal::CoreRangeSet core_set(const Rects& rects) {
        std::vector<tt::tt_metal::CoreRange> ranges;
        for (const auto& [x0, y0, x1, y1] : rects) {
            ranges.emplace_back(
                tt::tt_metal::CoreCoord{x0, y0}, tt::tt_metal::CoreCoord{x1, y1}
            );
        }
        return tt::tt_metal::CoreRangeSet(ranges);
    }

    tt::DataFormat data_format(const std::string& name) {
        static const std::vector<std::pair<std::string, tt::DataFormat>> formats = {
            {"bfloat16", tt::DataFormat::Float16_b},
            {"float32", tt::DataFormat::Float32},
            {"uint32", tt::DataFormat::UInt32},
            {"int32", tt::DataFormat::Int32},
            {"uint16", tt::DataFormat::UInt16},
            {"uint8", tt::DataFormat::UInt8},
            {"bfloat8_b", tt::DataFormat::Bfp8_b},
            {"bfloat4_b", tt::DataFormat::Bfp4_b},
        };
        for (const auto& [format_name, format] : formats) {
            if (format_name == name) {
                return format;
            }
        }
        throw std::invalid_argument("no data format " + name);
    }

    tt::tt_metal::KernelBuildOptLevel opt_level(const std::string& name) {
        using Level = tt::tt_metal::KernelBuildOptLevel;
        static const std::vector<std::pair<std::string, Level>> levels = {
            {"O0", Level::O0},
            {"O1", Level::O1},
            {"O2", Level::O2},
            {"O3", Level::O3},
            {"Os", Level::Os},
            {"Ofast", Level::Ofast},
            {"Oz", Level::Oz},
        };
        for (const auto& [level_name, level] : levels) {
            if (level_name == name) {
                return level;
            }
        }
        throw std::invalid_argument("no optimisation level " + name);
    }

    // The compute kernel's options: the fidelity's name, an unpack mode per circular
    // buffer (empty: the runtime's default), and the flags by field name.
    tt::tt_metal::ComputeConfigDescriptor compute_config(
        const std::optional<std::string>& fidelity,
        const std::vector<std::string>& unpack,
        const std::vector<std::pair<std::string, bool>>& flags
    ) {
        tt::tt_metal::ComputeConfigDescriptor config;
        if (fidelity) {
            using Fidelity = tt::tt_metal::MathFidelity;
            const std::vector<std::pair<std::string, Fidelity>> names = {
                {"LoFi", Fidelity::LoFi},
                {"HiFi2", Fidelity::HiFi2},
                {"HiFi3", Fidelity::HiFi3},
                {"HiFi4", Fidelity::HiFi4}
            };
            const auto found = std::find_if(
                names.begin(), names.end(), [&](const auto& entry) {
                    return entry.first == *fidelity;
                }
            );
            if (found == names.end()) {
                throw std::invalid_argument("no math fidelity " + *fidelity);
            }
            config.math_fidelity = found->second;
        }
        for (const auto& mode : unpack) {
            if (mode == "Default") {
                config.unpack_to_dest_mode.push_back(
                    tt::tt_metal::UnpackToDestMode::Default
                );
            } else if (mode == "UnpackToDestFp32") {
                config.unpack_to_dest_mode.push_back(
                    tt::tt_metal::UnpackToDestMode::UnpackToDestFp32
                );
            } else {
                throw std::invalid_argument("no unpack-to-dest mode " + mode);
            }
        }
        for (const auto& [name, on] : flags) {
            if (name == "fp32_dest_acc_en") {
                config.fp32_dest_acc_en = on;
            } else if (name == "dst_full_sync_en") {
                config.dst_full_sync_en = on;
            } else if (name == "bfp8_pack_precise") {
                config.bfp8_pack_precise = on;
            } else if (name == "math_approx_mode") {
                config.math_approx_mode = on;
            } else if (name == "enable_trisc2_rvv") {
                config.enable_trisc2_rvv = on;
            } else {
                throw std::invalid_argument("no compute option " + name);
            }
        }
        return config;
    }

    // One program on the grid, described piece by piece (circular buffers, kernels
    // with their per-core arguments), then built and launched as generic_op does.
    class Launch {
    public:
        explicit Launch(const Device& device) : mesh_(device.shared()) {
        }

        void cb(
            const Rects& cores,
            uint8_t index,
            const std::string& format,
            uint32_t page_size,
            uint32_t total_size,
            bool first
        ) {
            tt::tt_metal::CBDescriptor descriptor;
            descriptor.total_size = total_size;
            descriptor.core_ranges = core_set(cores);
            descriptor.format_descriptors.push_back(
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = index,
                    .data_format = data_format(format),
                    .page_size = page_size
                }
            );
            // the runtime places a core's buffers in this order from its L1 base
            if (first) {
                desc_.cbs.insert(desc_.cbs.begin(), std::move(descriptor));
            } else {
                desc_.cbs.push_back(std::move(descriptor));
            }
        }

        // A kernel from a file on `cores`, args (x, y, words) per core; processor 0 or
        // 1: data movement on that RISC-V and NoC `noc` (default: its own), 2: compute.
        void kernel(
            const std::string& source,
            const Rects& cores,
            uint32_t processor,
            const std::vector<uint32_t>& compile_args,
            const std::vector<std::pair<std::string, std::string>>& defines,
            const std::vector<std::string>& includes,
            const std::vector<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>>&
                args,
            const std::optional<std::string>& level,
            const std::optional<std::string>& fidelity,
            const std::vector<std::string>& unpack,
            const std::vector<std::pair<std::string, bool>>& flags,
            std::optional<uint32_t> noc
        ) {
            using tt::tt_metal::DataMovementProcessor;
            using tt::tt_metal::NOC;
            tt::tt_metal::KernelDescriptor kernel_descriptor;
            kernel_descriptor.kernel_source = source;
            kernel_descriptor
                .source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            kernel_descriptor.core_ranges = core_set(cores);
            kernel_descriptor.compile_time_args = compile_args;
            kernel_descriptor.defines = defines;
            for (const auto& path : includes) {
                kernel_descriptor.compiler_include_paths.emplace_back(path);
            }
            for (const auto& [x, y, words] : args) {
                kernel_descriptor.runtime_args.emplace_back(
                    tt::tt_metal::CoreCoord{x, y}, words
                );
            }
            if (level) {
                kernel_descriptor.opt_level = opt_level(*level);
            }
            if (noc && (processor == 2 || *noc > 1)) {
                throw std::invalid_argument("noc: 0 or 1, for a data movement kernel");
            }
            if (processor == 0) {
                kernel_descriptor.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = static_cast<NOC>(noc.value_or(NOC::RISCV_0_default))
                };
            } else if (processor == 1) {
                kernel_descriptor.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = static_cast<NOC>(noc.value_or(NOC::RISCV_1_default))
                };
            } else if (processor == 2) {
                kernel_descriptor.config = compute_config(fidelity, unpack, flags);
            } else {
                throw std::invalid_argument(
                    "processor: 0, 1 (data movement) or 2 (compute)"
                );
            }
            desc_.kernels.push_back(std::move(kernel_descriptor));
        }

        KernelBuild build() const {
            return kernel_build(*mesh_, desc_);
        }

        KernelRecipes recipes() const {
            return kernel_recipes(*mesh_, desc_);
        }

        // Launches the program and returns without waiting for it (the launch call
        // of generic_op on a 1 x 1 mesh); a later run launches the kept program again.
        void run() {
            namespace distributed = tt::tt_metal::distributed;
            if (!workload_) {
                workload_ = std::make_unique<distributed::MeshWorkload>();
                workload_->add_program(
                    distributed::MeshCoordinateRange(mesh_->shape()),
                    tt::tt_metal::Program(desc_)
                );
            }
            distributed::EnqueueMeshWorkload(
                mesh_->mesh_command_queue(), *workload_, false
            );
        }

        // Resets core (x, y): `channels` closed, `drops` emptied; restarts, reloads.
        // Seconds of reset, firmware (-1: not ready, -2: not closed), load.
        std::array<double, 3> reset_core(
            uint32_t x,
            uint32_t y,
            double timeout_s,
            const std::vector<uint32_t>& channels = {},
            const std::vector<uint32_t>& drops = {}
        ) {
            namespace tm = tt::tt_metal;
            if (!workload_) {
                throw std::runtime_error("reset_core: the program was not launched");
            }
            if (drops.size() % 4 != 0) {
                throw std::invalid_argument("reset_core: drops come in fours");
            }
            auto& context = tm::MetalContext::instance();
            const auto& hal = context.hal();
            const auto& cluster = context.get_cluster();
            auto* dev = chip(*mesh_);
            const auto tensix = tm::HalProgrammableCoreType::TENSIX;
            const uint32_t index = hal.get_programmable_core_type_index(tensix);
            const tm::CoreCoord logical{x, y};
            const auto core = (
                dev->virtual_core_from_logical_core(logical, tt::CoreType::WORKER)
            );
            const tt_cxy_pair where(dev->id(), core);
            auto& program = workload_->get_programs().begin()->second;
            auto* group = program.impl().kernels_on_core(logical, index);
            if (group == nullptr) {
                throw std::invalid_argument("reset_core: no kernels on that core");
            }
            nb::gil_scoped_release release;
            std::array<double, 3> took{};
            auto started = Clock::now();
            const auto lap = [&](size_t part) {
                const auto now = Clock::now();
                took[part] = std::chrono::duration<double>(now - started).count();
                started = now;
            };

            // The open's firmware start, for this core: firmware images, initial launch
            // messages, go INIT, read pointers 0, BRISC's boot jump, BRISC released.
            cluster.assert_risc_reset_at_core(where, tt::umd::RiscType::ALL);
            if (!close_connections(dev, where, channels, timeout_s)) {
                lap(0);
                return {took[0], -2.0, 0.0};
            }
            drop_lines(where, drops);
            lap(0);
            auto& builds = (
                tm::BuildEnvManager::get_instance(tm::extract_context_id(dev))
            );
            for (
                uint32_t kind = 0; kind < hal.get_processor_classes_count(tensix);
                kind++
            ) {
                const uint32_t states = (
                    builds.get_build_index_and_state_count(index, kind, true).second
                );
                for (uint32_t risc = 0; risc < states; risc++) {
                    const auto& image = (
                        tt::llrt::get_risc_binary(builds.get_firmware_binary_path(
                            dev->id(), index, kind, risc
                        ))
                    );
                    const uint64_t local = (
                        hal.get_jit_build_config(index, kind, risc).local_init_addr
                    );
                    image.process_spans([&](auto words, uint64_t at, uint32_t count) {
                        cluster.write_core(
                            &*words,
                            count * sizeof(uint32_t),
                            where,
                            hal.relocate_dev_addr(at, local, false)
                        );
                    });
                }
            }
            for (const auto& [at, bytes] : firmware_tables(dev)) {
                cluster.write_core(bytes.data(), bytes.size(), where, at);
            }
            auto factory = hal.get_dev_msgs_factory(tensix);
            auto initial = factory.create<tm::dev_msgs::launch_msg_t>();
            initial.view().kernel_config().mode() = tm::dev_msgs::DISPATCH_MODE_HOST;
            std::vector<std::byte> launches;
            for (
                size_t entry = 0; entry < tm::dev_msgs::launch_msg_buffer_num_entries;
                entry++
            ) {
                launches.insert(
                    launches.end(), initial.data(), initial.data() + initial.size()
                );
            }
            auto init = factory.create<tm::dev_msgs::go_msg_t>();
            init.view().signal() = tm::dev_msgs::RUN_MSG_INIT;
            const uint32_t zero = 0;
            const auto address = [&](tm::HalL1MemAddrType type) {
                return hal.get_dev_addr(tensix, type);
            };
            cluster.write_core_immediate(
                launches.data(),
                launches.size(),
                where,
                address(tm::HalL1MemAddrType::LAUNCH)
            );
            cluster.write_core_immediate(
                init.data(), init.size(), where, address(tm::HalL1MemAddrType::GO_MSG)
            );
            cluster.write_core_immediate(
                &zero,
                sizeof(zero),
                where,
                address(tm::HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR)
            );
            cluster.write_core_immediate(
                &zero, sizeof(zero), where, address(tm::HalL1MemAddrType::GO_MSG_INDEX)
            );
            const auto& boot = hal.get_jit_build_config(index, 0, 0);
            cluster.write_core(
                &boot.fw_launch_addr_value, sizeof(uint32_t), where, boot.fw_launch_addr
            );
            cluster.l1_barrier(dev->id());
            cluster.deassert_risc_reset_at_core(where, tt::umd::RiscType::BRISC);
            std::unordered_set<tm::CoreCoord> left{core};
            try {
                tt::llrt::internal_::wait_until_cores_done(
                    context,
                    dev->id(),
                    tm::dev_msgs::RUN_MSG_INIT,
                    left,
                    static_cast<int>(timeout_s * 1000.0)
                );
            } catch (const std::exception&) {
                return {took[0], -1.0, 0.0};
            }
            lap(1);

            // The launch's writes for this core (slow dispatch): kernel binaries,
            // circular buffer configs, semaphores, runtime arguments, launch and go.
            const uint64_t config_base = address(tm::HalL1MemAddrType::KERNEL_CONFIG);
            for (const auto kernel_id : group->kernel_ids) {
                program.impl().get_kernel(kernel_id)->configure(
                    dev,
                    logical,
                    static_cast<uint32_t>(config_base),
                    group->kernel_text_offsets.data()
                );
            }
            const auto& config = program.impl().get_program_config(index);
            const uint32_t max_buffers = hal.get_num_dataflow_buffers();
            std::vector<uint32_t> cbs(config.cb_size / sizeof(uint32_t));
            const uint32_t remote_offset = config.local_cb_size / sizeof(uint32_t);
            for (
                const auto& buffer : program.impl().circular_buffers_on_core(logical)
            ) {
                for (const uint32_t buffer_index : buffer->local_buffer_indices()) {
                    const uint32_t base = (
                        buffer_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG
                    );
                    const uint32_t pages = buffer->num_pages(buffer_index);
                    cbs[base] = buffer->address();
                    cbs[base + 1] = buffer->size();
                    cbs[base + 2] = pages;
                    cbs[base + 3] = buffer->size() / pages;
                }
                for (const uint32_t buffer_index : buffer->remote_buffer_indices()) {
                    const uint32_t base = (
                        remote_offset +
                        (max_buffers - 1 - buffer_index) *
                            UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG
                    );
                    cbs[base] = buffer->config_address();
                    cbs[base + 1] = buffer->page_size(buffer_index);
                }
            }
            if (!cbs.empty()) {
                cluster.write_core(
                    dev->id(), core, cbs, config_base + config.cb_offset
                );
            }
            program.impl().init_semaphores(*dev, logical, index);
            auto launch = group->launch_msg.view();
            auto kernel_config = launch.kernel_config();
            const uint64_t args_base = (
                static_cast<uint64_t>(kernel_config.kernel_config_base()[index]) +
                hal.get_l1_noc_offset(tensix)
            );
            for (const auto kernel_id : group->kernel_ids) {
                const auto& kernel = program.impl().get_kernel(kernel_id);
                const uint32_t processor = (
                    hal.get_processor_index(
                        kernel->get_kernel_programmable_core_type(),
                        kernel->get_kernel_processor_class(),
                        kernel->get_kernel_processor_type(0)
                    )
                );
                const auto offsets = kernel_config.rta_offset()[processor];
                const auto& args = kernel->runtime_args(logical);
                if (!args.empty()) {
                    cluster.write_core(
                        dev->id(), core, args, args_base + offsets.rta_offset()
                    );
                }
                const auto& common = kernel->common_runtime_args();
                if (!common.empty()) {
                    cluster.write_core(
                        dev->id(), core, common, args_base + offsets.crta_offset()
                    );
                }
            }
            cluster.l1_barrier(dev->id());
            kernel_config.host_assigned_id() = program.get_runtime_id();
            kernel_config.mode() = tm::dev_msgs::DISPATCH_MODE_HOST;
            cluster.write_core_immediate(
                launch.data(),
                launch.size(),
                where,
                address(tm::HalL1MemAddrType::LAUNCH)
            );
            tt_driver_atomics::sfence();
            cluster.write_core_immediate(
                group->go_msg.data(),
                group->go_msg.size(),
                where,
                address(tm::HalL1MemAddrType::GO_MSG)
            );
            lap(2);
            return took;
        }

        // Per core (x + y * grid x), the number of its kernels' group among the cores'
        // groups in core order, -1 for none: one reset list (every) serves a group.
        std::vector<int32_t> reset_groups() {
            namespace tm = tt::tt_metal;
            if (!workload_) {
                throw std::runtime_error("reset_groups: the program was not launched");
            }
            const auto& hal = tm::MetalContext::instance().hal();
            const uint32_t index = (
                hal.get_programmable_core_type_index(
                    tm::HalProgrammableCoreType::TENSIX
                )
            );
            auto& program = workload_->get_programs().begin()->second;
            const auto grid = mesh_->compute_with_storage_grid_size();
            std::vector<const void*> seen;
            std::vector<int32_t> out;
            for (uint32_t core = 0; core < grid.x * grid.y; core++) {
                const tm::CoreCoord logical{core % grid.x, core / grid.x};
                const void* group = program.impl().kernels_on_core(logical, index);
                const auto found = std::find(seen.begin(), seen.end(), group);
                if (group == nullptr) {
                    out.push_back(-1);
                } else if (found == seen.end()) {
                    out.push_back(static_cast<int32_t>(seen.size()));
                    seen.push_back(group);
                } else {
                    out.push_back(static_cast<int32_t>(found - seen.begin()));
                }
            }
            return out;
        }

        // The dispatcher's reset of core (x, y): reset_core's writes as a list of
        // km_dispatch.h; `ready` (not 0) zeroed, then waited for (& mask == value).
        nb::bytes reset_list(
            uint32_t x,
            uint32_t y,
            bool every,
            uint32_t ready,
            uint32_t ready_mask,
            uint32_t ready_value
        ) {
            namespace tm = tt::tt_metal;
            if (!workload_) {
                throw std::runtime_error("reset_list: the program was not launched");
            }
            const auto& hal = tm::MetalContext::instance().hal();
            auto* dev = chip(*mesh_);
            const auto tensix = tm::HalProgrammableCoreType::TENSIX;
            const uint32_t index = hal.get_programmable_core_type_index(tensix);
            const tm::CoreCoord logical{x, y};
            auto& program = workload_->get_programs().begin()->second;
            auto* group = program.impl().kernels_on_core(logical, index);
            if (group == nullptr) {
                throw std::invalid_argument("reset_list: no kernels on that core");
            }
            if (
                !program.impl()
                                .semaphores_on_core(logical, tt::CoreType::WORKER)
                .empty()
            ) {
                throw std::invalid_argument("reset_list: the core has semaphores");
            }
            std::vector<uint32_t> list;
            const auto entry = [&](
                uint32_t kind, uint64_t at, uint32_t size, uint32_t value
            ) {
                list.insert(list.end(), {kind, static_cast<uint32_t>(at), size, value});
            };
            const auto write = [&](uint64_t at, const void* data, size_t size) {
                if (size % 4 != 0 || at % 4 != 0 || at + size > KMD_L1_BYTES) {
                    throw std::runtime_error("reset_list: a write not of whole words");
                }
                entry(KMD_RESET_WRITE, at, static_cast<uint32_t>(size), 0);
                const size_t start = list.size();
                list.resize(start + (size + 15) / 16 * 4, 0);
                std::memcpy(list.data() + start, data, size);
            };
            const auto address = [&](tm::HalL1MemAddrType type) {
                return hal.get_dev_addr(tensix, type);
            };
            const auto image = [&](const ll_api::memory& memory, auto at) {
                memory.process_spans([&](auto words, uint64_t span, uint32_t count) {
                    write(at(span), &*words, count * sizeof(uint32_t));
                });
            };

            entry(KMD_RESET_ASSERT, 0, 0, 0);
            if (ready != 0) {  // the last run's word no longer reads as ready
                const uint32_t cleared = 0;
                write(ready, &cleared, 4);
            }
            auto& builds = (
                tm::BuildEnvManager::get_instance(tm::extract_context_id(dev))
            );
            for (
                uint32_t kind = 0; kind < hal.get_processor_classes_count(tensix);
                kind++
            ) {
                const uint32_t states = (
                    builds.get_build_index_and_state_count(index, kind, true).second
                );
                for (uint32_t risc = 0; risc < states; risc++) {
                    const uint64_t local = (
                        hal.get_jit_build_config(index, kind, risc).local_init_addr
                    );
                    image(
                        tt::llrt::get_risc_binary(builds.get_firmware_binary_path(
                            dev->id(), index, kind, risc
                        )),
                        [&](uint64_t span) {
                            return hal.relocate_dev_addr(span, local, false);
                        }
                    );
                }
            }
            for (const auto& [at, bytes] : firmware_tables(dev)) {
                write(at, bytes.data(), bytes.size());
            }
            auto factory = hal.get_dev_msgs_factory(tensix);
            auto initial = factory.create<tm::dev_msgs::launch_msg_t>();
            initial.view().kernel_config().mode() = tm::dev_msgs::DISPATCH_MODE_HOST;
            std::vector<std::byte> launches;
            for (
                size_t entry = 0; entry < tm::dev_msgs::launch_msg_buffer_num_entries;
                entry++
            ) {
                launches.insert(
                    launches.end(), initial.data(), initial.data() + initial.size()
                );
            }
            write(
                address(tm::HalL1MemAddrType::LAUNCH), launches.data(), launches.size()
            );
            auto go = factory.create<tm::dev_msgs::go_msg_t>();
            go.view().signal() = tm::dev_msgs::RUN_MSG_INIT;
            write(address(tm::HalL1MemAddrType::GO_MSG), go.data(), go.size());
            const uint32_t zero = 0;
            write(address(tm::HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR), &zero, 4);
            write(address(tm::HalL1MemAddrType::GO_MSG_INDEX), &zero, 4);
            const auto& boot = hal.get_jit_build_config(index, 0, 0);
            write(boot.fw_launch_addr, &boot.fw_launch_addr_value, 4);
            entry(KMD_RESET_RELEASE, 0, 0, 0);
            // The firmware's DONE: the go message's signal byte (its mask from a go
            // message whose signal is all ones) reads RUN_MSG_DONE.
            auto mask = factory.create<tm::dev_msgs::go_msg_t>();
            mask.view().signal() = 0xFF;
            uint32_t mask_word = 0, done_word = 0;
            std::memcpy(&mask_word, mask.data(), 4);
            go.view().signal() = tm::dev_msgs::RUN_MSG_DONE;
            std::memcpy(&done_word, go.data(), 4);
            entry(
                KMD_RESET_WAIT,
                address(tm::HalL1MemAddrType::GO_MSG),
                mask_word,
                done_word & mask_word
            );

            const uint64_t config_base = address(tm::HalL1MemAddrType::KERNEL_CONFIG);
            const uint64_t key = (
                builds.get_device_build_env(dev->build_id()).build_key()
            );
            for (const auto kernel_id : group->kernel_ids) {
                const auto& kernel = program.impl().get_kernel(kernel_id);
                const auto& binaries = kernel->binaries(key);
                for (size_t binary = 0; binary < binaries.size(); binary++) {
                    const uint32_t processor = (
                        hal.get_processor_index(
                            kernel->get_kernel_programmable_core_type(),
                            kernel->get_kernel_processor_class(),
                            kernel->get_kernel_processor_type(static_cast<int>(binary))
                        )
                    );
                    const uint64_t at = (
                        config_base + group->kernel_text_offsets[processor]
                    );
                    image(*binaries[binary], [&](uint64_t) {
                        return at;
                    });
                }
            }
            const auto& config = program.impl().get_program_config(index);
            const uint32_t max_buffers = hal.get_num_dataflow_buffers();
            std::vector<uint32_t> cbs(config.cb_size / sizeof(uint32_t));
            const uint32_t remote_offset = config.local_cb_size / sizeof(uint32_t);
            for (
                const auto& buffer : program.impl().circular_buffers_on_core(logical)
            ) {
                for (const uint32_t buffer_index : buffer->local_buffer_indices()) {
                    const uint32_t base = (
                        buffer_index * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG
                    );
                    const uint32_t pages = buffer->num_pages(buffer_index);
                    cbs[base] = buffer->address();
                    cbs[base + 1] = buffer->size();
                    cbs[base + 2] = pages;
                    cbs[base + 3] = buffer->size() / pages;
                }
                for (const uint32_t buffer_index : buffer->remote_buffer_indices()) {
                    const uint32_t base = (
                        remote_offset +
                        (max_buffers - 1 - buffer_index) *
                            UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG
                    );
                    cbs[base] = buffer->config_address();
                    cbs[base + 1] = buffer->page_size(buffer_index);
                }
            }
            if (!cbs.empty()) {
                write(config_base + config.cb_offset, cbs.data(), cbs.size() * 4);
            }
            auto launch = group->launch_msg.view();
            auto kernel_config = launch.kernel_config();
            const uint64_t args_base = (
                static_cast<uint64_t>(kernel_config.kernel_config_base()[index]) +
                hal.get_l1_noc_offset(tensix)
            );
            // every: each core's runtime arguments, core i (x + y * grid x) the i-th
            // part, zero for a core with other kernels (reset_groups: not this list's).
            const auto grid = mesh_->compute_with_storage_grid_size();
            const uint32_t cores = static_cast<uint32_t>(grid.x * grid.y);
            std::vector<bool> mine(cores);
            for (uint32_t core = 0; core < cores; core++) {
                const tm::CoreCoord other{core % grid.x, core / grid.x};
                mine[core] = program.impl().kernels_on_core(other, index) == group;
            }
            const auto each = [&](uint64_t at, tm::Kernel& kernel) {
                size_t size = 0;
                for (uint32_t core = 0; core < cores; core++) {
                    if (mine[core]) {
                        const tm::CoreCoord other{core % grid.x, core / grid.x};
                        size = std::max(size, kernel.runtime_args(other).size() * 4);
                    }
                }
                const size_t padded = (size + 15) / 16 * 16;
                entry(KMD_RESET_EACH, at, static_cast<uint32_t>(size), cores);
                const size_t start = list.size();
                list.resize(start + padded / 4 * cores, 0);
                for (uint32_t core = 0; core < cores; core++) {
                    if (!mine[core]) {
                        continue;
                    }
                    const auto& args = kernel.runtime_args(
                        tm::CoreCoord{core % grid.x, core / grid.x}
                    );
                    std::copy(
                        args.begin(),
                        args.end(),
                        list.begin() + start + core * padded / 4
                    );
                }
            };
            for (const auto kernel_id : group->kernel_ids) {
                const auto& kernel = program.impl().get_kernel(kernel_id);
                const uint32_t processor = (
                    hal.get_processor_index(
                        kernel->get_kernel_programmable_core_type(),
                        kernel->get_kernel_processor_class(),
                        kernel->get_kernel_processor_type(0)
                    )
                );
                const auto offsets = kernel_config.rta_offset()[processor];
                if (every) {
                    each(args_base + offsets.rta_offset(), *kernel);
                } else if (
                    const auto& args = kernel->runtime_args(logical); !args.empty()
                ) {
                    write(
                        args_base + offsets.rta_offset(), args.data(), args.size() * 4
                    );
                }
                const auto& common = kernel->common_runtime_args();
                if (!common.empty()) {
                    write(
                        args_base + offsets.crta_offset(),
                        common.data(),
                        common.size() * 4
                    );
                }
            }
            kernel_config.host_assigned_id() = program.get_runtime_id();
            kernel_config.mode() = tm::dev_msgs::DISPATCH_MODE_HOST;
            write(address(tm::HalL1MemAddrType::LAUNCH), launch.data(), launch.size());
            write(
                address(tm::HalL1MemAddrType::GO_MSG),
                group->go_msg.data(),
                group->go_msg.size()
            );
            if (ready != 0) {  // the kernels started: nothing is put in the core early
                entry(KMD_RESET_WAIT, ready, ready_mask, ready_value);
            }
            entry(KMD_RESET_END, 0, 0, 0);
            list.resize((list.size() + 15) / 16 * 16, 0);
            return (
                nb::bytes(
                    reinterpret_cast<const char*>(list.data()),
                    list.size() * sizeof(uint32_t)
                )
            );
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
        tt::tt_metal::ProgramDescriptor desc_;
        std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload_;
    };

    // [n, 4] events (index, req, status, ticks) as a numpy array that owns its words.
    Events as_array(std::vector<uint32_t>&& words) {
        auto* owned = new std::vector<uint32_t>(std::move(words));
        nb::capsule owner(owned, [](void* memory) noexcept {
            delete static_cast<std::vector<uint32_t>*>(memory);
        });
        const size_t shape[2] = {owned->size() / 4, 4};
        return Events(owned->data(), 2, shape, owner);
    }

    // The X280 runtime on one card and the dispatcher's port on cluster 0: its
    // LIM layout, the host command ring, the event ring and the result block.
    class Dispatch {
    public:
        Dispatch(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz) :
            mesh_(std::move(mesh_device)) {
            opened_ = std::make_shared<x280::RuntimeOpenResult>(x280::X280Runtime::open(
                mesh_,
                x280::RuntimeOpenConfig{
                    .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
                }
            ));
            if (!opened_->ready()) {
                const std::string why = opened_->diagnostic;
                opened_.reset();
                throw std::runtime_error("X280 runtime not ready: " + why);
            }
            try {
                boot(pll_mhz);
                locate();
            } catch (...) {
                close();
                throw;
            }
        }

        // An island's dispatch: the mesh's shared runtime, this island's chip only.
        Dispatch(
            std::shared_ptr<MeshDevice> island,
            std::shared_ptr<x280::RuntimeOpenResult> shared,
            uint32_t pll_mhz
        ) :
            mesh_(std::move(island)), opened_(std::move(shared)) {
            if (!opened_ || !opened_->ready()) {
                throw std::runtime_error("the islands' X280 runtime is not open");
            }
            chip_id_ = chip(*mesh_)->id();
            try {
                boot(pll_mhz);
                locate();
            } catch (...) {
                close();
                throw;
            }
        }

        // A dispatch on X280 cluster `cluster` of `first`'s chip, beside it: its
        // runtime, its tiles booted and clocked by `first` (a card split into islands).
        Dispatch(const Dispatch& first, uint32_t cluster) :
            mesh_(first.mesh_),
            opened_(first.opened_),
            chip_id_(first.chip_id_),
            cluster_(cluster) {
            require_open();
            if (cluster_ == 0 || cluster_ >= tile_ids().size()) {
                throw std::runtime_error("a dispatch beside another: clusters 1 to 3");
            }
            try {
                locate();
            } catch (...) {
                close();
                throw;
            }
        }

        ~Dispatch() {
            close();
        }

        bool is_open() const {
            return opened_ != nullptr;
        }

        bool running() const {
            return session_.has_value();
        }

        // Whether its cluster runs its idle image and no session: an image can start.
        bool idle() const {
            return (
                opened_ && !session_ && tile_ != nullptr &&
                tile_->inspect().resident_idle_live
            );
        }

        // Stops a running kernel, releases the host buffer and the window, closes the
        // X280 runtime. Returns whether a running kernel was stopped cleanly.
        bool close() {
            close_failed_.clear();
            bool cleaned = true;
            for (uint32_t cluster = 1; cluster < work_.size(); cluster++) {
                cleaned = work_abort(cluster) && cleaned;
            }
            cleaned = abort() && cleaned;
            win_.reset();
            buffer_.reset();
            host_ = nullptr;
            tile_ = nullptr;
            opened_.reset();
            return cleaned;
        }

        // Core index -> translated NoC0 x | y << 16; core index = y * grid_x + x.
        std::vector<uint32_t> core_table() const {
            std::vector<uint32_t> table(KM_CORES_MAX, 0);
            for (uint32_t y = 0; y < grid_y_; y++) {
                for (uint32_t x = 0; x < grid_x_; x++) {
                    const auto core = translated(x, y);
                    table[y * grid_x_ + x] = core[0] | (core[1] << 16);
                }
            }
            return table;
        }

        // Picks the run and the tag, zeroes LIM layout and host buffer, writes the
        // argument block (state_ddr: the kernel's state; upload_bytes: -1 all free).
        uint32_t prepare(
            uint32_t depth,
            uint32_t host_depth,
            uint32_t event_depth,
            double t_start_s,
            double t_command_s,
            double t_idle_s,
            uint64_t state_ddr,
            uint64_t work_cmds,
            uint64_t work_depth,
            uint64_t work_event_depth,
            const std::vector<uint64_t>& work_noc,
            const std::vector<uint64_t>& work_noc_addr,
            int64_t upload_bytes,
            uint64_t messages_handler,
            const std::vector<uint8_t>& programs,
            uint64_t island,
            const std::vector<uint8_t>& link,
            uint32_t first_row,
            uint32_t rows
        ) {
            require_open();
            rows = rows == 0 ? grid_y_ - std::min(first_row, grid_y_) : rows;
            if (rows == 0 || first_row + rows > grid_y_) {
                throw std::runtime_error("rows: a range of the grid's rows");
            }
            if (link.size() > KMD_LINK_COUNT || link.size() % 4 != 0) {
                throw std::runtime_error(
                    "link: the link line and the island table, at most KMD_LINK_COUNT "
                    "bytes"
                );
            }
            if (programs.size() > KM_CORES_MAX) {
                throw std::runtime_error(
                    "programs: a byte per core, at most KM_CORES_MAX"
                );
            }
            if (session_) {
                throw std::runtime_error("a kernel is running");
            }
            if (
                depth < 1 || depth > KM_RING_SLOTS || !power_of_two(host_depth) ||
                host_depth > KMD_HOST_DEPTH_MAX || !power_of_two(event_depth) ||
                event_depth > kEventDepthMax
            ) {
                throw std::runtime_error(
                    "depth out of range, or host_depth or event_depth not a power of "
                    "two in range"
                );
            }
            if (state_ddr == 0 || state_ddr % 64 != 0) {
                throw std::runtime_error("state_ddr is not a nonzero multiple of 64");
            }
            // The work clusters' rings: per cluster 1 to 3 its command lines and read
            // line after the layout; work_noc and work_noc_addr per cluster (0: none).
            const uint64_t work_bytes = (
                work_cmds ? KMD_WORK_CLUSTERS * KMD_WORK_RING_BYTES(work_depth) : 0
            );
            if (
                work_noc.size() > KMD_WORK_CLUSTERS ||
                work_noc_addr.size() != work_noc.size()
            ) {
                throw std::runtime_error("work_noc and work_noc_addr: one per cluster");
            }
            if (
                work_cmds &&
                (
                    work_cmds % 64 != 0 || work_cmds < KMD_LIM_BYTES ||
                    base_ > tile_->available_lim().end ||
                    work_cmds > tile_->available_lim().end - base_ ||
                    work_bytes > tile_->available_lim().end - base_ - work_cmds
                )
            ) {
                throw std::runtime_error("the work ring's lines are not inside LIM");
            }
            // The upload region ends at the work ring, at the end of LIM or of the
            // window.
            const uint64_t end = (
                std::min<uint64_t>(
                    work_cmds ? base_ + work_cmds : tile_->available_lim().end,
                    base_ + win_->get_size()
                )
            );
            const uint64_t upload_start = base_ + KMD_LIM_UPLOAD;
            const uint64_t room = (
                end > upload_start ? (end - upload_start) & ~63ull : 0
            );
            if (
                upload_bytes > static_cast<int64_t>(room) ||
                (upload_bytes >= 0 && upload_bytes % 64 != 0)
            ) {
                throw std::runtime_error(
                    "upload_bytes: a multiple of 64, at most the room"
                );
            }
            upload_bytes_ = (
                upload_bytes < 0 ? room : static_cast<uint64_t>(upload_bytes)
            );
            depth_ = depth;
            host_depth_ = host_depth;
            event_depth_ = event_depth;
            std::random_device entropy;
            run_ = (
                (0x6b6d680000000000ull ^ (static_cast<uint64_t>(entropy()) << 16)) +
                entropy()
            );
            do {
                tag_ = (entropy() & 0xFFFFu) | 1u;
            } while (tag_ == 0xFFFFu);
            next_ = 1;
            seen_ = published_ = 0;
            queue_.clear();
            head_ = 0;
            std::memset(const_cast<uint32_t*>(host_), 0, kHostBufferBytes);
            tile_->write_available_lim(base_, std::vector<uint8_t>(KMD_LIM_ZEROED, 0));
            if (work_cmds) {
                tile_->write_available_lim(
                    base_ + work_cmds, std::vector<uint8_t>(work_bytes, 0)
                );
            }

            kmd_args_t arguments{};
            arguments.run_value = run_;
            arguments.depth = depth_;
            arguments.host_depth = host_depth_;
            arguments.event_depth = event_depth_;
            arguments.columns = grid_x_;
            arguments.cores = grid_x_ * grid_y_;
            arguments.panels = panels_.size();
            for (size_t panel = 0; panel < panels_.size(); panel++) {
                const auto first_corner = translated(panels_[panel][0], first_row);
                const auto last_corner = (
                    translated(panels_[panel][1], first_row + rows - 1)
                );
                arguments.rect[panel][0] = first_corner[0];
                arguments.rect[panel][1] = first_corner[1];
                arguments.rect[panel][2] = last_corner[0];
                arguments.rect[panel][3] = last_corner[1];
            }
            arguments.host_noc_addr = host_noc_;
            arguments.pcie_x = pcie_x_;
            arguments.pcie_y = pcie_y_;
            arguments.t_start = static_cast<uint64_t>(t_start_s * kMtimeHz);
            arguments.t_command = static_cast<uint64_t>(t_command_s * kMtimeHz);
            arguments.t_idle = static_cast<uint64_t>(t_idle_s * kMtimeHz);
            arguments.state_ddr = state_ddr;
            arguments.work_cmds = work_cmds;
            arguments.work_depth = work_depth;
            arguments.work_event_depth = work_event_depth;
            for (
                size_t work_cluster = 0; work_cluster < work_noc.size(); work_cluster++
            ) {
                arguments.work_noc[work_cluster] = work_noc[work_cluster];
                arguments.work_noc_addr[work_cluster] = work_noc_addr[work_cluster];
            }
            arguments.version = KMD_ARGS_VERSION;
            arguments.upload_bytes = upload_bytes_;
            arguments.messages_handler = messages_handler;
            arguments.programs = programs.empty() ? 0 : 1;
            arguments.island = island;
            if (!programs.empty()) {
                std::vector<uint8_t> held(KM_CORES_MAX, 0xFF);
                std::copy(programs.begin(), programs.end(), held.begin());
                tile_->write_available_lim(base_ + KMD_LIM_PROGRAMS, held);
            }
            const std::vector<uint32_t> table = core_table();  // the inspect's windows
            std::vector<uint8_t> coordinates(4 * table.size());
            std::memcpy(coordinates.data(), table.data(), coordinates.size());
            tile_->write_available_lim(base_ + KMD_LIM_COORDS, coordinates);
            if (!link.empty()) {  // the dispatcher reads it after the hello
                tile_->write_available_lim(base_ + KMD_LIM_LINK, link);
            }
            std::vector<uint8_t> block(kArgsBytes, 0);
            std::memcpy(block.data(), &arguments, sizeof(arguments));
            tile_->write_available_lim(base_ + KMD_LIM_ARGS, block);
            return tag_;
        }

        // Starts the kernel image on cluster 0 with the prepared argument block.
        void start(const std::string& image) {
            require_open();
            if (session_) {
                throw std::runtime_error("a kernel is running");
            }
            x280::ActiveRunRequest request;
            request.image_path = image;
            request.run_value = run_;
            request.args_addr = base_ + KMD_LIM_ARGS;
            request.args_bytes = kArgsBytes;
            const auto enqueued = x280_runtime().enqueue(*tile_, request);
            if (!enqueued.session) {
                throw std::runtime_error(
                    "X280 enqueue failed: " + x280::describe(enqueued)
                );
            }
            session_ = enqueued.session;
            aborted_.reset();
        }

        // Stores whole 64-byte lines at `offset` of the upload region, then a fence.
        void upload(uint64_t offset, const nb::bytes& data) {
            require_open();
            const uint64_t bytes = data.size();
            if (
                offset % 64 != 0 || bytes % 64 != 0 || offset > upload_bytes_ ||
                bytes > upload_bytes_ - offset
            ) {
                throw std::runtime_error("upload: whole lines inside the region");
            }
            settle();
            win_->write_block(KMD_LIM_UPLOAD + offset, data.c_str(), bytes);
            _mm_sfence();
            fenced_ = Clock::now();
        }

        // Stores the read count and a fence: an engine's posts wait for event room.
        void publish() {
            require_open();
            if (seen_ != published_) {
                settle();
                store_read();
                _mm_sfence();
                fenced_ = Clock::now();
            }
        }

        uint32_t host_word(uint32_t word) const {
            require_open();
            if (word >= KMD_HOST_RING / 4) {
                throw std::runtime_error("no such host word");
            }
            return host_[word];
        }

        // Waits until the kernel reports READY or a final state; returns that state, or
        // 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            require_open();
            const auto deadline = after(timeout_s);
            for (;;) {
                const uint32_t state = host_[KMD_HOST_STATE];
                if (state >= KMD_STATE_READY) {
                    return state;
                }
                if (Clock::now() > deadline) {
                    return 0;
                }
            }
        }

        // Commands that fit now: free host ring slots, and room in the event ring, so
        // that unread events plus commands in flight never exceed it. Counters wrap.
        uint32_t room() const {
            require_open();
            const uint32_t taken = host_[KMD_HOST_CONSUMED];
            const int64_t ring = (
                static_cast<int64_t>(host_depth_) -
                static_cast<int32_t>(next_ - 1u - taken)
            );
            const int64_t events = (
                static_cast<int64_t>(event_depth_) -
                static_cast<int32_t>(next_ - 1u - seen_)
            );
            return static_cast<uint32_t>(std::max<int64_t>(0, std::min(ring, events)));
        }

        // Stores the read count line (KMD_LIM_READ: events the host has read), without
        // a fence. The kernel takes a command only while its events fit the ring.
        void store_read() {
            alignas(64) uint32_t line[16] = {seen_};
            win_->write_block(KMD_LIM_READ, line, 64);
            published_ = seen_;
        }

        // While waiting: the read count and a fence, only when the kernel may hold a
        // command for it (command i waits only while i - 1 - published reaches depth).
        void publish_if_held() {
            if (
                seen_ != published_ && next_ - 1u - published_ >= event_depth_ &&
                Clock::now() >= settled()
            ) {
                store_read();
                _mm_sfence();
                fenced_ = Clock::now();
            }
        }

        // Spins until settle_microseconds_ passed since the last fence: a store that
        // follows a fence too closely often stalls about 12 us.
        void settle() const {
            const auto until = settled();
            while (Clock::now() < until) {
                _mm_pause();
            }
        }

        Clock::time_point settled() const {
            return (
                fenced_ +
                std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double, std::micro>(settle_microseconds_)
                )
            );
        }

        // Stores the first k commands that fit as commands next .. next + k - 1 (number
        // in sequence and the landing words), the read count, one fence. Returns k.
        uint32_t push(const uint32_t* commands, size_t count) {
            const uint32_t stored = (
                static_cast<uint32_t>(std::min<size_t>(count, room()))
            );
            store(commands, stored);
            return stored;
        }

        // Queues n commands for wait(), stored in order as the ring has room (the
        // kernel orders KM_FLAG_EXCLUSIVE ones). Returns the number queued.
        size_t enqueue(const uint32_t* commands, size_t count) {
            require_open();
            queue_.insert(queue_.end(), commands, commands + 16 * count);
            return queued();
        }

        size_t queued() const {
            return queue_.size() / 16 - head_;
        }

        // Stores queued commands up to `upto` and reads events until all `wanted` have
        // one, a non-DONE event or a final kernel state arrives, or timeout_s passes.
        std::vector<uint32_t> wait(
            const std::vector<uint32_t>& wanted, uint32_t upto, double timeout_s
        ) {
            require_open();
            // wanted is in command order: offsets from its first fit one table.
            const uint32_t first_wanted = wanted.empty() ? 0 : wanted.front();
            std::vector<uint8_t> need(
                wanted.empty() ? 0 : wanted.back() - first_wanted + 1u, 0
            );
            size_t left = 0;
            for (const uint32_t command : wanted) {
                left += need[command - first_wanted] == 0;
                need[command - first_wanted] = 1;
            }
            const auto deadline = after(timeout_s);
            std::vector<uint32_t> out;
            for (;;) {
                store_queued(upto);
                const size_t had = out.size();
                read_events(out);
                publish_if_held();
                bool other = false;
                for (size_t event = had; event < out.size(); event += 4) {
                    const uint32_t offset = out[event + 1] - first_wanted;
                    if (offset < need.size() && need[offset] != 0) {
                        need[offset] = 0;
                        left--;
                    }
                    other = other || (out[event + 2] & 0xFFu) != KM_EVENT_DONE;
                }
                if (
                    left == 0 || other || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > deadline
                ) {
                    return out;
                }
                if (out.size() == had) {
                    _mm_pause();
                }
            }
        }

        // Each dispatcher's batch up to its upto (as its wait(.., 0.0)), back to back:
        // every line, every read count, one fence; then each one's events.
        static std::vector<std::vector<uint32_t>> store_together(
            const std::vector<Dispatch*>& ports, const std::vector<uint32_t>& uptos
        ) {
            if (ports.size() != uptos.size()) {
                throw std::runtime_error("a dispatcher's upto each");
            }
            std::vector<uint32_t> counts(ports.size());
            for (size_t i = 0; i < ports.size(); i++) {
                ports[i]->require_open();
                counts[i] = ports[i]->batch_upto(uptos[i]);
                if (counts[i] != 0) {
                    ports[i]->settle();
                }
            }
            for (size_t i = 0; i < ports.size(); i++) {
                Dispatch& port = *ports[i];
                port.store_lines(port.queue_.data() + 16 * port.head_, counts[i]);
            }
            for (size_t i = 0; i < ports.size(); i++) {
                if (counts[i] != 0) {
                    ports[i]->store_read();
                }
            }
            _mm_sfence();
            const auto now = Clock::now();
            std::vector<std::vector<uint32_t>> out(ports.size());
            for (size_t i = 0; i < ports.size(); i++) {
                Dispatch& port = *ports[i];
                if (counts[i] != 0) {
                    port.fenced_ = now;
                    port.next_ += counts[i];
                    port.taken(counts[i]);
                }
                port.read_events(out[i]);
                port.publish_if_held();
            }
            return out;
        }

        // The events that have arrived since the last call, in order.
        std::vector<uint32_t> poll() {
            require_open();
            std::vector<uint32_t> out;
            read_events(out);
            return out;
        }

        // Appends the events that have arrived to out, in order.
        void read_events(std::vector<uint32_t>& out) {
            const auto* ring = (
                reinterpret_cast<const volatile uint32_t*>(
                    reinterpret_cast<const volatile uint8_t*>(host_) + KMD_HOST_RING
                )
            );
            for (;;) {
                // A sequence lock: index, fields, index again; a change means the
                // event was being written, so it is read again.
                const uint32_t number = seen_ + 1;
                const volatile uint32_t* line = ring + (number % event_depth_) * 4;
                if (line[0] != number) {
                    return;
                }
                _mm_lfence();
                const uint32_t request = line[1], status = line[2], ticks = line[3];
                _mm_lfence();
                if (line[0] != number) {
                    continue;
                }
                out.insert(out.end(), {number, request, status, ticks});
                seen_ = number;
            }
        }

        // poll() until at least one event arrives, the kernel reaches a final state, or
        // timeout_s passes.
        std::vector<uint32_t> wait_events(double timeout_s) {
            const auto deadline = after(timeout_s);
            for (;;) {
                auto out = poll();
                publish_if_held();
                if (
                    !out.empty() || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > deadline
                ) {
                    auto rest = poll();
                    out.insert(out.end(), rest.begin(), rest.end());
                    return out;
                }
                _mm_pause();
            }
        }

        // Timed wait for the kernel's core result; stops the session, reads the result
        // block: result ready, core result, cleaned, what failed, words, cores.
        nb::dict finish(double timeout_s) {
            require_open();
            if (!session_) {
                throw std::runtime_error("no kernel running");
            }
            bool ready = false, cleaned = false;
            uint64_t result = 0;
            std::string failed;
            {
                nb::gil_scoped_release release;
                const auto wait_result = (
                    x280_runtime().session(*session_).wait_until_result_ready(
                        millis(timeout_s)
                    )
                );
                const auto done = (
                    x280_runtime().cleanup(*session_, std::chrono::seconds(5))
                );
                session_.reset();
                ready = wait_result.succeeded();
                if (ready) {
                    result = wait_result.hartstates[0].state.result;
                } else {
                    failed = x280::describe(wait_result);
                }
                cleaned = done.cleaned_up();
                if (!cleaned) {
                    failed += (failed.empty() ? "" : "; ") + x280::describe(done);
                }
            }
            const auto block = (
                tile_->read_available_lim(base_ + KMD_LIM_RESULT, 4 * KMD_R_WORDS)
            );
            std::vector<uint32_t> words(KMD_R_WORDS);
            std::memcpy(words.data(), block.data(), 4 * KMD_R_WORDS);
            const auto core_block = (
                tile_->read_available_lim(base_ + KMD_LIM_CORES, 4 * KMD_H_WORDS)
            );
            std::vector<uint32_t> x280_cores(KMD_H_WORDS);
            std::memcpy(x280_cores.data(), core_block.data(), 4 * KMD_H_WORDS);
            nb::dict out;
            out["result_ready"] = ready;
            out["result"] = ready ? nb::object(nb::int_(result)) : nb::none();
            out["cleaned"] = cleaned;
            out["failed"] = failed;
            out["words"] = words;
            out["x280_cores"] = x280_cores;  // the core block (KMD_H_*)
            return out;
        }

        // Stops the session now if one runs (any failure path): whether the runtime
        // restored the tile (else close_failed says why); aborted() keeps the blocks.
        bool abort() {
            if (!session_ || !opened_) {
                session_.reset();
                return true;
            }
            const auto done = (
                x280_runtime().cleanup(*session_, std::chrono::seconds(5))
            );
            session_.reset();
            note_failed("cluster 0", done);
            try {
                aborted_ = {
                    lim_words(KMD_LIM_RESULT, KMD_R_WORDS),
                    lim_words(KMD_LIM_CORES, KMD_H_WORDS)
                };
            } catch (const std::exception& error) {
                note_failed("cluster 0's blocks", error.what());
            }
            return done.cleaned_up();
        }

        // After abort() stopped a kernel: {words, cores} as it left them, or None.
        nb::object aborted() const {
            if (!aborted_) {
                return nb::none();
            }
            nb::dict out;
            out["words"] = aborted_->first;
            out["x280_cores"] = aborted_->second;  // the core block (KMD_H_*)
            return out;
        }

        // While the kernel runs: the result and core blocks (core 0's pass marker) read
        // twice gap_s apart, and the runtime's view of each core.
        nb::dict probe(double gap_s) {
            require_open();
            nb::dict out;
            out["running"] = session_.has_value();
            if (!session_) {
                return out;
            }
            std::vector<uint32_t> before, after, words;
            std::string x280_states, failed;
            {
                nb::gil_scoped_release release;
                try {
                    before = lim_words(KMD_LIM_CORES, KMD_H_WORDS);
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(std::clamp(gap_s, 0.0, 1.0))
                    );
                    after = lim_words(KMD_LIM_CORES, KMD_H_WORDS);
                    words = lim_words(KMD_LIM_RESULT, KMD_R_WORDS);
                } catch (const std::exception& error) {
                    failed = std::string("LIM not readable: ") + error.what();
                }
                try {
                    x280_states = (
                        x280::describe(x280_runtime().session(*session_).inspect())
                    );
                } catch (const std::exception& error) {
                    x280_states = (
                        std::string("core states not readable: ") + error.what()
                    );
                }
            }
            if (failed.empty()) {
                out["marker_before"] = before[KMD_H_MARKER0];
                out["marker_after"] = after[KMD_H_MARKER0];
                out["words"] = words;
                out["x280_cores"] = after;  // the core block (KMD_H_*)
            }
            out["failed"] = failed;
            out["x280_states"] = x280_states;  // the runtime's view of each core
            return out;
        }

        // While the kernel runs: the inspect block's header and its first `count`
        // records, and the commands given up so far (the core block's word).
        nb::dict inspect(uint32_t count) {
            require_open();
            if (!session_ || count > KM_CORES_MAX) {
                throw std::runtime_error(
                    "no kernel is running, or more records than KM_CORES_MAX"
                );
            }
            std::vector<uint32_t> block;
            uint32_t given_up = 0;
            const uint32_t words = (KMD_INSPECT_HEAD + count * KMD_INSPECT_RECORD) / 4;
            {
                nb::gil_scoped_release release;
                block = lim_words(KMD_LIM_INSPECT, words);
                given_up = lim_words(KMD_LIM_CORES, KMD_H_WORDS)[KMD_H_GIVEN_UP];
            }
            nb::dict out;
            out["block"] = as_array(std::move(block));
            out["given_up"] = given_up;
            return out;
        }

        // Work clusters 1 to 3: one image each, started with an argument block at the
        // start of that cluster's available LIM. Cluster 0 is the dispatcher's.
        uint32_t clusters() const {
            return static_cast<uint32_t>(work_.size());
        }

        // The cluster's hardware index, NoC0 coordinate and available LIM.
        nb::dict work_info(uint32_t cluster) const {
            const Work& cluster_state = work(cluster, true);
            nb::dict out;
            out["hw_index"] = cluster_state.tile->id().hw_index;
            out["noc_x"] = cluster_state.tile->id().noc_x;
            out["noc_y"] = cluster_state.tile->id().noc_y;
            // The coordinate another cluster's window aims at (as for cluster 0's).
            const auto virtual_core = (
                x280_runtime().virtual_core_from_tile(
                    cluster_state.tile->id(), tt::tt_metal::NOC::NOC_0
                )
            );
            out["virtual_x"] = static_cast<uint32_t>(virtual_core.x());
            out["virtual_y"] = static_cast<uint32_t>(virtual_core.y());
            out["lim_base"] = cluster_state.base;
            out["lim_bytes"] = cluster_state.bytes;
            return out;
        }

        // Whole 64-byte lines at `offset` from the cluster's LIM base.
        void work_write(uint32_t cluster, uint64_t offset, const nb::bytes& data) {
            Work& cluster_state = work(cluster, true);
            if (
                offset % 64 != 0 || data.size() % 64 != 0 ||
                data.size() > cluster_state.bytes ||
                offset > cluster_state.bytes - data.size()
            ) {
                throw std::runtime_error("LIM writes are whole lines inside the area");
            }
            const auto* source = static_cast<const uint8_t*>(data.data());
            cluster_state.tile->write_available_lim(
                cluster_state.base + offset, {source, data.size()}
            );
        }

        nb::bytes work_read(uint32_t cluster, uint64_t offset, uint64_t n) const {
            const Work& cluster_state = work(cluster, true);
            if (
                offset % 4 != 0 || n % 4 != 0 || n > cluster_state.bytes ||
                offset > cluster_state.bytes - n
            ) {
                throw std::runtime_error("LIM reads are whole words inside the area");
            }
            const auto got = (
                cluster_state.tile->read_available_lim(cluster_state.base + offset, n)
            );
            return nb::bytes(reinterpret_cast<const char*>(got.data()), got.size());
        }

        // Writes `block` (whole lines; its first 8 bytes become the run value) at the
        // cluster's LIM base and starts `image` with it. Returns the run value.
        uint64_t work_start(
            uint32_t cluster, const std::string& image, nb::bytes block
        ) {
            Work& cluster_state = work(cluster, true);
            if (cluster_state.session) {
                throw std::runtime_error("an image is running on that cluster");
            }
            if (
                block.size() < 64 || block.size() % 64 != 0 ||
                block.size() > cluster_state.bytes
            ) {
                throw std::runtime_error("the argument block is whole 64-byte lines");
            }
            std::vector<uint8_t> bytes(
                static_cast<const uint8_t*>(block.data()),
                static_cast<const uint8_t*>(block.data()) + block.size()
            );
            std::random_device entropy;
            cluster_state.run = (
                (0x6b6d780000000000ull ^ (static_cast<uint64_t>(entropy()) << 16)) +
                entropy() + cluster
            );
            std::memcpy(bytes.data(), &cluster_state.run, sizeof(cluster_state.run));
            cluster_state.tile->write_available_lim(cluster_state.base, bytes);
            x280::ActiveRunRequest request;
            request.image_path = image;
            request.run_value = cluster_state.run;
            request.args_addr = cluster_state.base;
            request.args_bytes = bytes.size();
            const auto enqueued = x280_runtime().enqueue(*cluster_state.tile, request);
            if (!enqueued.session) {
                throw std::runtime_error(
                    "X280 enqueue failed: " + x280::describe(enqueued)
                );
            }
            cluster_state.session = enqueued.session;
            return cluster_state.run;
        }

        bool work_running(uint32_t cluster) const {
            return work(cluster, false).session.has_value();
        }

        // Waits (with a timeout) until every core of the cluster has returned from
        // main: ready, the four core results, and the runtime's description otherwise.
        nb::dict work_wait(uint32_t cluster, double timeout_s) {
            Work& cluster_state = work(cluster, true);
            if (!cluster_state.session) {
                throw std::runtime_error("no image running on that cluster");
            }
            x280::WaitResult wait_result;
            {
                nb::gil_scoped_release release;
                wait_result = (
                    x280_runtime()
                        .session(*cluster_state.session)
                        .wait_until_result_ready(millis(timeout_s))
                );
            }
            std::vector<uint64_t> results, stages;
            for (const auto& hart : wait_result.hartstates) {
                results.push_back(hart.state.result);
                stages.push_back(hart.state.stage);
            }
            nb::dict out;
            out["ready"] = wait_result.outcome == x280::WaitOutcome::ResultReady;
            out["results"] = results;
            out["stages"] = stages;
            out["failed"] = wait_result.succeeded() ? std::string()
                                                    : x280::describe(wait_result);
            return out;
        }

        // Stops the cluster's image through the runtime (with a timeout): cleaned, and
        // the runtime's description when it did not return to idle.
        nb::dict work_stop(uint32_t cluster, double timeout_s) {
            Work& cluster_state = work(cluster, true);
            if (!cluster_state.session) {
                throw std::runtime_error("no image running on that cluster");
            }
            x280::CleanupResult done;
            {
                nb::gil_scoped_release release;
                done = (
                    x280_runtime().cleanup(*cluster_state.session, millis(timeout_s))
                );
            }
            cluster_state.session.reset();
            nb::dict out;
            out["cleaned"] = done.cleaned_up();
            out["failed"] = done.cleaned_up() ? std::string() : x280::describe(done);
            return out;
        }

        // Stops the cluster's image if one runs (any failure path); whether it
        // returned to idle.
        bool work_abort(uint32_t cluster) {
            if (cluster >= work_.size() || !work_[cluster].session || !opened_) {
                return true;
            }
            const auto done = (
                x280_runtime().cleanup(*work_[cluster].session, std::chrono::seconds(5))
            );
            work_[cluster].session.reset();
            note_failed("cluster " + std::to_string(cluster), done);
            return done.cleaned_up();
        }

        // The runtime's error message for every clean-up since the last close() that
        // did not restore its cluster ("" when all did).
        const std::string& close_failed() const {
            return close_failed_;
        }

        uint32_t batch_ = 64;  // commands per store: every line, then one fence
        uint32_t grid_x_ = 0, grid_y_ = 0, l2x_ = 0, l2y_ = 0;
        uint64_t base_ = 0;  // LIM address of the layout (the argument block)
        std::vector<std::array<uint32_t, 2>> panels_;  // logical column ranges
        uint32_t tag_ = 0, depth_ = 1, host_depth_ = 1, event_depth_ = 1;
        uint32_t next_ = 1;                 // number of the next command
        uint32_t seen_ = 0;                 // events read from the ring
        uint32_t published_ = 0;            // the read count last stored into LIM
        double settle_microseconds_ = 1.0;  // pause between a fence and the next store
        Clock::time_point fenced_{};
        uint64_t upload_bytes_ = 0;  // the upload region's size

    private:
        std::string close_failed_;
        std::optional<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> aborted_;

        x280::X280Runtime& x280_runtime() const {
            return *opened_->runtime;
        }

        void note_failed(const std::string& what, const std::string& why) {
            close_failed_ += (close_failed_.empty() ? "" : "; ") + what + ": " + why;
        }

        void note_failed(const std::string& what, const x280::CleanupResult& done) {
            if (!done.cleaned_up()) {
                note_failed(what, x280::describe(done));
            }
        }

        std::vector<uint32_t> lim_words(uint64_t offset, uint32_t count) const {
            const auto block = tile_->read_available_lim(base_ + offset, 4 * count);
            std::vector<uint32_t> words(count);
            std::memcpy(words.data(), block.data(), 4 * count);
            return words;
        }

        void require_open() const {
            if (!opened_) {
                throw std::runtime_error("the X280 runtime is closed");
            }
        }

        // Stores k commands as the next commands, every line then one fence.
        void store(const uint32_t* commands, uint32_t count) {
            if (count == 0) {
                return;
            }
            settle();
            store_lines(commands, count);
            store_read();
            _mm_sfence();
            fenced_ = Clock::now();
            next_ += count;
        }

        // The lines of k commands as commands next .. next + k - 1, no fence.
        void store_lines(const uint32_t* commands, uint32_t count) const {
            alignas(64) uint32_t line[16];
            for (uint32_t index = 0; index < count; index++) {
                const uint32_t number = next_ + index;
                std::memcpy(line, commands + 16 * index, sizeof(line));
                line[0] = line[4] = line[8] = line[12] = number;
                win_->write_block(
                    KMD_LIM_HRING + (number % host_depth_) * 64u, line, 64
                );
            }
        }

        // The batch store_queued(upto) takes from the queue: those up to upto that fit.
        uint32_t batch_upto(uint32_t upto) const {
            const int64_t due = static_cast<int32_t>(upto - next_) + int64_t{1};
            size_t count = std::min<size_t>({queued(), batch_, room()});
            count = (
                std::min<int64_t>(
                    static_cast<int64_t>(count), std::max<int64_t>(0, due)
                )
            );
            return static_cast<uint32_t>(count);
        }

        // One batch from the queue: the commands up to upto that fit. An exclusive one
        // is stored as any other; the kernel's KM_FLAG_EXCLUSIVE check orders it.
        void store_queued(uint32_t upto) {
            const uint32_t count = batch_upto(upto);
            store(queue_.data() + 16 * head_, count);
            taken(count);
        }

        // `count` commands left the queue's head.
        void taken(size_t count) {
            head_ += count;
            if (head_ == queue_.size() / 16) {
                queue_.clear();
                head_ = 0;
            } else if (head_ > 4096 && 2 * head_ > queue_.size() / 16) {
                queue_.erase(queue_.begin(), queue_.begin() + 16 * head_);
                head_ = 0;
            }
        }

        struct Work {
            x280::X280Tile* tile = nullptr;
            uint64_t base = 0, bytes = 0, run = 0;  // available LIM, the run value
            std::optional<x280::SessionId> session;
        };

        // A work cluster (1 to 3); `open` also requires the runtime to be open.
        const Work& work(uint32_t cluster, bool open) const {
            if (open) {
                require_open();
            }
            if (cluster == 0 || cluster >= work_.size()) {
                throw std::runtime_error("work clusters are 1 to clusters() - 1");
            }
            return work_[cluster];
        }

        Work& work(uint32_t cluster, bool open) {
            return const_cast<Work&>(std::as_const(*this).work(cluster, open));
        }

        // The runtime's tiles, or an island's chip's tiles only.
        std::vector<x280::TileId> tile_ids() const {
            auto ids = x280_runtime().tile_ids();
            if (chip_id_ >= 0) {
                std::erase_if(ids, [&](const x280::TileId& id) {
                    return id.chip_id != chip_id_;
                });
                if (ids.empty()) {
                    throw std::runtime_error("the island's chip has no X280 tile");
                }
            }
            return ids;
        }

        // Boots the tiles to their idle image if they are not idle, then sets the
        // clock.
        void boot(uint32_t pll_mhz) {
            auto& runtime = x280_runtime();
            bool all_idle = true;
            for (const auto& id : tile_ids()) {
                all_idle = all_idle && runtime.tile(id).inspect().resident_idle_live;
            }
            if (!all_idle) {
                if (idle_image_path.empty()) {
                    throw std::runtime_error(
                        "X280 boot: set_idle_image was not called"
                    );
                }
                x280::MultiTileBootRequest request;
                request.tiles = tile_ids();
                request.idle_image_path = idle_image_path;
                request.allow_reset_release = true;
                if (!runtime.boot_tiles(request).ready()) {
                    throw std::runtime_error("X280 boot_tiles failed");
                }
            }
            runtime.set_pll(tile_ids().front().chip_id, pll_mhz);
        }

        std::array<uint32_t, 2> translated(uint32_t x, uint32_t y) const {
            const auto core = x280::X280Runtime::virtual_core_from_logical_core(
                *mesh_,
                tt::tt_metal::CoreCoord{x, y},
                tt::CoreType::WORKER,
                tt::tt_metal::NOC::NOC_0
            );
            return {static_cast<uint32_t>(core.x()), static_cast<uint32_t>(core.y())};
        }

        // Cluster 0, its LIM layout, the grid's panels, the host window and buffer.
        void locate() {
            auto& runtime = x280_runtime();
            const auto ids = tile_ids();
            tile_ = &runtime.tile(ids.at(cluster_));
            work_.resize(ids.size());
            for (
                size_t cluster_index = 0; cluster_index < ids.size(); cluster_index++
            ) {
                work_[cluster_index].tile = &runtime.tile(ids[cluster_index]);
                const auto lim = work_[cluster_index].tile->available_lim();
                work_[cluster_index].base = (lim.begin + 63) / 64 * 64;
                work_[cluster_index].bytes = (
                    lim.end > work_[cluster_index].base
                        ? lim.end - work_[cluster_index].base
                        : 0
                );
            }
            const auto l2 = (
                runtime.virtual_core_from_tile(
                    ids.at(cluster_), tt::tt_metal::NOC::NOC_0
                )
            );
            // The writers push completions on NoC1: the same coordinate must reach it.
            const auto l2_noc1 = (
                runtime.virtual_core_from_tile(
                    ids.at(cluster_), tt::tt_metal::NOC::NOC_1
                )
            );
            if (l2.x() != l2_noc1.x() || l2.y() != l2_noc1.y()) {
                throw std::runtime_error(
                    "cluster 0 has different NoC0 and NoC1 coordinates"
                );
            }
            l2x_ = static_cast<uint32_t>(l2.x());
            l2y_ = static_cast<uint32_t>(l2.y());
            base_ = tile_->available_lim().begin;
            if (base_ % 64 != 0 || base_ + KMD_LIM_BYTES > 0xFFFFFFFFull) {
                throw std::runtime_error(
                    "available LIM is not a 64-byte line below 4 GiB"
                );
            }
            const auto grid = mesh_->compute_with_storage_grid_size();
            grid_x_ = grid.x;
            grid_y_ = grid.y;
            // Panels: runs of logical columns whose NoC0 x are consecutive.
            const auto& soc = (
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                    mesh_->get_devices().front()->id()
                )
            );
            auto phys_x = [&](uint32_t logical_x) {
                const auto core = translated(logical_x, 0);
                return (
                    soc.translate_coord_to(
                        tt::umd::CoreCoord(
                            core[0],
                            core[1],
                            tt::CoreType::TENSIX,
                            tt::CoordSystem::TRANSLATED
                        ),
                        tt::CoordSystem::NOC0
                    )
                    .x
                );
            };
            for (uint32_t x = 0; x < grid_x_; x++) {
                if (!panels_.empty() && phys_x(x) == phys_x(panels_.back()[1]) + 1) {
                    panels_.back()[1] = x;
                } else {
                    panels_.push_back({x, x});
                }
            }
            if (
                panels_.size() > 2 || grid_x_ > KMD_READY_MAX ||
                grid_x_ * grid_y_ > KM_CORES_MAX
            ) {
                throw std::runtime_error("grid not supported: more than two panels");
            }
            const auto chip = mesh_->get_devices().front()->id();
            auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            auto* dev = cluster.get_driver()->get_tt_device(chip);
            win_ = dev->create_io_window(
                tt::umd::TargetIoWindowConfig{
                    .core_start = {l2x_, l2y_},
                    .addr = base_,
                    .noc = tt::umd::NocId::NOC0
                },
                tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC},
                tt::umd::IoOrdering::Strict
            );
            buffer_ = cluster.allocate_sysmem_buffer(chip, kHostBufferBytes, true);
            host_ = static_cast<volatile uint32_t*>(buffer_->get_va());
            host_noc_ = buffer_->get_noc_address().value();
            if ((host_noc_ & 0x1FFFFFu) + kHostBufferBytes > 0x200000u) {
                throw std::runtime_error("host buffer crosses a 2 MiB window");
            }
            const auto pcie = (
                cluster.get_soc_desc(chip).get_cores(
                    tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED
                )
            );
            pcie_x_ = pcie.front().x;
            pcie_y_ = pcie.front().y;
        }

        std::shared_ptr<MeshDevice> mesh_;
        std::shared_ptr<x280::RuntimeOpenResult> opened_;  // shared on islands
        int chip_id_ = -1;      // an island's chip: only its tiles; -1 every tile
        uint32_t cluster_ = 0;  // the X280 cluster it runs on (tile_ids() order)
        x280::X280Tile* tile_ = nullptr;
        uint64_t run_ = 0;
        std::shared_ptr<tt::umd::IoWindow> win_;         // host -> LIM, write-combining
        std::shared_ptr<tt::umd::SysmemBuffer> buffer_;  // kernel -> pinned host memory
        volatile uint32_t* host_ = nullptr;
        uint64_t host_noc_ = 0;
        uint32_t pcie_x_ = 0, pcie_y_ = 0;
        std::optional<x280::SessionId> session_;
        std::vector<uint32_t> queue_;  // commands not stored yet, 16 words each
        size_t head_ = 0;              // the first command not stored
        std::vector<Work> work_;  // every cluster in tile_ids() order; 0 unused here
    };

}  // namespace

NB_MODULE(_km_host, m) {
    m.attr("MTIME_HZ") = kMtimeHz;
    m.attr("EVENT_DEPTH_MAX") = kEventDepthMax;
    m.def(
        "set_idle_image",
        [](std::string path) {
            idle_image_path = std::move(path);
        },
        nb::arg("path")
    );
    nb::class_<Device>(m, "Device")
        .def(nb::init<std::optional<size_t>>(), nb::arg("worker_l1_size") = nb::none())
        .def_prop_ro("is_open", &Device::is_open)
        .def("close", &Device::close)
        .def("grid", &Device::grid)
        .def("worker_core", &Device::worker_core, nb::arg("x"), nb::arg("y"))
        .def("dram_core", &Device::dram_core, nb::arg("bank"))
        .def("host_channel", &Device::host_channel)
        // A std::shared_ptr<MeshDevice>, for native modules of their own that take
        // the device; the capsule's name is checked on the receiving side.
        .def(
            "mesh_capsule",
            [](const Device& device) {
                using Held = std::shared_ptr<MeshDevice>;
                return nb::capsule(
                    new Held(device.shared()),
                    "komorebi.mesh_device",
                    [](void* pointer) noexcept {
                        delete static_cast<Held*>(pointer);
                    }
                );
            }
        ).def("finish", [](const Device& device) {
            nb::gil_scoped_release release;
            device.finish();
        });
    nb::class_<Islands>(m, "Islands")
        .def(
            nb::init<uint32_t, std::optional<size_t>, uint32_t>(),
            nb::arg("count"),
            nb::arg("worker_l1_size") = nb::none(),
            nb::arg("packet") = 0
        ).def_prop_ro("is_open", &Islands::is_open).def("count", &Islands::count)
        .def("chips", &Islands::chips)
        .def("device", &Islands::device, nb::arg("island"))
        .def(
            "link",
            &Islands::link,
            nb::arg("source"),
            nb::arg("target"),
            nb::arg("link") = 0
        ).def(
            "open_x280",
            [](Islands& islands) {
                nb::gil_scoped_release release;
                islands.open_x280();
            }
        ).def("close", [](Islands& islands) {
            nb::gil_scoped_release release;
            islands.close();
        });
    nb::class_<DeviceBuffer>(m, "Buffer")
        .def(
            nb::init<
                const Device&,
                uint64_t,
                uint64_t,
                std::optional<std::array<uint32_t, 2>>,
            std::optional<uint64_t>>(),
            nb::arg("device"),
            nb::arg("size"),
            nb::arg("page_size"),
            nb::arg("shard_grid") = nb::none(),
            nb::arg("address") = nb::none()
        ).def_prop_ro("address", &DeviceBuffer::address)
        .def_prop_ro("size", &DeviceBuffer::size)
        .def_prop_ro("page_size", &DeviceBuffer::page_size)
        .def_prop_ro("aligned_page_size", &DeviceBuffer::aligned_page_size)
        .def_prop_ro("num_pages", &DeviceBuffer::num_pages)
        .def("accessor_args", &DeviceBuffer::accessor_args)
        .def("write", &DeviceBuffer::write, nb::arg("data"))
        .def("read", &DeviceBuffer::read)
        .def("free", &DeviceBuffer::free);
    nb::class_<Launch>(m, "Launch")
        .def(nb::init<const Device&>(), nb::arg("device"))
        .def(
            "cb",
            &Launch::cb,
            nb::arg("cores"),
            nb::arg("index"),
            nb::arg("format"),
            nb::arg("page_size"),
            nb::arg("total_size"),
            nb::arg("first") = false
        ).def(
            "kernel",
            &Launch::kernel,
            nb::arg("source"),
            nb::arg("cores"),
            nb::arg("processor"),
            nb::arg("compile_args"),
            nb::arg("defines"),
            nb::arg("includes"),
            nb::arg("args"),
            nb::arg("opt_level") = nb::none(),
            nb::arg("math_fidelity") = nb::none(),
            nb::arg("unpack_to_dest_mode") = std::vector<std::string>{},
            nb::arg("flags") = std::vector<std::pair<std::string, bool>>{},
            nb::arg("noc") = nb::none()
        ).def("build", &Launch::build).def("recipes", &Launch::recipes)
        .def("run", &Launch::run)
        .def(
            "reset_core",
            &Launch::reset_core,
            nb::arg("x"),
            nb::arg("y"),
            nb::arg("timeout_s"),
            nb::arg("channels") = std::vector<uint32_t>{},
            nb::arg("drops") = std::vector<uint32_t>{}
        ).def("reset_groups", &Launch::reset_groups).def(
            "reset_list",
            &Launch::reset_list,
            nb::arg("x"),
            nb::arg("y"),
            nb::arg("every") = false,
            nb::arg("ready") = 0,
            nb::arg("ready_mask") = 0,
            nb::arg("ready_value") = 0
        );
    m.def(
        "write_l1",
        [](
            const Device& device,
            uint32_t x,
            uint32_t y,
            uint32_t address,
            std::vector<uint32_t> words
        ) {
            write_l1(device.mesh(), x, y, address, std::move(words));
        },
        nb::arg("device"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("address"),
        nb::arg("words")
    );
    m.def(
        "core_nocs",
        [](const Device& device) {
            return core_nocs(device.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "dram_banks",
        [](const Device& device) {
            return dram_banks(device.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "kernel_buffer",
        [](const Device& device) {
            return kernel_buffer(device.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "read_dram",
        [](const Device& device, std::vector<std::array<uint64_t, 3>> spans) {
            return read_dram(device.mesh(), spans);
        },
        nb::arg("device"),
        nb::arg("spans")
    );
    m.def(
        "write_dram",
        [](
            const Device& device,
            std::vector<uint32_t> banks,
            uint64_t address,
            const Numbers8& data
        ) {
            write_dram(device.mesh(), banks, address, data);
        },
        nb::arg("device"),
        nb::arg("banks"),
        nb::arg("address"),
        nb::arg("data")
    );
    m.def(
        "dram_barrier",
        [](const Device& device) {
            dram_barrier(device.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "wait_grid",
        [](
            const Device& device,
            uint32_t grid_x,
            uint32_t grid_y,
            double timeout_s,
            uint32_t first_row
        ) {
            return wait_grid(device.mesh(), grid_x, grid_y, timeout_s, first_row);
        },
        nb::arg("device"),
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("timeout_s"),
        nb::arg("first_row") = 0
    );
    m.def(
        "read_l1",
        [](
            const Device& device,
            uint32_t x,
            uint32_t y,
            uint32_t address,
            uint32_t count
        ) {
            return read_l1(device.mesh(), x, y, address, count);
        },
        nb::arg("device"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("address"),
        nb::arg("count")
    );

    nb::class_<Dispatch>(m, "Dispatch")
        .def(
            "__init__",
            [](Dispatch* self, const Device& device, uint32_t pll_mhz) {
                if (device.x280()) {
                    new (self)
                        Dispatch(device.shared(), device.x280()->opened, pll_mhz);
                } else {
                    new (self) Dispatch(device.shared(), pll_mhz);
                }
            },
            nb::arg("device"),
            nb::arg("pll_mhz") = 1000
        ).def(
            "__init__",
            [](Dispatch* self, const Dispatch& first, uint32_t cluster) {
                new (self) Dispatch(first, cluster);
            },
            nb::arg("first"),
            nb::arg("cluster")
        ).def_ro("grid_x", &Dispatch::grid_x_).def_ro("grid_y", &Dispatch::grid_y_)
        .def_ro("panels", &Dispatch::panels_)
        .def_ro("l2_x", &Dispatch::l2x_)
        .def_ro("l2_y", &Dispatch::l2y_)
        .def_ro("lim_base", &Dispatch::base_)
        .def_ro("tag", &Dispatch::tag_)
        .def_ro("depth", &Dispatch::depth_)
        .def_ro("host_depth", &Dispatch::host_depth_)
        .def_ro("event_depth", &Dispatch::event_depth_)
        .def_ro("next", &Dispatch::next_)
        .def_ro("seen", &Dispatch::seen_)
        .def_rw("settle_microseconds", &Dispatch::settle_microseconds_)
        .def_rw("batch", &Dispatch::batch_)
        .def_prop_ro("queued", &Dispatch::queued)
        .def_prop_ro("is_open", &Dispatch::is_open)
        .def_prop_ro("running", &Dispatch::running)
        .def_prop_ro("idle", &Dispatch::idle)
        .def("core_table", &Dispatch::core_table)
        .def(
            "prepare",
            &Dispatch::prepare,
            nb::arg("depth"),
            nb::arg("host_depth"),
            nb::arg("event_depth"),
            nb::arg("t_start_s"),
            nb::arg("t_command_s"),
            nb::arg("t_idle_s"),
            nb::arg("state_ddr"),
            nb::arg("work_cmds") = 0,
            nb::arg("work_depth") = 0,
            nb::arg("work_event_depth") = 0,
            nb::arg("work_noc") = std::vector<uint64_t>{},
            nb::arg("work_noc_addr") = std::vector<uint64_t>{},
            nb::arg("upload_bytes") = -1,
            nb::arg("messages_handler") = 0,
            nb::arg("programs") = std::vector<uint8_t>{},
            nb::arg("island") = 0,
            nb::arg("link") = std::vector<uint8_t>{},
            nb::arg("first_row") = 0,
            nb::arg("rows") = 0
        ).def("start", &Dispatch::start, nb::arg("image"))
        .def("host_word", &Dispatch::host_word, nb::arg("word"))
        .def_ro("upload_bytes", &Dispatch::upload_bytes_)
        .def("upload", &Dispatch::upload, nb::arg("offset"), nb::arg("data"))
        .def("publish", &Dispatch::publish)
        .def(
            "wait_ready",
            [](const Dispatch& device, double timeout_s) {
                nb::gil_scoped_release release;
                return device.wait_ready(timeout_s);
            },
            nb::arg("timeout_s")
        ).def("room", &Dispatch::room).def(
            "push",
            [](Dispatch& device, Commands commands) {
                if (commands.shape(1) != 16) {
                    throw std::runtime_error("commands are [n, 16] words");
                }
                return device.push(commands.data(), commands.shape(0));
            },
            nb::arg("commands")
        ).def(
            "enqueue",
            [](Dispatch& device, Commands commands) {
                if (commands.shape(1) != 16) {
                    throw std::runtime_error("commands are [n, 16] words");
                }
                return device.enqueue(commands.data(), commands.shape(0));
            },
            nb::arg("commands")
        ).def(
            "wait",
            [](Dispatch& device, Numbers wanted, uint32_t upto, double timeout_s) {
                std::vector<uint32_t> want(
                    wanted.data(), wanted.data() + wanted.shape(0)
                );
                std::vector<uint32_t> out;
                {
                    nb::gil_scoped_release release;
                    out = device.wait(want, upto, timeout_s);
                }
                return as_array(std::move(out));
            },
            nb::arg("wanted"),
            nb::arg("upto"),
            nb::arg("timeout_s")
        ).def_static(
            "store_together",
            [](std::vector<Dispatch*> ports, std::vector<uint32_t> uptos) {
                std::vector<std::vector<uint32_t>> out;
                {
                    nb::gil_scoped_release release;
                    out = Dispatch::store_together(ports, uptos);
                }
                nb::list events;
                for (auto& words : out) {
                    events.append(as_array(std::move(words)));
                }
                return events;
            },
            nb::arg("ports"),
            nb::arg("uptos")
        ).def(
            "poll",
            [](Dispatch& device) {
                return as_array(device.poll());
            }
        ).def(
            "wait_events",
            [](Dispatch& device, double timeout_s) {
                std::vector<uint32_t> out;
                {
                    nb::gil_scoped_release release;
                    out = device.wait_events(timeout_s);
                }
                return as_array(std::move(out));
            },
            nb::arg("timeout_s")
        ).def("finish", &Dispatch::finish, nb::arg("timeout_s"))
        .def("abort", &Dispatch::abort)
        .def("close", &Dispatch::close)
        .def_prop_ro("close_failed", &Dispatch::close_failed)
        .def("aborted", &Dispatch::aborted)
        .def("probe", &Dispatch::probe, nb::arg("gap_s"))
        .def("inspect", &Dispatch::inspect, nb::arg("count"))
        .def("clusters", &Dispatch::clusters)
        .def("work_info", &Dispatch::work_info, nb::arg("cluster"))
        .def(
            "work_write",
            &Dispatch::work_write,
            nb::arg("cluster"),
            nb::arg("offset"),
            nb::arg("data")
        ).def(
            "work_read",
            &Dispatch::work_read,
            nb::arg("cluster"),
            nb::arg("offset"),
            nb::arg("n")
        ).def(
            "work_start",
            &Dispatch::work_start,
            nb::arg("cluster"),
            nb::arg("image"),
            nb::arg("block")
        ).def("work_running", &Dispatch::work_running, nb::arg("cluster")).def(
            "work_wait", &Dispatch::work_wait, nb::arg("cluster"), nb::arg("timeout_s")
        ).def(
            "work_stop", &Dispatch::work_stop, nb::arg("cluster"), nb::arg("timeout_s")
        ).def("work_abort", &Dispatch::work_abort, nb::arg("cluster"));
}
