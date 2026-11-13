// _km_host: the X280 runtime, the dispatch kernel's memory layout (km_dispatch.h), its
// start and stop, batched host commands and the event ring. Every wait is bounded.
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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

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

    bool power_of_two(uint32_t n) {
        return n != 0 && (n & (n - 1)) == 0;
    }

    // The X280 idle image of the tree in use (firmware-lim-idle.bin), set by the
    // package when it loads this module: the module does not name a build directory.
    std::string idle_image_path;

    static_assert(sizeof(km_record_t) == 64);
    static_assert(sizeof(km_event_t) == 16 && sizeof(kmd_args_t) <= kArgsBytes);
    static_assert(kArgsBytes <= KMD_LIM_MAILBOX);

    using Records = nb::
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
    // (not the X280): the stop word when the dispatch kernel is not there to send it.
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

    // The grid (x, y), then per core in core order its translated NoC address (the
    // core table's) and its NoC0 coordinates, as x | y << 16 words; no X280 needed.
    std::tuple<uint32_t, uint32_t, std::vector<uint32_t>, std::vector<uint32_t>>
    core_nocs(MeshDevice& mesh) {
        const auto g = mesh.compute_with_storage_grid_size();
        const auto& soc = (
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                chip(mesh)->id()
            )
        );
        std::vector<uint32_t> table, noc0;
        for (uint32_t y = 0; y < g.y; y++) {
            for (uint32_t x = 0; x < g.x; x++) {
                const auto t = x280::X280Runtime::virtual_core_from_logical_core(
                    mesh,
                    tt::tt_metal::CoreCoord{x, y},
                    tt::CoreType::WORKER,
                    tt::tt_metal::NOC::NOC_0
                );
                const auto p = (
                    soc.translate_coord_to(
                        tt::umd::CoreCoord(
                            t.x(), t.y(), tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED
                        ),
                        tt::CoordSystem::NOC0
                    )
                );
                table.push_back(static_cast<uint32_t>(t.x() | t.y() << 16));
                noc0.push_back(static_cast<uint32_t>(p.x | p.y << 16));
            }
        }
        return {static_cast<uint32_t>(g.x), static_cast<uint32_t>(g.y), table, noc0};
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
        const uint64_t l1 =
            (hal.get_dev_addr(tensix, tt::tt_metal::HalL1MemAddrType::BASE) +
             hal.get_dev_size(tensix, tt::tt_metal::HalL1MemAddrType::BASE));
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
        for (uint32_t t = 0; t < types; t++) {
            for (const auto& [id, kernel] : program.impl().get_kernels(t)) {
                const auto& src = kernel->kernel_source();
                kernels.emplace_back(
                    src.path_.empty() ? kernel->name() : src.path_.string(),
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
        } catch (const std::exception& e) {
            error = e.what();
        }
        // The text sizes are set before the size check, so they are there either way.
        std::vector<uint32_t> text(hal.get_num_risc_processors(tensix), 0);
        for (const auto& group : program.impl().get_kernel_groups(index)) {
            const auto sizes = (
                group->launch_msg.view().kernel_config().kernel_text_size()
            );
            for (size_t p = 0; p < text.size(); p++) {
                text[p] = std::max(text[p], static_cast<uint32_t>(sizes[p]));
            }
        }
        return {kernels, config_bytes, kernel_buffer(mesh)[0], text, error};
    }

    // Reads (bank, address, bytes) spans of DRAM after one DRAM barrier, one read each.
    // No wait for idle cores: the caller reads only what no kernel is still writing.
    std::vector<Bytes> read_dram(
        const MeshDevice& mesh, const std::vector<std::array<uint64_t, 3>>& spans
    ) {
        auto* dev = chip(mesh);
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        std::vector<std::vector<uint8_t>*> data;
        for (const auto& span : spans) {
            if (span[0] >= dram_banks(mesh) || span[2] > UINT32_MAX) {
                throw std::runtime_error("a DRAM span outside the banks or over 4 GB");
            }
            data.push_back(new std::vector<uint8_t>(span[2]));
        }
        {
            nb::gil_scoped_release release;
            cluster.dram_barrier(dev->id());
            for (size_t i = 0; i < spans.size(); i++) {
                const auto& [bank, address, bytes] = spans[i];
                cluster.read_dram_vec(
                    data[i]->data(),
                    static_cast<uint32_t>(bytes),
                    dev->id(),
                    static_cast<int>(bank),
                    address
                );
            }
        }
        std::vector<Bytes> out;
        for (auto* owned : data) {
            nb::capsule owner(owned, [](void* p) noexcept {
                delete static_cast<std::vector<uint8_t>*>(p);
            });
            const size_t shape[1] = {owned->size()};
            out.emplace_back(owned->data(), 1, shape, owner);
        }
        return out;
    }

    // Waits up to timeout_s for the kernels on the gx x gy worker grid to return, in
    // slices of at most a second with signals checked between them. True: returned.
    bool wait_grid(const MeshDevice& mesh, uint32_t gx, uint32_t gy, double timeout_s) {
        auto* dev = chip(mesh);
        std::unordered_set<tt::tt_metal::CoreCoord> left;
        for (uint32_t y = 0; y < gy; y++) {
            for (uint32_t x = 0; x < gx; x++) {
                left.insert(dev->virtual_core_from_logical_core(
                    tt::tt_metal::CoreCoord{x, y}, tt::CoreType::WORKER
                ));
            }
        }
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto by = after(timeout_s);
        for (;;) {
            const auto rest = (
                std::chrono::duration_cast<std::chrono::milliseconds>(by - Clock::now())
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
            if (Clock::now() >= by) {
                return false;
            }
            if (PyErr_CheckSignals() != 0) {
                throw nb::python_error();
            }
        }
    }

    // An open card: the 1 x 1 mesh of the first visible device, opened as ttnn's
    // open_mesh_device opens it, with the program cache off.
    class Device {
    public:
        explicit Device(std::optional<size_t> worker_l1_size) {
            namespace d = tt::tt_metal::distributed;
            mesh_ = MeshDevice::create(
                d::MeshDeviceConfig(d::MeshShape(1, 1)),
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
            const auto g = mesh().compute_with_storage_grid_size();
            return {static_cast<uint32_t>(g.x), static_cast<uint32_t>(g.y)};
        }

        // A Tensix core's NoC coordinates (the runtime's virtual ones) from its
        // logical (x, y).
        std::tuple<uint32_t, uint32_t> worker_core(uint32_t x, uint32_t y) const {
            const auto c = mesh().worker_core_from_logical_core({x, y});
            return {static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
        }

        // The NoC coordinates of DRAM bank `bank`'s core.
        std::tuple<uint32_t, uint32_t> dram_core(uint32_t bank) const {
            const auto c = mesh().virtual_core_from_logical_core(
                {bank, 0}, tt::CoreType::DRAM
            );
            return {static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)};
        }

        // Waits until every launched program has finished.
        void finish() const {
            tt::tt_metal::distributed::Finish(mesh().mesh_command_queue());
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
    };

    using MeshBuffer = tt::tt_metal::distributed::MeshBuffer;

    // A device buffer: interleaved in DRAM, or (shard_grid set) height-sharded in L1
    // with one page on each core of the grid, cores in row-major order.
    class DeviceBuffer {
    public:
        DeviceBuffer(
            const Device& device,
            uint64_t size,
            uint64_t page_size,
            std::optional<std::array<uint32_t, 2>> shard_grid
        )
            : mesh_(device.shared()) {
            namespace d = tt::tt_metal::distributed;
            using tt::tt_metal::BufferType;
            if (page_size == 0 || size % page_size != 0) {
                throw std::invalid_argument("the size is not a whole number of pages");
            }
            d::DeviceLocalBufferConfig local{
                .page_size = page_size, .buffer_type = BufferType::DRAM
            };
            if (shard_grid) {
                const auto [gx, gy] = *shard_grid;
                const uint32_t cores = gx * gy;
                if (size != uint64_t{cores} * page_size || page_size % 4 != 0) {
                    throw std::invalid_argument(
                        "a sharded buffer holds one page of whole words per core"
                    );
                }
                const auto words = static_cast<uint32_t>(page_size / 4);
                const tt::tt_metal::CoreRangeSet grid(
                    tt::tt_metal::CoreRange({0, 0}, {gx - 1, gy - 1})
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
                d::ReplicatedBufferConfig{.size = size}, local, mesh_.get()
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
            nb::capsule owner(owned, [](void* p) noexcept {
                delete static_cast<std::vector<uint8_t>*>(p);
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
        for (const auto& [n, f] : formats) {
            if (n == name) {
                return f;
            }
        }
        throw std::invalid_argument("no data format " + name);
    }

    tt::tt_metal::KernelBuildOptLevel opt_level(const std::string& name) {
        using L = tt::tt_metal::KernelBuildOptLevel;
        static const std::vector<std::pair<std::string, L>> levels = {
            {"O0", L::O0},
            {"O1", L::O1},
            {"O2", L::O2},
            {"O3", L::O3},
            {"Os", L::Os},
            {"Ofast", L::Ofast},
            {"Oz", L::Oz},
        };
        for (const auto& [n, l] : levels) {
            if (n == name) {
                return l;
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
        tt::tt_metal::ComputeConfigDescriptor c;
        if (fidelity) {
            using F = tt::tt_metal::MathFidelity;
            const std::vector<std::pair<std::string, F>> names = {
                {"LoFi", F::LoFi},
                {"HiFi2", F::HiFi2},
                {"HiFi3", F::HiFi3},
                {"HiFi4", F::HiFi4}
            };
            const auto it = std::find_if(
                names.begin(), names.end(), [&](const auto& n) {
                    return n.first == *fidelity;
                }
            );
            if (it == names.end()) {
                throw std::invalid_argument("no math fidelity " + *fidelity);
            }
            c.math_fidelity = it->second;
        }
        for (const auto& mode : unpack) {
            if (mode == "Default") {
                c.unpack_to_dest_mode.push_back(
                    tt::tt_metal::UnpackToDestMode::Default
                );
            } else if (mode == "UnpackToDestFp32") {
                c.unpack_to_dest_mode.push_back(
                    tt::tt_metal::UnpackToDestMode::UnpackToDestFp32
                );
            } else {
                throw std::invalid_argument("no unpack-to-dest mode " + mode);
            }
        }
        for (const auto& [name, on] : flags) {
            if (name == "fp32_dest_acc_en") {
                c.fp32_dest_acc_en = on;
            } else if (name == "dst_full_sync_en") {
                c.dst_full_sync_en = on;
            } else if (name == "bfp8_pack_precise") {
                c.bfp8_pack_precise = on;
            } else if (name == "math_approx_mode") {
                c.math_approx_mode = on;
            } else if (name == "enable_trisc2_rvv") {
                c.enable_trisc2_rvv = on;
            } else {
                throw std::invalid_argument("no compute option " + name);
            }
        }
        return c;
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
            uint32_t total_size
        ) {
            tt::tt_metal::CBDescriptor cb;
            cb.total_size = total_size;
            cb.core_ranges = core_set(cores);
            cb.format_descriptors.push_back(
                tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = index,
                    .data_format = data_format(format),
                    .page_size = page_size
                }
            );
            desc_.cbs.push_back(std::move(cb));
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
            tt::tt_metal::KernelDescriptor k;
            k.kernel_source = source;
            k.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            k.core_ranges = core_set(cores);
            k.compile_time_args = compile_args;
            k.defines = defines;
            for (const auto& path : includes) {
                k.compiler_include_paths.emplace_back(path);
            }
            for (const auto& [x, y, words] : args) {
                k.runtime_args.emplace_back(tt::tt_metal::CoreCoord{x, y}, words);
            }
            if (level) {
                k.opt_level = opt_level(*level);
            }
            if (noc && (processor == 2 || *noc > 1)) {
                throw std::invalid_argument("noc: 0 or 1, for a data movement kernel");
            }
            if (processor == 0) {
                k.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = static_cast<NOC>(noc.value_or(NOC::RISCV_0_default))
                };
            } else if (processor == 1) {
                k.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = static_cast<NOC>(noc.value_or(NOC::RISCV_1_default))
                };
            } else if (processor == 2) {
                k.config = compute_config(fidelity, unpack, flags);
            } else {
                throw std::invalid_argument(
                    "processor: 0, 1 (data movement) or 2 (compute)"
                );
            }
            desc_.kernels.push_back(std::move(k));
        }

        KernelBuild build() const {
            return kernel_build(*mesh_, desc_);
        }

        // Launches the program and returns without waiting for it (the launch call
        // of generic_op on a 1 x 1 mesh); a later run launches the kept program again.
        void run() {
            namespace d = tt::tt_metal::distributed;
            if (!workload_) {
                workload_ = std::make_unique<d::MeshWorkload>();
                workload_->add_program(
                    d::MeshCoordinateRange(mesh_->shape()), tt::tt_metal::Program(desc_)
                );
            }
            d::EnqueueMeshWorkload(mesh_->mesh_command_queue(), *workload_, false);
        }

    private:
        std::shared_ptr<MeshDevice> mesh_;
        tt::tt_metal::ProgramDescriptor desc_;
        std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload_;
    };

    // [n, 4] events (index, req, status, ticks) as a numpy array that owns its words.
    Events as_array(std::vector<uint32_t>&& words) {
        auto* owned = new std::vector<uint32_t>(std::move(words));
        nb::capsule owner(owned, [](void* p) noexcept {
            delete static_cast<std::vector<uint32_t>*>(p);
        });
        const size_t shape[2] = {owned->size() / 4, 4};
        return Events(owned->data(), 2, shape, owner);
    }

    // The X280 runtime on one card and the dispatch kernel's port on cluster 0: its
    // LIM layout, the host command ring, the event ring and the result block.
    class Dispatch {
    public:
        Dispatch(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz)
            : mesh_(std::move(mesh_device)) {
            opened_ = std::make_unique<x280::RuntimeOpenResult>(x280::X280Runtime::open(
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

        ~Dispatch() {
            close();
        }

        bool is_open() const {
            return opened_ != nullptr;
        }

        bool running() const {
            return session_.has_value();
        }

        // Whether cluster 0 runs its idle image and no session: an image can start.
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
            for (uint32_t c = 1; c < work_.size(); c++) {
                cleaned = work_abort(c) && cleaned;
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
            std::vector<uint32_t> t(KM_CORES_MAX, 0);
            for (uint32_t y = 0; y < gy_; y++) {
                for (uint32_t x = 0; x < gx_; x++) {
                    const auto c = translated(x, y);
                    t[y * gx_ + x] = c[0] | (c[1] << 16);
                }
            }
            return t;
        }

        // Picks the run and the tag, zeroes LIM layout and host buffer, writes the
        // argument block (state_ddr: the kernel's state; upload_bytes: -1 all free).
        uint32_t prepare(
            uint32_t depth,
            uint32_t host_depth,
            uint32_t event_depth,
            double t_start_s,
            double t_record_s,
            double t_idle_s,
            uint64_t state_ddr,
            uint64_t work_cmds,
            uint64_t work_depth,
            uint64_t work_event_depth,
            const std::vector<uint64_t>& work_noc,
            const std::vector<uint64_t>& work_noc_addr,
            int64_t upload_bytes,
            uint64_t dyn_seg
        ) {
            require_open();
            if (session_) {
                throw std::runtime_error("a kernel is running");
            }
            if (depth < 1 || depth > KM_RING_SLOTS || !power_of_two(host_depth) ||
                host_depth > KMD_HOST_DEPTH_MAX || !power_of_two(event_depth) ||
                event_depth > kEventDepthMax) {
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
            const uint64_t work_bytes =
                (work_cmds ? KMD_WORK_CLUSTERS * KMD_WORK_RING_BYTES(work_depth) : 0);
            if (work_noc.size() > KMD_WORK_CLUSTERS ||
                work_noc_addr.size() != work_noc.size()) {
                throw std::runtime_error("work_noc and work_noc_addr: one per cluster");
            }
            if (work_cmds &&
                (work_cmds % 64 != 0 || work_cmds < KMD_LIM_BYTES ||
                 base_ + work_cmds + work_bytes > tile_->available_lim().end)) {
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
            const uint64_t lo = base_ + KMD_LIM_UPLOAD;
            const uint64_t room = end > lo ? (end - lo) & ~63ull : 0;
            if (upload_bytes > static_cast<int64_t>(room) ||
                (upload_bytes >= 0 && upload_bytes % 64 != 0)) {
                throw std::runtime_error(
                    "upload_bytes: a multiple of 64, at most the room"
                );
            }
            upload_bytes_ =
                (upload_bytes < 0 ? room : static_cast<uint64_t>(upload_bytes));
            depth_ = depth;
            host_depth_ = host_depth;
            event_depth_ = event_depth;
            std::random_device rd;
            run_ = (0x6b6d680000000000ull ^ (static_cast<uint64_t>(rd()) << 16)) + rd();
            do {
                tag_ = (rd() & 0xFFFFu) | 1u;
            } while (tag_ == 0xFFFFu);
            next_ = 1;
            seen_ = published_ = 0;
            queue_.clear();
            head_ = 0;
            std::memset(const_cast<uint32_t*>(host_), 0, kHostBufferBytes);
            tile_->write_available_lim(base_, std::vector<uint8_t>(KMD_LIM_BYTES, 0));
            if (work_cmds) {
                tile_->write_available_lim(
                    base_ + work_cmds, std::vector<uint8_t>(work_bytes, 0)
                );
            }

            kmd_args_t a{};
            a.run_value = run_;
            a.depth = depth_;
            a.host_depth = host_depth_;
            a.event_depth = event_depth_;
            a.columns = gx_;
            a.cores = gx_ * gy_;
            a.panels = panels_.size();
            for (size_t r = 0; r < panels_.size(); r++) {
                const auto p0 = translated(panels_[r][0], 0);
                const auto p1 = translated(panels_[r][1], gy_ - 1);
                a.rect[r][0] = p0[0];
                a.rect[r][1] = p0[1];
                a.rect[r][2] = p1[0];
                a.rect[r][3] = p1[1];
            }
            a.host_noc_addr = host_noc_;
            a.pcie_x = pcie_x_;
            a.pcie_y = pcie_y_;
            a.t_start = static_cast<uint64_t>(t_start_s * kMtimeHz);
            a.t_record = static_cast<uint64_t>(t_record_s * kMtimeHz);
            a.t_idle = static_cast<uint64_t>(t_idle_s * kMtimeHz);
            a.state_ddr = state_ddr;
            a.work_cmds = work_cmds;
            a.work_depth = work_depth;
            a.work_event_depth = work_event_depth;
            for (size_t w = 0; w < work_noc.size(); w++) {
                a.work_noc[w] = work_noc[w];
                a.work_noc_addr[w] = work_noc_addr[w];
            }
            a.version = KMD_ARGS_VERSION;
            a.upload_bytes = upload_bytes_;
            a.dyn_seg = dyn_seg;
            std::vector<uint8_t> block(kArgsBytes, 0);
            std::memcpy(block.data(), &a, sizeof(a));
            tile_->write_available_lim(base_ + KMD_LIM_ARGS, block);
            return tag_;
        }

        // Starts the kernel image on cluster 0 with the prepared argument block.
        void start(const std::string& image) {
            require_open();
            if (session_) {
                throw std::runtime_error("a kernel is running");
            }
            x280::ActiveRunRequest req;
            req.image_path = image;
            req.run_value = run_;
            req.args_addr = base_ + KMD_LIM_ARGS;
            req.args_bytes = kArgsBytes;
            const auto enq = rt().enqueue(*tile_, req);
            if (!enq.session) {
                throw std::runtime_error("X280 enqueue failed: " + x280::describe(enq));
            }
            session_ = enq.session;
            aborted_.reset();
        }

        // Stores whole 64-byte lines at `offset` of the upload region, then a fence.
        void upload(uint64_t offset, const nb::bytes& data) {
            require_open();
            const uint64_t n = data.size();
            if (offset % 64 != 0 || n % 64 != 0 || offset > upload_bytes_ ||
                n > upload_bytes_ - offset) {
                throw std::runtime_error("upload: whole lines inside the region");
            }
            settle();
            win_->write_block(KMD_LIM_UPLOAD + offset, data.c_str(), n);
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

        uint32_t host_word(uint32_t w) const {
            require_open();
            if (w >= KMD_HOST_RING / 4) {
                throw std::runtime_error("no such host word");
            }
            return host_[w];
        }

        // Waits until the kernel reports READY or a final state; returns that state, or
        // 0 after the timeout.
        uint32_t wait_ready(double timeout_s) const {
            require_open();
            const auto by = after(timeout_s);
            for (;;) {
                const uint32_t s = host_[KMD_HOST_STATE];
                if (s >= KMD_STATE_READY) {
                    return s;
                }
                if (Clock::now() > by) {
                    return 0;
                }
            }
        }

        // Commands that fit now: free host ring slots, and room in the event ring, so
        // that unread events plus commands in flight never exceed it. Counters wrap.
        uint32_t room() const {
            require_open();
            const uint32_t taken = host_[KMD_HOST_CONSUMED];
            const int64_t ring =
                (static_cast<int64_t>(host_depth_) -
                 static_cast<int32_t>(next_ - 1u - taken));
            const int64_t events =
                (static_cast<int64_t>(event_depth_) -
                 static_cast<int32_t>(next_ - 1u - seen_));
            return static_cast<uint32_t>(std::max<int64_t>(0, std::min(ring, events)));
        }

        // Stores the read count line (KMD_LIM_READ: events the host has read), without
        // a fence. The kernel takes a command only while its events fit the ring.
        void store_read() {
            alignas(64) uint32_t w[16] = {seen_};
            win_->write_block(KMD_LIM_READ, w, 64);
            published_ = seen_;
        }

        // While waiting: the read count and a fence, only when the kernel may hold a
        // command for it (command i waits only while i - 1 - published reaches depth).
        void publish_if_held() {
            if (seen_ != published_ && next_ - 1u - published_ >= event_depth_ &&
                Clock::now() >= settled()) {
                store_read();
                _mm_sfence();
                fenced_ = Clock::now();
            }
        }

        // Spins until settle_us_ passed since the last fence: a store that follows a
        // fence too closely often stalls about 12 us.
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
                    std::chrono::duration<double, std::micro>(settle_us_)
                )
            );
        }

        // Stores the first k records that fit as commands next .. next + k - 1 (number
        // in seq and the landing words), the read count, one fence. Returns k.
        uint32_t push(const uint32_t* recs, size_t n) {
            const uint32_t k = static_cast<uint32_t>(std::min<size_t>(n, room()));
            store(recs, k);
            return k;
        }

        // Queues n records for wait(), stored in order as the ring has room (the
        // kernel orders KM_F_EXCL ones). Returns the number queued.
        size_t enqueue(const uint32_t* recs, size_t n) {
            require_open();
            queue_.insert(queue_.end(), recs, recs + 16 * n);
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
            const uint32_t lo = wanted.empty() ? 0 : wanted.front();
            std::vector<uint8_t> need(wanted.empty() ? 0 : wanted.back() - lo + 1u, 0);
            size_t left = 0;
            for (const uint32_t c : wanted) {
                left += need[c - lo] == 0;
                need[c - lo] = 1;
            }
            const auto by = after(timeout_s);
            std::vector<uint32_t> out;
            for (;;) {
                store_queued(upto);
                const size_t had = out.size();
                read_events(out);
                publish_if_held();
                bool other = false;
                for (size_t e = had; e < out.size(); e += 4) {
                    const uint32_t off = out[e + 1] - lo;
                    if (off < need.size() && need[off] != 0) {
                        need[off] = 0;
                        left--;
                    }
                    other = other || (out[e + 2] & 0xFFu) != KM_EV_DONE;
                }
                if (left == 0 || other || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > by) {
                    return out;
                }
                if (out.size() == had) {
                    _mm_pause();
                }
            }
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
                const uint32_t i = seen_ + 1;
                const volatile uint32_t* e = ring + (i % event_depth_) * 4;
                if (e[0] != i) {
                    return;
                }
                _mm_lfence();
                const uint32_t req = e[1], status = e[2], ticks = e[3];
                _mm_lfence();
                if (e[0] != i) {
                    continue;
                }
                out.insert(out.end(), {i, req, status, ticks});
                seen_ = i;
            }
        }

        // poll() until at least one event arrives, the kernel reaches a final state, or
        // timeout_s passes.
        std::vector<uint32_t> wait_events(double timeout_s) {
            const auto by = after(timeout_s);
            for (;;) {
                auto out = poll();
                publish_if_held();
                if (!out.empty() || host_[KMD_HOST_STATE] >= KMD_STATE_DONE ||
                    Clock::now() > by) {
                    auto rest = poll();
                    out.insert(out.end(), rest.begin(), rest.end());
                    return out;
                }
                _mm_pause();
            }
        }

        // Waits (bounded) for the kernel's hart result, stops the session, reads the
        // result block: result ready, hart result, cleaned, what failed, words, harts.
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
                const auto res = (
                    rt().session(*session_).wait_until_result_ready(millis(timeout_s))
                );
                const auto done = rt().cleanup(*session_, std::chrono::seconds(5));
                session_.reset();
                ready = res.succeeded();
                if (ready) {
                    result = res.hartstates[0].state.result;
                } else {
                    failed = x280::describe(res);
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
            const auto hb = (
                tile_->read_available_lim(base_ + KMD_LIM_HARTS, 4 * KMD_H_WORDS)
            );
            std::vector<uint32_t> harts(KMD_H_WORDS);
            std::memcpy(harts.data(), hb.data(), 4 * KMD_H_WORDS);
            nb::dict out;
            out["result_ready"] = ready;
            out["result"] = ready ? nb::object(nb::int_(result)) : nb::none();
            out["cleaned"] = cleaned;
            out["failed"] = failed;
            out["words"] = words;
            out["harts"] = harts;  // the hart block (KMD_H_*)
            return out;
        }

        // Stops the session now if one runs (any failure path): whether the runtime
        // restored the tile (else close_failed says why); aborted() keeps the blocks.
        bool abort() {
            if (!session_ || !opened_) {
                session_.reset();
                return true;
            }
            const auto done = rt().cleanup(*session_, std::chrono::seconds(5));
            session_.reset();
            note_failed("cluster 0", done);
            try {
                aborted_ = {
                    lim_words(KMD_LIM_RESULT, KMD_R_WORDS),
                    lim_words(KMD_LIM_HARTS, KMD_H_WORDS)
                };
            } catch (const std::exception& error) {
                note_failed("cluster 0's blocks", error.what());
            }
            return done.cleaned_up();
        }

        // After abort() stopped a kernel: {words, harts} as it left them, or None.
        nb::object aborted() const {
            if (!aborted_) {
                return nb::none();
            }
            nb::dict out;
            out["words"] = aborted_->first;
            out["harts"] = aborted_->second;  // the hart block (KMD_H_*)
            return out;
        }

        // While the kernel runs: the result and hart blocks (core 0's pass stamp) read
        // twice gap_s apart, and the runtime's view of each hart.
        nb::dict probe(double gap_s) {
            require_open();
            nb::dict out;
            out["running"] = session_.has_value();
            if (!session_) {
                return out;
            }
            std::vector<uint32_t> before, after, words;
            std::string harts, failed;
            {
                nb::gil_scoped_release release;
                try {
                    before = lim_words(KMD_LIM_HARTS, KMD_H_WORDS);
                    std::this_thread::sleep_for(
                        std::chrono::duration<double>(std::clamp(gap_s, 0.0, 1.0))
                    );
                    after = lim_words(KMD_LIM_HARTS, KMD_H_WORDS);
                    words = lim_words(KMD_LIM_RESULT, KMD_R_WORDS);
                } catch (const std::exception& error) {
                    failed = std::string("LIM not readable: ") + error.what();
                }
                try {
                    harts = x280::describe(rt().session(*session_).inspect());
                } catch (const std::exception& error) {
                    harts = std::string("hart states not readable: ") + error.what();
                }
            }
            if (failed.empty()) {
                out["stamp_before"] = before[KMD_H_STAMP0];
                out["stamp_after"] = after[KMD_H_STAMP0];
                out["words"] = words;
            }
            out["failed"] = failed;
            out["harts"] = harts;
            return out;
        }

        // Work clusters 1 to 3: one image each, started with an argument block at the
        // start of that cluster's available LIM. Cluster 0 is the dispatch kernel's.
        uint32_t clusters() const {
            return static_cast<uint32_t>(work_.size());
        }

        // The cluster's hardware index, NoC0 coordinate and available LIM.
        nb::dict work_info(uint32_t c) const {
            const Work& w = work(c, true);
            nb::dict out;
            out["hw_index"] = w.tile->id().hw_index;
            out["noc_x"] = w.tile->id().noc_x;
            out["noc_y"] = w.tile->id().noc_y;
            // The coordinate another cluster's window aims at (as for cluster 0's).
            const auto v = (
                rt().virtual_core_from_tile(w.tile->id(), tt::tt_metal::NOC::NOC_0)
            );
            out["virtual_x"] = static_cast<uint32_t>(v.x());
            out["virtual_y"] = static_cast<uint32_t>(v.y());
            out["lim_base"] = w.base;
            out["lim_bytes"] = w.bytes;
            return out;
        }

        // Whole 64-byte lines at `offset` from the cluster's LIM base.
        void work_write(uint32_t c, uint64_t offset, const nb::bytes& data) {
            Work& w = work(c, true);
            if (offset % 64 != 0 || data.size() % 64 != 0 ||
                offset + data.size() > w.bytes) {
                throw std::runtime_error("LIM writes are whole lines inside the area");
            }
            const auto* p = static_cast<const uint8_t*>(data.data());
            w.tile->write_available_lim(w.base + offset, {p, data.size()});
        }

        nb::bytes work_read(uint32_t c, uint64_t offset, uint64_t n) const {
            const Work& w = work(c, true);
            if (offset % 4 != 0 || n % 4 != 0 || offset + n > w.bytes) {
                throw std::runtime_error("LIM reads are whole words inside the area");
            }
            const auto got = w.tile->read_available_lim(w.base + offset, n);
            return nb::bytes(reinterpret_cast<const char*>(got.data()), got.size());
        }

        // Writes `block` (whole lines; its first 8 bytes become the run value) at the
        // cluster's LIM base and starts `image` with it. Returns the run value.
        uint64_t work_start(uint32_t c, const std::string& image, nb::bytes block) {
            Work& w = work(c, true);
            if (w.session) {
                throw std::runtime_error("an image is running on that cluster");
            }
            if (block.size() < 64 || block.size() % 64 != 0 || block.size() > w.bytes) {
                throw std::runtime_error("the argument block is whole 64-byte lines");
            }
            std::vector<uint8_t> bytes(
                static_cast<const uint8_t*>(block.data()),
                static_cast<const uint8_t*>(block.data()) + block.size()
            );
            std::random_device rd;
            w.run =
                ((0x6b6d780000000000ull ^ (static_cast<uint64_t>(rd()) << 16)) + rd() +
                 c);
            std::memcpy(bytes.data(), &w.run, sizeof(w.run));
            w.tile->write_available_lim(w.base, bytes);
            x280::ActiveRunRequest req;
            req.image_path = image;
            req.run_value = w.run;
            req.args_addr = w.base;
            req.args_bytes = bytes.size();
            const auto enq = rt().enqueue(*w.tile, req);
            if (!enq.session) {
                throw std::runtime_error("X280 enqueue failed: " + x280::describe(enq));
            }
            w.session = enq.session;
            return w.run;
        }

        bool work_running(uint32_t c) const {
            return work(c, false).session.has_value();
        }

        // Waits (bounded) until every hart of the cluster has returned from main:
        // ready, the four hart results, and the runtime's description otherwise.
        nb::dict work_wait(uint32_t c, double timeout_s) {
            Work& w = work(c, true);
            if (!w.session) {
                throw std::runtime_error("no image running on that cluster");
            }
            x280::WaitResult res;
            {
                nb::gil_scoped_release release;
                res = (
                    rt().session(*w.session).wait_until_result_ready(millis(timeout_s))
                );
            }
            std::vector<uint64_t> results, stages;
            for (const auto& h : res.hartstates) {
                results.push_back(h.state.result);
                stages.push_back(h.state.stage);
            }
            nb::dict out;
            out["ready"] = res.outcome == x280::WaitOutcome::ResultReady;
            out["results"] = results;
            out["stages"] = stages;
            out["failed"] = res.succeeded() ? std::string() : x280::describe(res);
            return out;
        }

        // Stops the cluster's image through the runtime (bounded): cleaned, and the
        // runtime's description when it did not return to idle.
        nb::dict work_stop(uint32_t c, double timeout_s) {
            Work& w = work(c, true);
            if (!w.session) {
                throw std::runtime_error("no image running on that cluster");
            }
            x280::CleanupResult done;
            {
                nb::gil_scoped_release release;
                done = rt().cleanup(*w.session, millis(timeout_s));
            }
            w.session.reset();
            nb::dict out;
            out["cleaned"] = done.cleaned_up();
            out["failed"] = done.cleaned_up() ? std::string() : x280::describe(done);
            return out;
        }

        // Stops the cluster's image if one runs (any failure path); whether it
        // returned to idle.
        bool work_abort(uint32_t c) {
            if (c >= work_.size() || !work_[c].session || !opened_) {
                return true;
            }
            const auto done = rt().cleanup(*work_[c].session, std::chrono::seconds(5));
            work_[c].session.reset();
            note_failed("cluster " + std::to_string(c), done);
            return done.cleaned_up();
        }

        // The runtime's account of every clean-up since the last close() that did not
        // restore its cluster ("" when all did).
        const std::string& close_failed() const {
            return close_failed_;
        }

        uint32_t batch_ = 64;  // commands per store: every line, then one fence
        uint32_t gx_ = 0, gy_ = 0, l2x_ = 0, l2y_ = 0;
        uint64_t base_ = 0;  // LIM address of the layout (the argument block)
        std::vector<std::array<uint32_t, 2>> panels_;  // logical column ranges
        uint32_t tag_ = 0, depth_ = 1, host_depth_ = 1, event_depth_ = 1;
        uint32_t next_ = 1;       // number of the next command
        uint32_t seen_ = 0;       // events read from the ring
        uint32_t published_ = 0;  // the read count last stored into LIM
        double settle_us_ = 1.0;  // pause between a fence and the next store
        Clock::time_point fenced_{};
        uint64_t upload_bytes_ = 0;  // the upload region's size

    private:
        std::string close_failed_;
        std::optional<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> aborted_;

        x280::X280Runtime& rt() const {
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

        std::vector<uint32_t> lim_words(uint64_t offset, uint32_t n) const {
            const auto block = tile_->read_available_lim(base_ + offset, 4 * n);
            std::vector<uint32_t> words(n);
            std::memcpy(words.data(), block.data(), 4 * n);
            return words;
        }

        void require_open() const {
            if (!opened_) {
                throw std::runtime_error("the X280 runtime is closed");
            }
        }

        // Stores k records as the next commands, every line then one fence.
        void store(const uint32_t* recs, uint32_t k) {
            if (k == 0) {
                return;
            }
            settle();
            alignas(64) uint32_t w[16];
            for (uint32_t j = 0; j < k; j++) {
                const uint32_t i = next_ + j;
                std::memcpy(w, recs + 16 * j, sizeof(w));
                w[0] = w[4] = w[8] = w[12] = i;
                win_->write_block(KMD_LIM_HRING + (i % host_depth_) * 64u, w, 64);
            }
            store_read();
            _mm_sfence();
            fenced_ = Clock::now();
            next_ += k;
        }

        // One batch from the queue: the commands up to upto that fit. An exclusive
        // record is stored like any other; the kernel's KM_F_EXCL check orders it.
        void store_queued(uint32_t upto) {
            const int64_t due = static_cast<int32_t>(upto - next_) + int64_t{1};
            size_t k = std::min<size_t>({queued(), batch_, room()});
            k = std::min<int64_t>(static_cast<int64_t>(k), std::max<int64_t>(0, due));
            store(queue_.data() + 16 * head_, static_cast<uint32_t>(k));
            head_ += k;
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

        // Work cluster c (1 to 3); `open` also requires the runtime to be open.
        const Work& work(uint32_t c, bool open) const {
            if (open) {
                require_open();
            }
            if (c == 0 || c >= work_.size()) {
                throw std::runtime_error("work clusters are 1 to clusters() - 1");
            }
            return work_[c];
        }

        Work& work(uint32_t c, bool open) {
            return const_cast<Work&>(std::as_const(*this).work(c, open));
        }

        // Boots the tiles to their idle image if they are not idle, then sets the
        // clock.
        void boot(uint32_t pll_mhz) {
            auto& r = rt();
            bool all_idle = true;
            for (const auto& id : r.tile_ids()) {
                all_idle = all_idle && r.tile(id).inspect().resident_idle_live;
            }
            if (!all_idle) {
                if (idle_image_path.empty()) {
                    throw std::runtime_error(
                        "X280 boot: set_idle_image was not called"
                    );
                }
                x280::MultiTileBootRequest req;
                req.tiles = r.tile_ids();
                req.idle_image_path = idle_image_path;
                req.allow_reset_release = true;
                if (!r.boot_tiles(req).ready()) {
                    throw std::runtime_error("X280 boot_tiles failed");
                }
            }
            r.set_pll(r.tile_ids().front().chip_id, pll_mhz);
        }

        std::array<uint32_t, 2> translated(uint32_t x, uint32_t y) const {
            const auto t = x280::X280Runtime::virtual_core_from_logical_core(
                *mesh_,
                tt::tt_metal::CoreCoord{x, y},
                tt::CoreType::WORKER,
                tt::tt_metal::NOC::NOC_0
            );
            return {static_cast<uint32_t>(t.x()), static_cast<uint32_t>(t.y())};
        }

        // Cluster 0, its LIM layout, the grid's panels, the host window and buffer.
        void locate() {
            auto& r = rt();
            const auto ids = r.tile_ids();
            tile_ = &r.tile(ids.front());
            work_.resize(ids.size());
            for (size_t c = 0; c < ids.size(); c++) {
                work_[c].tile = &r.tile(ids[c]);
                const auto lim = work_[c].tile->available_lim();
                work_[c].base = (lim.begin + 63) / 64 * 64;
                work_[c].bytes = lim.end > work_[c].base ? lim.end - work_[c].base : 0;
            }
            const auto l2 = (
                r.virtual_core_from_tile(ids.front(), tt::tt_metal::NOC::NOC_0)
            );
            // The writers push completions on NoC1: the same coordinate must reach it.
            const auto l2_noc1 = (
                r.virtual_core_from_tile(ids.front(), tt::tt_metal::NOC::NOC_1)
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
            const auto g = mesh_->compute_with_storage_grid_size();
            gx_ = g.x;
            gy_ = g.y;
            // Panels: runs of logical columns whose NoC0 x are consecutive.
            const auto& soc = (
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                    mesh_->get_devices().front()->id()
                )
            );
            auto phys_x = [&](uint32_t lx) {
                const auto t = translated(lx, 0);
                return (
                    soc.translate_coord_to(
                                      tt::umd::CoreCoord(
                                          t[0],
                                          t[1],
                                          tt::CoreType::TENSIX,
                                          tt::CoordSystem::TRANSLATED
                                      ),
                                      tt::CoordSystem::NOC0
                    )
                    .x
                );
            };
            for (uint32_t x = 0; x < gx_; x++) {
                if (!panels_.empty() && phys_x(x) == phys_x(panels_.back()[1]) + 1) {
                    panels_.back()[1] = x;
                } else {
                    panels_.push_back({x, x});
                }
            }
            if (panels_.size() > 2 || gx_ > KMD_READY_MAX || gx_ * gy_ > KM_CORES_MAX) {
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
        std::unique_ptr<x280::RuntimeOpenResult> opened_;
        x280::X280Tile* tile_ = nullptr;
        uint64_t run_ = 0;
        std::shared_ptr<tt::umd::IoWindow> win_;         // host -> LIM, write-combining
        std::shared_ptr<tt::umd::SysmemBuffer> buffer_;  // kernel -> pinned host memory
        volatile uint32_t* host_ = nullptr;
        uint64_t host_noc_ = 0;
        uint32_t pcie_x_ = 0, pcie_y_ = 0;
        std::optional<x280::SessionId> session_;
        std::vector<uint32_t> queue_;  // records not stored yet, 16 words each
        size_t head_ = 0;              // the first record not stored
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
        // A std::shared_ptr<MeshDevice>, for native modules of their own that take
        // the device; the capsule's name is checked on the receiving side.
        .def(
            "mesh_capsule",
            [](const Device& d) {
                using Held = std::shared_ptr<MeshDevice>;
                return nb::capsule(
                    new Held(d.shared()), "komorebi.mesh_device", [](void* p) noexcept {
                        delete static_cast<Held*>(p);
                    }
                );
            }
        )
        .def("finish", [](const Device& d) {
            nb::gil_scoped_release release;
            d.finish();
        });
    nb::class_<DeviceBuffer>(m, "Buffer")
        .def(
            nb::init<
                const Device&,
                uint64_t,
                uint64_t,
                std::optional<std::array<uint32_t, 2>>>(),
            nb::arg("device"),
            nb::arg("size"),
            nb::arg("page_size"),
            nb::arg("shard_grid") = nb::none()
        )
        .def_prop_ro("address", &DeviceBuffer::address)
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
            nb::arg("total_size")
        )
        .def(
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
        )
        .def("build", &Launch::build)
        .def("run", &Launch::run);
    m.def(
        "write_l1",
        [](const Device& d,
           uint32_t x,
           uint32_t y,
           uint32_t address,
           std::vector<uint32_t> words) {
            write_l1(d.mesh(), x, y, address, std::move(words));
        },
        nb::arg("device"),
        nb::arg("x"),
        nb::arg("y"),
        nb::arg("address"),
        nb::arg("words")
    );
    m.def(
        "core_nocs",
        [](const Device& d) {
            return core_nocs(d.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "dram_banks",
        [](const Device& d) {
            return dram_banks(d.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "kernel_buffer",
        [](const Device& d) {
            return kernel_buffer(d.mesh());
        },
        nb::arg("device")
    );
    m.def(
        "read_dram",
        [](const Device& d, std::vector<std::array<uint64_t, 3>> spans) {
            return read_dram(d.mesh(), spans);
        },
        nb::arg("device"),
        nb::arg("spans")
    );
    m.def(
        "wait_grid",
        [](const Device& d, uint32_t gx, uint32_t gy, double timeout_s) {
            return wait_grid(d.mesh(), gx, gy, timeout_s);
        },
        nb::arg("device"),
        nb::arg("grid_x"),
        nb::arg("grid_y"),
        nb::arg("timeout_s")
    );

    nb::class_<Dispatch>(m, "Dispatch")
        .def(
            "__init__",
            [](Dispatch* self, const Device& d, uint32_t pll_mhz) {
                new (self) Dispatch(d.shared(), pll_mhz);
            },
            nb::arg("device"),
            nb::arg("pll_mhz") = 1000
        )
        .def_ro("grid_x", &Dispatch::gx_)
        .def_ro("grid_y", &Dispatch::gy_)
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
        .def_rw("settle_us", &Dispatch::settle_us_)
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
            nb::arg("t_record_s"),
            nb::arg("t_idle_s"),
            nb::arg("state_ddr"),
            nb::arg("work_cmds") = 0,
            nb::arg("work_depth") = 0,
            nb::arg("work_event_depth") = 0,
            nb::arg("work_noc") = std::vector<uint64_t>{},
            nb::arg("work_noc_addr") = std::vector<uint64_t>{},
            nb::arg("upload_bytes") = -1,
            nb::arg("dyn_seg") = 0
        )
        .def("start", &Dispatch::start, nb::arg("image"))
        .def("host_word", &Dispatch::host_word, nb::arg("word"))
        .def_ro("upload_bytes", &Dispatch::upload_bytes_)
        .def("upload", &Dispatch::upload, nb::arg("offset"), nb::arg("data"))
        .def("publish", &Dispatch::publish)
        .def(
            "wait_ready",
            [](const Dispatch& d, double timeout_s) {
                nb::gil_scoped_release release;
                return d.wait_ready(timeout_s);
            },
            nb::arg("timeout_s")
        )
        .def("room", &Dispatch::room)
        .def(
            "push",
            [](Dispatch& d, Records recs) {
                if (recs.shape(1) != 16) {
                    throw std::runtime_error("records are [n, 16] words");
                }
                return d.push(recs.data(), recs.shape(0));
            },
            nb::arg("records")
        )
        .def(
            "enqueue",
            [](Dispatch& d, Records recs) {
                if (recs.shape(1) != 16) {
                    throw std::runtime_error("records are [n, 16] words");
                }
                return d.enqueue(recs.data(), recs.shape(0));
            },
            nb::arg("records")
        )
        .def(
            "wait",
            [](Dispatch& d, Numbers wanted, uint32_t upto, double timeout_s) {
                std::vector<uint32_t> want(
                    wanted.data(), wanted.data() + wanted.shape(0)
                );
                std::vector<uint32_t> out;
                {
                    nb::gil_scoped_release release;
                    out = d.wait(want, upto, timeout_s);
                }
                return as_array(std::move(out));
            },
            nb::arg("wanted"),
            nb::arg("upto"),
            nb::arg("timeout_s")
        )
        .def(
            "poll",
            [](Dispatch& d) {
                return as_array(d.poll());
            }
        )
        .def(
            "wait_events",
            [](Dispatch& d, double timeout_s) {
                std::vector<uint32_t> out;
                {
                    nb::gil_scoped_release release;
                    out = d.wait_events(timeout_s);
                }
                return as_array(std::move(out));
            },
            nb::arg("timeout_s")
        )
        .def("finish", &Dispatch::finish, nb::arg("timeout_s"))
        .def("abort", &Dispatch::abort)
        .def("close", &Dispatch::close)
        .def_prop_ro("close_failed", &Dispatch::close_failed)
        .def("aborted", &Dispatch::aborted)
        .def("probe", &Dispatch::probe, nb::arg("gap_s"))
        .def("clusters", &Dispatch::clusters)
        .def("work_info", &Dispatch::work_info, nb::arg("cluster"))
        .def(
            "work_write",
            &Dispatch::work_write,
            nb::arg("cluster"),
            nb::arg("offset"),
            nb::arg("data")
        )
        .def(
            "work_read",
            &Dispatch::work_read,
            nb::arg("cluster"),
            nb::arg("offset"),
            nb::arg("n")
        )
        .def(
            "work_start",
            &Dispatch::work_start,
            nb::arg("cluster"),
            nb::arg("image"),
            nb::arg("block")
        )
        .def("work_running", &Dispatch::work_running, nb::arg("cluster"))
        .def(
            "work_wait", &Dispatch::work_wait, nb::arg("cluster"), nb::arg("timeout_s")
        )
        .def(
            "work_stop", &Dispatch::work_stop, nb::arg("cluster"), nb::arg("timeout_s")
        )
        .def("work_abort", &Dispatch::work_abort, nb::arg("cluster"));
}
