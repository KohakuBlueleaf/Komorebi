// _collision_x280: the landing experiment's host module under another name, starting
// x280/command_seq.c, plus Plan.translated(x, y) for the broadcast rectangles.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ttx_dispatch_host.hpp"  // tt-x280-dispatch v1, included unchanged

namespace nb = nanobind;
using tt::tt_metal::distributed::MeshDevice;

namespace {

    // The device in a komorebi Device.mesh_capsule().
    std::shared_ptr<MeshDevice> mesh_of(const nb::capsule& device) {
        const char* name = device.name();
        if (name == nullptr || std::strcmp(name, "komorebi.mesh_device") != 0) {
            throw std::invalid_argument("not a Device.mesh_capsule() capsule");
        }
        return *static_cast<std::shared_ptr<MeshDevice>*>(device.data());
    }

    // Must match landing_args_t in x280/command_seq.c.
    struct LandingArgs {
        uint64_t run_value, mailbox, acks, n, k, nrect;
        uint32_t rect[2][4];
        uint64_t ack_stride, ordering, proto, depth, tag;
    };

    static_assert(sizeof(LandingArgs) <= 0x100, "args must end before the mailbox");

    struct Runtime {
        std::shared_ptr<MeshDevice> mesh;
        std::unique_ptr<x280::RuntimeOpenResult> opened;

        Runtime(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz) :
            mesh(std::move(mesh_device)) {
            opened = std::make_unique<x280::RuntimeOpenResult>(x280::X280Runtime::open(
                mesh,
                x280::RuntimeOpenConfig{
                    .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
                }
            ));
            if (!opened->ready()) {
                throw std::runtime_error(
                    "X280 runtime not ready: " + opened->diagnostic
                );
            }
            auto& rt = runtime();
            bool all_idle = true;
            for (const auto& id : rt.tile_ids()) {
                all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
            }
            if (!all_idle) {
                x280::MultiTileBootRequest req;
                req.tiles = rt.tile_ids();
                req.idle_image_path = (
                    std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin"
                );
                req.allow_reset_release = true;
                if (!rt.boot_tiles(req).ready()) {
                    throw std::runtime_error("X280 boot_tiles failed");
                }
            }
            rt.set_pll(rt.tile_ids().front().chip_id, pll_mhz);
        }

        x280::X280Runtime& runtime() {
            return *opened->runtime;
        }
    };

    // ttx's DispatchPlan with this experiment's start().
    struct Plan : ttx::DispatchPlan {
        std::string image;
        uint32_t proto = 0;

        x280::SessionId start(uint64_t n_commands) {
            run = (
                0x6c616e6400000000ull ^
                (static_cast<uint64_t>(std::random_device{}()) << 8)
            ) +
                n_commands;
            tag = ((std::random_device{}() ^ static_cast<uint32_t>(run)) & 0xFFFFu) |
                1u;
            tile->write_available_lim(mailbox, std::vector<uint8_t>(64, 0));
            tile->write_available_lim(acks, std::vector<uint8_t>(2048, 0));
            LandingArgs a{};
            a.run_value = run;
            a.mailbox = mailbox;
            a.acks = acks;
            a.n = n_commands;
            a.k = gx;
            a.nrect = panels.size();
            for (size_t r = 0; r < panels.size(); r++) {
                const auto p0 = translated(panels[r][0], 0);
                const auto p1 = translated(panels[r][1], gy - 1);
                a.rect[r][0] = p0[0];
                a.rect[r][1] = p0[1];
                a.rect[r][2] = p1[0];
                a.rect[r][3] = p1[1];
            }
            a.ack_stride = 16;
            a.ordering = ordering;
            a.proto = proto;
            a.depth = depth;
            a.tag = tag;
            // LIM writes are whole 64 B lines; the mailbox starts at +0x100.
            std::vector<uint8_t> block((sizeof(LandingArgs) + 63) / 64 * 64, 0);
            std::memcpy(block.data(), &a, sizeof(a));
            tile->write_available_lim(args_addr, block);
            x280::ActiveRunRequest req;
            req.image_path = image;
            req.run_value = run;
            req.args_addr = args_addr;
            req.args_bytes = block.size();
            const auto enq = rt->enqueue(*tile, req);
            if (!enq.session) {
                throw std::runtime_error("X280 enqueue failed: " + x280::describe(enq));
            }
            return enq.session;
        }
    };

}  // namespace

NB_MODULE(_collision_x280, m) {
    m.attr("MTIME_HZ") = ttx::kMtimeHz;

    nb::class_<x280::SessionId>(m, "Session");

    nb::class_<Runtime>(m, "Runtime")
        .def(
            "__init__",
            [](Runtime* self, const nb::capsule& device, uint32_t pll_mhz) {
                new (self) Runtime(mesh_of(device), pll_mhz);
            },
            nb::arg("device"),
            nb::arg("pll_mhz") = 1750
        ).def(
            "plan",
            [](
                Runtime& self, const std::string& image, uint32_t depth, uint32_t proto
            ) {
                if (depth < 1 || depth > 4 || proto > 1) {
                    throw std::runtime_error("depth 1 to 4, proto 0 (A) or 1 (B)");
                }
                Plan p;
                static_cast<ttx::DispatchPlan&>(p) = ttx::DispatchPlan::make(
                    self.mesh, self.runtime(), depth
                );
                p.ring_stride = 64;
                p.image = image;
                p.proto = proto;
                return p;
            },
            nb::arg("image"),
            nb::arg("depth"),
            nb::arg("proto"),
            nb::keep_alive<0, 1>()
        );

    nb::class_<Plan>(m, "Plan")
        .def_ro("grid_x", &Plan::gx)
        .def_ro("grid_y", &Plan::gy)
        .def_ro("panels", &Plan::panels)
        .def_ro("tag", &Plan::tag)
        .def_ro("depth", &Plan::depth)
        .def_ro("proto", &Plan::proto)
        .def("aiclk_mhz", &Plan::aiclk_mhz)
        .def("translated", &Plan::translated, nb::arg("x"), nb::arg("y"))
        .def("worker_args", &Plan::worker_args, nb::arg("x"), nb::arg("y"))
        .def("start", &Plan::start, nb::arg("n_commands"))
        .def(
            "finish",
            [](Plan& p, const x280::SessionId& session, double timeout_s) {
                nb::gil_scoped_release release;
                return (
                    p.finish(
                        session, std::chrono::seconds(static_cast<int64_t>(timeout_s))
                    )
                );
            },
            nb::arg("session"),
            nb::arg("timeout_s") = 60.0
        );
}
