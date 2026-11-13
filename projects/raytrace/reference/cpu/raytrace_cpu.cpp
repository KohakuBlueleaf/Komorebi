// Ray tracer on the host CPU: one ray at a time with a stack, threads over pixels.
// Usage: raytrace_cpu --scene DIR --tree DIR --out DIR [--threads N] [--reps R]
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "host.hpp"
#include "raytrace_core.hpp"

using namespace kr;

constexpr int PIXELS = WIDTH * HEIGHT;

struct Buffers {
    std::vector<Hit> hit0, hit1;
    std::vector<float> normal, origin, albedo, direct0, bounce_dir, radiance1, colour;

    Buffers() :
        hit0(PIXELS),
        hit1(PIXELS),
        normal(3 * PIXELS),
        origin(3 * PIXELS),
        albedo(3 * PIXELS),
        direct0(PIXELS),
        bounce_dir(3 * PIXELS),
        radiance1(3 * PIXELS),
        colour(3 * PIXELS) {
    }
};

static Frame frame_of(Buffers& b) {
    return {
        b.hit0.data(),
        b.hit1.data(),
        b.normal.data(),
        b.origin.data(),
        b.albedo.data(),
        b.direct0.data(),
        b.bounce_dir.data(),
        b.radiance1.data(),
        b.colour.data()
    };
}

struct Times {
    double primary, shadow1, bounce, shadow2, shade;
    long rays_shadow1, rays_bounce, rays_shadow2;
};

constexpr int GRAIN = 256;

template <class F>
static double stage(Pool& pool, F body) {
    double t0 = now_ms();
    pool.for_chunks(PIXELS, GRAIN, [&](int64_t a, int64_t e) {
        for (int64_t p = a; p < e; p++) {
            body((int)p);
        }
    });
    return now_ms() - t0;
}

static Times run(const Scene& s, Pool& pool, Buffers& buffers) {
    Frame b = frame_of(buffers);
    Times t;
    t.primary = stage(pool, [&](int p) {
        primary_ray(s, b, p);
    });
    t.shadow1 = stage(pool, [&](int p) {
        shadow_primary(s, b, p);
    });
    t.bounce = stage(pool, [&](int p) {
        bounce_ray(s, b, p);
    });
    t.shadow2 = stage(pool, [&](int p) {
        shadow_bounce(s, b, p);
    });
    t.shade = stage(pool, [&](int p) {
        shade(s, b, p);
    });
    long hit0 = 0, hit1 = 0;
    for (int p = 0; p < PIXELS; p++) {
        hit0 += b.hit0[p].leaf >= 0;
        hit1 += b.hit0[p].leaf >= 0 && b.hit1[p].leaf >= 0;
    }
    t.rays_shadow1 = t.rays_bounce = hit0;
    t.rays_shadow2 = hit1;
    return t;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene_dir = args.get("scene"), tree_dir = args.get("tree");
    std::string out = args.get("out");
    int n_threads = args.geti("threads", (int)std::thread::hardware_concurrency());
    int reps = args.geti("reps", 3);
    Array colors = load_npy(scene_dir + "/colors.npy", 4);
    Array indices = load_npy(scene_dir + "/indices.npy", 4);
    Array camera = load_npy(scene_dir + "/camera.npy", 4);
    Array light = load_npy(scene_dir + "/light.npy", 4);
    Array eps = load_npy(scene_dir + "/eps.npy", 4);
    Array nodes = load_npy(tree_dir + "/nodes.npy", 4);
    Array tris = load_npy(tree_dir + "/tris.npy", 4);
    Array leaf_tri = load_npy(tree_dir + "/leaf_tri.npy", 4);
    Scene s;
    s.tree = {nodes.data<uint32_t>(), tris.data<float>(), leaf_tri.data<int32_t>()};
    s.colors = colors.data<float>();
    s.indices = indices.data<int32_t>();
    std::copy(camera.data<float>(), camera.data<float>() + 16, s.camera);
    std::copy(light.data<float>(), light.data<float>() + 3, s.light);
    s.eps = eps.data<float>()[0];

    Pool pool(n_threads);
    Buffers b;
    Times last = run(s, pool, b);  // warm-up: page faults, thread start
    RepLog log(reps);
    for (int rep = 0; rep < reps; rep++) {
        last = run(s, pool, b);
        char record[320];
        std::snprintf(
            record,
            sizeof record,
            "{\"primary_ms\": %.4f, \"shadow_ms\": %.4f, \"bounce_ms\": %.4f, "
            "\"shadow2_ms\": %.4f, \"shade_ms\": %.4f, \"total_ms\": %.4f}",
            last.primary,
            last.shadow1,
            last.bounce,
            last.shadow2,
            last.shade,
            last.primary + last.shadow1 + last.bounce + last.shadow2 + last.shade
        );
        log.add(record);
    }
    double loop_ms = log.finish();
    std::printf(
        "{\"impl\": \"cpu\", \"threads\": %d, \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"rays_primary\": %d, \"rays_shadow\": %ld, "
        "\"rays_bounce\": %ld, \"rays_shadow2\": %ld, \"reps\": [%s]}\n",
        n_threads,
        (int)leaf_tri.shape[0],
        reps,
        loop_ms,
        PIXELS,
        last.rays_shadow1,
        last.rays_bounce,
        last.rays_shadow2,
        log.body.c_str()
    );
    if (!out.empty()) {
        std::vector<float> t(PIXELS);
        std::vector<int32_t> tri(PIXELS), visits(PIXELS), leaves(PIXELS);
        for (int p = 0; p < PIXELS; p++) {
            const Hit& h = b.hit0[p];
            t[p] = h.t;
            tri[p] = h.leaf < 0 ? -1 : s.tree.leaf_tri[h.leaf];
            visits[p] = h.nodes;
            leaves[p] = h.leaves;
        }
        save_f32(out + "/image.npy", b.colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out + "/t.npy", t.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tri.npy", tri.data(), {HEIGHT, WIDTH});
        save_i32(out + "/nodes.npy", visits.data(), {HEIGHT, WIDTH});
        save_i32(out + "/leaves.npy", leaves.data(), {HEIGHT, WIDTH});
    }
    return 0;
}
