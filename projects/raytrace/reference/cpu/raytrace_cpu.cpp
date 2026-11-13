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
    std::vector<float> normal, origin, albedo, direct0, bounce_directions, radiance1,
        colour;

    Buffers() :
        hit0(PIXELS),
        hit1(PIXELS),
        normal(3 * PIXELS),
        origin(3 * PIXELS),
        albedo(3 * PIXELS),
        direct0(PIXELS),
        bounce_directions(3 * PIXELS),
        radiance1(3 * PIXELS),
        colour(3 * PIXELS) {
    }
};

static Frame frame_of(Buffers& buffers) {
    return {
        buffers.hit0.data(),
        buffers.hit1.data(),
        buffers.normal.data(),
        buffers.origin.data(),
        buffers.albedo.data(),
        buffers.direct0.data(),
        buffers.bounce_directions.data(),
        buffers.radiance1.data(),
        buffers.colour.data()
    };
}

struct Times {
    double primary, shadow1, bounce, shadow2, shade;
    long rays_shadow1, rays_bounce, rays_shadow2;
};

constexpr int GRAIN = 256;

template <class F>
static double stage(Pool& pool, F body) {
    double started = now_ms();
    pool.for_chunks(PIXELS, GRAIN, [&](int64_t first, int64_t end) {
        for (int64_t pixel = first; pixel < end; pixel++) {
            body((int)pixel);
        }
    });
    return now_ms() - started;
}

static Times run(const Scene& scene, Pool& pool, Buffers& buffers) {
    Frame frame = frame_of(buffers);
    Times times;
    times.primary = stage(pool, [&](int pixel) {
        primary_ray(scene, frame, pixel);
    });
    times.shadow1 = stage(pool, [&](int pixel) {
        shadow_primary(scene, frame, pixel);
    });
    times.bounce = stage(pool, [&](int pixel) {
        bounce_ray(scene, frame, pixel);
    });
    times.shadow2 = stage(pool, [&](int pixel) {
        shadow_bounce(scene, frame, pixel);
    });
    times.shade = stage(pool, [&](int pixel) {
        shade(scene, frame, pixel);
    });
    long primary_hit_count = 0, bounce_hit_count = 0;
    for (int pixel = 0; pixel < PIXELS; pixel++) {
        primary_hit_count += frame.hit0[pixel].leaf >= 0;
        bounce_hit_count += frame.hit0[pixel].leaf >= 0 && frame.hit1[pixel].leaf >= 0;
    }
    times.rays_shadow1 = times.rays_bounce = primary_hit_count;
    times.rays_shadow2 = bounce_hit_count;
    return times;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene_dir = args.get("scene"), tree_dir = args.get("tree");
    std::string out_folder = args.get("out");
    int thread_count = args.geti("threads", (int)std::thread::hardware_concurrency());
    int repetitions = args.geti("reps", 3);
    Array colors = load_npy(scene_dir + "/colors.npy", 4);
    Array indices = load_npy(scene_dir + "/indices.npy", 4);
    Array camera = load_npy(scene_dir + "/camera.npy", 4);
    Array light = load_npy(scene_dir + "/light.npy", 4);
    Array eps = load_npy(scene_dir + "/eps.npy", 4);
    Array nodes = load_npy(tree_dir + "/nodes.npy", 4);
    Array triangles = load_npy(tree_dir + "/tris.npy", 4);
    Array leaf_tri = load_npy(tree_dir + "/leaf_tri.npy", 4);
    Scene scene;
    scene.tree = {
        nodes.data<uint32_t>(), triangles.data<float>(), leaf_tri.data<int32_t>()
    };
    scene.colors = colors.data<float>();
    scene.indices = indices.data<int32_t>();
    std::copy(camera.data<float>(), camera.data<float>() + 16, scene.camera);
    std::copy(light.data<float>(), light.data<float>() + 3, scene.light);
    scene.eps = eps.data<float>()[0];

    Pool pool(thread_count);
    Buffers buffers;
    Times last = run(scene, pool, buffers);  // warm-up: page faults, thread start
    RepLog repetition_log(repetitions);
    for (int repetition = 0; repetition < repetitions; repetition++) {
        last = run(scene, pool, buffers);
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
        repetition_log.add(record);
    }
    double loop_ms = repetition_log.finish();
    std::printf(
        "{\"impl\": \"cpu\", \"threads\": %d, \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"rays_primary\": %d, \"rays_shadow\": %ld, "
        "\"rays_bounce\": %ld, \"rays_shadow2\": %ld, \"reps\": [%s]}\n",
        thread_count,
        (int)leaf_tri.shape[0],
        repetitions,
        loop_ms,
        PIXELS,
        last.rays_shadow1,
        last.rays_bounce,
        last.rays_shadow2,
        repetition_log.body.c_str()
    );
    if (!out_folder.empty()) {
        std::vector<float> distances(PIXELS);
        std::vector<int32_t> triangle_ids(PIXELS), visits(PIXELS), leaves(PIXELS);
        for (int pixel = 0; pixel < PIXELS; pixel++) {
            const Hit& hit = buffers.hit0[pixel];
            distances[pixel] = hit.t;
            triangle_ids[pixel] = hit.leaf < 0 ? -1 : scene.tree.leaf_tri[hit.leaf];
            visits[pixel] = hit.nodes;
            leaves[pixel] = hit.leaves;
        }
        save_f32(out_folder + "/image.npy", buffers.colour.data(), {HEIGHT, WIDTH, 3});
        save_f32(out_folder + "/t.npy", distances.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/tri.npy", triangle_ids.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/nodes.npy", visits.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/leaves.npy", leaves.data(), {HEIGHT, WIDTH});
    }
    return 0;
}
