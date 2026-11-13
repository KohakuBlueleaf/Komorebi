// Ray tracer on one GPU: one thread per pixel for each stage (primary rays, shadow
// rays, bounce rays, shadow rays at the bounce hits, shade), stack in local memory.
// Usage: raytrace_gpu --scene DIR --tree DIR --out DIR [--reps R]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "host.hpp"
#include "raytrace_core.hpp"

using namespace kr;

#define CHECK(call)                                                                 \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            std::fprintf(                                                           \
                stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_) \
            );                                                                      \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

constexpr int PIXELS = WIDTH * HEIGHT;
constexpr int BLOCK = 128;

__global__ void primary_kernel(Scene scene, Frame frame) {
    int pixel = blockIdx.x * BLOCK + threadIdx.x;
    primary_ray(scene, frame, pixel);
}

__global__ void shadow_kernel(Scene scene, Frame frame) {
    int pixel = blockIdx.x * BLOCK + threadIdx.x;
    shadow_primary(scene, frame, pixel);
}

__global__ void bounce_kernel(Scene scene, Frame frame) {
    int pixel = blockIdx.x * BLOCK + threadIdx.x;
    bounce_ray(scene, frame, pixel);
}

__global__ void shadow2_kernel(Scene scene, Frame frame) {
    int pixel = blockIdx.x * BLOCK + threadIdx.x;
    shadow_bounce(scene, frame, pixel);
}

__global__ void shade_kernel(Scene scene, Frame frame) {
    int pixel = blockIdx.x * BLOCK + threadIdx.x;
    shade(scene, frame, pixel);
}

struct Times {
    float stage[5];
};

static Times run(const Scene& scene, const Frame& frame) {
    cudaEvent_t events[6];
    for (auto& event : events) {
        CHECK(cudaEventCreate(&event));
    }
    const int blocks = PIXELS / BLOCK;
    CHECK(cudaEventRecord(events[0]));
    primary_kernel<<<blocks, BLOCK>>>(scene, frame);
    CHECK(cudaEventRecord(events[1]));
    shadow_kernel<<<blocks, BLOCK>>>(scene, frame);
    CHECK(cudaEventRecord(events[2]));
    bounce_kernel<<<blocks, BLOCK>>>(scene, frame);
    CHECK(cudaEventRecord(events[3]));
    shadow2_kernel<<<blocks, BLOCK>>>(scene, frame);
    CHECK(cudaEventRecord(events[4]));
    shade_kernel<<<blocks, BLOCK>>>(scene, frame);
    CHECK(cudaEventRecord(events[5]));
    CHECK(cudaEventSynchronize(events[5]));
    CHECK(cudaGetLastError());
    Times times;
    for (int index = 0; index < 5; index++) {
        CHECK(
            cudaEventElapsedTime(&times.stage[index], events[index], events[index + 1])
        );
    }
    for (auto& event : events) {
        CHECK(cudaEventDestroy(event));
    }
    return times;
}

template <class T>
static T* upload(const Array& array) {
    T* device;
    CHECK(cudaMalloc(&device, array.bytes.size()));
    CHECK(cudaMemcpy(
        device, array.bytes.data(), array.bytes.size(), cudaMemcpyHostToDevice
    ));
    return device;
}

template <class T>
static T* device_buffer(size_t count) {
    T* device;
    CHECK(cudaMalloc(&device, count * sizeof(T)));
    CHECK(cudaMemset(device, 0, count * sizeof(T)));
    return device;
}

template <class T>
static std::vector<T> download(const T* device, size_t count) {
    std::vector<T> host(count);
    CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    std::string scene_dir = args.get("scene"), tree_dir = args.get("tree");
    std::string out_folder = args.get("out");
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
        upload<uint32_t>(nodes), upload<float>(triangles), upload<int32_t>(leaf_tri)
    };
    scene.colors = upload<float>(colors);
    scene.indices = upload<int32_t>(indices);
    std::copy(camera.data<float>(), camera.data<float>() + 16, scene.camera);
    std::copy(light.data<float>(), light.data<float>() + 3, scene.light);
    scene.eps = eps.data<float>()[0];
    Frame frame = {
        device_buffer<Hit>(PIXELS),
        device_buffer<Hit>(PIXELS),
        device_buffer<float>(3 * PIXELS),
        device_buffer<float>(3 * PIXELS),
        device_buffer<float>(3 * PIXELS),
        device_buffer<float>(PIXELS),
        device_buffer<float>(3 * PIXELS),
        device_buffer<float>(3 * PIXELS),
        device_buffer<float>(3 * PIXELS),
    };
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    run(scene, frame);  // warm-up: loads the code
    RepLog repetition_log(repetitions);
    for (int repetition = 0; repetition < repetitions; repetition++) {
        Times times = run(scene, frame);
        char record[320];
        std::snprintf(
            record,
            sizeof record,
            "{\"primary_ms\": %.4f, \"shadow_ms\": %.4f, \"bounce_ms\": %.4f, "
            "\"shadow2_ms\": %.4f, \"shade_ms\": %.4f, \"total_ms\": %.4f}",
            times.stage[0],
            times.stage[1],
            times.stage[2],
            times.stage[3],
            times.stage[4],
            times.stage[0] + times.stage[1] + times.stage[2] + times.stage[3] +
                times.stage[4]
        );
        repetition_log.add(record);
    }
    double loop_ms = repetition_log.finish();
    std::vector<Hit> primary_hits = download(frame.hit0, PIXELS);
    std::vector<Hit> bounce_hits = download(frame.hit1, PIXELS);
    long primary_hit_count = 0, bounce_hit_count = 0;
    for (int pixel = 0; pixel < PIXELS; pixel++) {
        primary_hit_count += primary_hits[pixel].leaf >= 0;
        bounce_hit_count += (
            primary_hits[pixel].leaf >= 0 && bounce_hits[pixel].leaf >= 0
        );
    }
    std::printf(
        "{\"impl\": \"gpu\", \"device\": \"%s\", \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"rays_primary\": %d, \"rays_shadow\": %ld, "
        "\"rays_bounce\": %ld, \"rays_shadow2\": %ld, \"reps\": [%s]}\n",
        properties.name,
        (int)leaf_tri.shape[0],
        repetitions,
        loop_ms,
        PIXELS,
        primary_hit_count,
        primary_hit_count,
        bounce_hit_count,
        repetition_log.body.c_str()
    );
    if (!out_folder.empty()) {
        std::vector<float> distances(PIXELS);
        std::vector<int32_t> triangle_ids(PIXELS), visits(PIXELS), leaves(PIXELS);
        for (int pixel = 0; pixel < PIXELS; pixel++) {
            distances[pixel] = primary_hits[pixel].t;
            triangle_ids[pixel] = (
                primary_hits[pixel].leaf < 0
                    ? -1
                    : leaf_tri.data<int32_t>()[primary_hits[pixel].leaf]
            );
            visits[pixel] = primary_hits[pixel].nodes;
            leaves[pixel] = primary_hits[pixel].leaves;
        }
        std::vector<float> image = download(frame.colour, 3 * PIXELS);
        save_f32(out_folder + "/image.npy", image.data(), {HEIGHT, WIDTH, 3});
        save_f32(out_folder + "/t.npy", distances.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/tri.npy", triangle_ids.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/nodes.npy", visits.data(), {HEIGHT, WIDTH});
        save_i32(out_folder + "/leaves.npy", leaves.data(), {HEIGHT, WIDTH});
    }
    return 0;
}
