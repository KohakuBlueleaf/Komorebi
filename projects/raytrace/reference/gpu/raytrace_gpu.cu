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

__global__ void primary_kernel(Scene s, Frame f) {
    int p = blockIdx.x * BLOCK + threadIdx.x;
    primary_ray(s, f, p);
}

__global__ void shadow_kernel(Scene s, Frame f) {
    int p = blockIdx.x * BLOCK + threadIdx.x;
    shadow_primary(s, f, p);
}

__global__ void bounce_kernel(Scene s, Frame f) {
    int p = blockIdx.x * BLOCK + threadIdx.x;
    bounce_ray(s, f, p);
}

__global__ void shadow2_kernel(Scene s, Frame f) {
    int p = blockIdx.x * BLOCK + threadIdx.x;
    shadow_bounce(s, f, p);
}

__global__ void shade_kernel(Scene s, Frame f) {
    int p = blockIdx.x * BLOCK + threadIdx.x;
    shade(s, f, p);
}

struct Times {
    float stage[5];
};

static Times run(const Scene& s, const Frame& f) {
    cudaEvent_t e[6];
    for (auto& ev : e) {
        CHECK(cudaEventCreate(&ev));
    }
    const int blocks = PIXELS / BLOCK;
    CHECK(cudaEventRecord(e[0]));
    primary_kernel<<<blocks, BLOCK>>>(s, f);
    CHECK(cudaEventRecord(e[1]));
    shadow_kernel<<<blocks, BLOCK>>>(s, f);
    CHECK(cudaEventRecord(e[2]));
    bounce_kernel<<<blocks, BLOCK>>>(s, f);
    CHECK(cudaEventRecord(e[3]));
    shadow2_kernel<<<blocks, BLOCK>>>(s, f);
    CHECK(cudaEventRecord(e[4]));
    shade_kernel<<<blocks, BLOCK>>>(s, f);
    CHECK(cudaEventRecord(e[5]));
    CHECK(cudaEventSynchronize(e[5]));
    CHECK(cudaGetLastError());
    Times t;
    for (int i = 0; i < 5; i++) {
        CHECK(cudaEventElapsedTime(&t.stage[i], e[i], e[i + 1]));
    }
    for (auto& ev : e) {
        CHECK(cudaEventDestroy(ev));
    }
    return t;
}

template <class T>
static T* upload(const Array& a) {
    T* p;
    CHECK(cudaMalloc(&p, a.bytes.size()));
    CHECK(cudaMemcpy(p, a.bytes.data(), a.bytes.size(), cudaMemcpyHostToDevice));
    return p;
}

template <class T>
static T* device_buffer(size_t count) {
    T* p;
    CHECK(cudaMalloc(&p, count * sizeof(T)));
    CHECK(cudaMemset(p, 0, count * sizeof(T)));
    return p;
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
    std::string out = args.get("out");
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
    s.tree = {upload<uint32_t>(nodes), upload<float>(tris), upload<int32_t>(leaf_tri)};
    s.colors = upload<float>(colors);
    s.indices = upload<int32_t>(indices);
    std::copy(camera.data<float>(), camera.data<float>() + 16, s.camera);
    std::copy(light.data<float>(), light.data<float>() + 3, s.light);
    s.eps = eps.data<float>()[0];
    Frame f = {
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
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    run(s, f);  // warm-up: loads the code
    RepLog log(reps);
    for (int rep = 0; rep < reps; rep++) {
        Times t = run(s, f);
        char record[320];
        std::snprintf(
            record,
            sizeof record,
            "{\"primary_ms\": %.4f, \"shadow_ms\": %.4f, \"bounce_ms\": %.4f, "
            "\"shadow2_ms\": %.4f, \"shade_ms\": %.4f, \"total_ms\": %.4f}",
            t.stage[0],
            t.stage[1],
            t.stage[2],
            t.stage[3],
            t.stage[4],
            t.stage[0] + t.stage[1] + t.stage[2] + t.stage[3] + t.stage[4]
        );
        log.add(record);
    }
    double loop_ms = log.finish();
    std::vector<Hit> hit0 = download(f.hit0, PIXELS), hit1 = download(f.hit1, PIXELS);
    long n0 = 0, n1 = 0;
    for (int p = 0; p < PIXELS; p++) {
        n0 += hit0[p].leaf >= 0;
        n1 += hit0[p].leaf >= 0 && hit1[p].leaf >= 0;
    }
    std::printf(
        "{\"impl\": \"gpu\", \"device\": \"%s\", \"triangles\": %d, \"reps_n\": %d, "
        "\"loop_ms\": %.4f, \"rays_primary\": %d, \"rays_shadow\": %ld, "
        "\"rays_bounce\": %ld, \"rays_shadow2\": %ld, \"reps\": [%s]}\n",
        prop.name,
        (int)leaf_tri.shape[0],
        reps,
        loop_ms,
        PIXELS,
        n0,
        n0,
        n1,
        log.body.c_str()
    );
    if (!out.empty()) {
        std::vector<float> t(PIXELS);
        std::vector<int32_t> tri(PIXELS), visits(PIXELS), leaves(PIXELS);
        for (int p = 0; p < PIXELS; p++) {
            t[p] = hit0[p].t;
            tri[p] = hit0[p].leaf < 0 ? -1 : leaf_tri.data<int32_t>()[hit0[p].leaf];
            visits[p] = hit0[p].nodes;
            leaves[p] = hit0[p].leaves;
        }
        std::vector<float> image = download(f.colour, 3 * PIXELS);
        save_f32(out + "/image.npy", image.data(), {HEIGHT, WIDTH, 3});
        save_f32(out + "/t.npy", t.data(), {HEIGHT, WIDTH});
        save_i32(out + "/tri.npy", tri.data(), {HEIGHT, WIDTH});
        save_i32(out + "/nodes.npy", visits.data(), {HEIGHT, WIDTH});
        save_i32(out + "/leaves.npy", leaves.data(), {HEIGHT, WIDTH});
    }
    return 0;
}
