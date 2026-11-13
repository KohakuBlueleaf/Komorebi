// Matrix product Y[m, n] = A[m, k] W[k, n] (row major) with cuBLAS on one GPU.
// Usage: gemm_gpu --m M --k K --n N --type fp32|fp16 [--seconds S]
// fp32: float in and out, float accumulation. fp16: half in and out, float accumulation
// (tensor cores). Prints one JSON line; LOOP_START and LOOP_END on stderr.
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#define CHECK_BLAS(call)                                                          \
    do {                                                                          \
        cublasStatus_t st_ = (call);                                              \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "%s:%d: cuBLAS %d\n", __FILE__, __LINE__, st_); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

    // Uniform in [-1, 1) from a 64-bit linear congruential generator.
    std::vector<float> uniform(size_t n, uint64_t seed) {
        std::vector<float> out(n);
        for (auto& x : out) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            x = static_cast<float>((seed >> 40) * (2.0 / 16777216.0) - 1.0);
        }
        return out;
    }

    struct Gemm {
        cublasHandle_t h;
        int m, k, n;
        bool half;
        void *a, *w, *y;

        void call() const {
            const float one = 1.0f, zero = 0.0f;
            const cudaDataType t = half ? CUDA_R_16F : CUDA_R_32F;
            // Row-major Y = A W is column-major Y^T = W^T A^T.
            CHECK_BLAS(cublasGemmEx(
                h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, w, t, n, a, t, k, &zero, y, t,
                n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
            ));
        }
    };

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

}  // namespace

int main(int argc, char** argv) {
    int m = 0, k = 0, n = 0;
    double seconds = 4.0;
    std::string type = "fp32";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--m") {
            m = std::atoi(argv[i + 1]);
        } else if (key == "--k") {
            k = std::atoi(argv[i + 1]);
        } else if (key == "--n") {
            n = std::atoi(argv[i + 1]);
        } else if (key == "--type") {
            type = argv[i + 1];
        } else if (key == "--seconds") {
            seconds = std::atof(argv[i + 1]);
        }
    }
    if (m <= 0 || k <= 0 || n <= 0 || (type != "fp32" && type != "fp16")) {
        std::fprintf(stderr, "usage: gemm_gpu --m M --k K --n N --type fp32|fp16\n");
        return 2;
    }
    const bool half = type == "fp16";
    const size_t elem = half ? 2 : 4;
    std::vector<float> a = uniform(size_t(m) * k, 1), w = uniform(size_t(k) * n, 2);
    std::vector<__half> ah, wh;
    if (half) {
        // The values the GPU sees, so the check below uses them too.
        for (auto* v : {&a, &w}) {
            auto& hv = v == &a ? ah : wh;
            hv.resize(v->size());
            for (size_t i = 0; i < v->size(); ++i) {
                hv[i] = __float2half((*v)[i]);
                (*v)[i] = __half2float(hv[i]);
            }
        }
    }
    Gemm g{nullptr, m, k, n, half, nullptr, nullptr, nullptr};
    CHECK(cudaMalloc(&g.a, a.size() * elem));
    CHECK(cudaMalloc(&g.w, w.size() * elem));
    CHECK(cudaMalloc(&g.y, size_t(m) * n * elem));
    CHECK(cudaMemcpy(g.a, half ? (void*)ah.data() : a.data(), a.size() * elem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(g.w, half ? (void*)wh.data() : w.data(), w.size() * elem, cudaMemcpyHostToDevice));
    CHECK_BLAS(cublasCreate(&g.h));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float ms = 0;
    CHECK(cudaEventRecord(e0));
    g.call();
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double first_ms = ms;
    // Check 256 entries against a double-precision dot product of the same values.
    std::vector<uint8_t> yraw(size_t(m) * n * elem);
    CHECK(cudaMemcpy(yraw.data(), g.y, yraw.size(), cudaMemcpyDeviceToHost));
    double err = 0, scale = 0;
    for (int s = 0; s < 256; ++s) {
        const size_t i = (size_t(s) * 2654435761ULL) % m, j = (size_t(s) * 40503ULL) % n;
        double want = 0;
        for (int q = 0; q < k; ++q) {
            want += double(a[i * k + q]) * w[size_t(q) * n + j];
        }
        const size_t at = i * n + j;
        const double got = half ? double(__half2float(reinterpret_cast<__half*>(yraw.data())[at]))
                                : double(reinterpret_cast<float*>(yraw.data())[at]);
        err = std::max(err, std::fabs(got - want));
        scale = std::max(scale, std::fabs(want));
    }
    // Synced calls, then a back-to-back loop of about `seconds`.
    std::vector<double> synced;
    for (int r = 0; r < 20; ++r) {
        CHECK(cudaEventRecord(e0));
        g.call();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        synced.push_back(ms);
    }
    const double one_ms = median(synced);
    const int reps = std::max(10, int(seconds * 1e3 / std::max(one_ms, 1e-3)));
    std::fprintf(stderr, "LOOP_START\n");
    std::fflush(stderr);
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) {
        g.call();
    }
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    std::fprintf(stderr, "LOOP_END\n");
    std::fflush(stderr);
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf(
        "{\"device\": \"%s\", \"type\": \"%s\", \"m\": %d, \"k\": %d, \"n\": %d, "
        "\"first_ms\": %.4f, \"synced_ms_median\": %.5f, \"synced_ms_min\": %.5f, "
        "\"reps_n\": %d, \"loop_ms\": %.4f, \"microseconds_per_call\": %.4f, \"tflops\": %.3f, "
        "\"sample_max_error_over_max\": %.3g}\n",
        prop.name, type.c_str(), m, k, n, first_ms, one_ms,
        *std::min_element(synced.begin(), synced.end()), reps, ms, ms * 1e3 / reps,
        2.0 * m * k * n / (ms * 1e-3 / reps) / 1e12, err / scale
    );
    cublasDestroy(g.h);
    return 0;
}
