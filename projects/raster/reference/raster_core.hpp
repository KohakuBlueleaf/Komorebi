// Triangle setup and pixel fill of the rasterizer, as written in ref/raster.py.
// float32 only, no fused multiply-add (build with -ffp-contract=off / --fmad=false).
#pragma once

#include <cstdint>

#include "kr.hpp"

#ifdef __CUDACC__
#define KR_NOINLINE __noinline__
#else
#define KR_NOINLINE inline
#endif

namespace kr {

    struct Mesh {
        const float* positions;  // (V, 3)
        const float* colors;     // (V, 3)
        const int32_t* indices;  // (T, 3)
        float camera[16];
        int n_tris;
    };

    // One setup record: what the fill needs of one triangle.
    struct Record {
        int32_t tri;             // triangle index, -1 when dropped
        int32_t box[4];          // pixel box x0, x1, y0, y1 (inclusive)
        float a[3], b[3], c[3];  // edge values  a x + b y + c
        uint32_t top_left;       // bit i: edge i is a top or left edge
        float w[3];              // 1/z plane
        float cp[3][3];          // colour planes [x, y, 1][channel]
    };

    // Camera-space x, y, z (along right, up, forward) of point p, as in ref/raster.py.
    KR_HD inline void camera_space(const float* cam, const float* p, float* c) {
        float dx = p[0] - cam[0], dy = p[1] - cam[1], dz = p[2] - cam[2];
        c[0] = (dx * cam[3] + dy * cam[4]) + dz * cam[5];
        c[1] = (dx * cam[6] + dy * cam[7]) + dz * cam[8];
        c[2] = (dx * cam[9] + dy * cam[10]) + dz * cam[11];
    }

    // Screen x, y and 1/z of a camera-space point beyond the near distance.
    KR_HD inline void to_screen(const float* cam, const float* c, float* v) {
        float inv = 1.0f / c[2];
        v[0] = cam[13] + cam[12] * (c[0] * inv);
        v[1] = cam[14] - cam[12] * (c[1] * inv);
        v[2] = inv;
    }

    KR_HD inline float plane(const float* coef, const float* values, float inv_area) {
        return (
            ((coef[0] * values[0] + coef[1] * values[1]) + coef[2] * values[2]) *
            inv_area
        );
    }

    // Fills `r` for the piece with screen corners v (x, y, 1/z) and colours col of
    // triangle t; r.tri stays -1 when it covers no pixel centre box.
    KR_HD inline void build_record(float v[3][3], float col[3][3], int t, Record& r) {
        r.tri = -1;
        float area2 = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]);
        area2 = area2 - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
        if (area2 == 0.0f) {
            return;
        }
        if (area2 < 0.0f) {
            for (int k = 0; k < 3; k++) {
                float s = v[1][k];
                v[1][k] = v[2][k];
                v[2][k] = s;
                float q = col[1][k];
                col[1][k] = col[2][k];
                col[2][k] = q;
            }
            area2 = -area2;
        }
        float lo_x = fminf(fminf(v[0][0], v[1][0]), v[2][0]);
        float hi_x = fmaxf(fmaxf(v[0][0], v[1][0]), v[2][0]);
        float lo_y = fminf(fminf(v[0][1], v[1][1]), v[2][1]);
        float hi_y = fmaxf(fmaxf(v[0][1], v[1][1]), v[2][1]);
        r.box[0] = (int)fminf(fmaxf(ceilf(lo_x - 0.5f), 0.0f), (float)WIDTH);
        r.box[1] = (int)fmaxf(fminf(floorf(hi_x - 0.5f), (float)(WIDTH - 1)), -1.0f);
        r.box[2] = (int)fminf(fmaxf(ceilf(lo_y - 0.5f), 0.0f), (float)HEIGHT);
        r.box[3] = (int)fmaxf(fminf(floorf(hi_y - 0.5f), (float)(HEIGHT - 1)), -1.0f);
        if (r.box[0] > r.box[1] || r.box[2] > r.box[3]) {
            return;
        }
        r.top_left = 0;
        for (int i = 0; i < 3; i++) {
            int p = (i + 1) % 3, q = (i + 2) % 3;
            r.a[i] = v[p][1] - v[q][1];
            r.b[i] = v[q][0] - v[p][0];
            r.c[i] = -(r.a[i] * v[p][0] + r.b[i] * v[p][1]);
            if (r.a[i] > 0.0f || (r.a[i] == 0.0f && r.b[i] > 0.0f)) {
                r.top_left |= 1u << i;
            }
        }
        float inv_area = 1.0f / area2;
        float w[3] = {v[0][2], v[1][2], v[2][2]};
        r.w[0] = plane(r.a, w, inv_area);
        r.w[1] = plane(r.b, w, inv_area);
        r.w[2] = plane(r.c, w, inv_area);
        for (int ch = 0; ch < 3; ch++) {
            float f[3] = {col[0][ch], col[1][ch], col[2][ch]};
            r.cp[0][ch] = plane(r.a, f, inv_area);
            r.cp[1][ch] = plane(r.b, f, inv_area);
            r.cp[2][ch] = plane(r.c, f, inv_area);
        }
        r.tri = t;
    }

    // The one or two pieces of triangle t, corners c (x, y, z, colour) in camera space,
    // cut at z = near (ref/raster.py, clip_polygon); kept out of line for the GPU.
    KR_HD KR_NOINLINE int clip_triangle(
        const float* cam, const float (*c)[6], int t, Record& r0, Record& r1
    ) {
        const float near = cam[15];
        float poly[4][6];
        int n = 0;
        for (int k = 0; k < 3; k++) {
            const float* a = c[k];
            const float* b = c[(k + 1) % 3];
            bool in_a = a[2] > near, in_b = b[2] > near;
            if (in_a) {
                for (int j = 0; j < 6; j++) {
                    poly[n][j] = a[j];
                }
                n++;
            }
            if (in_a != in_b) {
                const float* in = in_a ? a : b;
                const float* out = in_a ? b : a;
                float s = (near - in[2]) / (out[2] - in[2]);
                for (int j = 0; j < 6; j++) {
                    poly[n][j] = in[j] + s * (out[j] - in[j]);
                }
                poly[n][2] = near;
                n++;
            }
        }
        for (int piece = 0; piece + 2 < n; piece++) {
            const int fan[3] = {0, piece + 1, piece + 2};
            float v[3][3], col[3][3];
            for (int i = 0; i < 3; i++) {
                to_screen(cam, poly[fan[i]], v[i]);
                for (int ch = 0; ch < 3; ch++) {
                    col[i][ch] = poly[fan[i]][3 + ch];
                }
            }
            build_record(v, col, t, piece == 0 ? r0 : r1);
        }
        return (r0.tri >= 0 ? 1 : 0) | (n == 4 && r1.tri >= 0 ? 2 : 0);
    }

    // Fills r0 (and r1) with the pieces of triangle t, whole or clipped at z = near.
    // Returns bit 0 when r0 holds a piece, bit 1 when r1 holds one.
    KR_HD inline int setup_triangle(const Mesh& m, int t, Record& r0, Record& r1) {
        r0.tri = -1;
        float c[3][6];
        int vi[3], beyond = 0;
        for (int i = 0; i < 3; i++) {
            vi[i] = m.indices[3 * t + i];
            camera_space(m.camera, m.positions + 3 * vi[i], c[i]);
            beyond += c[i][2] > m.camera[15];
        }
        if (beyond == 0) {
            return 0;
        }
        for (int i = 0; i < 3; i++) {
            for (int ch = 0; ch < 3; ch++) {
                c[i][3 + ch] = m.colors[3 * vi[i] + ch];
            }
        }
        if (beyond < 3) {
            return clip_triangle(m.camera, c, t, r0, r1);
        }
        float v[3][3], col[3][3];
        for (int i = 0; i < 3; i++) {
            to_screen(m.camera, c[i], v[i]);
            for (int ch = 0; ch < 3; ch++) {
                col[i][ch] = c[i][3 + ch];
            }
        }
        build_record(v, col, t, r0);
        return r0.tri >= 0 ? 1 : 0;
    }

    // True when the pixel centre (px, py) is inside the triangle (top-left fill rule).
    KR_HD inline bool covers(const Record& r, float px, float py) {
        for (int i = 0; i < 3; i++) {
            float e = (r.a[i] * px + r.b[i] * py) + r.c[i];
            if (!(e > 0.0f || (e == 0.0f && ((r.top_left >> i) & 1u)))) {
                return false;
            }
        }
        return true;
    }

    KR_HD inline float depth_at(const Record& r, float px, float py) {
        return (r.w[0] * px + r.w[1] * py) + r.w[2];
    }

    KR_HD inline bool nearer(float w, int tri, float depth, int owner) {
        return w > depth || (w == depth && tri < owner);
    }

    KR_HD inline void colour_at(const Record& r, float px, float py, float* out) {
        for (int ch = 0; ch < 3; ch++) {
            out[ch] = (r.cp[0][ch] * px + r.cp[1][ch] * py) + r.cp[2][ch];
        }
    }

    KR_HD inline float background(int channel) {
        return channel == 2 ? 0.08f : 0.05f;
    }

}  // namespace kr
