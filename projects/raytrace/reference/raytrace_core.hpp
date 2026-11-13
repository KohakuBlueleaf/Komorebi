// Tree walk, triangle test and shading of ref/raytrace.py: float32 only, no fused
// multiply-add (build with -ffp-contract=off / --fmad=false), one ray with a stack.
#pragma once

#include <cmath>
#include <cstdint>

#include "kr.hpp"

namespace kr {

    constexpr uint32_t LEAF_BIT = 0x80000000u;
    constexpr int STACK = 64;
    constexpr float TMAX = 1e30f;
    constexpr float SLAB_SCALE = 1.0000004f;
    constexpr float MIN_DIR = 1e-12f;
    constexpr float AMBIENT = 0.1f;
    constexpr float GI = 0.5f;
    constexpr uint32_t BOUNCE_SEED = 7;

    struct Tree {
        const uint32_t* nodes;    // (M, 8): box min, box max, child 0, child 1
        const float* tris;        // (N, 9) in leaf order: v0, e1, e2
        const int32_t* leaf_tri;  // leaf -> triangle index of the scene
    };

    struct Scene {
        Tree tree;
        const float* colors;
        const int32_t* indices;
        float camera[16];
        float light[3];
        float eps;
    };

    struct Ray {
        float o[3], d[3], inv[3];
    };

    struct Hit {
        float t, u, v;
        int32_t leaf;    // -1: none
        int32_t nodes;   // box tests
        int32_t leaves;  // triangle tests
    };

    // Per-pixel buffers of one frame (pointers, so the same code runs on host and
    // device).
    struct Frame {
        Hit *hit0, *hit1;
        float *normal, *origin, *albedo, *direct0, *bounce_dir, *radiance1, *colour;
    };

    KR_HD inline float dot3(const float* a, const float* b) {
        return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
    }

    KR_HD inline void cross3(const float* a, const float* b, float* r) {
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
    }

    KR_HD inline void make_ray(Ray& r, const float* o, const float* d) {
        for (int k = 0; k < 3; k++) {
            r.o[k] = o[k];
            r.d[k] = d[k];
            float safe = fabsf(d[k]) < MIN_DIR ? copysignf(MIN_DIR, d[k]) : d[k];
            r.inv[k] = 1.0f / safe;
        }
    }

    KR_HD inline float as_float(uint32_t w) {
        union {
            uint32_t u;
            float f;
        } c;

        c.u = w;
        return c.f;
    }

    // Box of a node against the ray: true when hit, with the entry distance.
    KR_HD inline bool box_test(
        const uint32_t* node, const Ray& r, float best, float& entry
    ) {
        float near_[3], far_[3];
        for (int k = 0; k < 3; k++) {
            float t0 = (as_float(node[k]) - r.o[k]) * r.inv[k];
            float t1 = (as_float(node[3 + k]) - r.o[k]) * r.inv[k];
            near_[k] = t0 < t1 ? t0 : t1;
            far_[k] = t0 < t1 ? t1 : t0;
        }
        float a = near_[0] > near_[1] ? near_[0] : near_[1];
        float b = near_[2] > 0.0f ? near_[2] : 0.0f;
        entry = a > b ? a : b;
        float c = far_[0] < far_[1] ? far_[0] : far_[1];
        float exit_ = (c < far_[2] ? c : far_[2]) * SLAB_SCALE;
        return entry <= (exit_ < best ? exit_ : best);
    }

    KR_HD inline bool triangle_test(
        const float* g, const Ray& r, float& t, float& u, float& v
    ) {
        float pv[3], tv[3], qv[3];
        cross3(r.d, g + 6, pv);
        float det = dot3(g + 3, pv);
        if (det == 0.0f) {
            return false;
        }
        float inv = 1.0f / det;
        for (int k = 0; k < 3; k++) {
            tv[k] = r.o[k] - g[k];
        }
        u = dot3(tv, pv) * inv;
        if (!(u >= 0.0f && u <= 1.0f)) {
            return false;
        }
        cross3(tv, g + 3, qv);
        v = dot3(r.d, qv) * inv;
        if (!(v >= 0.0f && u + v <= 1.0f)) {
            return false;
        }
        t = dot3(g + 6, qv) * inv;
        return t > 0.0f;
    }

    // Closest hit with t < tmax (any_hit: stops at the first hit; best stays tmax).
    KR_HD inline void trace(
        const Tree& tree, const Ray& r, float tmax, bool any_hit, Hit& h
    ) {
        h.t = tmax;
        h.u = h.v = 0.0f;
        h.leaf = -1;
        h.nodes = 1;
        h.leaves = 0;
        float best = tmax, entry;
        int32_t stack_node[STACK];
        float stack_t[STACK];
        if (!box_test(tree.nodes, r, best, entry)) {
            return;
        }
        stack_node[0] = 0;
        stack_t[0] = entry;
        int sp = 1;
        while (sp > 0) {
            sp--;
            if (!(stack_t[sp] <= best)) {
                continue;
            }
            const uint32_t* node = tree.nodes + 8 * (size_t)stack_node[sp];
            if (node[6] & LEAF_BIT) {
                int leaf = (int)(node[6] & ~LEAF_BIT);
                float t, u, v;
                h.leaves++;
                if (!triangle_test(tree.tris + 9 * (size_t)leaf, r, t, u, v)) {
                    continue;
                }
                bool better = t < best;
                if (!any_hit && !better && t == best && h.leaf >= 0) {
                    better = tree.leaf_tri[leaf] < tree.leaf_tri[h.leaf];
                }
                if (better) {
                    h.t = t;
                    h.u = u;
                    h.v = v;
                    h.leaf = leaf;
                    if (any_hit) {
                        return;
                    }
                    best = t;
                }
                continue;
            }
            float t0, t1;
            bool h0 = box_test(tree.nodes + 8 * (size_t)node[6], r, best, t0);
            bool h1 = box_test(tree.nodes + 8 * (size_t)node[7], r, best, t1);
            h.nodes += 2;
            if (h0 && h1) {
                bool near1 = t1 < t0;
                stack_node[sp] = (int32_t)(near1 ? node[6] : node[7]);
                stack_t[sp++] = near1 ? t0 : t1;
                stack_node[sp] = (int32_t)(near1 ? node[7] : node[6]);
                stack_t[sp++] = near1 ? t1 : t0;
            } else if (h0 || h1) {
                stack_node[sp] = (int32_t)(h0 ? node[6] : node[7]);
                stack_t[sp++] = h0 ? t0 : t1;
            }
        }
    }

    KR_HD inline void camera_ray(const float* cam, int pixel, Ray& r) {
        float x = (float)(pixel % WIDTH) + 0.5f;
        float y = (float)(pixel / WIDTH) + 0.5f;
        float sx = (x - cam[13]) / cam[12];
        float sy = (y - cam[14]) / cam[12];
        float d[3];
        for (int k = 0; k < 3; k++) {
            d[k] = (cam[9 + k] + cam[3 + k] * sx) - cam[6 + k] * sy;
        }
        make_ray(r, cam, d);
    }

    KR_HD inline void sky(const float* d, float* c) {
        float s = d[1] / sqrtf(dot3(d, d));
        s = fminf(fmaxf(s, 0.0f), 1.0f);
        const float horizon[3] = {0.85f, 0.88f, 0.95f};
        const float zenith[3] = {0.35f, 0.55f, 0.9f};
        for (int k = 0; k < 3; k++) {
            c[k] = horizon[k] + (zenith[k] - horizon[k]) * s;
        }
    }

    // Normal facing the ray direction d, and the colour interpolated at a hit.
    KR_HD inline void surface(
        const Scene& s, const Hit& h, const float* d, float* n, float* albedo
    ) {
        const float* g = s.tree.tris + 9 * (size_t)h.leaf;
        cross3(g + 3, g + 6, n);
        float len = sqrtf(dot3(n, n));
        for (int k = 0; k < 3; k++) {
            n[k] = n[k] / len;
        }
        if (dot3(n, d) > 0.0f) {
            for (int k = 0; k < 3; k++) {
                n[k] = -n[k];
            }
        }
        const int32_t* tri = s.indices + 3 * (size_t)s.tree.leaf_tri[h.leaf];
        float w1 = h.u, w2 = h.v, w0 = (1.0f - w1) - w2;
        for (int k = 0; k < 3; k++) {
            albedo[k] =
                ((s.colors[3 * tri[0] + k] * w0 + s.colors[3 * tri[1] + k] * w1) +
                 s.colors[3 * tri[2] + k] * w2);
        }
    }

    // The point o above the surface at p, along the normal n.
    KR_HD inline void offset_point(
        const Scene& s, const float* p, const float* n, float* o
    ) {
        for (int k = 0; k < 3; k++) {
            o[k] = p[k] + n[k] * s.eps;
        }
    }

    // The shadow ray from o: direction light - o, t in (0, 1); cos_ is n . l / |l|.
    KR_HD inline void shadow_ray(
        const Scene& s, const float* o, const float* n, Ray& r, float& cos_
    ) {
        float l[3];
        for (int k = 0; k < 3; k++) {
            l[k] = s.light[k] - o[k];
        }
        cos_ = dot3(n, l) * (1.0f / sqrtf(dot3(l, l)));
        make_ray(r, o, l);
    }

    KR_HD inline float diffuse(float cos_, bool occluded) {
        return occluded ? 0.0f : (cos_ > 0.0f ? cos_ : 0.0f);
    }

    // A unit direction along n plus a random unit vector, from the hash of the pixel.
    KR_HD inline void bounce_direction(int pixel, const float* n, float* b) {
        float g[3];
        for (int k = 0; k < 3; k++) {
            float r0 = uniform(pixel, 4 * k + 0, BOUNCE_SEED);
            float r1 = uniform(pixel, 4 * k + 1, BOUNCE_SEED);
            float r2 = uniform(pixel, 4 * k + 2, BOUNCE_SEED);
            float r3 = uniform(pixel, 4 * k + 3, BOUNCE_SEED);
            g[k] = (((r0 + r1) + r2) + r3) - 2.0f;
        }
        float g2 = dot3(g, g);
        float gu[3];
        for (int k = 0; k < 3; k++) {
            gu[k] = g2 < 1e-12f ? n[k] : g[k] / sqrtf(g2 > 1e-12f ? g2 : 1e-12f);
        }
        float v[3] = {n[0] + gu[0], n[1] + gu[1], n[2] + gu[2]};
        float b2 = dot3(v, v);
        for (int k = 0; k < 3; k++) {
            b[k] = b2 < 1e-6f ? n[k] : v[k] / sqrtf(b2 > 1e-6f ? b2 : 1e-6f);
        }
    }

    // The five stages of one pixel.
    KR_HD inline void primary_ray(const Scene& s, const Frame& b, int p) {
        Ray r;
        camera_ray(s.camera, p, r);
        trace(s.tree, r, TMAX, false, b.hit0[p]);
    }

    // Normal, colour, offset origin and the shadow ray's diffuse term at the primary
    // hit.
    KR_HD inline void shadow_primary(const Scene& s, const Frame& b, int p) {
        const Hit& h = b.hit0[p];
        if (h.leaf < 0) {
            return;
        }
        Ray r;
        camera_ray(s.camera, p, r);
        float point[3];
        for (int k = 0; k < 3; k++) {
            point[k] = r.o[k] + r.d[k] * h.t;
        }
        float* n = &b.normal[3 * p];
        surface(s, h, r.d, n, &b.albedo[3 * p]);
        float* o2 = &b.origin[3 * p];
        offset_point(s, point, n, o2);
        Ray shadow;
        float cos_;
        shadow_ray(s, o2, n, shadow, cos_);
        Hit sh;
        trace(s.tree, shadow, 1.0f, true, sh);
        b.direct0[p] = diffuse(cos_, sh.leaf >= 0);
    }

    KR_HD inline void bounce_ray(const Scene& s, const Frame& b, int p) {
        if (b.hit0[p].leaf < 0) {
            return;
        }
        float* dir = &b.bounce_dir[3 * p];
        bounce_direction(p, &b.normal[3 * p], dir);
        Ray r;
        make_ray(r, &b.origin[3 * p], dir);
        trace(s.tree, r, TMAX, false, b.hit1[p]);
    }

    KR_HD inline void shadow_bounce(const Scene& s, const Frame& b, int p) {
        if (b.hit0[p].leaf < 0) {
            return;
        }
        const float* dir = &b.bounce_dir[3 * p];
        float* rad = &b.radiance1[3 * p];
        const Hit& h = b.hit1[p];
        if (h.leaf < 0) {
            sky(dir, rad);
            return;
        }
        const float* o2 = &b.origin[3 * p];
        float point[3], n[3], albedo[3];
        for (int k = 0; k < 3; k++) {
            point[k] = o2[k] + dir[k] * h.t;
        }
        surface(s, h, dir, n, albedo);
        float o3[3];
        offset_point(s, point, n, o3);
        Ray shadow;
        float cos_;
        shadow_ray(s, o3, n, shadow, cos_);
        Hit sh;
        trace(s.tree, shadow, 1.0f, true, sh);
        float direct = diffuse(cos_, sh.leaf >= 0);
        for (int k = 0; k < 3; k++) {
            rad[k] = albedo[k] * (AMBIENT + direct);
        }
    }

    KR_HD inline void shade(const Scene& s, const Frame& b, int p) {
        float* c = &b.colour[3 * p];
        if (b.hit0[p].leaf < 0) {
            Ray r;
            camera_ray(s.camera, p, r);
            sky(r.d, c);
            return;
        }
        for (int k = 0; k < 3; k++) {
            c[k] =
                (b.albedo[3 * p + k] *
                 ((AMBIENT + b.direct0[p]) + GI * b.radiance1[3 * p + k]));
        }
    }

}  // namespace kr
