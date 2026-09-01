// SPDX-License-Identifier: MIT
// be_micro.c
// Micro-benchmark: cost of big-endian 32-bit loads in wasm.  Three
// formulations over the same 64 KB buffer, both as independent (throughput)
// and dependent (latency-chained, like fetch->dispatch) loops.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
static double now_ms(void) { return emscripten_get_now(); }
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
#endif

#define BUF_WORDS 16384 // 64 KB
static uint32_t buf[BUF_WORDS];

// Variant 0: plain little-endian load (the no-swap upper bound)
static inline uint32_t ld_le(const uint32_t *p) { return *p; }

// Variant 1: the repo's LOAD_BE32 — __builtin_bswap32 (LLVM lowers to
// 4 shifts + 2 ands + 3 ors in wasm)
static inline uint32_t ld_bswap(const uint32_t *p) { return __builtin_bswap32(*p); }

// Variant 2: rotate-based byte swap — rot16 then swap adjacent bytes
// (1 rot + 2 shifts + 2 ands + 1 or); compilers turn the (x<<16|x>>16)
// idiom into i32.rotl
static inline uint32_t ld_rot(const uint32_t *p) {
    uint32_t x = *p;
    uint32_t t = (x << 16) | (x >> 16);
    return ((t << 8) & 0xFF00FF00u) | ((t >> 8) & 0x00FF00FFu);
}

// One timed pass; kind selects variant, dep selects dependent chaining.
// Returns checksum so nothing is optimized away.
static uint64_t run(int kind, int dep, uint32_t iters, double *ms) {
    uint64_t sum = 0;
    uint32_t lin = 0x9E3779B9u; // LCG index stream (opaque to loop folding)
    uint32_t idx = 0; // load-value-dependent index chain
    double t0 = now_ms();
    for (uint32_t i = 0; i < iters; i++) {
        lin = lin * 1664525u + 1013904223u;
        const uint32_t *p = &buf[(dep ? idx : (lin >> 8)) & (BUF_WORDS - 1)];
        uint32_t v = (kind == 0) ? ld_le(p) : (kind == 1) ? ld_bswap(p) : ld_rot(p);
        sum += v;
        // dep=1: next address depends on the load, via a raw byte that is
        // identical for all variants so every kind walks the same addresses
        idx = idx * 5 + *(const uint8_t *)p + 1;
    }
    *ms = now_ms() - t0;
    return sum;
}

int main(int argc, char **argv) {
    uint32_t iters = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 0) : 400u * 1000 * 1000;
    uint32_t s = 12345;
    for (int i = 0; i < BUF_WORDS; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = s;
    }
    static const char *const names[3] = { "le-load ", "bswap   ", "rot-swap" };
    for (int dep = 0; dep <= 1; dep++) {
        printf("%s:\n", dep ? "dependent (fetch-like)" : "independent (throughput)");
        for (int kind = 0; kind < 3; kind++) {
            double best = 1e30;
            uint64_t sum = 0;
            for (int r = 0; r < 5; r++) {
                double ms;
                sum = run(kind, dep, iters, &ms);
                if (ms < best)
                    best = ms;
            }
            printf("  %s %8.1f Mloads/s  (checksum %016llx)\n", names[kind],
                   (double)iters / 1e3 / best, (unsigned long long)sum);
        }
    }
    return 0;
}
