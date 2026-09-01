// SPDX-License-Identifier: MIT
// bench_main.c
// Harness for the PPC dispatch-architecture benchmark: sets up the real SoA
// memory fast path over an 8 MB guest RAM, assembles three benchmark kernels
// into guest memory, verifies both backends produce bit-identical final
// state, then times them with alternating repetitions.

#include "ppc_bench.h"

#include "mem_env.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <time.h>
#endif

#define RAM_SIZE (8u << 20) // 8 MB guest RAM at guest address 0

static uint8_t *g_ram; // host view of guest RAM (from mem_env)

// Diagnostic for instructions outside the benchmark subset
void ppc_illegal(ppc_t *c, uint32_t insn) {
    fprintf(stderr, "FATAL: illegal/unimplemented insn %08x near pc %08x\n", insn, c->pc);
    exit(1);
}

// === PPC instruction encoders (host-side mini assembler) ===
#define E_D(op, d, a, imm) (((uint32_t)(op) << 26) | ((uint32_t)(d) << 21) | ((uint32_t)(a) << 16) | ((uint32_t)(imm) & 0xFFFFu))
#define E_ADDI(d, a, i)    E_D(14, d, a, i)
#define E_ADDIS(d, a, i)   E_D(15, d, a, i)
#define E_MULLI(d, a, i)   E_D(7, d, a, i)
#define E_ORI(a, s, ui)    E_D(24, s, a, ui) // note rS in the D slot
#define E_XORI(a, s, ui)   E_D(26, s, a, ui)
#define E_ANDI_RC(a, s, ui) E_D(28, s, a, ui)
#define E_CMPWI(crf, a, i)  E_D(11, (crf) << 2, a, i)
#define E_CMPLWI(crf, a, i) E_D(10, (crf) << 2, a, i)
#define E_LWZ(d, a, off)   E_D(32, d, a, off)
#define E_LHZ(d, a, off)   E_D(40, d, a, off)
#define E_LBZ(d, a, off)   E_D(34, d, a, off)
#define E_STW(s, a, off)   E_D(36, s, a, off)
#define E_STH(s, a, off)   E_D(44, s, a, off)
#define E_STB(s, a, off)   E_D(38, s, a, off)

#define E_X(d, a, b, xo, rc) (((uint32_t)31 << 26) | ((uint32_t)(d) << 21) | ((uint32_t)(a) << 16) | ((uint32_t)(b) << 11) | ((uint32_t)(xo) << 1) | (rc))
#define E_ADD(d, a, b)     E_X(d, a, b, 266, 0)
#define E_ADD_RC(d, a, b)  E_X(d, a, b, 266, 1)
#define E_SUBF(d, a, b)    E_X(d, a, b, 40, 0)
#define E_MULLW(d, a, b)   E_X(d, a, b, 235, 0)
#define E_AND(a, s, b)     E_X(s, a, b, 28, 0)
#define E_OR(a, s, b)      E_X(s, a, b, 444, 0)
#define E_XOR(a, s, b)     E_X(s, a, b, 316, 0)
#define E_SRAWI(a, s, sh)  E_X(s, a, sh, 824, 0)
#define E_LWZX(d, a, b)    E_X(d, a, b, 23, 0)
#define E_LBZX(d, a, b)    E_X(d, a, b, 87, 0)
#define E_STWX(s, a, b)    E_X(s, a, b, 151, 0)
#define E_MTCTR(s)         E_X(s, 9, 0, 467, 0)

#define E_RLWINM(a, s, sh, mb, me) (((uint32_t)21 << 26) | ((uint32_t)(s) << 21) | ((uint32_t)(a) << 16) | ((uint32_t)(sh) << 11) | ((uint32_t)(mb) << 6) | ((uint32_t)(me) << 1))
#define E_BC(bo, bi, off) (((uint32_t)16 << 26) | ((uint32_t)(bo) << 21) | ((uint32_t)(bi) << 16) | ((uint32_t)(off) & 0xFFFCu))
#define E_BDNZ(off)       E_BC(16, 0, off)
#define E_B(off)          (((uint32_t)18 << 26) | ((uint32_t)(off) & 0x03FFFFFCu))

static uint32_t g_emit_pc; // guest address the next emit() lands at

// Store one big-endian instruction word into guest RAM and advance
static void emit(uint32_t insn) {
    STORE_BE32(g_ram + g_emit_pc, insn);
    g_emit_pc += 4;
}

// Deterministic PRNG for kernel data and the random instruction stream
static uint32_t lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

#define CODE_BASE  0x001000u // all kernels are emitted here
#define DATA_BASE  0x100000u // read window (64 KB of PRNG fill)
#define STORE_BASE 0x180000u // write window

// Kernel 1: tight predictable ALU loop (10 instructions/iteration).
// Small code footprint, dispatch trivially predictable — the switch
// backend's best case.
static void emit_kernel_alu(void) {
    emit(E_ADDIS(4, 0, 0x1234));
    emit(E_ORI(4, 4, 0x5678)); // r4 = 0x12345678
    emit(E_ADDI(5, 0, 0)); // r5 = accumulator
    emit(E_ADDIS(7, 0, 0x7FFF));
    emit(E_ORI(7, 7, 0xFFFF));
    emit(E_MTCTR(7)); // ctr = 0x7FFFFFFF (never exits within budget)
    uint32_t loop = g_emit_pc;
    emit(E_ADD(5, 5, 4));
    emit(E_RLWINM(8, 5, 7, 0, 31)); // rotate
    emit(E_XOR(5, 5, 8));
    emit(E_ADDI(4, 4, 3));
    emit(E_ADD_RC(9, 5, 4)); // records CR0
    emit(E_BC(12, 0, 8)); // blt cr0 -> skip the subf
    emit(E_SUBF(5, 4, 5));
    emit(E_MULLI(10, 4, 13));
    emit(E_XOR(5, 5, 10));
    emit(E_BDNZ((int32_t)(loop - g_emit_pc)));
}

// Kernel 2: memory-streaming loop, 3 memory ops per 8 instructions through
// the real SoA fast path, wrapping in a 16 KB window.
static void emit_kernel_mem(void) {
    emit(E_ADDIS(20, 0, DATA_BASE >> 16)); // r20 = read base
    emit(E_ADDIS(21, 0, STORE_BASE >> 16)); // r21 = write base
    emit(E_ADDI(3, 0, 0)); // r3 = index
    emit(E_ADDI(4, 0, 0)); // r4 = checksum
    emit(E_ADDIS(7, 0, 0x7FFF));
    emit(E_ORI(7, 7, 0xFFFF));
    emit(E_MTCTR(7));
    uint32_t loop = g_emit_pc;
    emit(E_LWZX(6, 20, 3));
    emit(E_ADD(4, 4, 6));
    emit(E_STWX(4, 21, 3));
    emit(E_LBZX(8, 20, 3));
    emit(E_XOR(4, 4, 8));
    emit(E_ADDI(3, 3, 4));
    emit(E_ANDI_RC(3, 3, 0x3FFC)); // wrap index in 16 KB
    emit(E_BDNZ((int32_t)(loop - g_emit_pc)));
}

#define STREAM_LEN 16384 // instructions in the random straight-line stream

// Kernel 3: dispatch-stress — a 64 KB straight-line stream of random
// instructions (targets r3..r12, fixed-base D-form memory ops, occasional
// forward bc), looped with an unconditional b.  Code footprint and opcode
// sequence defeat trivial dispatch prediction; this is where per-handler
// dispatch sites should differ most from a single shared br_table.
static void emit_kernel_stress(void) {
    // seed r3..r12 deterministically, set up base registers
    uint32_t seed = 0xC0FFEE01u;
    for (int r = 3; r <= 12; r++) {
        uint32_t v = lcg(&seed);
        emit(E_ADDIS(r, 0, (int32_t)(v >> 16)));
        emit(E_ORI(r, r, v & 0xFFFF));
    }
    emit(E_ADDIS(20, 0, DATA_BASE >> 16));
    emit(E_ADDIS(21, 0, STORE_BASE >> 16));
    uint32_t loop = g_emit_pc;
    for (int i = 0; i < STREAM_LEN; i++) {
        uint32_t r = lcg(&seed);
        int rd = 3 + (int)((r >> 8) % 10); // targets r3..r12
        int ra = 3 + (int)((r >> 12) % 10);
        int rb = 3 + (int)((r >> 16) % 10);
        uint32_t off = (r >> 20) & 0x7FC; // safe offset in the data windows
        int pick = (int)(r % 100);
        bool last = (i == STREAM_LEN - 1); // no bc as the final instruction
        if (pick < 8)
            emit(E_ADD(rd, ra, rb));
        else if (pick < 14)
            emit(E_SUBF(rd, ra, rb));
        else if (pick < 20)
            emit(E_XOR(rd, ra, rb));
        else if (pick < 24)
            emit(E_OR(rd, ra, rb));
        else if (pick < 28)
            emit(E_AND(rd, ra, rb));
        else if (pick < 36)
            emit(E_RLWINM(rd, ra, (r >> 3) & 31, (r >> 26) & 31, 31));
        else if (pick < 44)
            emit(E_ADDI(rd, ra, (int32_t)(r & 0xFF) - 128));
        else if (pick < 48)
            emit(E_ADDIS(rd, ra, 1));
        else if (pick < 52)
            emit(E_MULLI(rd, ra, 13));
        else if (pick < 56)
            emit(E_MULLW(rd, ra, rb));
        else if (pick < 61)
            emit(E_SRAWI(rd, ra, (r >> 3) & 31));
        else if (pick < 68)
            emit(E_LWZ(rd, 20, off));
        else if (pick < 72)
            emit(E_LHZ(rd, 20, off));
        else if (pick < 76)
            emit(E_LBZ(rd, 20, off));
        else if (pick < 82)
            emit(E_STW(rd, 21, off));
        else if (pick < 85)
            emit(E_STH(rd, 21, off));
        else if (pick < 88)
            emit(E_STB(rd, 21, off));
        else if (pick < 94)
            emit(E_CMPWI(7, ra, (int32_t)(r & 0x7FFF) - 16384));
        else if (!last) // bc forward +8: conditionally skip one instruction
            emit(E_BC(((r >> 5) & 1) ? 12 : 4, 28 + (int)((r >> 6) & 3), 8));
        else
            emit(E_ADD(rd, ra, rb));
    }
    emit(E_B((int32_t)(loop - g_emit_pc)));
}

// Reset guest RAM + CPU state and assemble kernel `k`; both backends must
// start every run from this exact state
static void setup_kernel(int k, ppc_t *c) {
    memset(g_ram, 0, RAM_SIZE);
    uint32_t seed = 0xDEADBEEFu;
    for (uint32_t i = 0; i < 0x10000; i += 4) { // 64 KB PRNG read window
        uint32_t v = lcg(&seed);
        STORE_BE32(g_ram + DATA_BASE + i, v);
    }
    g_emit_pc = CODE_BASE;
    if (k == 0)
        emit_kernel_alu();
    else if (k == 1)
        emit_kernel_mem();
    else
        emit_kernel_stress();
    memset(c, 0, sizeof(*c));
    c->pc = CODE_BASE;
}

// FNV-1a over CPU state and all of guest RAM: any divergence between the
// two backends shows up here
static uint64_t state_hash(const ppc_t *c) {
    uint64_t h = 0xcbf29ce484222325ull;
#define MIX(v) do { h ^= (uint64_t)(v); h *= 0x100000001b3ull; } while (0)
    for (int i = 0; i < 32; i++)
        MIX(c->gpr[i]);
    MIX(c->pc);
    MIX(c->lr);
    MIX(c->ctr);
    MIX(c->cr);
    MIX(c->xer_ca);
    for (uint32_t i = 0; i < RAM_SIZE; i += 8) {
        uint64_t w;
        memcpy(&w, g_ram + i, 8);
        MIX(w);
    }
#undef MIX
    return h;
}

// Monotonic wall-clock in milliseconds
static double now_ms(void) {
#ifdef __EMSCRIPTEN__
    return emscripten_get_now();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
#endif
}

typedef void run_fn(ppc_t *c, uint32_t budget);
#define NBACKENDS 6
static run_fn *const g_backends[NBACKENDS] = { ppc_run_switch, ppc_run_tail, ppc_run_call,
                                               ppc_run_pd, ppc_run_pdx8, ppc_run_pdx16 };
static const char *const g_backend_names[NBACKENDS] = { "switch", "tail-call", "call-loop",
                                                        "pd-raw", "pdx-8B", "pdx-16B" };
static const char *const g_kernel_names[3] = { "alu-loop", "mem-stream", "dispatch-stress" };

#ifndef BENCH_DEFAULT_BUDGET
#define BENCH_DEFAULT_BUDGET (100u * 1000 * 1000)
#endif
#ifndef BENCH_DEFAULT_REPS
#define BENCH_DEFAULT_REPS 5
#endif

int main(int argc, char **argv) {
    uint32_t verify_budget = 20u * 1000 * 1000;
    uint32_t bench_budget = BENCH_DEFAULT_BUDGET;
    int reps = BENCH_DEFAULT_REPS;
    if (argc > 1)
        bench_budget = (uint32_t)strtoul(argv[1], NULL, 0);
    if (argc > 2)
        reps = atoi(argv[2]);

    mem_env_init(RAM_SIZE);
    g_ram = mem_env_ram();

    printf("ppc dispatch benchmark: budget=%u reps=%d musttail=%s\n", bench_budget, reps,
#if defined(__clang__) && __has_attribute(musttail)
           "yes"
#else
           "no (sibling-call fallback)"
#endif
    );

    ppc_t cpu;
    for (int k = 0; k < 3; k++) {
        // 1. correctness: both backends must produce bit-identical state
        uint64_t hash[NBACKENDS];
        for (int b = 0; b < NBACKENDS; b++) {
            setup_kernel(k, &cpu);
            g_backends[b](&cpu, verify_budget);
            hash[b] = state_hash(&cpu);
        }
        for (int b = 1; b < NBACKENDS; b++) {
            if (hash[b] != hash[0]) {
                fprintf(stderr, "FAIL %s: backend %s diverged (%016" PRIx64 " vs %016" PRIx64 ")\n",
                        g_kernel_names[k], g_backend_names[b], hash[0], hash[b]);
                return 1;
            }
        }
        // 2. warmup: let the engine's optimizing tier compile both backends
        for (int b = 0; b < NBACKENDS; b++) {
            setup_kernel(k, &cpu);
            g_backends[b](&cpu, bench_budget / 2);
        }
        // 3. timed, alternating reps; report the best (least-disturbed) run
        double best[NBACKENDS];
        for (int b = 0; b < NBACKENDS; b++)
            best[b] = 1e30;
        for (int r = 0; r < reps; r++) {
            for (int b = 0; b < NBACKENDS; b++) {
                setup_kernel(k, &cpu);
                double t0 = now_ms();
                g_backends[b](&cpu, bench_budget);
                double dt = now_ms() - t0;
                if (dt < best[b])
                    best[b] = dt;
            }
        }
        double mips[NBACKENDS];
        for (int b = 0; b < NBACKENDS; b++)
            mips[b] = (double)bench_budget / 1e3 / best[b];
        for (int b = 0; b < NBACKENDS; b++) {
            printf("%-16s %-10s %8.1f MIPS", g_kernel_names[k], g_backend_names[b], mips[b]);
            if (b == 0)
                printf("  (baseline, verify=ok)\n");
            else
                printf("  (%+.1f%%)\n", (mips[b] / mips[0] - 1.0) * 100.0);
        }
    }
    printf("benchmark done\n");
    return 0;
}
