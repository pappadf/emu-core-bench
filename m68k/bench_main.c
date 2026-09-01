// SPDX-License-Identifier: MIT
// bench_main.c (m68k)
// Harness for the M68K interpreter-format benchmark: assembles three
// kernels into guest RAM, verifies backend state equivalence, and times
// alternating repetitions.  Mirrors ppc/bench_main.c.
//
// Verification note: the flag-elision backend intentionally leaves stale
// NZVC at sprint boundaries (the elided flags are dead ON the guest's own
// instruction path, but a sprint can end between definer and overwriter),
// so it is compared on a reduced hash that excludes CCR; registers, pc and
// all of RAM must still match bit-for-bit — kernels branch on flags
// constantly, so any elision unsoundness diverges that hash immediately.

#include "m68k_bench.h"

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

#define RAM_SIZE (8u << 20)

static uint8_t *g_ram; // host view of guest RAM (from mem_env)

// Diagnostic for instructions outside the benchmark subset
void m68k_illegal(m68k_t *m, uint32_t opcode) {
    fprintf(stderr, "FATAL: illegal/unimplemented m68k opcode %04x near pc %08x\n", opcode & 0xFFFF, m->pc);
    exit(1);
}

// === Mini assembler (subset encodings, big-endian words) ===
static uint32_t g_pc; // guest address the next ew() lands at

static void ew(uint16_t w) {
    STORE_BE16(g_ram + g_pc, w);
    g_pc += 2;
}

#define MOVEQ(dn, imm)          ew(0x7000 | (dn) << 9 | ((imm) & 0xFF))
#define MOVE_L_IMM_DN(dn, imm)  do { ew(0x203C | (dn) << 9); ew((uint16_t)((uint32_t)(imm) >> 16)); ew((uint16_t)(imm)); } while (0)
#define MOVE_W_IMM_DN(dn, imm)  do { ew(0x303C | (dn) << 9); ew((uint16_t)(imm)); } while (0)
#define MOVEA_L_IMM(an, imm)    do { ew(0x207C | (an) << 9); ew((uint16_t)((uint32_t)(imm) >> 16)); ew((uint16_t)(imm)); } while (0)
#define MOVE_L_DD(dst, src)     ew(0x2000 | (dst) << 9 | (src))
#define MOVE_L_ANP_DN(dn, an)   ew(0x2018 | (dn) << 9 | (an))
#define MOVE_L_DN_ANP(an, dn)   ew(0x20C0 | (an) << 9 | (dn))
#define MOVE_L_D16_DN(dn, an, d) do { ew(0x2028 | (dn) << 9 | (an)); ew((uint16_t)(d)); } while (0)
#define MOVE_L_DN_D16(an, d, dn) do { ew(0x2140 | (an) << 9 | (dn)); ew((uint16_t)(d)); } while (0)
#define MOVE_W_D16_DN(dn, an, d) do { ew(0x3028 | (dn) << 9 | (an)); ew((uint16_t)(d)); } while (0)
#define MOVE_W_DN_D16(an, d, dn) do { ew(0x3140 | (an) << 9 | (dn)); ew((uint16_t)(d)); } while (0)
#define MOVE_B_D16_DN(dn, an, d) do { ew(0x1028 | (dn) << 9 | (an)); ew((uint16_t)(d)); } while (0)
#define MOVE_B_DN_D16(an, d, dn) do { ew(0x1140 | (an) << 9 | (dn)); ew((uint16_t)(d)); } while (0)
#define ADD_L_DD(dst, src)      ew(0xD080 | (dst) << 9 | (src))
#define SUB_L_DD(dst, src)      ew(0x9080 | (dst) << 9 | (src))
#define CMP_L_DD(dn, src)       ew(0xB080 | (dn) << 9 | (src))
#define EOR_L_DD(dst, dn)       ew(0xB180 | (dn) << 9 | (dst))
#define AND_L_DD(dst, src)      ew(0xC080 | (dst) << 9 | (src))
#define OR_L_DD(dst, src)       ew(0x8080 | (dst) << 9 | (src))
#define ADDQ_L_DN(q, dn)        ew(0x5080 | (((q) & 7) << 9) | (dn))
#define ADDQ_L_AN(q, an)        ew(0x5088 | (((q) & 7) << 9) | (an))
#define SUBQ_L_DN(q, dn)        ew(0x5180 | (((q) & 7) << 9) | (dn))
#define CMPI_L_DN(imm, dn)      do { ew(0x0C80 | (dn)); ew((uint16_t)((uint32_t)(imm) >> 16)); ew((uint16_t)(imm)); } while (0)
#define TST_L_DN(dn)            ew(0x4A80 | (dn))
#define TST_W_DN(dn)            ew(0x4A40 | (dn))
#define TST_B_DN(dn)            ew(0x4A00 | (dn))
#define LSL_L(q, dn)            ew(0xE188 | (((q) & 7) << 9) | (dn))
#define LSR_L(q, dn)            ew(0xE088 | (((q) & 7) << 9) | (dn))
#define LEA_D16(dst, src, d)    do { ew(0x41E8 | (dst) << 9 | (src)); ew((uint16_t)(d)); } while (0)
#define NOP()                   ew(0x4E71)
// branch displacements are from CIA+2; helpers take absolute targets
#define BRA_B(target)           ew(0x6000 | (uint8_t)((target) - (g_pc + 2)))
#define BRA_W(target)           do { uint32_t _t = (target); ew(0x6000); ew((uint16_t)(_t - g_pc)); } while (0)
#define BCC_B(cond, target)     ew(0x6000 | (cond) << 8 | (uint8_t)((target) - (g_pc + 2)))
#define DBRA(dn, target)        do { uint32_t _t = (target); ew(0x51C8 | (dn)); ew((uint16_t)(_t - g_pc)); } while (0)

// 68000 condition encodings used by the kernels
#define CC_NE 6
#define CC_EQ 7
#define CC_PL 10
#define CC_MI 11
#define CC_GE 12
#define CC_LT 13
#define CC_GT 14
#define CC_LE 15

#define CODE_BASE  0x001000u
#define DATA_BASE  0x100000u
#define STORE_BASE 0x180000u

// Deterministic PRNG (same as the PPC harness)
static uint32_t lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

// Kernel 1: tight predictable ALU loop with the idiomatic dbra inner loop
static void emit_kernel_alu(void) {
    MOVE_L_IMM_DN(2, 0x12345678); // d2 = seed
    uint32_t top = g_pc;
    MOVE_W_IMM_DN(0, 0x0FFF); // d0 = inner counter (4096 passes)
    uint32_t loop = g_pc;
    ADD_L_DD(1, 2); // d1 += d2
    MOVE_L_DD(3, 1); // d3 = d1
    LSR_L(7, 3); // d3 >>= 7
    EOR_L_DD(1, 3); // d1 ^= d3
    ADDQ_L_DN(3, 2); // d2 += 3
    CMP_L_DD(1, 2); // flags from d1 - d2
    BCC_B(CC_GT, g_pc + 2 + 2); // bgt.s: skip the sub
    SUB_L_DD(1, 2); // d1 -= d2
    LSL_L(1, 4); // d4 <<= 1
    DBRA(0, loop);
    BRA_B(top);
}

// Kernel 2: memory streaming — 3 memory accesses per 6 instructions
static void emit_kernel_mem(void) {
    uint32_t top = g_pc;
    MOVEA_L_IMM(0, DATA_BASE); // a0 = read pointer
    MOVEA_L_IMM(1, STORE_BASE); // a1 = write pointer
    MOVE_W_IMM_DN(0, 0x0FFF); // 4096 iterations -> walks 16 KB per pass
    uint32_t loop = g_pc;
    MOVE_L_ANP_DN(1, 0); // move.l (a0)+,d1
    ADD_L_DD(2, 1); // d2 += d1
    MOVE_L_DN_ANP(1, 2); // move.l d2,(a1)+
    MOVE_B_D16_DN(3, 0, 2); // move.b 2(a0),d3
    EOR_L_DD(2, 3); // d2 ^= d3
    DBRA(0, loop);
    BRA_W(top);
}

#define STREAM_LEN 8192 // instructions in the random straight-line stream

// Kernel 3: dispatch stress — random straight-line stream (~26 KB of code)
static void emit_kernel_stress(void) {
    uint32_t seed = 0xC0FFEE02u;
    MOVEA_L_IMM(5, DATA_BASE); // a5 = read base
    MOVEA_L_IMM(6, STORE_BASE); // a6 = write base
    for (int r = 0; r < 8; r++)
        MOVE_L_IMM_DN(r, lcg(&seed));
    uint32_t loop = g_pc;
    int force_short = 0; // after a bcc.s +2, the next instruction must be 1 word
    for (int i = 0; i < STREAM_LEN; i++) {
        uint32_t r = lcg(&seed);
        int dn = (int)((r >> 8) & 7), dm = (int)((r >> 12) & 7);
        uint32_t off = (r >> 16) & 0x7FC; // aligned, within the 2 KB window
        int q = 1 + (int)((r >> 26) & 7);
        int pick = force_short ? (int)(r % 40) : (int)(r % 100);
        force_short = 0;
        bool last = (i >= STREAM_LEN - 2);
        if (pick < 8)       ADD_L_DD(dn, dm);
        else if (pick < 14) SUB_L_DD(dn, dm);
        else if (pick < 19) EOR_L_DD(dn, dm);
        else if (pick < 23) AND_L_DD(dn, dm);
        else if (pick < 26) OR_L_DD(dn, dm);
        else if (pick < 30) MOVE_L_DD(dn, dm);
        else if (pick < 33) MOVEQ(dn, (int)(r & 0xFF) - 128);
        else if (pick < 36) { if (r & 0x100000) ADDQ_L_DN(q, dn); else SUBQ_L_DN(q, dn); }
        else if (pick < 40) { if (r & 0x100000) LSR_L(q, dn); else LSL_L(q, dn); }
        else if (pick < 46) MOVE_L_D16_DN(dn, 5, off);
        else if (pick < 50) MOVE_W_D16_DN(dn, 5, off);
        else if (pick < 53) MOVE_B_D16_DN(dn, 5, off);
        else if (pick < 59) MOVE_L_DN_D16(6, off, dn);
        else if (pick < 62) MOVE_W_DN_D16(6, off, dn);
        else if (pick < 65) MOVE_B_DN_D16(6, off, dn);
        else if (pick < 70) MOVE_L_IMM_DN(dn, lcg(&seed));
        else if (pick < 76) CMP_L_DD(dn, dm);
        else if (pick < 80) CMPI_L_DN((int32_t)lcg(&seed), dn);
        else if (pick < 84) { TST_L_DN(dn); }
        else if (!last) { // conditional skip of exactly one 1-word instruction
            static const int conds[6] = { CC_EQ, CC_NE, CC_LT, CC_GE, CC_MI, CC_PL };
            BCC_B(conds[(r >> 5) % 6], g_pc + 2 + 2);
            force_short = 1;
        } else
            ADD_L_DD(dn, dm);
    }
    BRA_W(loop);
}

// Reset guest RAM + CPU state and assemble kernel `k`
static void setup_kernel(int k, m68k_t *m) {
    memset(g_ram, 0, RAM_SIZE);
    uint32_t seed = 0xDEADBEEFu;
    for (uint32_t i = 0; i < 0x10000; i += 4) {
        uint32_t v = lcg(&seed);
        STORE_BE32(g_ram + DATA_BASE + i, v);
    }
    g_pc = CODE_BASE;
    if (k == 0)
        emit_kernel_alu();
    else if (k == 1)
        emit_kernel_mem();
    else
        emit_kernel_stress();
    memset(m, 0, sizeof(*m));
    m->pc = CODE_BASE;
}

// FNV-1a over architectural state and all of RAM; with_flags=false is the
// reduced hash used for the elision backend (see the header comment)
static uint64_t state_hash(const m68k_t *m, bool with_flags) {
    uint64_t h = 0xcbf29ce484222325ull;
#define MIX(vv) do { h ^= (uint64_t)(vv); h *= 0x100000001b3ull; } while (0)
    for (int i = 0; i < 16; i++)
        MIX(m->regs[i]);
    MIX(m->pc);
    if (with_flags) {
        MIX(m->n);
        MIX(m->z);
        MIX(m->v);
        MIX(m->cf);
        MIX(m->x);
    }
    for (uint32_t i = 0; i < RAM_SIZE; i += 8) {
        uint64_t w;
        memcpy(&w, g_ram + i, 8);
        MIX(w);
    }
#undef MIX
    return h;
}

static double now_ms(void) {
#ifdef __EMSCRIPTEN__
    return emscripten_get_now();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
#endif
}

typedef void run_fn(m68k_t *m, uint32_t budget);
#define NBACKENDS 3
static run_fn *const g_backends[NBACKENDS] = { m68k_run_switch, m68k_run_pd, m68k_run_pd_elide };
static const char *const g_backend_names[NBACKENDS] = { "switch", "pd-8B", "pd-elide" };
static const bool g_exact_flags[NBACKENDS] = { true, true, false };
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

    printf("m68k format benchmark: budget=%u reps=%d\n", bench_budget, reps);

    m68k_t cpu;
    for (int k = 0; k < 3; k++) {
        uint64_t full[NBACKENDS], reduced[NBACKENDS];
        for (int b = 0; b < NBACKENDS; b++) {
            setup_kernel(k, &cpu);
            g_backends[b](&cpu, verify_budget);
            full[b] = state_hash(&cpu, true);
            reduced[b] = state_hash(&cpu, false);
        }
        for (int b = 1; b < NBACKENDS; b++) {
            bool ok = g_exact_flags[b] ? (full[b] == full[0]) : (reduced[b] == reduced[0]);
            if (!ok) {
                fprintf(stderr, "FAIL %s: backend %s diverged (%016" PRIx64 " vs %016" PRIx64 ")\n",
                        g_kernel_names[k], g_backend_names[b],
                        g_exact_flags[b] ? full[0] : reduced[0], g_exact_flags[b] ? full[b] : reduced[b]);
                return 1;
            }
        }
        for (int b = 0; b < NBACKENDS; b++) { // warmup
            setup_kernel(k, &cpu);
            g_backends[b](&cpu, bench_budget / 2);
        }
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
