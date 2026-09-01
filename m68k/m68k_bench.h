// SPDX-License-Identifier: MIT
// m68k_bench.h
// Shared definitions for the M68K (68000-subset) interpreter-format
// benchmark.  Mirrors the PPC lane: several backends stamped over the same
// semantics, running against the shared SoA memory fast path.

#ifndef M68K_BENCH_H
#define M68K_BENCH_H

#include "mem_fastpath.h"

#include <stdbool.h>
#include <stdint.h>

// 68000-subset state.  D0-D7 at byte offsets 0..28, A0-A7 at 32..60.
// Condition codes are kept decomposed (0/1 words) for cheap eager update
// and cheap condition evaluation; both backends share this representation.
typedef struct m68k {
    uint32_t regs[16]; // D0-D7, A0-A7
    uint32_t pc; // next-instruction address at sprint boundaries
    uint32_t n, z, v, cf; // CCR N/Z/V/C as 0/1
    uint32_t x; // CCR X as 0/1
} m68k_t;

#define DR_OFF(n) ((uint8_t)((n) * 4)) // data register byte offset
#define AR_OFF(n) ((uint8_t)(32 + (n) * 4)) // address register byte offset
#define REGO(m, off) (*(uint32_t *)((uint8_t *)(m)->regs + (off)))

// Backends; each executes exactly `budget` instructions from m->pc
void m68k_run_switch(m68k_t *m, uint32_t budget);
void m68k_run_pd(m68k_t *m, uint32_t budget);
void m68k_run_pd_elide(m68k_t *m, uint32_t budget);

// Fatal diagnostic for opcodes outside the benchmark subset (bench_main.c)
void m68k_illegal(m68k_t *m, uint32_t opcode);

#endif // M68K_BENCH_H
