// SPDX-License-Identifier: MIT
// ppc_bench.h
// Shared definitions for the PPC dispatch-architecture benchmark prototype.
// Two interpreter backends (big-switch and musttail tail-call) are stamped
// from the same opcode header (ppc_ops.h) and run against the emulator's
// real SoA memory fast path (src/core/memory/memory.h).

#ifndef PPC_BENCH_H
#define PPC_BENCH_H

#include "mem_fastpath.h" // standalone mock of granny-smith's memory fast path

#include <stdbool.h>
#include <stdint.h>

// Benchmark-subset PPC601-class integer state (no FPU, no MSR, no XER.SO)
typedef struct ppc {
    uint32_t gpr[32]; // general-purpose registers
    uint32_t zero_base; // always 0: base-register slot for ra=0 forms (pdx backends)
    uint32_t pc; // next-instruction address at sprint boundaries
    uint32_t lr; // link register
    uint32_t ctr; // count register
    uint32_t cr; // full condition register, CR0.LT at bit 31
    uint32_t xer_ca; // XER carry bit only (benchmark subset)
} ppc_t;

// musttail is required for the wasm build (emcc/clang emits return_call_indirect
// with -mtail-call); native gcc falls back to sibling-call optimization at -O2
#if defined(__clang__) && __has_attribute(musttail)
#define MUSTTAIL __attribute__((musttail))
#else
#define MUSTTAIL
#endif

// The backends; each executes exactly `budget` instructions from c->pc
void ppc_run_switch(ppc_t *c, uint32_t budget);
void ppc_run_tail(ppc_t *c, uint32_t budget);
void ppc_run_call(ppc_t *c, uint32_t budget);
void ppc_run_pd(ppc_t *c, uint32_t budget);
void ppc_run_pdx8(ppc_t *c, uint32_t budget);
void ppc_run_pdx16(ppc_t *c, uint32_t budget);

// Fatal diagnostic for opcodes outside the benchmark subset (bench_main.c)
void ppc_illegal(ppc_t *c, uint32_t insn);

#endif // PPC_BENCH_H
