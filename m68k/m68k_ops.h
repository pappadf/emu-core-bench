// SPDX-License-Identifier: MIT
// m68k_ops.h
// Shared 68000-subset semantics: condition-code helpers and condition
// evaluation, used identically by every backend so that final state is
// bit-comparable.

#ifndef M68K_OPS_H
#define M68K_OPS_H

#include "m68k_bench.h"

// Move/logical result flags: N and Z from the result, V=C=0 (X unchanged)
static inline void cc_nz_l(m68k_t *m, uint32_t r) {
    m->n = r >> 31;
    m->z = (r == 0);
    m->v = 0;
    m->cf = 0;
}
static inline void cc_nz_w(m68k_t *m, uint16_t r) {
    m->n = r >> 15;
    m->z = (r == 0);
    m->v = 0;
    m->cf = 0;
}
static inline void cc_nz_b(m68k_t *m, uint8_t r) {
    m->n = r >> 7;
    m->z = (r == 0);
    m->v = 0;
    m->cf = 0;
}

// ADD.L flags (r = d + s), sets X
static inline void cc_add_l(m68k_t *m, uint32_t s, uint32_t d, uint32_t r) {
    m->cf = r < s;
    m->x = m->cf;
    m->v = ((s ^ r) & (d ^ r)) >> 31;
    m->n = r >> 31;
    m->z = (r == 0);
}

// SUB/SUBQ.L flags (r = d - s), sets X
static inline void cc_sub_l(m68k_t *m, uint32_t s, uint32_t d, uint32_t r) {
    m->cf = s > d;
    m->x = m->cf;
    m->v = ((d ^ s) & (d ^ r)) >> 31;
    m->n = r >> 31;
    m->z = (r == 0);
}

// CMP.L flags (r = d - s), X unchanged
static inline void cc_cmp_l(m68k_t *m, uint32_t s, uint32_t d, uint32_t r) {
    m->cf = s > d;
    m->v = ((d ^ s) & (d ^ r)) >> 31;
    m->n = r >> 31;
    m->z = (r == 0);
}

// Bcc condition codes used by the benchmark subset (68000 encodings)
#define M68K_CONDS(X) \
    X(CC, 4,  !m->cf) \
    X(CS, 5,  m->cf) \
    X(NE, 6,  !m->z) \
    X(EQ, 7,  m->z) \
    X(PL, 10, !m->n) \
    X(MI, 11, m->n) \
    X(GE, 12, m->n == m->v) \
    X(LT, 13, m->n != m->v) \
    X(GT, 14, !m->z && m->n == m->v) \
    X(LE, 15, m->z || m->n != m->v)

// Runtime condition evaluation (switch backend); the predecode backend
// instead bakes the condition into the handler id
static inline bool m68k_cond(m68k_t *m, unsigned cond) {
    switch (cond) {
#define CASE(name, code, expr) \
    case code: \
        return expr;
        M68K_CONDS(CASE)
#undef CASE
    default:
        return false; // 0/1 (BRA/BSR forms) handled by callers
    }
}

#endif // M68K_OPS_H
