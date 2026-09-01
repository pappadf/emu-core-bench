// SPDX-License-Identifier: MIT
// mem_fastpath.h
// Standalone mock of granny-smith's SoA memory fast path, kept structurally
// identical to src/core/memory/memory.h (MIT, Copyright (c) pappadf) so that
// per-instruction memory-access cost in these benchmarks matches the real
// emulator: same adjusted-base page-table trick, same page-crossing checks,
// same big-endian byteswap macros, same value-trap branch on the write path.
// Only the emulator-specific slow-path machinery (devices, MMU, bus errors,
// I/O penalties) is reduced to the extern declarations the inline accessors
// reference; the benchmark environment (mem_env.c) supplies them.

#ifndef MEM_FASTPATH_H
#define MEM_FASTPATH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// === Big-endian access macros (guest memory image is big-endian) ===
#define LOAD_BE8(p)  (*(const uint8_t *)(p))
#define LOAD_BE16(p) (__builtin_bswap16(*(const uint16_t *)(p)))
#define LOAD_BE32(p) (__builtin_bswap32(*(const uint32_t *)(p)))

#define STORE_BE8(p, v)  (*(uint8_t *)(p) = (uint8_t)(v))
#define STORE_BE16(p, v) (*(uint16_t *)(p) = __builtin_bswap16((uint16_t)(v)))
#define STORE_BE32(p, v) (*(uint32_t *)(p) = __builtin_bswap32((uint32_t)(v)))

// === Page table geometry ===
#define PAGE_SHIFT    12
#define MEM_PAGE_SIZE (1 << PAGE_SHIFT) // 4096
#define PAGE_MASK     (MEM_PAGE_SIZE - 1) // 0xFFF

// SoA (Struct-of-Arrays) fast-path page tables.
// Each entry stores an adjusted host address: (uintptr_t)host_base - page_guest_base
// so that (uint8_t *)(entry + masked_addr) yields the correct host pointer directly.
// Zero entry = slow path (device I/O, unmapped, or MMU miss in the emulator;
// a fatal diagnostic in the benchmark environment).
extern uintptr_t *g_active_read;
extern uintptr_t *g_active_write;
extern uint32_t g_address_mask; // e.g. RAM_SIZE-1 in the benchmark env

// Slow-path handlers (benchmark env: abort with a diagnostic — kernels must
// never leave mapped RAM, so any hit is a bug in the benchmark itself)
uint8_t memory_read_uint8_slow(uint32_t addr);
uint16_t memory_read_uint16_slow(uint32_t addr);
uint32_t memory_read_uint32_slow(uint32_t addr);
void memory_write_uint8_slow(uint32_t addr, uint8_t value);
void memory_write_uint16_slow(uint32_t addr, uint16_t value);
void memory_write_uint32_slow(uint32_t addr, uint32_t value);

// Value trap: granny-smith's fast-path needle search.  Kept (disabled) so
// the write path carries the same inline check as the real emulator.
extern uint32_t g_value_trap_active; // 0 = disabled
extern uint32_t g_value_trap_value;
extern uint32_t g_value_trap_size;
void value_trap_check(uint32_t logical_addr, uint32_t value, unsigned size);

// === Inline accessors (verbatim structure from granny-smith memory.h) ===

static inline uint8_t memory_read_uint8(uint32_t addr) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_read[masked >> PAGE_SHIFT];
    if (__builtin_expect(base != 0, 1))
        return LOAD_BE8((uint8_t *)(base + masked));
    return memory_read_uint8_slow(masked);
}

static inline uint16_t memory_read_uint16(uint32_t addr) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_read[masked >> PAGE_SHIFT];
    // Fast path: non-zero entry and access doesn't cross page boundary
    if (__builtin_expect(base != 0 && (masked & PAGE_MASK) <= MEM_PAGE_SIZE - 2, 1))
        return LOAD_BE16((uint8_t *)(base + masked));
    return memory_read_uint16_slow(masked);
}

static inline uint32_t memory_read_uint32(uint32_t addr) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_read[masked >> PAGE_SHIFT];
    // Fast path: non-zero entry and access doesn't cross page boundary
    if (__builtin_expect(base != 0 && (masked & PAGE_MASK) <= MEM_PAGE_SIZE - 4, 1)) {
        return LOAD_BE32((uint8_t *)(base + masked));
    }
    return memory_read_uint32_slow(masked);
}

static inline void memory_write_uint8(uint32_t addr, uint8_t value) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_write[masked >> PAGE_SHIFT];
    if (__builtin_expect(g_value_trap_active && g_value_trap_size == 1 && (uint8_t)value == (uint8_t)g_value_trap_value,
                         0))
        value_trap_check(masked, value, 1);
    if (__builtin_expect(base != 0, 1)) {
        STORE_BE8((uint8_t *)(base + masked), value);
        return;
    }
    memory_write_uint8_slow(masked, value);
}

static inline void memory_write_uint16(uint32_t addr, uint16_t value) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_write[masked >> PAGE_SHIFT];
    if (__builtin_expect(
            g_value_trap_active && g_value_trap_size == 2 && (uint16_t)value == (uint16_t)g_value_trap_value, 0))
        value_trap_check(masked, value, 2);
    // Fast path: non-zero entry and access doesn't cross page boundary
    if (__builtin_expect(base != 0 && (masked & PAGE_MASK) <= MEM_PAGE_SIZE - 2, 1)) {
        STORE_BE16((uint8_t *)(base + masked), value);
        return;
    }
    memory_write_uint16_slow(masked, value);
}

static inline void memory_write_uint32(uint32_t addr, uint32_t value) {
    uint32_t masked = addr & g_address_mask;
    uintptr_t base = g_active_write[masked >> PAGE_SHIFT];
    if (__builtin_expect(g_value_trap_active && g_value_trap_size == 4 && value == g_value_trap_value, 0))
        value_trap_check(masked, value, 4);
    // Fast path: non-zero entry and access doesn't cross page boundary
    if (__builtin_expect(base != 0 && (masked & PAGE_MASK) <= MEM_PAGE_SIZE - 4, 1)) {
        STORE_BE32((uint8_t *)(base + masked), value);
        return;
    }
    memory_write_uint32_slow(masked, value);
}

#endif // MEM_FASTPATH_H
