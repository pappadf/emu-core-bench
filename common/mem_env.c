// SPDX-License-Identifier: MIT
// mem_env.c
// Benchmark memory environment: definitions for the globals the fast-path
// accessors reference (the real emulator defines these in memory.c), plus
// fatal slow-path stubs — benchmark kernels must never leave mapped RAM.

#include "mem_env.h"

#include <stdio.h>
#include <stdlib.h>

uint32_t g_address_mask;
uintptr_t *g_active_read;
uintptr_t *g_active_write;
uint32_t g_value_trap_active;
uint32_t g_value_trap_value;
uint32_t g_value_trap_size;

static uint8_t *g_ram; // host backing store for guest RAM at guest address 0

// Any slow-path hit means a benchmark kernel escaped RAM: fail loudly
static void bad_access(const char *what, uint32_t addr) {
    fprintf(stderr, "FATAL: slow-path %s at %08x (kernel escaped RAM)\n", what, addr);
    exit(1);
}
uint8_t memory_read_uint8_slow(uint32_t addr) { bad_access("read8", addr); return 0; }
uint16_t memory_read_uint16_slow(uint32_t addr) { bad_access("read16", addr); return 0; }
uint32_t memory_read_uint32_slow(uint32_t addr) { bad_access("read32", addr); return 0; }
void memory_write_uint8_slow(uint32_t addr, uint8_t v) { (void)v; bad_access("write8", addr); }
void memory_write_uint16_slow(uint32_t addr, uint16_t v) { (void)v; bad_access("write16", addr); }
void memory_write_uint32_slow(uint32_t addr, uint32_t v) { (void)v; bad_access("write32", addr); }
void value_trap_check(uint32_t la, uint32_t v, unsigned s) { (void)la; (void)v; (void)s; }

// Allocate guest RAM and point every SoA entry at it.  Guest RAM is
// contiguous at guest address 0, so each adjusted-base entry
// (host_base - guest_page_base) is simply the host base pointer.
void mem_env_init(uint32_t ram_bytes) {
    uint32_t pages = ram_bytes >> PAGE_SHIFT;
    g_ram = malloc(ram_bytes);
    uintptr_t *rd = malloc(pages * sizeof(uintptr_t));
    uintptr_t *wr = malloc(pages * sizeof(uintptr_t));
    if (!g_ram || !rd || !wr) {
        fprintf(stderr, "FATAL: mem_env_init(%u) allocation failed\n", ram_bytes);
        exit(1);
    }
    for (uint32_t p = 0; p < pages; p++) {
        rd[p] = (uintptr_t)g_ram;
        wr[p] = (uintptr_t)g_ram;
    }
    g_address_mask = ram_bytes - 1;
    g_active_read = rd;
    g_active_write = wr;
}

uint8_t *mem_env_ram(void) {
    return g_ram;
}
