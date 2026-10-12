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

// The device window (mem_env.h)
static uint32_t g_dev_base, g_dev_end;
static uint64_t g_dev_state = 0xcbf29ce484222325ull;
static uint64_t g_dev_accesses;
uint64_t (*g_dev_cpu_snapshot)(void);

static void dev_mix(uint64_t v) {
    g_dev_state ^= v;
    g_dev_state *= 0x100000001b3ull;
}
static bool in_device(uint32_t addr) {
    return addr >= g_dev_base && addr < g_dev_end;
}
static uint32_t dev_access(uint32_t addr, uint32_t value, unsigned size, bool write) {
    g_dev_accesses++;
    dev_mix(((uint64_t)addr << 8) | (size << 1) | (write ? 1u : 0u));
    if (write)
        dev_mix(value);
    if (g_dev_cpu_snapshot)
        dev_mix(g_dev_cpu_snapshot());
    return (uint32_t)(g_dev_state ^ (g_dev_state >> 29));
}

// Code-page marks (mem_env.h)
static uint8_t *g_code_marks; // per page
static uint32_t *g_marked_list;
static uint32_t g_marked_count, g_pages_total;
void (*g_code_write_hook)(uint32_t page);

void mem_env_code_mark(uint32_t page) {
    if (g_code_marks[page])
        return;
    g_code_marks[page] = 1;
    g_marked_list[g_marked_count++] = page;
    g_active_write[page] = 0;
}
void mem_env_code_unmark(uint32_t page) {
    if (!g_code_marks[page])
        return;
    g_code_marks[page] = 0;
    g_active_write[page] = (uintptr_t)g_ram;
    for (uint32_t i = 0; i < g_marked_count; i++)
        if (g_marked_list[i] == page) {
            g_marked_list[i] = g_marked_list[--g_marked_count];
            break;
        }
}
void mem_env_code_unmark_all(void) {
    while (g_marked_count)
        mem_env_code_unmark(g_marked_list[g_marked_count - 1]);
}
bool mem_env_code_marked(uint32_t page) {
    return g_code_marks[page] != 0;
}

// A store into a marked page: perform it (it cannot cross out of RAM; a
// page-crossing access is split byte-wise by its size), then tell the owner.
static bool code_store(uint32_t addr, uint32_t v, unsigned size) {
    uint32_t page = addr >> PAGE_SHIFT;
    uint32_t last = (addr + size - 1) >> PAGE_SHIFT;
    if (!g_code_marks[page] && !(last < g_pages_total && g_code_marks[last]))
        return false;
    for (unsigned i = 0; i < size; i++)
        g_ram[(addr + i) & g_address_mask] = (uint8_t)(v >> (8 * (size - 1 - i)));
    if (g_code_write_hook) {
        g_code_write_hook(page);
        if (last != page)
            g_code_write_hook(last);
    }
    return true;
}

// Outside the device window, any slow-path hit means a benchmark kernel
// escaped RAM: fail loudly
static void bad_access(const char *what, uint32_t addr) {
    fprintf(stderr, "FATAL: slow-path %s at %08x (kernel escaped RAM)\n", what, addr);
    exit(1);
}
uint8_t memory_read_uint8_slow(uint32_t addr) {
    if (in_device(addr))
        return (uint8_t)dev_access(addr, 0, 1, false);
    bad_access("read8", addr);
    return 0;
}
uint16_t memory_read_uint16_slow(uint32_t addr) {
    if (in_device(addr))
        return (uint16_t)dev_access(addr, 0, 2, false);
    if (g_active_read[addr >> PAGE_SHIFT]) // a page-crossing read in RAM
        return (uint16_t)((g_ram[addr & g_address_mask] << 8) | g_ram[(addr + 1) & g_address_mask]);
    bad_access("read16", addr);
    return 0;
}
uint32_t memory_read_uint32_slow(uint32_t addr) {
    if (in_device(addr))
        return dev_access(addr, 0, 4, false);
    if (g_active_read[addr >> PAGE_SHIFT]) { // a page-crossing read in RAM
        uint32_t v = 0;
        for (int i = 0; i < 4; i++)
            v = (v << 8) | g_ram[(addr + i) & g_address_mask];
        return v;
    }
    bad_access("read32", addr);
    return 0;
}
void memory_write_uint8_slow(uint32_t addr, uint8_t v) {
    if (in_device(addr)) {
        dev_access(addr, v, 1, true);
        return;
    }
    if (code_store(addr, v, 1))
        return;
    bad_access("write8", addr);
}
void memory_write_uint16_slow(uint32_t addr, uint16_t v) {
    if (in_device(addr)) {
        dev_access(addr, v, 2, true);
        return;
    }
    if (code_store(addr, v, 2))
        return;
    bad_access("write16", addr);
}
void memory_write_uint32_slow(uint32_t addr, uint32_t v) {
    if (in_device(addr)) {
        dev_access(addr, v, 4, true);
        return;
    }
    if (code_store(addr, v, 4))
        return;
    bad_access("write32", addr);
}

void mem_env_device(uint32_t base, uint32_t bytes) {
    g_dev_base = base;
    g_dev_end = base + bytes;
    for (uint32_t a = base; a < base + bytes; a += MEM_PAGE_SIZE) {
        g_active_read[a >> PAGE_SHIFT] = 0;
        g_active_write[a >> PAGE_SHIFT] = 0;
    }
}
void mem_env_device_reset(void) {
    g_dev_state = 0xcbf29ce484222325ull;
    g_dev_accesses = 0;
}
uint64_t mem_env_device_state(void) {
    return g_dev_state;
}
uint64_t mem_env_device_accesses(void) {
    return g_dev_accesses;
}
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
    g_pages_total = pages;
    g_code_marks = calloc(pages, 1);
    g_marked_list = malloc(pages * sizeof(uint32_t));
}

uint8_t *mem_env_ram(void) {
    return g_ram;
}
