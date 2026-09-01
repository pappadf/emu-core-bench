# emu-core-bench: interpreter-core design benchmarks for CPU emulators.
# Codegen flags mirror granny-smith's release wasm build (-O2, no LTO);
# -mtail-call is required for the return_call_indirect backend.

CFLAGS   := -O2 -DNDEBUG -Wall -Icommon $(BENCH_DEFS)
EMFLAGS  := -mtail-call -Wno-initializer-overrides
BUILD    := build

COMMON_SRCS := common/mem_env.c
M68K_SRCS := m68k/bench_main.c m68k/m68k_switch.c m68k/m68k_pd.c m68k/m68k_pd_elide.c $(COMMON_SRCS)
M68K_HDRS := m68k/m68k_bench.h m68k/m68k_ops.h m68k/m68k_pd_impl.h common/mem_fastpath.h common/mem_env.h
PPC_SRCS := ppc/bench_main.c ppc/ppc_switch.c ppc/ppc_tail.c ppc/ppc_call.c ppc/ppc_predecode.c ppc/ppc_pdx8.c ppc/ppc_pdx16.c $(COMMON_SRCS)
PPC_HDRS := ppc/ppc_bench.h ppc/ppc_ops.h common/mem_fastpath.h common/mem_env.h

all: $(BUILD)/ppc-bench-native $(BUILD)/ppc-bench.js $(BUILD)/ppc-bench-web.html $(BUILD)/m68k-bench-native $(BUILD)/m68k-bench.js $(BUILD)/be-micro-native $(BUILD)/be-micro.js

$(BUILD):
	mkdir -p $(BUILD)

# --- PPC dispatch benchmark ---
$(BUILD)/ppc-bench-native: $(PPC_SRCS) $(PPC_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(PPC_SRCS) -o $@

$(BUILD)/ppc-bench.js: $(PPC_SRCS) $(PPC_HDRS) | $(BUILD)
	emcc $(CFLAGS) $(EMFLAGS) $(PPC_SRCS) -o $@ -sENVIRONMENT=node

$(BUILD)/ppc-bench-web.html: $(PPC_SRCS) $(PPC_HDRS) | $(BUILD)
	emcc $(CFLAGS) $(EMFLAGS) $(PPC_SRCS) -o $@ -sENVIRONMENT=web

# --- byte-swap microbenchmark ---
$(BUILD)/be-micro-native: micro/be_micro.c | $(BUILD)
	$(CC) $(CFLAGS) micro/be_micro.c -o $@

$(BUILD)/be-micro.js: micro/be_micro.c | $(BUILD)
	emcc $(CFLAGS) micro/be_micro.c -o $@ -sENVIRONMENT=node

# --- convenience runners ---
run-native: $(BUILD)/ppc-bench-native
	$(BUILD)/ppc-bench-native

run-wasm: $(BUILD)/ppc-bench.js
	node $(BUILD)/ppc-bench.js

run-browsers: $(BUILD)/ppc-bench-web.html
	cd $(BUILD) && BENCH_PAGE=ppc-bench-web.html node ../tools/run_browsers.mjs

clean:
	rm -rf $(BUILD)

.PHONY: all run-native run-wasm run-browsers clean

$(BUILD)/m68k-bench-native: $(M68K_SRCS) $(M68K_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(M68K_SRCS) -o $@

$(BUILD)/m68k-bench.js: $(M68K_SRCS) $(M68K_HDRS) | $(BUILD)
	emcc $(CFLAGS) -Wno-initializer-overrides $(M68K_SRCS) -o $@ -sENVIRONMENT=node
