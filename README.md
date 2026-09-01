# emu-core-bench: Measured Comparisons of Interpreter-Core Designs for CPU Emulators on WebAssembly and Native Hosts

## Abstract

Retro-computer emulators targeting the browser must interpret guest CPU
instructions inside WebAssembly, where the folklore of interpreter design —
computed-goto dispatch, tail-call threading, dynarecs — meets an unusual set
of constraints: no computed goto, no runtime code generation from within a
module, and an indirect-call primitive that carries a per-call signature
check.  This project measures, with controlled experiments, two questions:
**which dispatch architecture is fastest**, and **what the optimal
predecoded intermediate format looks like**, for two guest ISAs of opposite
character — fixed-width PowerPC and variable-length, EA-mode-rich,
eager-flags M68K.

A PowerPC-subset core is implemented six times from one shared opcode
header, varying only the interpreter architecture: (A) a single-function
big-switch loop, (B) `musttail` tail-call threading, (C) call-threading,
(D) predecode into raw `(handler id, instruction)` entries, (E) predecode
into 8-byte entries with pre-extracted operands and a specialized id space,
and (F) a 16-byte fully-resolved variant.  An M68K-subset core is
implemented three times: big switch, predecode with the format adapted to
the 68K's shape, and predecode plus a flag-liveness pass that elides dead
condition-code computation at decode time.  All backends are verified to
produce equivalent architectural state before timing.

On V8 (Node 22, Chromium 145): tail-call threading — the celebrated
wasm3/CPython-3.14 technique — is **16–59% slower** than the plain switch,
attributable to `call_indirect`'s per-dispatch overhead rather than
tail-chaining itself.  Predecoding is the design that pays: raw predecode
gains **+57…+100%**, the pre-extracted 8-byte format **+74…+146%**, while
the 16-byte format shows the cache-footprint penalty and wins nothing
further.  On M68K the payoff is larger still — **+82…+142%** for the
8-byte format, and decode-time flag elision lifts it to **+89…+174%** —
because predecoding deletes the 68K's runtime EA-mode dispatch,
variable-length fetch, and most eager flag writes.  We conclude that the
optimal C-expressible emulator core on today's wasm engines is a flat
`br_table` switch over a **specialized-id, operand-pre-extracted, 8-byte
per-instruction predecoded format**, with the id space (not entry width)
carrying the specialization.

## 1. Introduction

Interpreter dispatch is one of the most retold performance stories in
systems programming: computed goto beats switch, tail-call threading beats
computed goto, and dynamic translation beats them all.  Most of the
supporting data comes from native-code language runtimes (wasm3, upb,
CPython 3.14, LuaJIT's interpreter).  A browser-hosted emulator lives in a
different regime:

- The interpreter itself is compiled **to** WebAssembly and then
  JIT-compiled by the browser engine; the engine's code generator, not the
  C compiler, has the final word.
- wasm has **no computed goto**; a C `switch` becomes a single `br_table`.
  The only way to express "replicated dispatch sites" (the mechanism behind
  computed-goto's and tail-threading's predicted-branch advantage) is tail
  calls (`return_call_indirect`, shipped in all major engines).
- wasm's indirect call checks a signature on **every** call.  The checkless
  typed-reference form (`call_ref`) exists in the spec (WasmGC) but is
  unreachable from C/LLVM, which lowers all function pointers to indices
  into an untyped `funcref` table.
- Runtime code generation from inside a module is impossible; JIT-style
  translation requires round-tripping generated modules through JavaScript
  — out of scope for projects committed to plain portable C.

Within those constraints, the classical alternative to better dispatch is a
better *intermediate format*: predecode the guest code once into a side
table and interpret that.  A predecoded format is a design space of its own
— what the entry holds (raw word vs pre-extracted operands), how wide it
is, how much decode-time *specialization* the handler-id space absorbs, and
(for an ISA with condition codes) whether decode-time analysis can delete
work outright.  This project walks that ladder experimentally.

The immediate consumer is
[granny-smith](https://github.com/pappadf/granny-smith), a browser-based
Macintosh/Lisa emulator whose 68K-family cores use a shared-header
big-switch design and which contemplates a future PowerPC core; the
questions and answers generalize to any emulator with a similar deployment
target.

## 2. Benchmark design

### 2.1 Common method

Guest memory accesses go through `common/mem_fastpath.h`, a structurally
line-for-line mock of granny-smith's production SoA (struct-of-arrays)
software-TLB fast path: 4 KB pages, per-page *adjusted host base* entries
(`host_base − guest_page_base`, so one add yields the host pointer),
page-crossing checks on 16/32-bit accesses, big-endian storage with
`__builtin_bswap` accessors, and the (disabled) value-trap compare on the
write path.  Per-instruction memory cost therefore matches a real emulator,
not a toy `uint8_t mem[]`.  The benchmarks map 8 MB of RAM at guest address
0; slow-path entry aborts the run (kernels are constructed never to leave
mapped RAM).

Each architecture lane assembles three kernels into guest RAM with a
host-side mini-assembler:

1. **alu-loop** — a ~10-instruction predictable loop; small code footprint,
   trivially predictable dispatch — the switch backend's best case.
2. **mem-stream** — ~3 memory operations per 6–8 instructions through the
   SoA fast path, walking a bounded window.
3. **dispatch-stress** — thousands of pseudo-random straight-line
   instructions (26–64 KB of code, looped), with occasional data-dependent
   conditional skips: opcode sequence and code footprint defeat trivial
   dispatch prediction — the workload the threading literature says should
   favor replicated dispatch sites.

Runs are **budget-driven**: a run executes exactly N guest instructions
(default 100 M; compile-time override `BENCH_DEFAULT_BUDGET`/`_REPS`,
runtime override via argv on native/node builds).  Before any timing, every
backend executes 20 M instructions of each kernel from identical initial
state and an FNV-1a hash over the complete architectural state **and all
8 MB of guest RAM** must match across backends (one deliberate exception
for flag elision, §2.4); a mismatch aborts.  Timing runs each backend for a
half-budget warm-up (letting the engine's optimizing tier compile), then R
alternating repetitions, reporting best-of-R.  MIPS = budget / best time.

### 2.2 The PPC lane: six backends, one opcode header

The unit under test is a PowerPC-subset integer core (48 instructions:
D/X-form loads/stores including indexed and update forms, XO-form
arithmetic with `Rc`, immediates, `rlwinm`, compares, the full `bc` BO/BI
machinery, `b`, `bclr`/`bcctr`, LR/CTR moves).  All semantics live once in
`ppc/ppc_ops.h` as generic `static inline` bodies stamped through X-macro
instruction lists; each backend expands the same lists into its own
dispatch shape, so the interpreter architecture is the only variable:

| backend | file | shape |
|---|---|---|
| A: switch | `ppc_switch.c` | one function, `while` + nested `switch` (one `br_table`) |
| B: tail-call | `ppc_tail.c` | function per opcode, `musttail` through a const pointer table (`return_call_indirect` per instruction) |
| C: call-loop | `ppc_call.c` | same functions, plain `call_indirect` from a central loop |
| D: pd-raw | `ppc_predecode.c` | predecode to flat `(id, raw insn)` 8-byte entries; fields extracted at run time |
| E: pdx-8B | `ppc_pdx8.c` | predecode with **pre-extracted operands and specialized ids**, 8-byte entries |
| F: pdx-16B | `ppc_pdx16.c` | same as E with 16-byte entries (full `rlwinm` pre-extraction) |

Backend C is a control: B−C isolates tail-chaining, A−C isolates the
indirect-call primitive.  D isolates "predecode at all."  E and F walk the
format ladder (§2.3).

### 2.3 The intermediate-format ladder (backends D → E → F)

Backend D's entry is `{u16 handler-id, u32 raw instruction}`: it removes
the guest fetch (SoA lookup + byte swap) and flattens two-level decode to
one `br_table`, but handlers still extract fields with shifts and masks.

Backend E keeps 8 bytes per instruction — `{u16 id; u8 a; u8 b; u32 c}` —
but changes what they carry:

- **Registers as pre-scaled byte offsets** into the register file (no
  shift/mask at run time; `ra=0` addressing folds to an always-zero slot in
  the state struct).
- **The id space is the specialization axis**: `Rc` record forms, `ra=0`
  forms, `addi`/`addis` (immediate pre-shifted) merging, and the common
  `bc` shapes (`bdnz`, branch-on-CR-bit-true/false with a precomputed bit
  shift) each resolve to distinct leaf ids at decode time — runtime flag
  tests disappear into dispatch.
- **Branch targets are precomputed entry indices**; the hot loop walks an
  entry pointer and never materializes the guest pc (reconstructed as
  `(cur − tab) << 2` only at sprint exit and in rare generic fallbacks).
- **Per-id raw fallback is part of the format**: awkward shapes (`rlwinm`'s
  sh+mask+2 registers exceed 8 bytes; uncommon BO combinations; `bclr`)
  keep the raw word and reuse the generic semantics — the optimal format is
  allowed to be non-uniform.

Backend F doubles the entry to 16 bytes, buying full `rlwinm`
pre-extraction (mask and rotate stored) and headroom a fused-pair format
would need — at twice the table footprint (the dispatch-stress kernel's
side table grows from ~130 KB to ~260 KB).

### 2.4 The M68K lane: the format follows the ISA

`m68k/` implements a 68000-subset core (MOVE B/W/L across Dn/(An)+/d16(An)/
immediate forms, MOVEA, ADD/SUB/CMP/CMPI/TST, AND/OR/EOR, ADDQ/SUBQ,
LSL/LSR, LEA, MOVEQ, Bcc/BRA/DBRA, NOP — with 68000-exact condition codes
including X) three times:

- **switch** (`m68k_switch.c`): conventional runtime decode — 16-bit opcode
  fetch, nested opcode-group switches, runtime EA-mode switches with
  extension-word fetches, eager condition codes.
- **pd-8B** (`m68k_pd.c`): predecode with the format adapted to the 68K's
  shape: **one entry per 16-bit word** (`index = pc >> 1`; multi-word
  instructions leave never-executed filler entries), **instruction length
  folded into the specialized id** (each handler advances by its own
  constant — no length field), extension words (displacements, immediates)
  pre-extracted, **EA modes and sizes resolved into the id space** (one
  leaf id per op × size × EA combination), branch targets as entry indices
  with conditions baked into per-condition ids.
- **pd-elide** (`m68k_pd_elide.c`): the same, plus a decode-time
  **flag-liveness pass**: an instruction whose eagerly-computed NZVC is
  overwritten by the next sequential instruction before any reader is
  retargeted to a no-flags twin handler; compare/test instructions whose
  only effect is flags become sized NOPs.  The rule is a 1-lookahead
  (nothing reads CCR between adjacent instructions; the property is
  transitive across elided successors; X is conservatively preserved).
  This is analysis the *format* can carry and a runtime interpreter cannot
  — lazy-flags schemes recompute on demand, whereas elision deletes the
  work statically.

Verification note: pd-elide intentionally leaves stale NZVC at sprint
boundaries (dead on the guest's own path, but a sprint can cut between
definer and overwriter), so it is verified on a reduced hash excluding CCR;
registers, pc and all of RAM must still match bit-for-bit — the kernels
branch on flags constantly, so unsound elision diverges that hash
immediately.

## 3. Experimental setup

| component | value |
|---|---|
| host | GitHub Codespaces devcontainer, Intel Xeon Platinum 8370C @ 2.80 GHz, 2 vCPU (shared/cloud — see §6), Linux 6.8 |
| C compiler (native) | gcc 13.3.0, `-O2 -DNDEBUG` |
| wasm toolchain | Emscripten 4.0.10 (clang-based), `-O2 -DNDEBUG -mtail-call` (mirrors granny-smith's release flags; no LTO, no SIMD) |
| engines | Node 22.23.2 (V8), Chromium 145 and Firefox 146 via Playwright |
| date | 2026-09-01 |

`musttail` is honored by clang/emcc (verified: one `return_call_indirect`
per handler in the module).  The native gcc build lacks `musttail` and
relies on `-O2` sibling-call optimization — its backend-B numbers are
labeled accordingly.

## 4. Results

Percentages are relative to the switch backend of the same run.  V8 ranges
span two independent node passes (best-of-5 each, 100 M instructions).

### 4.1 PPC on V8 (Node 22)

| kernel | switch | B: tail-call | C: call-loop | D: pd-raw | E: pdx-8B | F: pdx-16B |
|---|---|---|---|---|---|---|
| alu-loop | 141–157 MIPS | −53…−58% | −26…−27% | +87% | **+93%** | +104…+107% |
| mem-stream | 145–165 MIPS | −55…−59% | −26% | +97…+100% | **+133…+138%** | +120…+122% |
| dispatch-stress | 48–53 MIPS | −16…−19% | −19…−23% | +60…+63% | **+79…+84%** | +77…+84% |

### 4.2 PPC in Chromium 145 (same V8 family; confirms §4.1)

| kernel | switch | tail-call | call-loop | pd-raw | pdx-8B | pdx-16B |
|---|---|---|---|---|---|---|
| alu-loop | 143.5 MIPS | −54% | −35% | +75% | **+95%** | +105% |
| mem-stream | 145.9 MIPS | −54% | −33% | +86% | **+146%** | +138% |
| dispatch-stress | 50.8 MIPS | −24% | −29% | +57% | **+74%** | +65% |

**Firefox 146 (Playwright build; backends A–D, 30 M budget).**  This
environment executes the whole module ~8× slower than V8, a control module
with zero tail-call instructions is equally slow, and wasm-JIT prefs change
nothing — this Firefox appears pinned to a non-optimizing tier.  With that
caveat: tail ≈ parity with switch, and **predecode still wins +55…+99%**,
i.e. on both ends of the engine-sophistication spectrum
(`results-firefox.log`).

### 4.3 PPC native x86-64 (gcc 13; B via sibling calls, not musttail; 30 M budget)

| kernel | switch | tail-call | call-loop | pd-raw | pdx-8B | pdx-16B |
|---|---|---|---|---|---|---|
| alu-loop | 213.2 MIPS | −0.5% | −3.7% | +41% | +79% | **+92%** |
| mem-stream | 248.4 MIPS | −9.4% | −6.7% | +48% | +108% | **+110%** |
| dispatch-stress | 57.7 MIPS | +6.4% | +6.4% | +52% | +67% | **+69%** |

### 4.4 M68K on V8 (Node 22; Chromium 145 in parentheses)

| kernel | switch | pd-8B | pd-elide |
|---|---|---|---|
| alu-loop | 160–161 MIPS (171) | +140…+142% (+134%) | **+147…+154% (+150%)** |
| mem-stream | 116–118 MIPS (128) | +127…+130% (+138%) | **+162…+174% (+153%)** |
| dispatch-stress | 43 MIPS (47) | +104% (+82%) | **+111…+114% (+89%)** |

Native (gcc, 30 M): pd-8B +65…+119%, pd-elide +73…+137%.

### 4.5 Byte-swap microbenchmark (`micro/be_micro.c`)

wasm has no scalar byte-swap instruction; LLVM lowers `__builtin_bswap32`
to 9 ALU operations and canonicalizes hand-written rotate variants to
identical code (0 `i32.rotl` in the binary).  Measured cost vs a raw
little-endian load, identical address streams: **−4.1%** independent /
**−3.7%** dependent on V8; ~0% native (single `bswap` insn — but gcc takes
the rotate idiom literally: −12…−20%, a pessimization).  Use the intrinsic,
always; the swap is a minor cost and unimprovable from C.

## 5. Analysis

**5.1 The indirect-call tax, isolated.**  B and C share small per-opcode
functions and differ only in chaining; both lose heavily on V8.  Since C
differs from A only by dispatching through `call_indirect` instead of
`br_table`, the ~26–35% A→C gap prices wasm's indirect call (function
pointer load from linear memory + table bounds check + signature check +
call) — paid per emulated instruction.  B's deeper deficit on predictable
kernels reflects V8's costlier `return_call_indirect`.  The literature's
tail-threading win presupposes an indirect jump no dearer than a switch's;
wasm breaks that premise.  Natively, B beats the switch only under dispatch
stress (+6%) — the folklore reproduced, in the one place it applies.

**5.2 Where interpreter time actually goes.**  The switch backend drops
~3× between alu-loop and dispatch-stress while the memory-heavy kernel
costs nothing extra: fetch/decode/dispatch dominates, not memory access.
Predecode attacks exactly that, which is why it wins on every engine
tested, including Firefox's baseline tier.

**5.3 The format ladder has a clear sweet spot: 8 bytes, specialized ids.**
Pre-extraction (D→E) is worth another 6–46 points on V8 — largest on
mem-stream, where address arithmetic collapses into `base + preextracted
imm` with pre-scaled register offsets and entry-pointer walking removes the
per-instruction pc maintenance.  Doubling the entry (E→F) buys nothing
robust: it wins a few points on the tiny alu-loop (footprint irrelevant,
`rlwinm` fully pre-extracted) and **loses** on mem-stream and
dispatch-stress on V8, where the side table's cache footprint doubles.
Natively, with bigger caches, F ties or edges E.  Conclusion: keep entries
at 8 bytes; let the *id space*, not the entry width, absorb specialization;
allow per-id raw fallback for shapes that don't fit.

**5.4 M68K profits more, and differently.**  The 68K switch baseline pays
for what PPC never had: runtime EA-mode dispatch, extension-word fetches,
variable-length pc advance, and eager NZVCX on nearly every instruction.
The predecoded format deletes the first three statically (entry-per-word
indexing, length-in-id, pre-extracted extension words, EA-in-id), which is
why pd-8B's +104…+142% on V8 exceeds the PPC lane's +79…+138% despite an
identical memory model.  Flag elision then deletes a fraction of the
fourth: +7…+44 further points (largest on mem-stream, whose MOVE-heavy
stream is almost entirely dead flag writes).  Elision is a *format-level*
optimization — a one-pass, 1-lookahead liveness scan at decode time, no
runtime lazy-flags machinery — and its soundness is mechanically checked
here by cross-backend state hashing.

**5.5 Implications for emulator cores.**  For a wasm-deployed emulator in
portable C: (i) keep big-switch dispatch — on today's V8 it beats every
C-expressible threading alternative; (ii) the profitable evolution is a
predecoded 8-byte-per-instruction format with specialized handler ids,
pre-extracted operands, and precomputed branch-target indices; (iii) on
68K-class ISAs, add decode-time flag-liveness elision — it is cheap, sound,
and worth more than any dispatch trick measured here; (iv) production
concerns are lazy per-page decode, invalidation on code writes (an SoA
memory fast path supports this cheaply: zero the page's write entry so
stores take a slow path that invalidates), and treating the cache as
derived data for snapshot/restore; (v) re-measure tail calls only if
engines gain signature-elided indirect calls.

## 6. Threats to validity

- **Shared cloud CPU.**  2 vCPUs of a multi-tenant Xeon; absolute MIPS vary
  between sessions (up to ~2.5× under load).  Mitigations: alternating
  repetitions, best-of-N, deltas quoted within single runs; deltas were
  stable across passes and engines.  Treat MIPS as indicative, deltas as
  the result.
- **Subset ISAs, synthetic kernels.**  48 PPC / ~35 M68K instructions, no
  FPU/exceptions/supervisor state; the M68K EA repertoire is the common
  subset, not all eleven modes.  dispatch-stress brackets the pessimal
  case, alu/mem the friendly ones; real workloads lie between.  Richer EA
  modes would *increase* the M68K predecode advantage (more runtime decode
  to delete), so the conclusion is conservative.
- **Predecode cost is charged but idealized**: whole-region redecode per
  run (<0.1% of a measurement) rather than production lazy per-page fill
  with write invalidation; the steady-state cost is what's measured either
  way.
- **Flag-elision verification** excludes CCR from the hash (§2.4); flag
  correctness is enforced indirectly through the flag-consuming branches in
  every kernel.
- **Budget in a register.**  All backends carry the sprint budget in a
  local; emulators that must abort sprints from memory slow paths would
  re-load it per handler.
- **Engine versions.**  One V8 lineage dominates the browser conclusions;
  the Firefox datapoint is compromised (§4.2); JavaScriptCore is unmeasured.
- **gcc-native backend B is not musttail** — it measures sibling-call
  codegen, not the guaranteed-tail-call regime.

## 7. Reproducing

```
make                        # native + node + web builds into build/
./build/ppc-bench-native [budget] [reps]
node build/ppc-bench.js  [budget] [reps]     # §4.1
./build/m68k-bench-native [budget] [reps]
node build/m68k-bench.js [budget] [reps]     # §4.4
make run-browsers                            # Chromium/Firefox (npm i, or PW_CORE=<path>)
node build/be-micro.js                       # §4.5
```

Compile-time knobs: `make BENCH_DEFS='-DBENCH_DEFAULT_BUDGET=30000000
-DBENCH_DEFAULT_REPS=3'` (used for the slow-engine Firefox page).
Requirements: gcc/clang, GNU make; Emscripten ≥ 4.0.x and Node ≥ 20 (wasm
tail calls) for wasm targets.  Every number in §4 regenerates from these
commands; the `results-*.log` files in the repository root are the quoted
runs.

## 8. Future work

- Superinstruction fusion at predecode time (cmp+bcc / cmpwi+bc, addq+dbra)
  — the format headroom exists in the id space; attacks the residual
  dispatch-stress gap.
- Production-shaped predecode: lazy per-page decode, write invalidation via
  SoA write-entry zeroing, physically-indexed entries under an MMU.
- Wider M68K EA coverage and a 68020+ variant (scaled index, memory
  indirect) to quantify how the predecode advantage grows with EA
  complexity.
- Engines: desktop Firefox and JavaScriptCore datapoints; periodic re-runs
  as `call_indirect`/`return_call` implementations evolve.
- SoA entry layout (separate id and operand arrays) to shrink the
  dispatch-critical working set on large code footprints.

## 9. Provenance and license

`common/mem_fastpath.h` is modelled line-for-line on granny-smith's
`src/core/memory/memory.h` (MIT, Copyright (c) pappadf); all other code is
original to this project.  SPDX headers mark all files MIT.
