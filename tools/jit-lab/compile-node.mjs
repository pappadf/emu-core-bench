// SPDX-License-Identifier: MIT
// compile-node.mjs — what a JIT-compiled page costs the thread that calls it.
//
//   JIT_DUMP=<dir> node build/ppc-bench.js 20000000 1   # dump the modules
//   node [v8 flags] tools/jit-lab/compile-node.mjs <dir> <mode>
//
// mode: sync (new WebAssembly.Module), async (WebAssembly.compile),
// worker (compiled on a worker, posted back), worker-warm (compiled on a
// worker AND called once there before posting).  Per page: the calling
// thread's time to get the module, to instantiate it, its first call and its
// second call (budget 0: the first block's guard leaves at once, so a call
// times only the call and any compilation V8 deferred to it).
import { readFileSync, readdirSync } from 'node:fs';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';

const mem = new WebAssembly.Memory({ initial: 1024 });
const table = new WebAssembly.Table({ initial: 4, element: 'anyfunc' });
const imports = { e: { m: mem, t: table } };

if (!isMainThread) {
  // Worker: compile, instantiate against its own memory, call once (budget 0),
  // then post the module back.
  const mod = await WebAssembly.compile(workerData.bytes);
  const m2 = new WebAssembly.Memory({ initial: 1024 });
  const t2 = new WebAssembly.Table({ initial: 4, element: 'anyfunc' });
  const inst = new WebAssembly.Instance(mod, { e: { m: m2, t: t2 } });
  if (workerData.warm) inst.exports.f(0, 0, 0);
  parentPort.postMessage(mod);
} else {
  const dir = process.argv[2], mode = process.argv[3];
  const files = readdirSync(dir).filter((f) => f.endsWith('.wasm')).sort();
  const tot = { get: 0, inst: 0, first: 0, second: 0 };
  for (const f of files) {
    const bytes = readFileSync(dir + '/' + f);
    let mod;
    const t0 = performance.now();
    let blocked; // calling-thread time to obtain the module
    if (mode === 'sync') {
      mod = new WebAssembly.Module(bytes);
      blocked = performance.now() - t0;
    } else if (mode === 'async') {
      const p = WebAssembly.compile(bytes);
      blocked = performance.now() - t0; // time until compile() returned
      mod = await p;
    } else {
      const w = new Worker(fileURLToPath(import.meta.url), { workerData: { bytes, warm: mode === 'worker-warm' } });
      mod = await new Promise((r) => w.once('message', r));
      await w.terminate();
      blocked = 0; // the main thread did nothing but wait on its event loop
    }
    const t1 = performance.now();
    const inst = new WebAssembly.Instance(mod, imports);
    const t2 = performance.now();
    inst.exports.f(0, 0, 0);
    const t3 = performance.now();
    inst.exports.f(0, 0, 0);
    const t4 = performance.now();
    tot.get += blocked; tot.inst += t2 - t1; tot.first += t3 - t2; tot.second += t4 - t3;
  }
  const n = files.length;
  console.log(`${mode.padEnd(12)} ${process.execArgv.join(' ').padEnd(30)} per page: on-thread compile ${(tot.get/n).toFixed(3)} ms, Instance ${(tot.inst/n).toFixed(3)} ms, first call ${(tot.first/n).toFixed(3)} ms, second call ${(tot.second/n).toFixed(4)} ms`);
}
