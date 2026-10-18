// SPDX-License-Identifier: MIT
// run-browser.mjs — serve tools/jit-lab and run compile-browser.html in
// headless Chromium (playwright-core: PW_CORE=<path to its index.mjs>).
// Put the dumped modules in tools/jit-lab/dump/ first (JIT_DUMP=...).
//   cd tools/jit-lab && PW_CORE=... node run-browser.mjs
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
const pw = await import(process.env.PW_CORE);
const server = createServer(async (req, res) => { try { const p = '.' + req.url.split('?')[0]; const b = await readFile(p);
  res.writeHead(200, { 'content-type': p.endsWith('.html') ? 'text/html' : p.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream' }); res.end(b); } catch { res.writeHead(404); res.end(); } });
await new Promise(r => server.listen(0, '127.0.0.1', r));
const b = await pw.chromium.launch({ headless: true }); const page = await b.newPage();
const done = new Promise(r => page.on('console', m => { const t = m.text(); if (t.startsWith('LAZY')) { console.log(t); if (t === 'LAZY done') r(); } }));
await page.goto(`http://127.0.0.1:${server.address().port}/compile-browser.html`); await done; await b.close(); server.close();
