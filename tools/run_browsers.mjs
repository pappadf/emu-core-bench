// Drive the wasm benchmark inside real browser engines (Chromium/V8 and
// Firefox/SpiderMonkey) via playwright-core, capturing the console output.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";

// Resolve playwright-core from node_modules, or point PW_CORE at an
// existing installation (e.g. another repo's node_modules copy)
const pw = await import(process.env.PW_CORE ?? "playwright-core");

const mime = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };
const server = createServer(async (req, res) => {
    const path = "." + (req.url === "/" ? "/" + (process.env.BENCH_PAGE ?? "bench_web.html") : req.url.split("?")[0]);
    try {
        const body = await readFile(path);
        res.writeHead(200, { "content-type": mime[extname(path)] ?? "application/octet-stream" });
        res.end(body);
    } catch {
        res.writeHead(404);
        res.end();
    }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

for (const name of process.argv.slice(2).length ? process.argv.slice(2) : ["chromium", "firefox"]) {
    const browser = await pw[name].launch({
        headless: true,
        firefoxUserPrefs: {
            "javascript.options.wasm_optimizingjit": true,
            "javascript.options.wasm_baselinejit": true,
        },
    });
    const page = await browser.newPage();
    const lines = [];
    let done;
    const finished = new Promise((r) => (done = r));
    page.on("console", (msg) => {
        const t = msg.text();
        if (/^(ppc dispatch|m68k format|alu-loop|mem-stream|dispatch-stress|benchmark done|FAIL|FATAL)/.test(t)) {
            if (!t.startsWith("benchmark done")) lines.push(t);
            if (t.startsWith("benchmark done") || t.startsWith("FAIL") || t.startsWith("FATAL")) done();
        }
    });
    await page.goto(`http://127.0.0.1:${port}/${process.env.BENCH_PAGE ?? "bench_web.html"}`);
    await Promise.race([finished, new Promise((r) => setTimeout(r, 300000))]);
    console.log(`=== ${name} (${browser.version()}) ===`);
    for (const l of lines) console.log(l);
    await browser.close();
}
server.close();
