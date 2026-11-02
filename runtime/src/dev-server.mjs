/**
 * Local stand-in for `vercel dev`: serves public/ and routes /api/* to the same
 * handler modules that Vercel runs, so benchmarks measured here use the same code.
 *
 *   node runtime/src/dev-server.mjs [port]
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const port = Number(process.argv[2] || process.env.PORT || 3000);

const routes = new Map();
for (const f of fs.readdirSync(path.join(root, "api"))) {
  if (!f.endsWith(".mjs")) continue;
  const mod = await import(pathToFileURL(path.join(root, "api", f)).href);
  routes.set("/api/" + f.replace(/\.mjs$/, ""), mod.default);
}

const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml",
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const handler = routes.get(url.pathname.replace(/\/$/, ""));
  if (handler) {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");
    let body = {};
    try { body = raw ? JSON.parse(raw) : {}; } catch { body = raw; }
    const shim = {
      status(code) { res.statusCode = code; return shim; },
      json(obj) {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(obj));
      },
      setHeader(k, v) { res.setHeader(k, v); },
    };
    try {
      await handler({ method: req.method, body, query: Object.fromEntries(url.searchParams) }, shim);
    } catch (err) {
      res.statusCode = 500;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ error: String(err && err.stack || err) }));
    }
    return;
  }
  let p = path.join(root, "public", url.pathname === "/" ? "index.html" : url.pathname);
  if (!p.startsWith(path.join(root, "public"))) { res.statusCode = 403; res.end("no"); return; }
  if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    res.statusCode = 404;
    res.setHeader("content-type", "text/plain");
    res.end("not found");
    return;
  }
  res.setHeader("content-type", types[path.extname(p)] || "application/octet-stream");
  fs.createReadStream(p).pipe(res);
});

server.listen(port, "0.0.0.0", () => {
  console.log(`cadence dev server on http://0.0.0.0:${port} (routes: ${[...routes.keys()].join(", ")})`);
});
