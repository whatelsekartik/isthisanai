#!/usr/bin/env node
/**
 * Local stand-in for `vercel dev`: serves `public/` and routes `/api/*` to the same
 * handler modules Vercel runs, so the numbers measured here come from the deployed code.
 *
 *   node dev.mjs [port]
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const port = Number(process.argv[2] || process.env.PORT || 3000);
const root = process.cwd();
const routes = new Map();
for (const f of fs.readdirSync(path.join(root, "api"))) {
  if (!f.endsWith(".mjs") || f.startsWith("_")) continue;
  const mod = await import(pathToFileURL(path.join(root, "api", f)).href);
  routes.set("/api/" + f.replace(/\.mjs$/, ""), mod.default);
}
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".css": "text/css", ".svg": "image/svg+xml", ".ico": "image/x-icon" };

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const key = url.pathname.replace(/\/$/, "");
  const handler = routes.get(key);
  if (handler) {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");
    req.body = raw ? (() => { try { return JSON.parse(raw); } catch { return {}; } })() : {};
    res.status = (code) => { res.statusCode = code; return res; };
    try {
      await handler(req, res);
    } catch (err) {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: String(err && err.stack || err) }));
    }
    return;
  }
  const rel = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
  const file = path.join(root, "public", rel);
  if (!file.startsWith(path.join(root, "public")) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.statusCode = 404;
    res.setHeader("content-type", "text/plain");
    res.end("not found");
    return;
  }
  res.setHeader("content-type", TYPES[path.extname(file)] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
});

server.listen(port, "0.0.0.0", () => {
  console.log(`dev server http://0.0.0.0:${port}  routes: ${[...routes.keys()].join(", ")}`);
});
