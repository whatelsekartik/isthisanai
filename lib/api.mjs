/**
 * Shared model handle. Loading happens once per function instance (module scope), so a
 * warm invocation pays nothing and a cold invocation pays only the file read + parse.
 */
import fs from "node:fs";
import path from "node:path";
import { loadModel } from "../runtime/src/model.mjs";

const CANDIDATES = [
  path.join(process.cwd(), "runtime", "model"),
  path.join(process.cwd(), "model"),
  new URL("../runtime/model", import.meta.url).pathname,
];

function findDir() {
  for (const c of CANDIDATES) {
    try {
      if (fs.existsSync(path.join(c, "model.bin"))) return c;
    } catch { /* ignore */ }
  }
  return null;
}

let cached = null;
let loadMs = null;

export function getModel() {
  if (cached) return { model: cached, loadMs, cold: false };
  const dir = findDir();
  if (!dir) throw new Error("model.bin not found in the deployment bundle");
  const t = performance.now();
  cached = loadModel(dir);
  loadMs = +(performance.now() - t).toFixed(2);
  return { model: cached, loadMs, cold: true };
}

export function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") return JSON.parse(req.body || "{}");
  return {};
}

export function json(res, code, obj) {
  res.statusCode = code;
  res.setHeader("content-type", "application/json; charset=utf-8");
  res.setHeader("cache-control", "no-store");
  res.end(JSON.stringify(obj));
}
