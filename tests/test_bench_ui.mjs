/**
 * The /bench page: it must show what the server measured, say whether it is live or held
 * over from an earlier request, and never invent a number.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { createDom, settle } from "./dom.mjs";

const IDS = ["status", "status-text", "badge", "measured-at", "error", "run", "rows"];

const MEASURED = {
  weights_mb: 1.63,
  params: 1561170,
  cold_start_ms: 2.4,
  prefill_ms: 1.1,
  prefill_tokens: 12,
  ms_per_token: 5.1,
  tokens_per_second: 196,
  state_bytes: 221184,
  rss_mb: 71.2,
  heap_mb: 9.4,
  decode_tokens: 64,
  arch: "cadence",
  ckpt_step: 900,
  ckpt_val_ppl: 69.54,
  node: "v22.0.0",
  platform: "linux",
  cpus: 2,
  measured_at: 1762000000000,
  cached: false,
};

async function boot({ responses = [] } = {}) {
  const dom = createDom(IDS);
  globalThis.document = dom.document;
  globalThis.window = dom.window;
  const urls = [];
  let i = 0;
  globalThis.fetch = async (url) => {
    urls.push(url);
    const next = responses[Math.min(i++, responses.length - 1)];
    if (next instanceof Error) throw next;
    return { ok: next.status === undefined || next.status < 400, status: next.status || 200, json: async () => next.body };
  };
  await import(`../public/bench.js?t=${Math.random()}`);
  await settle(10);
  const byId = { get: (id) => dom.document.getElementById(id) };
  return { byId, urls };
}

const cells = (byId) =>
  Object.fromEntries(byId.get("rows").children.map((tr) => [tr.children[0].textContent, tr.children[1].textContent]));

test("a first measurement is labelled live and shows the server's numbers", async () => {
  const { byId, urls } = await boot({ responses: [{ body: MEASURED }] });
  assert.match(urls[0], /^\/api\/bench\?t=\d+$/, "the first load does not force a re-measure");
  assert.equal(byId.get("badge").textContent, "live");
  const table = cells(byId);
  assert.equal(table["model size"], "1.63 MB");
  assert.equal(table["parameters"], (1561170).toLocaleString());
  assert.equal(table["decode latency"], "5.1 ms/token");
  assert.equal(table["throughput"], "196 tok/s");
  assert.equal(table["session state"], "216 KB");
  assert.equal(table["architecture"], "cadence");
  assert.equal(table["RSS"], "71.2 MB");
  assert.match(byId.get("measured-at").textContent, /measured just now/);
  assert.equal(byId.get("error").hidden, true);
});

test("a cached measurement is labelled stored, not live", async () => {
  const { byId } = await boot({ responses: [{ body: { ...MEASURED, cached: true } }] });
  assert.equal(byId.get("badge").textContent, "stored");
  assert.match(byId.get("measured-at").textContent, /measured earlier/);
  // the numbers are still the real ones that were measured
  assert.equal(cells(byId)["decode latency"], "5.1 ms/token");
});

test("Run again asks for a fresh measurement", async () => {
  const { byId, urls } = await boot({
    responses: [{ body: { ...MEASURED, cached: true } }, { body: { ...MEASURED, cached: false, ms_per_token: 4.8 } }],
  });
  assert.equal(byId.get("badge").textContent, "stored");
  byId.get("run").click();
  await settle(10);
  assert.match(urls[1], /^\/api\/bench\?fresh=1&t=\d+$/);
  assert.equal(byId.get("badge").textContent, "live");
  assert.equal(cells(byId)["decode latency"], "4.8 ms/token");
  assert.equal(byId.get("run").disabled, false, "the button is usable again");
});

test("a failure is reported and no numbers are shown", async () => {
  const { byId } = await boot({ responses: [{ status: 500, body: { error: "checkpoint missing" } }] });
  assert.equal(byId.get("error").hidden, false);
  assert.match(byId.get("error").textContent, /checkpoint missing/);
  assert.equal(byId.get("rows").children.length, 0);
  assert.equal(byId.get("badge").textContent, "—");
  assert.equal(byId.get("status-text").textContent, "failed");
});

test("a network error does not leave stale numbers behind", async () => {
  const { byId } = await boot({ responses: [{ body: MEASURED }, new Error("offline")] });
  byId.get("run").click();
  await settle(10);
  assert.equal(byId.get("error").hidden, false);
  assert.match(byId.get("error").textContent, /offline/);
  assert.notEqual(byId.get("badge").textContent, "live");
});
