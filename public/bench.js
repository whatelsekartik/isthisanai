const $ = (id) => document.getElementById(id);
const rowsEl = $("rows");
const badge = $("badge");
const atEl = $("measured-at");
const errEl = $("error");
const runEl = $("run");
const statusEl = $("status");
const statusText = $("status-text");

const ROWS = [
  ["model size", (m) => `${m.weights_mb} MB`],
  ["parameters", (m) => m.params ? m.params.toLocaleString() : "—"],
  ["cold start (read + parse)", (m) => `${m.cold_start_ms} ms`],
  ["prefill", (m) => `${m.prefill_ms} ms / ${m.prefill_tokens} tok`],
  ["decode latency", (m) => `${m.ms_per_token} ms/token`],
  ["throughput", (m) => `${m.tokens_per_second} tok/s`],
  ["session state", (m) => `${(m.state_bytes / 1024).toFixed(0)} KB`],
  ["RSS", (m) => `${m.rss_mb} MB`],
  ["heap used", (m) => `${m.heap_mb} MB`],
  ["decode tokens measured", (m) => `${m.decode_tokens}`],
  ["architecture", (m) => m.arch],
  ["checkpoint", (m) => (m.ckpt_step ? `step ${m.ckpt_step} · val ppl ${m.ckpt_val_ppl}` : "—")],
  ["runtime", (m) => `${m.node} · ${m.platform} · ${m.cpus} vCPU`],
];

function render(data) {
  // Never assume — the server says whether these numbers were measured now or kept in-process.
  const live = !data.cached;
  rowsEl.textContent = "";
  for (const [label, fn] of ROWS) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    const td = document.createElement("td");
    let value;
    try { value = fn(data); } catch { value = "—"; }
    td.textContent = value ?? "—";
    tr.append(th, td);
    rowsEl.append(tr);
  }
  badge.textContent = live ? "live" : "stored";
  badge.className = `badge ${live ? "live" : "stored"}`;
  atEl.textContent = data.measured_at
    ? `${live ? "measured just now" : "measured earlier, served from this process"} · ${new Date(data.measured_at).toLocaleTimeString()}`
    : "";
}

async function run({ fresh = false } = {}) {
  runEl.disabled = true;
  errEl.hidden = true;
  statusEl.dataset.state = "generating";
  statusText.textContent = fresh ? "measuring" : "loading";
  try {
    const query = fresh ? `?fresh=1&t=${Date.now()}` : `?t=${Date.now()}`;
    const res = await fetch(`/api/bench${query}`, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `failed (${res.status})`);
    render(data);
    statusEl.dataset.state = "ready";
    statusText.textContent = "ready";
  } catch (err) {
    // a failed run must not leave the previous run's numbers on screen looking current
    rowsEl.textContent = "";
    badge.textContent = "—";
    badge.className = "badge";
    atEl.textContent = "";
    errEl.hidden = false;
    errEl.textContent = String(err.message || err);
    statusEl.dataset.state = "error";
    statusText.textContent = "failed";
  } finally {
    runEl.disabled = false;
  }
}

runEl.addEventListener("click", () => run({ fresh: true }));
run();
