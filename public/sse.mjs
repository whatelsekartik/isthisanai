/**
 * Server-sent-events reading, done by hand so the same code runs in the browser (fetch +
 * ReadableStream) and under `node --test` with a fake response object.
 */

/** Split a growing text buffer into complete SSE `data:` payloads. */
export function createEventParser(onEvent) {
  let buffer = "";
  return {
    push(chunk) {
      buffer += chunk;
      for (;;) {
        const sep = buffer.indexOf("\n\n");
        if (sep === -1) break;
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          let obj;
          try {
            obj = JSON.parse(payload);
          } catch {
            onEvent({ type: "error", error: "malformed stream frame" });
            continue;
          }
          onEvent(obj);
        }
      }
    },
    flush() {
      if (buffer.trim()) {
        const rest = buffer;
        buffer = "";
        for (const line of rest.split("\n")) {
          if (!line.startsWith("data:")) continue;
          try {
            onEvent(JSON.parse(line.slice(5).trim()));
          } catch { /* ignore a torn final frame */ }
        }
      }
    },
  };
}

/**
 * POST a streaming generation request and feed each event to `onEvent`.
 * Returns whatever the `done` event carried (or null).
 */
export async function streamGenerate({ body, signal, onEvent, fetchImpl = fetch, url = "/api/generate" }) {
  const res = await fetchImpl(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });

  if (!res.ok) {
    let message = `request failed (${res.status})`;
    try {
      const err = await res.json();
      if (err && err.error) message = err.error;
    } catch { /* keep the status-code message */ }
    throw new Error(message);
  }
  if (!res.body) throw new Error("streaming is not supported by this browser");

  const parser = createEventParser(onEvent);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.push(dec.decode(value, { stream: true }));
  }
  parser.flush();
  return true;
}
