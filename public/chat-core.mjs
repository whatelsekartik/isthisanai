/**
 * The conversation, with no DOM in it.
 *
 * `app.js` renders `subscribe()` snapshots; `node --test` drives the same object with a fake
 * transport. Everything the UI can do to a conversation — send, stop, retry, fail, retry
 * after failure — lives here, so those behaviours are testable without a browser.
 */
import { streamGenerate } from "./sse.mjs";

let seq = 0;
const nextId = () => `m${++seq}`;

export function createConversation({ transport = streamGenerate, now = () => Date.now() } = {}) {
  const state = {
    messages: [],
    busy: false,
    error: null,
    params: {},
    lastStats: null,
  };
  const listeners = new Set();
  let controller = null;

  const emit = () => {
    const snapshot = {
      messages: state.messages.map((m) => ({ ...m })),
      busy: state.busy,
      error: state.error,
      params: { ...state.params },
      lastStats: state.lastStats,
    };
    for (const fn of listeners) fn(snapshot);
    return snapshot;
  };

  const find = (id) => state.messages.find((m) => m.id === id);

  function history(beforeIndex) {
    return state.messages
      .slice(0, beforeIndex)
      .filter((m) => m.role !== "assistant" || m.content.trim())
      .map((m) => ({ role: m.role, content: m.content }));
  }

  function startAssistant(turn) {
    const msg = {
      id: nextId(),
      role: "assistant",
      content: "",
      status: "streaming",
      stats: null,
      error: null,
      startedAt: now(),
    };
    state.messages.push(msg);
    turn.target = msg;
    return msg;
  }

  async function run(turn) {
    state.busy = true;
    state.error = null;
    controller = new AbortController();
    const msg = turn.target;
    emit();

    try {
      await transport({
        signal: controller.signal,
        body: { messages: turn.messages, ...state.params },
        onEvent(event) {
          if (event.type === "token") {
            msg.content += event.text;
            emit();
          } else if (event.type === "done") {
            msg.stats = {
              tokens: event.tokens_generated,
              ms_per_token: event.ms_per_token,
              tokens_per_second: event.tokens_per_second,
            };
            state.lastStats = msg.stats;
          } else if (event.type === "error") {
            throw new Error(event.error || "generation failed");
          }
        },
      });
      msg.status = msg.content ? "done" : "empty";
      if (!msg.content) msg.error = "the model returned nothing";
    } catch (err) {
      const aborted = err && (err.name === "AbortError" || controller.signal.aborted);
      msg.status = aborted ? "stopped" : "error";
      if (!aborted) msg.error = String((err && err.message) || err);
      if (!msg.content) msg.error = msg.error || "generation stopped";
    } finally {
      msg.endedAt = now();
      state.busy = false;
      controller = null;
      emit();
    }
    return msg;
  }

  return {
    subscribe(fn) {
      listeners.add(fn);
      fn({
        messages: state.messages.map((m) => ({ ...m })),
        busy: state.busy,
        error: state.error,
        params: { ...state.params },
        lastStats: state.lastStats,
      });
      return () => listeners.delete(fn);
    },
    setParams(params) {
      state.params = { ...params };
      emit();
    },
    getParams() {
      return { ...state.params };
    },
    send(text) {
      const content = String(text || "").trim();
      if (!content || state.busy) return null;
      state.messages.push({ id: nextId(), role: "user", content, status: "done", error: null });
      const turn = { messages: history(state.messages.length), target: null };
      startAssistant(turn);
      run(turn);
      return turn.target.id;
    },
    /** Regenerate the reply with `id` (assistant message), discarding it and anything after. */
    retry(id) {
      if (state.busy) return null;
      const index = state.messages.findIndex((m) => m.id === id);
      if (index < 0) return null;
      state.messages.splice(index);
      const turn = { messages: history(index), target: null };
      startAssistant(turn);
      run(turn);
      return turn.target.id;
    },
    stop() {
      if (!state.busy || !controller) return false;
      controller.abort();
      return true;
    },
    clear() {
      if (state.busy) return false;
      state.messages = [];
      state.error = null;
      state.lastStats = null;
      emit();
      return true;
    },
    get busy() {
      return state.busy;
    },
    find,
  };
}
