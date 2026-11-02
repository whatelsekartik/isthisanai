/**
 * A very small DOM stand-in — just the surface `public/app.js` actually touches.
 *
 * Deliberately hand-written: adding jsdom or a browser driver would be a dependency the
 * project does not need, and this keeps `npm test` runnable with nothing installed.
 */

class ClassList {
  constructor(el) { this.el = el; }
  add(...c) { const s = new Set(this.el.className.split(" ").filter(Boolean)); c.forEach((x) => s.add(x)); this.el.className = [...s].join(" "); }
  remove(...c) { this.el.className = this.el.className.split(" ").filter((x) => x && !c.includes(x)).join(" "); }
  contains(c) { return this.el.className.split(" ").includes(c); }
}

let uid = 0;

export class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.uid = ++uid;
    this.children = [];
    this.parent = null;
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    this.attrs = {};
    this.className = "";
    this.classList = new ClassList(this);
    this.hidden = false;
    this.id = "";
    this._text = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this.clientHeight = 100;
  }

  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }

  set textContent(v) {
    this._text = v === "" || v === null || v === undefined ? "" : String(v);
    for (const c of this.children) c.parent = null;
    this.children = [];
  }

  append(...nodes) {
    for (const n of nodes) {
      if (n === null || n === undefined) continue;
      if (typeof n === "string") { this._text += n; continue; }
      // like the real DOM: appending an element that already has a parent moves it
      if (n.parent) n.parent.children = n.parent.children.filter((c) => c !== n);
      n.parent = this;
      this.children.push(n);
    }
  }

  appendChild(n) { this.append(n); return n; }

  remove() {
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }

  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "id") this.id = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }

  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(fn);
  }

  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }

  dispatchEvent(event) {
    const e = {
      type: event.type,
      target: this,
      currentTarget: this,
      key: event.key,
      defaultPrevented: false,
      shiftKey: Boolean(event.shiftKey),
      isComposing: Boolean(event.isComposing),
      preventDefault() { e.defaultPrevented = true; },
      stopPropagation() { e.propagationStopped = true; },
      ...event,
    };
    for (const fn of this.listeners.get(e.type) || []) fn(e);
    // bubble to parents (app.js relies on document-level click to close the panel)
    if (!e.propagationStopped && this.parent) this.parent.dispatchEvent(e);
    return e;
  }

  click() { this.dispatchEvent({ type: "click" }); }

  focus() { this.ownerDocument && (this.ownerDocument.activeElement = this); }
  blur() {}

  querySelector(sel) {
    const want = sel.toUpperCase();
    const walk = (node) => {
      for (const c of node.children) {
        if (c.tagName === want || (want.startsWith(".") && c.classList.contains(want.slice(1)))) return c;
        const found = walk(c);
        if (found) return found;
      }
      return null;
    };
    return walk(this);
  }

  querySelectorAll(sel) {
    const want = sel.toUpperCase();
    const out = [];
    const walk = (node) => {
      for (const c of node.children) {
        if (c.tagName === want) out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }
}

export function createDom(ids, { hidden = [], disabled = [], storage = new Map() } = {}) {
  const body = new El("body");
  const byId = new Map();
  for (const id of ids) {
    const el = new El(id === "input" ? "textarea" : "div");
    el.id = id;
    el.hidden = hidden.includes(id);
    el.disabled = disabled.includes(id);
    byId.set(id, el);
    body.append(el);
  }

  const document = {
    body,
    activeElement: null,
    documentElement: new El("html"),
    listeners: new Map(),
    createElement: (tag) => {
      const el = new El(tag);
      el.ownerDocument = document;
      return el;
    },
    getElementById: (id) => {
      // elements the page created dynamically (sliders, messages) live in the tree, not in `byId`
      const walk = (node) => {
        for (const c of node.children) {
          if (c.id === id) return c;
          const found = walk(c);
          if (found) return found;
        }
        return null;
      };
      const found = walk(body);
      if (found) return found;
      if (byId.has(id)) return byId.get(id);
      const el = new El("div");
      el.id = id;
      el.ownerDocument = document;
      byId.set(id, el);
      body.append(el);
      return el;
    },
    addEventListener(type, fn) {
      if (!document.listeners.has(type)) document.listeners.set(type, new Set());
      document.listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) { document.listeners.get(type)?.delete(fn); },
    dispatchEvent(event) {
      const e = { type: event.type, target: body, preventDefault() {}, stopPropagation() {}, ...event };
      for (const fn of document.listeners.get(e.type) || []) fn(e);
      return e;
    },
    execCommand: () => true,
  };

  const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)),
    removeItem: (k) => storage.delete(k),
    clear: () => storage.clear(),
    get _map() { return storage; },
  };

  const window = {
    innerHeight: 800,
    addEventListener() {},
    removeEventListener() {},
  };

  return { document, localStorage, window, byId, body };
}

/** Wait for pending microtasks + timers used by the app (rAF is synchronous here). */
export const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
