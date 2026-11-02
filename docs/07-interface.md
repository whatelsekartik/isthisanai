# 07 — The interface

The engine was the project. This document covers the page that sits on top of it, because a UI
is where claims quietly stop being true: sliders that change nothing, numbers that were typed
in by hand, spinners that pretend to think. Everything below is a rule the code enforces and
the tests check.

## What the page is

One column of text, one composer, one settings panel. No hero, no navigation, no capability
list. The empty state says *"A dumb little chatbot."* — which is accurate: a 1.56 M-parameter
model that has read 2.76 M tokens of public-domain books writes plausible 19th-century prose and
nothing else. The page does not apologise for that and does not oversell it.

- `/` — the conversation
- `/bench` — measurements taken by this process (see below)

There is no third route and nothing is fetched from anywhere but this origin.

## Streaming, stopping, failing

`POST /api/generate` with `stream: true` returns `text/event-stream`:

```
{type:"start", params, meta}      params the server actually used, after clamping
{type:"token", text}              zero or more, already valid UTF-8
{type:"done",  stats}             tokens_generated, ms_per_token, tokens_per_second
{type:"error", error}             only if generation fails after the headers are sent
```

A token boundary can fall inside a multi-byte character, so the endpoint decodes through a
`TextDecoder` with `{stream:true}` fed by `Bpe.bytesOf(id)`. `tests/test_api.mjs` asserts that
the streamed text equals the non-streamed completion for the same request, so the two paths
cannot drift.

**Stop** aborts the fetch (`AbortController`). The server notices the closed socket, stops
decoding, and emits nothing further. The partial reply is kept and marked `stopped` — it is
what the model actually wrote, and it says so.

**Failure** shows `no reply — <reason>`. There is no fallback string, no canned apology, no
fabricated answer, and no typing animation anywhere: text appears exactly when tokens arrive.

## Settings, and why the sliders are honest

The panel is built at runtime from `GET /api/params`, which is generated from the same
`lib/params.mjs` the generation handler clamps with. If a control has no implementation behind
it, it cannot appear — the list is not maintained by hand in the client. (The client keeps a
built-in copy of the ranges only as an offline fallback, and `tests/test_params.mjs` checks the
two agree.)

| control | range | default | what it really does |
|---|---|---|---|
| Max tokens | 1–512 | 128 | loop bound in `generate()` |
| Temperature | 0–2 | 0.8 | logit scale; **0 means greedy argmax** |
| Top-p | 0.05–1 | 0.9 | nucleus cut over the renormalised candidates |
| Top-k | 1–100 | 40 | candidate set before nucleus; 1 = greedy |
| Repetition penalty | 1–2 | 1.0 | divides/multiplies logits of the last 64 ids |

Defaults reproduce the previous sampler exactly, so this is not a behaviour change — it is the
same distribution with the dials exposed. At `temperature: 0` the top-p and repetition-penalty
sliders are provably inert (greedy ignores the distribution); the tests assert that rather than
hiding it. Values are clamped and snapped to their step on the client *and* on the server, so a
hand-written POST cannot reach a range the engine does not support.

Settings persist in `localStorage` under `cadence.settings.v1`; a value that fails validation on
load falls back to the default instead of breaking the page.

## `/bench` — live or stored, never invented

The page renders only what `GET /api/bench` returned:

- `cached: false` → badge **live**, "measured just now"
- `cached: true` → badge **stored**, "measured earlier, served from this process"

Loading the page shows the process's measurement (the first caller in a cold function pays for
it; later callers see `cached: true`). **Run again** issues `?fresh=1`, which re-measures on the
spot. A failed run clears the table and the badge — stale numbers are never left looking current.
Nothing on that page is hardcoded; `tests/test_bench_ui.mjs` feeds it two different payloads and
checks the labels follow the server, and that a failure shows no numbers at all.

## Test coverage

The UI is tested without a browser. `tests/dom.mjs` is a ~200-line stand-in for the handful of DOM
APIs `public/app.js` uses, so the real client code runs under `node --test`:

```bash
npm run test:ui      # app.js + bench.js + the DOM-free conversation core, 34 tests
npm run test:js      # all 64 JS tests, including the HTTP handlers
```

`tests/test_app.mjs` drives the actual page: builds sliders from `/api/params`, types, presses
Enter, streams a reply, presses Stop mid-stream, moves a slider and checks the next request body,
resets, reloads with persisted settings, opens/closes the panel with keyboard and mouse, and
forces a failing request to confirm the page does not invent an answer.

## Deliberate omissions

No markdown rendering (the model was trained on plain text; markdown would be decoration on top
of tokens that were not written in it). No message editing, branching, or history sidebar. No
avatar, no gradient, no animation beyond a 140 ms fade and a caret that exists only while tokens
are arriving. One stylesheet, no fonts downloaded, no third-party scripts, no analytics.

Accessibility: the log is `role="log"` and goes `aria-live="off"` while streaming so screen
readers are not spammed per token; the panel is a labelled `dialog` with `aria-expanded` on its
button, Escape to close, and focus returned to the button; every control is a native element;
`prefers-reduced-motion` disables the caret blink and the fade.
