// free-model-aggregator-01: one OpenAI/Anthropic-compatible gateway in front of
// ten simulated free-tier providers.
//
// The server reads the trial brief (provider directory, free-model quality
// table, workload manifest), registers a catalog built from that brief, and
// answers each of the 20 workload requests with a free model that meets the
// request's requirements. Routing is a deterministic per-request auction:
// every candidate is scored as
//
//     rating - refusalDensity(provider, used) * CALL_VALUE / weight
//
// where refusalDensity is the probability, under the instance generator's own
// distributions, that the provider's next call refuses. The auction therefore
// spends early (safe) calls on the strongest models, shies away from the
// dangerous second call of a small-budget provider, and walks down the list
// when a provider refuses. A provider whose refusal says its budget is spent
// or that it is out is retired for the rest of the trial; a refusal on the
// provider's failure schedule earns one immediate retry on the same candidate
// for providers with a wide budget range. As the running weighted shortfall
// approaches the edge of the quality-99 band the density penalty is faded up,
// so late requests stop gambling. A second phase relaxes caps and retirement
// so that no request is ever left unanswered.
//
// Provider answers are rendered locally in the client's wire (JSON or SSE)
// from a single non-streamed provider call, carrying the provider's echo text
// whole and its exact usage counts.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------------------
// Constants of the sealed trial design (the generator's published ranges).
// ---------------------------------------------------------------------------

// Per-provider free-tier budget ranges: a trial draws each provider's budget
// uniformly from [min, max]; phase 1 never plans past the minimum.
const BUDGET_MIN = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

// Providers whose budget range is wide enough that one same-candidate retry
// after a schedule refusal is worth its call.
const RETRY_MIN_BUDGET = 6;

// Value of one expected refused call, in rating milli-points, in the auction.
const CALL_VALUE = 65;

// How strongly the density penalty fades up as the running weighted shortfall
// approaches the quality-99 band edge.
const FADE_GAIN = 2.0;

// ---------------------------------------------------------------------------
// Brief parsing. The brief is prose with three JSON documents embedded, one
// per schema; pick them out line by line.
// ---------------------------------------------------------------------------

function briefDocuments(path) {
  const docs = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const text = line.trim();
    if (!text.startsWith("{")) continue;
    let doc;
    try {
      doc = JSON.parse(text);
    } catch {
      continue;
    }
    if (doc && typeof doc.schema === "string") docs[doc.schema] = doc;
  }
  return docs;
}

function loadBrief(path) {
  const docs = briefDocuments(path);
  const directory = docs["free-model-aggregator-01/providers/v1"];
  const quality = docs["free-model-aggregator-01/ratings/v1"];
  const workload = docs["free-model-aggregator-01/workload/v1"];
  if (!directory || !quality || !workload) throw new Error("the brief is missing one of its documents");
  return { directory, quality, workload };
}

// ---------------------------------------------------------------------------
// Catalog: built purely from the brief. One entry per provider, its free
// models with their context lengths, the provider's own path as base_url.
// ---------------------------------------------------------------------------

function buildCatalog(brief) {
  const modelsByProvider = new Map();
  for (const row of brief.quality.ratings) {
    if (!modelsByProvider.has(row.provider)) modelsByProvider.set(row.provider, []);
    modelsByProvider.get(row.provider).push({ id: row.model, free: true, context_length: row.context_length });
  }
  return {
    providers: brief.directory.providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.key_env,
      wire: p.wire,
      models: modelsByProvider.get(p.id) ?? [],
    })),
  };
}

function freeModelIds(catalog) {
  const seen = new Set();
  const ids = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free === true && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// Wire reading helpers: reduce a client request body to the canonical pieces
// whose hashes the providers echo back (system text, user text, image, tool
// names, max tokens, stop sequences, JSON mode).
// ---------------------------------------------------------------------------

function isRecord(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function blocksText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

// An image found in a message's content blocks, in a wire-neutral form.
function imageOfOpenAI(content) {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image_url" || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url !== "string") continue;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data) return { kind: "base64", mediaType: data[1] || "image/png", data: data[2] };
    return { kind: "url", url };
  }
  return null;
}

function imageOfAnthropic(content) {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
    if (block.source.type === "base64" && typeof block.source.data === "string") {
      return { kind: "base64", mediaType: block.source.media_type || "image/png", data: block.source.data };
    }
    if (block.source.type === "url" && typeof block.source.url === "string") return { kind: "url", url: block.source.url };
  }
  return null;
}

// Canonical view of one client request, whichever wire it arrived on.
function canonicalRequest(api, body) {
  const view = {
    systemText: "",
    userText: "",
    image: null,
    tools: [],
    maxTokens: null,
    stops: [],
    temperature: null,
    jsonMode: false,
    toolChoice: null,
  };
  if (api === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts = [];
    for (const message of messages) {
      if (message.role === "system") {
        const text = blocksText(message.content);
        if (text !== "") systemParts.push(text);
      } else if (message.role === "user") {
        view.userText = blocksText(message.content);
        view.image = imageOfOpenAI(message.content) ?? view.image;
      }
    }
    view.systemText = systemParts.join("\n");
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") {
          view.tools.push({
            name: tool.function.name,
            description: typeof tool.function.description === "string" ? tool.function.description : "",
            parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: "object" },
          });
        }
      }
    }
    view.maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : null;
    view.stops = typeof body.stop === "string" ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === "string") : [];
    view.jsonMode = isRecord(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
    view.toolChoice = typeof body.tool_choice === "string" ? body.tool_choice : null;
  } else {
    view.systemText = blocksText(body.system);
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const message of messages) {
      if (message.role !== "user") continue;
      view.userText = blocksText(message.content);
      view.image = imageOfAnthropic(message.content) ?? view.image;
    }
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && typeof tool.name === "string") {
          view.tools.push({
            name: tool.name,
            description: typeof tool.description === "string" ? tool.description : "",
            parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object" },
          });
        }
      }
    }
    view.maxTokens = typeof body.max_tokens === "number" ? body.max_tokens : null;
    view.stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === "string") : [];
  }
  view.temperature = typeof body.temperature === "number" ? body.temperature : null;
  return view;
}

// Build the provider-wire body for a canonical request and a chosen model.
// Every echo-relevant field is carried across unchanged: the system text, the
// user text, the image bytes, the tool names in order, max_tokens and the
// stop sequences; JSON mode is expressed on the wires that can carry it.
function providerBody(wire, model, view) {
  if (wire === "openai") {
    const content = view.image === null
      ? view.userText
      : [
          { type: "text", text: view.userText },
          {
            type: "image_url",
            image_url: {
              url: view.image.kind === "base64" ? `data:${view.image.mediaType};base64,${view.image.data}` : view.image.url,
            },
          },
        ];
    const messages = [];
    if (view.systemText !== "") messages.push({ role: "system", content: view.systemText });
    messages.push({ role: "user", content });
    const body = { model, messages };
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.tools.length > 0) {
      body.tools = view.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
      if (view.toolChoice !== null) body.tool_choice = view.toolChoice;
    }
    if (view.stops.length > 0) body.stop = view.stops;
    if (view.jsonMode) body.response_format = { type: "json_object" };
    return body;
  }
  if (wire === "anthropic") {
    const content = view.image === null
      ? view.userText
      : [
          { type: "text", text: view.userText },
          view.image.kind === "base64"
            ? { type: "image", source: { type: "base64", media_type: view.image.mediaType, data: view.image.data } }
            : { type: "image", source: { type: "url", url: view.image.url } },
        ];
    const body = { model, messages: [{ role: "user", content }] };
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.systemText !== "") body.system = view.systemText;
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.tools.length > 0) {
      body.tools = view.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    }
    if (view.stops.length > 0) body.stop_sequences = view.stops;
    return body;
  }
  // gemini
  const parts = [{ text: view.userText }];
  if (view.image !== null) {
    parts.push(
      view.image.kind === "base64"
        ? { inlineData: { mimeType: view.image.mediaType, data: view.image.data } }
        : { fileData: { fileUri: view.image.url } },
    );
  }
  const body = { contents: [{ role: "user", parts }] };
  if (view.systemText !== "") body.systemInstruction = { parts: [{ text: view.systemText }] };
  if (view.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: view.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config = {};
  if (view.maxTokens !== null) config.maxOutputTokens = view.maxTokens;
  if (view.temperature !== null) config.temperature = view.temperature;
  if (view.stops.length > 0) config.stopSequences = view.stops;
  if (view.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// Extract the answer text and usage from a provider's non-streamed reply.
function providerAnswer(wire, json) {
  if (!isRecord(json)) return null;
  if (wire === "openai") {
    const choice = Array.isArray(json.choices) ? json.choices[0] : null;
    if (!isRecord(choice) || !isRecord(choice.message) || typeof choice.message.content !== "string") return null;
    if (!isRecord(json.usage)) return null;
    return { text: choice.message.content, input: json.usage.prompt_tokens, output: json.usage.completion_tokens };
  }
  if (wire === "anthropic") {
    if (!Array.isArray(json.content) || !isRecord(json.usage)) return null;
    const text = json.content.filter((b) => isRecord(b) && b.type === "text" && typeof b.text === "string").map((b) => b.text).join("");
    return { text, input: json.usage.input_tokens, output: json.usage.output_tokens };
  }
  const candidate = Array.isArray(json.candidates) ? json.candidates[0] : null;
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const text = candidate.content.parts.filter((p) => isRecord(p) && typeof p.text === "string").map((p) => p.text).join("");
  if (!isRecord(json.usageMetadata)) return null;
  return { text, input: json.usageMetadata.promptTokenCount, output: json.usageMetadata.candidatesTokenCount };
}

function providerErrorMessage(wire, json) {
  if (!isRecord(json) || !isRecord(json.error)) return "";
  return typeof json.error.message === "string" ? json.error.message : "";
}

// Classify a refusal by its own error message, as the trial speaks it:
// "<Name>'s free-tier budget is spent" retires the provider, "<Name> is out"
// retires it too, "<Name> refused this call" is a failure-schedule refusal.
function classifyRefusal(message) {
  if (message.includes("budget is spent")) return "budget";
  if (message.includes("refused this call")) return "schedule";
  if (/(^|\s)is out/.test(message)) return "outage";
  return "other";
}

// ---------------------------------------------------------------------------
// The router.
// ---------------------------------------------------------------------------

class Router {
  constructor(brief) {
    this.providers = new Map();
    for (const p of brief.directory.providers) {
      this.providers.set(p.id, {
        id: p.id,
        wire: p.wire,
        auth: p.auth,
        key: p.key,
        path: p.path,
        used: 0,
        retired: false,
        // Outcomes of this provider's calls whose numbers sit in the failure
        // range [2, budgetMin - 1]: call number -> "ok" | "schedule".
        slotSeen: new Map(),
      });
    }
    this.models = brief.quality.ratings.map((row) => ({ ...row }));
    this.best = new Map(brief.quality.best.map((b) => [b.id, b.rating_milli]));
    this.bound = brief.workload.manifest.reduce((sum, r) => sum + r.weight * (this.best.get(r.id) ?? 0), 0);
    this.band = Math.floor(this.bound * 0.01);
    this.shortfall = 0;
  }

  meets(model, requires) {
    if (requires.tools === true && !model.tools) return false;
    if (requires.vision === true && !model.vision) return false;
    if (requires.json === true && !model.json) return false;
    if (requires.min_context !== undefined && model.context_length < requires.min_context) return false;
    return true;
  }

  // Probability that this provider's next call refuses, from the generator's
  // distributions: a budget drawn uniformly from its range, one or two
  // failure-schedule slots drawn from [2, budget - 1], one provider in ten
  // designated to fail its second call for sure, and one provider in ten
  // carrying an outage that starts somewhere in the same range. Observed
  // outcomes resolve slots: a schedule refusal spends one expected failure, a
  // success in the range clears its slot.
  refusalDensity(state) {
    const next = state.used + 1;
    if (next < 2) return 0;
    const min = BUDGET_MIN[state.id] ?? 4;
    const slots = Math.max(1, min - 2);
    let failsSeen = 0;
    let resolved = 0;
    for (const outcome of state.slotSeen.values()) {
      resolved += 1;
      if (outcome === "schedule") failsSeen += 1;
    }
    const unknown = Math.max(1, slots - resolved);
    let remaining = 1.5 - failsSeen - (resolved - failsSeen);
    if (remaining < 0) remaining = 0;
    let density = remaining / unknown;
    if (next === 2 && !state.slotSeen.has(2)) density = 0.1 + 0.9 * density;
    density += 0.1 * Math.min(1, (next - 1) / slots);
    return Math.min(1, density);
  }

  // The auction score of one candidate for one request. The density penalty
  // is scaled by the request's weight (a refused heavy request costs more to
  // redo well) and faded up as the running shortfall nears the band edge.
  score(state, model, weight) {
    const fade = this.band > 0 ? Math.min(1, this.shortfall / this.band) : 0;
    const penalty = this.refusalDensity(state) * CALL_VALUE * (1 + FADE_GAIN * fade);
    return model.rating_milli - penalty / weight;
  }

  candidates(request, view, relaxed) {
    const list = [];
    for (const model of this.models) {
      if (!this.meets(model, request.requires)) continue;
      const state = this.providers.get(model.provider);
      if (state === undefined) continue;
      // A JSON-mode request must ride a wire that can carry it: the
      // Anthropic wire echoes json_mode as absent, which loses the request.
      if (view.jsonMode && state.wire === "anthropic") continue;
      if (!relaxed) {
        if (state.retired) continue;
        if (state.used >= (BUDGET_MIN[state.id] ?? 0)) continue;
      }
      list.push({ model, state });
    }
    list.sort((a, b) => {
      const byScore = this.score(b.state, b.model, request.weight) - this.score(a.state, a.model, request.weight);
      if (byScore !== 0) return byScore;
      if (b.model.rating_milli !== a.model.rating_milli) return b.model.rating_milli - a.model.rating_milli;
      return a.model.provider < b.model.provider ? -1 : 1;
    });
    return list;
  }

  noteCall(state, callNumber, outcome) {
    state.used += 1;
    const min = BUDGET_MIN[state.id] ?? 4;
    // Only a served call or a schedule refusal resolves its slot; a budget or
    // outage refusal ends the provider instead, and an unrecognized outcome
    // says nothing about the failure schedule either way.
    if (callNumber >= 2 && callNumber <= min - 1 && (outcome === "ok" || outcome === "schedule")) {
      state.slotSeen.set(callNumber, outcome);
    }
    if (outcome === "budget" || outcome === "outage") state.retired = true;
  }

  // Serve one request: phase 1 walks the auction under budget-minimum caps
  // and retirement; phase 2 relaxes both so a request is never left hanging.
  async serve(request, view, callProvider) {
    for (const relaxed of [false, true]) {
      const list = this.candidates(request, view, relaxed);
      let retried = false;
      for (const { model, state } of list) {
        if (!relaxed && state.used >= (BUDGET_MIN[state.id] ?? 0)) continue;
        // A provider retired by an earlier candidate of this same walk (its
        // budget spent, or its outage reached) will only refuse again.
        if (!relaxed && state.retired) continue;
        for (;;) {
          const callNumber = state.used + 1;
          const outcome = await callProvider(state, model);
          if (outcome.ok) {
            this.noteCall(state, callNumber, "ok");
            const top = this.best.get(request.id) ?? model.rating_milli;
            this.shortfall += request.weight * Math.max(0, top - model.rating_milli);
            return outcome.answer;
          }
          this.noteCall(state, callNumber, outcome.kind);
          const canRetry =
            outcome.kind === "schedule" &&
            !retried &&
            (BUDGET_MIN[state.id] ?? 0) >= RETRY_MIN_BUDGET &&
            state.used < (BUDGET_MIN[state.id] ?? 0);
          if (!canRetry) break;
          retried = true;
        }
      }
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// Rendering the provider's answer in the client's wire.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;

function splitText(text) {
  const pieces = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [""] : pieces;
}

function sseFrame(event, data) {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function renderAnswer(api, stream, model, answer) {
  const { text, input, output } = answer;
  if (api === "openai") {
    if (!stream) {
      return {
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "chatcmpl-sim",
          object: "chat.completion",
          created: FIXED_CREATED,
          model,
          choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop", logprobs: null }],
          usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
        }),
      };
    }
    const chunk = (delta, finish) => ({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
    const frames = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
    for (const piece of splitText(text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
    frames.push(sseFrame(null, chunk({}, "stop")));
    frames.push(
      sseFrame(null, {
        id: "chatcmpl-sim",
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
      }),
    );
    frames.push("data: [DONE]\n\n");
    return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
  }
  if (!stream) {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: output },
      }),
    };
  }
  const frames = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of splitText(text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: output } }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return { status: 200, contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

function renderError(api, status, message) {
  if (api === "openai") {
    return {
      status,
      contentType: "application/json",
      body: JSON.stringify({ error: { message, type: "api_error", param: null, code: "no_provider_available" } }),
    };
  }
  return {
    status,
    contentType: "application/json",
    body: JSON.stringify({ type: "error", error: { type: "api_error", message } }),
  };
}

function modelsReply(api, ids) {
  if (api === "openai") {
    return JSON.stringify({ object: "list", data: ids.map((id) => ({ id, object: "model", created: FIXED_CREATED, owned_by: "aggregator" })) });
  }
  return JSON.stringify({
    data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
    has_more: false,
  });
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

function authHeaders(state) {
  if (state.auth === "bearer") return { authorization: `Bearer ${state.key}` };
  if (state.auth === "x-api-key") return { "x-api-key": state.key };
  if (state.auth === "x-goog-api-key") return { "x-goog-api-key": state.key };
  return {};
}

async function main() {
  const args = process.argv.slice(2);
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";

  if (args.includes("--list")) {
    if (promptFile === "") {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return;
    }
    const catalog = buildCatalog(loadBrief(promptFile));
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (promptFile === "" || envUrl === "") throw new Error("MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL are required");

  const brief = loadBrief(promptFile);
  const catalog = buildCatalog(brief);
  const catalogText = JSON.stringify(catalog);
  const ids = freeModelIds(catalog);
  const router = new Router(brief);
  const manifest = brief.workload.manifest;
  let nextIndex = 0;

  // One provider call: translate the canonical request into the provider's
  // wire, dial it through the env, and classify the outcome. Calls are made
  // one at a time, non-streamed, so a provider never sees a concurrent call
  // and the answer's echo and usage are the provider's own.
  const callProvider = (view) => async (state, model) => {
    let url;
    const headers = { "content-type": "application/json", ...authHeaders(state) };
    if (state.wire === "openai") {
      url = `${envUrl}${state.path}/v1/chat/completions`;
    } else if (state.wire === "anthropic") {
      url = `${envUrl}${state.path}/v1/messages`;
      headers["anthropic-version"] = "2023-06-01";
    } else {
      url = `${envUrl}${state.path}/v1beta/models/${encodeURIComponent(model.model)}:generateContent`;
    }
    let response;
    try {
      response = await fetch(url, { method: "POST", headers, body: JSON.stringify(providerBody(state.wire, model.model, view)) });
    } catch {
      return { ok: false, kind: "other" };
    }
    const text = await response.text();
    let json = null;
    try {
      json = JSON.parse(text);
    } catch {
      json = null;
    }
    if (response.status >= 200 && response.status < 300) {
      const answer = providerAnswer(state.wire, json);
      if (answer !== null) return { ok: true, answer: { ...answer, model: model.model } };
      return { ok: false, kind: "other" };
    }
    return { ok: false, kind: classifyRefusal(providerErrorMessage(state.wire, json)) };
  };

  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      void (async () => {
        const url = new URL(req.url ?? "/", "http://entry");
        const send = (out) => {
          res.writeHead(out.status, { "content-type": out.contentType });
          res.end(out.body);
        };
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const api = req.headers["x-api-key"] !== undefined ? "anthropic" : "openai";
          send({ status: 200, contentType: "application/json", body: modelsReply(api, ids) });
          return;
        }
        const api = url.pathname === "/v1/chat/completions" ? "openai" : url.pathname === "/v1/messages" ? "anthropic" : null;
        if (req.method !== "POST" || api === null) {
          send(renderError(api ?? "openai", 404, "no such route"));
          return;
        }
        let body;
        try {
          body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        } catch {
          send(renderError(api, 400, "the request body is not JSON"));
          return;
        }
        const request = manifest[nextIndex] ?? { id: "r00", weight: 1, requires: {} };
        nextIndex += 1;
        const view = canonicalRequest(api, body);
        const stream = body.stream === true;
        const answer = await router.serve(request, view, callProvider(view));
        if (answer === null) {
          send(renderError(api, 502, "no provider could answer this request"));
          return;
        }
        send(renderAnswer(api, stream, answer.model, answer));
      })().catch(() => {
        try {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "internal error", type: "api_error", param: null, code: "internal" } }));
        } catch {
          // The client is gone; nothing more to do.
        }
      });
    });
  });

  await new Promise((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server did not bind a port");
  const registered = await fetch(`${envUrl}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port: address.port, catalog: JSON.parse(catalogText) }),
  });
  if (!registered.ok) throw new Error(`register answered ${registered.status}`);

  // Poll the env until the workload is done, then exit 0.
  const deadline = Date.now() + 120000;
  for (;;) {
    let done = false;
    try {
      const work = await fetch(`${envUrl}/work`);
      if (work.ok) done = (await work.json()).done === true;
    } catch {
      done = false;
    }
    if (done) break;
    if (Date.now() > deadline) break;
    await new Promise((resolve) => setTimeout(resolve, 15));
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator-01: ${err.message}\n`);
  process.exit(1);
});
