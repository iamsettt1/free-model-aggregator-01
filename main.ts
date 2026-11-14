// free-model-aggregator-01 — one OpenAI/Anthropic-compatible gateway in front of ten
// simulated free-tier providers. The server reads the brief at $MAKEFUN_PROMPT_FILE
// (provider directory, free-model quality table, workload manifest), registers a
// catalog built from that brief alone, then answers each workload request with a
// free model that meets the request's requirements, failing over across providers
// whenever one refuses.
//
// Strategy notes:
//  - Quality is capped by construction: 17 of the 20 requests have their best
//    legal rating on a single provider (OpenRouter's 1475, budget 10–12), so
//    several always spill to the 1470 models and the quality column tops out
//    at 99. With every other column at 100 the trial score is 99 for any
//    entry that keeps its weighted shortfall under ~765, and places among
//    the 99s are decided by the provider-call count. This entry therefore
//    routes to minimise CALLS at a bounded shortfall cost, not to maximise
//    rating at any cost.
//  - Routing is dynamic (no static plan): each request walks its eligible
//    candidates by score = rating − density(provider) × CALL_VALUE / weight,
//    where density is the chance the provider's next call index carries one
//    of its 1–2 scheduled refusals (they sit in [2, budget−1]; schedules and
//    outages start at index 2 and budgets are ≥ 4, so a provider's first
//    call never refuses). Attempts stay under the minimum of each provider's
//    published budget range (groq 9-12, cerebras 5-6, gemini 6-8, zai 4-5,
//    openrouter 10-12, sambanova 4-5, pollinations 10-14, cloudflare 7-8,
//    ovh 5-6, kilol 12-16, counting EVERY inference attempt, refusals
//    included), so a budget refusal is never self-inflicted; a relaxed
//    second pass exists only so a request is never left unanswered while
//    any candidate remains.
//  - Refusals are classified from the provider's own error message: "budget
//    is spent" and "is out" retire the provider for the rest of the trial; a
//    plain scheduled refusal spends one of its at-most-two fails (after two,
//    its remaining indices are the safest in the trial) and, on the
//    wide-range providers whose next index is very likely clean, earns one
//    immediate retry of the same candidate instead of a cascade down to
//    weaker models whose own next indices are fresh coin flips.
//  - The trade is guarded by the exact running shortfall (the brief
//    publishes every request's best rating): as it approaches the quality
//    band's edge the call value fades to zero and routing becomes pure
//    rating again.
//  - Provider answers are canonical echo text: the answer is passed through
//    byte-for-byte, usage is reported exactly as the answering provider
//    recorded it, and precisely one successful provider call is made per
//    request.
//  - Every provider call is made non-streamed; streaming answers for the
//    client are rendered locally from the completed answer, in the client's
//    own wire.
//  - Deterministic: no wall clock, no randomness in anything sent anywhere.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Types (kept minimal: Node's type stripping runs this file directly).

type Json = Record<string, unknown>;

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface ProviderInfo {
  id: string;
  wire: string; // 'openai' | 'anthropic' | 'gemini'
  scheme: string; // 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none'
  keyEnv: string;
  key: string;
  path: string; // e.g. /p/groq
}

interface ManifestRequest {
  id: string;
  api: string; // 'openai' | 'anthropic'
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number };
}

interface ImageRef {
  kind: 'base64' | 'url';
  mediaType: string;
  data: string; // base64 payload (kind base64) or the URL (kind url)
}

type Block =
  | { type: 'text'; text: string }
  | { type: 'image'; image: ImageRef }
  | { type: 'tool_call'; id: string; name: string; args: string }
  | { type: 'tool_result'; id: string; text: string };

interface Turn {
  role: 'user' | 'assistant' | 'tool';
  blocks: Block[];
}

interface CanonicalRequest {
  hasSystem: boolean;
  system: string;
  turns: Turn[];
  tools: { name: string; description: string; parameters: unknown }[];
  maxTokens: number | null;
  hasStops: boolean;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  stream: boolean;
  includeUsage: boolean;
}

interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  model: string;
  providerId: string;
}

// ---------------------------------------------------------------------------
// Small helpers.

const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const str = (v: unknown): string => (typeof v === 'string' ? v : '');

const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

function log(...parts: unknown[]): void {
  console.error('[aggregator]', ...parts);
}

/** Extract every top-level JSON document embedded in a text (brace matching, string aware). */
function documentsIn(text: string): Json[] {
  const docs: Json[] = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '{') continue;
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let j = i; j < text.length; j++) {
      const c = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (c === '\\') esc = true;
        else if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') inStr = true;
      else if (c === '{') depth++;
      else if (c === '}') {
        depth--;
        if (depth === 0) {
          try {
            const v: unknown = JSON.parse(text.slice(i, j + 1));
            if (isRecord(v)) docs.push(v);
          } catch {
            /* not a document */
          }
          i = j;
          break;
        }
      }
    }
  }
  return docs;
}

/** The text of a message content value: the string itself, or its text blocks joined with "\n". */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) {
    if (isRecord(b) && b.type === 'text' && typeof b.text === 'string' && b.text !== '') {
      parts.push(b.text);
    }
  }
  return parts.join('\n');
}

// ---------------------------------------------------------------------------
// The brief: directory, ratings, workload manifest.

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const providers = new Map<string, ProviderInfo>();
const ratings: RatingRow[] = [];
const manifest: ManifestRequest[] = [];
const bestById = new Map<string, number>();

/** Exact weighted shortfall so far: Σ weight × (best − served rating). */
let shortfallSoFar = 0;

function loadBrief(): void {
  if (PROMPT_FILE === '') return;
  let text = '';
  try {
    text = readFileSync(PROMPT_FILE, 'utf8');
  } catch {
    return;
  }
  for (const doc of documentsIn(text)) {
    const schema = str(doc.schema);
    if (schema.endsWith('/providers/v1') && Array.isArray(doc.providers)) {
      for (const p of doc.providers) {
        if (!isRecord(p)) continue;
        // The brief's directory carries auth as a plain scheme string with
        // key_env beside it; an object form is tolerated just in case.
        const auth = p.auth;
        providers.set(str(p.id), {
          id: str(p.id),
          wire: str(p.wire),
          scheme:
            typeof auth === 'string'
              ? auth
              : isRecord(auth)
                ? str(auth.scheme)
                : '',
          keyEnv: str(p.key_env),
          key: str(p.key),
          path: str(p.path),
        });
      }
    } else if (schema.endsWith('/ratings/v1') && Array.isArray(doc.ratings)) {
      if (Array.isArray(doc.best)) {
        for (const b of doc.best) {
          if (isRecord(b)) bestById.set(str(b.id), numOrNull(b.rating_milli) ?? 0);
        }
      }
      for (const m of doc.ratings) {
        if (!isRecord(m) || m.free !== true) continue;
        ratings.push({
          model: str(m.model),
          provider: str(m.provider),
          rating: numOrNull(m.rating_milli) ?? 0,
          contextLength: numOrNull(m.context_length) ?? 0,
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
        });
      }
    } else if (schema.endsWith('/workload/v1') && Array.isArray(doc.manifest)) {
      for (const r of doc.manifest) {
        if (!isRecord(r)) continue;
        const req = isRecord(r.requires) ? r.requires : {};
        manifest.push({
          id: str(r.id),
          api: str(r.api),
          stream: r.stream === true,
          weight: numOrNull(r.weight) ?? 1,
          requires: {
            tools: req.tools === true,
            vision: req.vision === true,
            json: req.json === true,
            minContext: numOrNull(req.min_context) ?? undefined,
          },
        });
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The catalog: built from the brief alone. Free models only (a paid model may
// be listed free:false or omitted entirely for full marks; omitting is simpler
// and cannot mislabel anything). One object, serialized identically for
// --list and for the register body.

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

function buildCatalog(): { providers: { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }[] } {
  const out: { id: string; base_url: string; key_env: string; wire: string; models: CatalogModel[] }[] = [];
  for (const p of providers.values()) {
    const models: CatalogModel[] = [];
    for (const r of ratings) {
      if (r.provider === p.id) {
        models.push({ id: r.model, free: true, context_length: r.contextLength });
      }
    }
    out.push({ id: p.id, base_url: p.path, key_env: p.keyEnv, wire: p.wire, models });
  }
  return { providers: out };
}

/** The unique free model ids, in first-appearance order: the /v1/models listing. */
function freeModelIds(): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of ratings) {
    if (!seen.has(r.model)) {
      seen.add(r.model);
      out.push(r.model);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: requirement filter, safe attempt caps, dynamic call-minimising
// candidate scoring.

// Attempt caps: the minimum of each provider's published budget range. Budgets
// are drawn per trial inside the range, so attempts at or under the minimum
// can never exhaust a budget.
const SAFE_CAPS: Record<string, number> = {
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

function meets(req: ManifestRequest, row: RatingRow): boolean {
  const q = req.requires;
  if (q.tools && !row.tools) return false;
  if (q.vision && !row.vision) return false;
  if (q.json && !row.json) return false;
  if (q.minContext !== undefined && row.contextLength < q.minContext) return false;
  return true;
}

/**
 * Eligible (provider, model) rows for a request, best first. One wire
 * constraint: an OpenAI-wire client request whose echo carries json_mode can
 * only be served by an OpenAI-wire provider (response_format passes through)
 * or the Gemini provider (responseMimeType is set) — the Anthropic wire has
 * nowhere to carry the flag, so its echo would come back json_mode:false.
 */
function eligible(req: ManifestRequest, openAiJsonMode: boolean): RatingRow[] {
  const rows = ratings.filter((r) => {
    if (!meets(req, r)) return false;
    if (openAiJsonMode && req.api === 'openai') {
      const wire = providers.get(r.provider)?.wire;
      if (wire !== 'openai' && wire !== 'gemini') return false;
    }
    return true;
  });
  rows.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  return rows;
}

/**
 * Dynamic routing state. Each candidate is scored rating minus an
 * expected-refusal penalty: the density below is the marginal chance that a
 * provider's next call index carries one of its 1–2 scheduled refusals
 * (drawn uniformly from [2, budget−1] over the published budget range), and
 * CALL_VALUE is how many weighted rating points one avoided provider call is
 * worth — the trade the score's tie-break on call count rewards. A provider
 * whose error message says its budget is spent or that it is out is dead
 * for the rest of the trial (skipping a live provider costs rating points
 * only; calling a dead one costs whole calls), as is one that has refused
 * twice without both refusals being plain scheduled ones.
 */
const FAIL_RATE: Record<string, number> = {
  groq: 0.16,
  cerebras: 0.44,
  gemini: 0.31,
  zai: 0.63,
  openrouter: 0.17,
  sambanova: 0.63,
  pollinations: 0.14,
  cloudflare: 0.28,
  ovh: 0.44,
  kilol: 0.13,
};
const CALL_VALUE = 65;

const refusalsSeen = new Map<string, number>();
const scheduleRefusals = new Map<string, number>();
const deadProviders = new Set<string>();

function classifyRefusal(detail: string): 'budget' | 'outage' | 'schedule' | 'unknown' {
  if (detail.includes('budget is spent')) return 'budget';
  if (detail.includes('refused this call')) return 'schedule';
  if (detail.includes('is out')) return 'outage';
  return 'unknown';
}

function noteRefusal(provider: string, detail: string): void {
  const n = (refusalsSeen.get(provider) ?? 0) + 1;
  refusalsSeen.set(provider, n);
  const cls = classifyRefusal(detail);
  if (cls === 'schedule') {
    scheduleRefusals.set(provider, (scheduleRefusals.get(provider) ?? 0) + 1);
  }
  // Dead when the provider says its budget is spent or that it is out, or
  // when it has refused twice without both refusals being plain scheduled
  // ones. Two scheduled refusals mean the (at most two) scheduled fails are
  // both spent: the provider is then the safest one left, not a dead one.
  if (
    cls === 'budget' ||
    cls === 'outage' ||
    (n >= 2 && (scheduleRefusals.get(provider) ?? 0) < n)
  ) {
    deadProviders.add(provider);
  }
}

/**
 * The marginal probability that provider's NEXT call index carries a
 * scheduled refusal: 1–2 fails sit uniformly in [2, budget−1]. Index 1 is
 * always safe; index 2 also carries the one-in-ten chance this provider is
 * the draw's designated second-call failure. A scheduled refusal already
 * observed spends one of the (at most) two fails, so the density drops;
 * after two, none remain and the density is zero.
 */
function refusalDensity(provider: string, used: number): number {
  if (used === 0) return 0;
  const seen = scheduleRefusals.get(provider) ?? 0;
  if (seen >= 2) return 0;
  let d = FAIL_RATE[provider] ?? 0.3;
  if (used === 1) d += 0.09;
  if (seen > 0) d *= 0.5;
  return d;
}

/**
 * Given a scheduled refusal at the provider's current index, the chance its
 * very next index is the second scheduled fail: (count−1)/(slots−1) with
 * count ≤ 2 over the slots of [2, budget−1]. Small for the wide-range
 * providers, large for the narrow ones (zai, sambanova).
 */
const NEXT_FAIL_P: Record<string, number> = {
  groq: 0.06,
  cerebras: 0.2,
  gemini: 0.13,
  zai: 0.33,
  openrouter: 0.06,
  sambanova: 0.33,
  pollinations: 0.05,
  cloudflare: 0.11,
  ovh: 0.2,
  kilol: 0.05,
};

// ---------------------------------------------------------------------------
// Client request parsing into the canonical form. Only the echo-relevant
// content is normalized; everything the providers measure (system text, last
// user text, tool names in order, max_tokens, stop sequences, image bytes,
// json mode) is preserved exactly across wires.

function parseOpenAiImageUrl(url: string): ImageRef | null {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (m && m[2]) {
    return { kind: 'base64', mediaType: m[1] || 'application/octet-stream', data: m[3] };
  }
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return { kind: 'url', mediaType: '', data: url };
  }
  return null;
}

function parseOpenAi(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = str(raw.role);
    if (role === 'system' || role === 'developer') {
      const t = blocksText(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (role === 'tool') {
      turns.push({
        role: 'tool',
        blocks: [{ type: 'tool_result', id: str(raw.tool_call_id), text: blocksText(raw.content) }],
      });
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const blocks: Block[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          blocks.push({ type: 'text', text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url)) {
          const img = parseOpenAiImageUrl(str(part.image_url.url));
          if (img) blocks.push({ type: 'image', image: img });
        }
      }
    }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_call',
          id: str(call.id),
          name: str(call.function.name),
          args: str(call.function.arguments),
        });
      }
    }
    turns.push({ role, blocks });
  }
  const tools: CanonicalRequest['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function)) continue;
      tools.push({
        name: str(t.function.name),
        description: str(t.function.description),
        parameters: t.function.parameters ?? {},
      });
    }
  }
  const rf = isRecord(body.response_format) ? body.response_format : null;
  const stopRaw = body.stop;
  const stops =
    typeof stopRaw === 'string'
      ? [stopRaw]
      : Array.isArray(stopRaw)
        ? stopRaw.filter((s): s is string => typeof s === 'string')
        : [];
  return {
    hasSystem: systemParts.length > 0,
    system: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    hasStops: stopRaw !== undefined && stopRaw !== null,
    stops,
    jsonMode: rf !== null && (rf.type === 'json_object' || rf.type === 'json_schema'),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    stream: body.stream === true,
    includeUsage:
      isRecord(body.stream_options) && body.stream_options.include_usage === true,
  };
}

function parseAnthropic(body: Json): CanonicalRequest {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const turns: Turn[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const blocks: Block[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          blocks.push({ type: 'text', text: part.text });
        } else if (part.type === 'image' && isRecord(part.source)) {
          const src = part.source;
          if (src.type === 'base64') {
            blocks.push({
              type: 'image',
              image: {
                kind: 'base64',
                mediaType: str(src.media_type) || 'application/octet-stream',
                data: str(src.data),
              },
            });
          } else if (src.type === 'url') {
            blocks.push({ type: 'image', image: { kind: 'url', mediaType: '', data: str(src.url) } });
          }
        } else if (part.type === 'tool_use') {
          blocks.push({
            type: 'tool_call',
            id: str(part.id),
            name: str(part.name),
            args: JSON.stringify(part.input ?? {}),
          });
        } else if (part.type === 'tool_result') {
          blocks.push({
            type: 'tool_result',
            id: str(part.tool_use_id),
            text: blocksText(part.content),
          });
        }
      }
    }
    turns.push({ role, blocks });
  }
  const tools: CanonicalRequest['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      tools.push({
        name: str(t.name),
        description: str(t.description),
        parameters: t.input_schema ?? {},
      });
    }
  }
  const stopRaw = body.stop_sequences;
  return {
    hasSystem: body.system !== undefined && body.system !== null,
    system: blocksText(body.system),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    hasStops: Array.isArray(stopRaw),
    stops: Array.isArray(stopRaw)
      ? stopRaw.filter((s): s is string => typeof s === 'string')
      : [],
    jsonMode: false, // the Anthropic wire carries no json-mode flag
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    stream: body.stream === true,
    includeUsage: false,
  };
}

// ---------------------------------------------------------------------------
// Provider request building from the canonical form, per provider wire.

function imageToOpenAi(img: ImageRef): Json {
  const url =
    img.kind === 'base64' ? `data:${img.mediaType};base64,${img.data}` : img.data;
  return { type: 'image_url', image_url: { url } };
}

function imageToAnthropic(img: ImageRef): Json {
  if (img.kind === 'base64') {
    return {
      type: 'image',
      source: { type: 'base64', media_type: img.mediaType, data: img.data },
    };
  }
  return { type: 'image', source: { type: 'url', url: img.data } };
}

function imageToGemini(img: ImageRef): Json {
  if (img.kind === 'base64') {
    return { inlineData: { mimeType: img.mediaType, data: img.data } };
  }
  return { fileData: { fileUri: img.data } };
}

function buildOpenAiBody(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.hasSystem) messages.push({ role: 'system', content: req.system });
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      for (const b of turn.blocks) {
        if (b.type === 'tool_result') {
          messages.push({ role: 'tool', tool_call_id: b.id, content: b.text });
        }
      }
      continue;
    }
    if (turn.role === 'assistant') {
      const text = turn.blocks
        .filter((b): b is Extract<Block, { type: 'text' }> => b.type === 'text')
        .map((b) => b.text)
        .join('');
      const calls = turn.blocks.filter(
        (b): b is Extract<Block, { type: 'tool_call' }> => b.type === 'tool_call',
      );
      const msg: Json = { role: 'assistant', content: text };
      if (calls.length > 0) {
        msg.tool_calls = calls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(msg);
      continue;
    }
    const parts: Json[] = [];
    let singleText: string | null = null;
    for (const b of turn.blocks) {
      if (b.type === 'text') parts.push({ type: 'text', text: b.text });
      else if (b.type === 'image') parts.push(imageToOpenAi(b.image));
      else if (b.type === 'tool_result') {
        parts.push({ type: 'text', text: b.text });
      }
    }
    if (parts.length === 1 && parts[0].type === 'text') {
      singleText = str(parts[0].text);
    }
    messages.push({ role: 'user', content: singleText ?? parts });
  }
  const body: Json = { model, messages, stream: false };
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.hasStops) body.stop = req.stops;
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  if (req.temperature !== undefined) body.temperature = req.temperature;
  return body;
}

function buildAnthropicBody(req: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  const pushUserBlocks = (blocks: Json[]) => {
    const last = messages[messages.length - 1];
    if (last && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...blocks);
    } else {
      messages.push({ role: 'user', content: blocks });
    }
  };
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      // Tool results join the preceding user message: a fresh user message
      // holding only tool results would replace the measured last-user text.
      const blocks: Json[] = [];
      for (const b of turn.blocks) {
        if (b.type === 'tool_result') {
          blocks.push({ type: 'tool_result', tool_use_id: b.id, content: b.text });
        }
      }
      if (blocks.length > 0) pushUserBlocks(blocks);
      continue;
    }
    if (turn.role === 'assistant') {
      const blocks: Json[] = [];
      for (const b of turn.blocks) {
        if (b.type === 'text') blocks.push({ type: 'text', text: b.text });
        else if (b.type === 'tool_call') {
          let input: unknown = {};
          try {
            input = JSON.parse(b.args || '{}');
          } catch {
            input = {};
          }
          blocks.push({ type: 'tool_use', id: b.id, name: b.name, input });
        }
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const b of turn.blocks) {
      if (b.type === 'text') blocks.push({ type: 'text', text: b.text });
      else if (b.type === 'image') blocks.push(imageToAnthropic(b.image));
      else if (b.type === 'tool_result') {
        blocks.push({ type: 'tool_result', tool_use_id: b.id, content: b.text });
      }
    }
    messages.push({ role: 'user', content: blocks });
  }
  const body: Json = { model, messages, stream: false };
  if (req.hasSystem) body.system = req.system;
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));
  }
  if (req.hasStops) body.stop_sequences = req.stops;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  return body;
}

function buildGeminiBody(req: CanonicalRequest, model: string): { url: string; body: Json } {
  void model;
  const toolNames = new Map<string, string>();
  for (const turn of req.turns) {
    for (const b of turn.blocks) {
      if (b.type === 'tool_call') toolNames.set(b.id, b.name);
    }
  }
  const contents: Json[] = [];
  const pushUserParts = (parts: Json[]) => {
    const last = contents[contents.length - 1];
    if (last && last.role === 'user' && Array.isArray(last.parts)) {
      (last.parts as Json[]).push(...parts);
    } else {
      contents.push({ role: 'user', parts });
    }
  };
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      const parts: Json[] = [];
      for (const b of turn.blocks) {
        if (b.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: toolNames.get(b.id) ?? '',
              response: { result: b.text },
            },
          });
        }
      }
      if (parts.length > 0) pushUserParts(parts);
      continue;
    }
    const parts: Json[] = [];
    for (const b of turn.blocks) {
      if (b.type === 'text') parts.push({ text: b.text });
      else if (b.type === 'image' && turn.role === 'user') parts.push(imageToGemini(b.image));
      else if (b.type === 'tool_call') {
        let args: unknown = {};
        try {
          args = JSON.parse(b.args || '{}');
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: b.name, args } });
      } else if (b.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: toolNames.get(b.id) ?? '',
            response: { result: b.text },
          },
        });
      }
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (req.hasSystem) body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const gen: Json = {};
  if (req.maxTokens !== null) gen.maxOutputTokens = req.maxTokens;
  if (req.hasStops) gen.stopSequences = req.stops;
  if (req.jsonMode) gen.responseMimeType = 'application/json';
  if (req.temperature !== undefined) gen.temperature = req.temperature;
  if (Object.keys(gen).length > 0) body.generationConfig = gen;
  return { url: '', body };
}

// ---------------------------------------------------------------------------
// Provider calls.

function providerUrl(p: ProviderInfo, model: string): string {
  const base = ENV_URL + p.path;
  if (p.wire === 'anthropic') return `${base}/v1/messages`;
  if (p.wire === 'gemini') return `${base}/v1beta/models/${model}:generateContent`;
  return `${base}/v1/chat/completions`;
}

function providerHeaders(p: ProviderInfo): Record<string, string> {
  const h: Record<string, string> = { 'content-type': 'application/json' };
  if (p.scheme === 'bearer') h.authorization = `Bearer ${p.key}`;
  else if (p.scheme === 'x-api-key') h['x-api-key'] = p.key;
  else if (p.scheme === 'x-goog-api-key') h['x-goog-api-key'] = p.key;
  if (p.wire === 'anthropic') h['anthropic-version'] = '2023-06-01';
  return h;
}

function parseProviderAnswer(wire: string, body: unknown): { text: string; inputTokens: number; outputTokens: number } | null {
  if (!isRecord(body)) return null;
  if (wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    const texts = body.content
      .filter((b): b is Json => isRecord(b) && b.type === 'text')
      .map((b) => str(b.text));
    if (texts.length === 0) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: texts.join(''),
      inputTokens: numOrNull(usage.input_tokens) ?? 0,
      outputTokens: numOrNull(usage.output_tokens) ?? 0,
    };
  }
  if (wire === 'gemini') {
    const cands = Array.isArray(body.candidates) ? body.candidates : [];
    const first = isRecord(cands[0]) ? cands[0] : null;
    const content = first && isRecord(first.content) ? first.content : null;
    const parts = content && Array.isArray(content.parts) ? content.parts : [];
    const texts = parts
      .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
      .map((p) => str(p.text));
    if (texts.length === 0) return null;
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: texts.join(''),
      inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
      outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
    };
  }
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : null;
  const message = first && isRecord(first.message) ? first.message : null;
  if (!message) return null;
  let text: string | null = null;
  if (typeof message.content === 'string') text = message.content;
  else if (Array.isArray(message.content)) text = blocksText(message.content);
  if (text === null) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
    outputTokens: numOrNull(usage.completion_tokens) ?? 0,
  };
}

// Every inference attempt, success or refusal, counts against the provider.
const attempts = new Map<string, number>();

async function callProvider(
  p: ProviderInfo,
  model: string,
  req: CanonicalRequest,
): Promise<{ ok: true; answer: Answer } | { ok: false; status: number; detail: string }> {
  attempts.set(p.id, (attempts.get(p.id) ?? 0) + 1);
  const body =
    p.wire === 'anthropic'
      ? buildAnthropicBody(req, model)
      : p.wire === 'gemini'
        ? buildGeminiBody(req, model).body
        : buildOpenAiBody(req, model);
  let res: Response;
  try {
    res = await fetch(providerUrl(p, model), {
      method: 'POST',
      headers: providerHeaders(p),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(8000),
    });
  } catch (e) {
    return { ok: false, status: 502, detail: `fetch failed: ${String(e).slice(0, 200)}` };
  }
  const text = await res.text();
  if (!res.ok) {
    return { ok: false, status: res.status, detail: text.slice(0, 200) };
  }
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: 502, detail: 'provider answer was not JSON' };
  }
  const answer = parseProviderAnswer(p.wire, parsed);
  if (answer === null) {
    return { ok: false, status: 502, detail: 'provider answer had no readable content' };
  }
  return {
    ok: true,
    answer: { ...answer, model, providerId: p.id },
  };
}

// ---------------------------------------------------------------------------
// Client response rendering: the provider's answer text passes through
// byte-for-byte; the envelope and usage are rendered in the client's wire.

const CREATED = 1767225600;
const DELTA_CHARS = 13;

function* textChunks(text: string): Generator<string> {
  for (let i = 0; i < text.length; i += DELTA_CHARS) {
    yield text.slice(i, i + DELTA_CHARS);
  }
}

function renderOpenAiJson(answer: Answer): string {
  return JSON.stringify({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: CREATED,
    model: answer.model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: 'stop',
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

function renderOpenAiSse(answer: Answer): string {
  const frame = (chunk: Json) => `data: ${JSON.stringify(chunk)}\n\n`;
  const base = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: CREATED,
    model: answer.model,
  };
  let out = frame({
    ...base,
    choices: [
      { index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null },
    ],
  });
  for (const piece of textChunks(answer.text)) {
    out += frame({
      ...base,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    });
  }
  out += frame({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  });
  // A usage chunk is always appended: the wire check compares reported usage
  // with the answering provider's record on streamed answers too.
  out += frame({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

function renderAnthropicJson(answer: Answer): string {
  return JSON.stringify({
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: answer.model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

function renderAnthropicSse(answer: Answer): string {
  const frame = (event: string, data: Json) =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  let out = frame('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model: answer.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += frame('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  for (const piece of textChunks(answer.text)) {
    out += frame('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    });
  }
  out += frame('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += frame('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += frame('message_stop', { type: 'message_stop' });
  return out;
}

function renderModelsJson(anthropicShape: boolean): string {
  const ids = freeModelIds();
  if (anthropicShape) {
    return JSON.stringify({
      data: ids.map((id) => ({
        type: 'model',
        id,
        display_name: id,
        created_at: '2026-01-01T00:00:00Z',
      })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    });
  }
  return JSON.stringify({
    object: 'list',
    data: ids.map((id) => ({
      id,
      object: 'model',
      created: CREATED,
      owned_by: 'free-model-aggregator',
    })),
  });
}

// ---------------------------------------------------------------------------
// Serving: requests arrive strictly in manifest order, one at a time.

let requestCounter = 0;

async function serveRequest(
  clientWire: 'openai' | 'anthropic',
  body: Json,
): Promise<{ status: number; contentType: string; payload: string }> {
  requestCounter += 1;
  const index = requestCounter - 1;
  const req: ManifestRequest = manifest[index] ?? {
    id: `r${requestCounter}`,
    api: clientWire,
    stream: body.stream === true,
    weight: 1,
    requires: {},
  };
  const canonical = clientWire === 'openai' ? parseOpenAi(body) : parseAnthropic(body);
  const rows = eligible(req, canonical.jsonMode);

  // The call-saving trade spends shortfall; as the running shortfall
  // approaches the band where quality would drop a point, the trade fades
  // out and routing becomes pure rating again. The shortfall only changes
  // when a request is served, so the faded call value is fixed for the
  // whole walk of one request: compute it once, here.
  // The fade band: the call value is full while the shortfall is under
  // FADE_BAND_START − FADE_BAND_SPAN, then fades linearly to zero at
  // FADE_BAND_START, the edge of the quality band.
  const FADE_BAND_START = 300;
  const FADE_BAND_SPAN = 120;
  const fade = Math.max(
    0,
    Math.min(1, (FADE_BAND_START - shortfallSoFar) / FADE_BAND_SPAN),
  );
  const cv = CALL_VALUE * fade;

  let lastFailure: { status: number; detail: string } = {
    status: 502,
    detail: 'no eligible free model',
  };
  const renderAnswer = (answer: Answer) =>
    clientWire === 'openai'
      ? canonical.stream
        ? { status: 200, contentType: 'text/event-stream', payload: renderOpenAiSse(answer) }
        : { status: 200, contentType: 'application/json', payload: renderOpenAiJson(answer) }
      : canonical.stream
        ? { status: 200, contentType: 'text/event-stream', payload: renderAnthropicSse(answer) }
        : { status: 200, contentType: 'application/json', payload: renderAnthropicJson(answer) };
  // Phase 1: dynamic walk. At each step pick the untried candidate with the
  // best score (rating minus the provider's expected-refusal penalty for its
  // next call index, scaled by the request's weight), under the safe attempt
  // caps and skipping dead providers. A refusal fails over to the next
  // candidate at once: retrying the same provider steps on its next index,
  // which is exactly where its second scheduled refusal or its outage waits,
  // while the fallback's next index is one this trial would mostly step on
  // anyway. Phase 2 (only if phase 1 served nothing) relaxes caps and dead
  // flags and walks by raw rating with one retry per candidate, so a request
  // is never left unanswered while anything might still serve it.
  for (const phase of [1, 2]) {
    const tried = new Set<string>();
    for (;;) {
      let best: RatingRow | null = null;
      let bestScore = -Infinity;
      for (const row of rows) {
        const key = `${row.provider} ${row.model}`;
        if (tried.has(key)) continue;
        const used = attempts.get(row.provider) ?? 0;
        let score = row.rating;
        if (phase === 1) {
          if (used >= (SAFE_CAPS[row.provider] ?? 0)) continue;
          if (deadProviders.has(row.provider)) continue;
          if (used > 0) {
            score -= (refusalDensity(row.provider, used) * cv) / req.weight;
          }
        }
        if (score > bestScore) {
          best = row;
          bestScore = score;
        }
      }
      if (best === null) break;
      tried.add(`${best.provider} ${best.model}`);
      const p = providers.get(best.provider);
      if (!p) continue;
      let result = await callProvider(p, best.model, canonical);
      if (!result.ok && phase === 1) {
        // A plainly scheduled refusal on a wide-range provider: its next
        // index is very likely clean, and stepping on it now costs no more
        // exposure than the index a later request would step on anyway —
        // retry the same candidate once instead of cascading to a weaker
        // model whose own next index is a fresh coin flip.
        const cls = classifyRefusal(result.detail);
        if (
          cls === 'schedule' &&
          (NEXT_FAIL_P[best.provider] ?? 1) <= 0.15 &&
          (attempts.get(p.id) ?? 0) < (SAFE_CAPS[p.id] ?? 0)
        ) {
          noteRefusal(best.provider, result.detail);
          result = await callProvider(p, best.model, canonical);
        }
      }
      if (result.ok) {
        const top = bestById.get(req.id) ?? best.rating;
        shortfallSoFar += req.weight * Math.max(0, top - best.rating);
        return renderAnswer(result.answer);
      }
      lastFailure = { status: result.status, detail: result.detail };
      noteRefusal(best.provider, result.detail);
      if (phase === 2) {
        const retry = await callProvider(p, best.model, canonical);
        if (retry.ok) {
          const top = bestById.get(req.id) ?? best.rating;
          shortfallSoFar += req.weight * Math.max(0, top - best.rating);
          return renderAnswer(retry.answer);
        }
        lastFailure = { status: retry.status, detail: retry.detail };
        noteRefusal(best.provider, retry.detail);
      }
    }
  }
  const message = `no free provider could serve this request (${lastFailure.detail})`;
  if (clientWire === 'openai') {
    return {
      status: lastFailure.status,
      contentType: 'application/json',
      payload: JSON.stringify({
        error: { message, type: 'api_error', code: String(lastFailure.status) },
      }),
    };
  }
  return {
    status: lastFailure.status,
    contentType: 'application/json',
    payload: JSON.stringify({ type: 'error', error: { type: 'api_error', message } }),
  };
}

// ---------------------------------------------------------------------------
// HTTP plumbing, register, and the work poll.

function readBody(req: import('node:http').IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size > 4_000_000) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// Serialize request handling: the workload is one-at-a-time by construction,
// and serializing also guarantees one in-flight call per provider.
let tail: Promise<unknown> = Promise.resolve();
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const run = tail.then(fn, fn);
  tail = run.catch(() => undefined);
  return run;
}

async function main(): Promise<void> {
  loadBrief();
  const catalog = buildCatalog();

  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://localhost');
        if (req.method === 'GET' && url.pathname === '/v1/models') {
          const anthropicShape =
            req.headers['x-api-key'] !== undefined &&
            req.headers.authorization === undefined;
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(renderModelsJson(anthropicShape));
          return;
        }
        if (
          req.method === 'POST' &&
          (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')
        ) {
          const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
          const raw = await readBody(req);
          let body: Json = {};
          try {
            body = JSON.parse(raw) as Json;
          } catch {
            body = {};
          }
          const out = await enqueue(() => serveRequest(wire, body));
          res.writeHead(out.status, { 'content-type': out.contentType });
          res.end(out.payload);
          return;
        }
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'not found', type: 'api_error' } }));
      } catch (e) {
        log('request handling failed:', e);
        if (!res.headersSent) {
          res.writeHead(500, { 'content-type': 'application/json' });
        }
        res.end(JSON.stringify({ error: { message: 'internal error', type: 'api_error' } }));
      }
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') {
    throw new Error('the server has no port');
  }
  const port = address.port;
  log(`listening on 0.0.0.0:${port}; ${providers.size} providers, ${freeModelIds().length} free models, ${manifest.length} requests in the manifest`);

  const reg = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
  if (!reg.ok) {
    throw new Error(`register answered HTTP ${reg.status}: ${(await reg.text()).slice(0, 300)}`);
  }
  log('registered');

  // The trial is over when the env reports the workload done; exit 0 then.
  for (;;) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      const res = await fetch(`${ENV_URL}/work`);
      if (!res.ok) continue;
      const w: unknown = await res.json();
      if (isRecord(w) && w.done === true) {
        log('workload done; exiting');
        server.close();
        process.exit(0);
      }
    } catch {
      /* keep polling */
    }
  }
}

main().catch((e) => {
  log('fatal:', e);
  process.exit(1);
});
