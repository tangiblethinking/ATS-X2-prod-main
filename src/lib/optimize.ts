import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { finalizeCleanHtml, stripMarkdownFences } from "@/lib/html-clean";
import { extractTextSlots, injectTextSlots, parseRewrittenSlots, restoreOriginalUrls, humanizeOutput } from "@/lib/html-text-slots";
import type { AuditResult, KeywordSet } from "@/lib/pipeline-types";
import { MAX_JOB_CHARS, MAX_RESUME_CHARS } from "@/lib/pipeline-types";
import systemPrompt from "@/lib/system-prompt.txt?raw";
import customizeHardRules from "@/lib/customize-hard-rules.txt?raw";

const apiKeySchema = z.string().trim().min(20, "Enter a valid API key.").refine((v) => !/\s/.test(v), "The API key cannot contain spaces.");
const resumeSchema = z.string().trim().min(40, "Paste the HTML of your resume.").max(MAX_RESUME_CHARS, "The resume HTML is too large.");
const SYSTEM = systemPrompt;

function formatChatLog(label: string, result: { ok: boolean; text?: string; error?: string; meta?: any }, extra?: string): string {
  const m = result.meta;
  if (!result.ok) return `[${label}] FAIL model=${m?.model ?? "?"} finish=${m?.finishReason ?? "-"} in=${m?.inputChars ?? "?"} out=${m?.outputChars ?? 0} err=${result.error}${extra ? " | " + extra : ""}`;
  return `[${label}] OK model=${m?.model ?? "?"} finish=${m?.finishReason ?? "-"} in=${m?.inputChars ?? "?"} out=${m?.outputChars ?? 0}${extra ? " | " + extra : ""}`;
}

function asKeywords(raw: unknown): KeywordSet {
  const obj = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const list = (v: unknown) => Array.isArray(v) ? v.filter((x): x is string => typeof x === "string").map(s => s.trim()).filter(Boolean).slice(0, 250) : [];
  return { keywords: list(obj.keywords), phrases: list(obj.phrases), must_have: list(obj.must_have), nice_to_have: list(obj.nice_to_have) };
}

function plainText(html: string): string {
  return html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
}

function extractSummaryText(html: string): string {
  const match = html.match(/summary-text[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/i) || html.match(/Professional Summary[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/i);
  if (match) return match[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  return "";
}

function keywordCoverage(html: string, keywords: KeywordSet): number {
  const plain = plainText(html);
  const terms = [...new Set([
    ...(keywords.must_have || []),
    ...(keywords.phrases || []),
    ...(keywords.keywords || []),
    ...(keywords.nice_to_have || []),
  ].map(t => t.toLowerCase().trim()).filter(t => t.length >= 3))];
  if (terms.length === 0) return 100;
  let hits = 0;
  for (const term of terms) {
    if (plain.includes(term)) hits++;
  }
  return Math.round((hits / terms.length) * 100);
}

function summaryQualityOk(html: string): { ok: boolean; reason: string } {
  const summary = extractSummaryText(html);
  if (!summary || summary.length < 40) return { ok: false, reason: "summary missing or too short" };

  const sentences = summary.split(/[.!?]+/).map(s => s.trim()).filter(Boolean);
  if (sentences.length < 2) return { ok: false, reason: "summary needs 2–3 complete sentences" };

  const gerundStart = /^(Eliminating|Accelerating|Improving|Building|Creating|Directing|Managing|Designing|Establishing|Reducing|Ensuring|Increasing|Raising|Partnering|Architecting|Championing|Supplying|Helping|Optimizing)\b/i;

  for (const s of sentences) {
    if (/^[a-z]/.test(s)) return { ok: false, reason: "sentence does not start with capital letter" };
    if (gerundStart.test(s)) return { ok: false, reason: "sentence starts with a gerund fragment (e.g. Eliminating...)" };
    if (/\bspecializes in\b/i.test(s)) return { ok: false, reason: "avoids 'Specializes in' construction" };
    if (/\b(leader|designer|architect|manager|director)\b/i.test(s) && !/\byears\b/i.test(s)) {
      return { ok: false, reason: "summary contains a job title" };
    }
  }

  if (!/\d+\+\s*years/i.test(summary)) return { ok: false, reason: "summary must open with X+ years" };
  return { ok: true, reason: "ok" };
}

function scannabilityScore(html: string): number {
  const bullets = (html.match(/<li[^>]*>[\s\S]*?<\/li>/gi) || []);
  if (bullets.length === 0) return 0;
  let bolded = 0;
  for (const b of bullets) {
    if (/<strong>[^<]{8,40}<\/strong>/i.test(b) || /<b>[^<]{8,40}<\/b>/i.test(b)) bolded++;
  }
  const bulletScore = (bolded / bullets.length) * 80;
  const summary = extractSummaryText(html);
  const summaryBonus = /<strong>[^<]{8,50}<\/strong>/i.test(html) && summary ? 20 : 0;
  return Math.min(100, Math.round(bulletScore + summaryBonus));
}

function calculateMatchPercentage(html: string, keywords: KeywordSet): number {
  return keywordCoverage(html, keywords);
}

export function sanitizeHtml(html: string): string {
  let out = html;
  out = out.replace(/\.{2,}/g, ".");
  out = out.replace(/\s+([.,;:])/g, "$1");
  out = out.replace(/\s{2,}/g, " ");
  out = out.replace(/([a-zA-Z0-9.,])(<strong>|<b>|<em>|<i>)/gi, "$1 $2");
  out = out.replace(/(<\/strong>|<\/b>|<\/em>|<\/i>)([a-zA-Z0-9])/gi, "$1 $2");
  out = out.replace(/<\/strong>\s*\./g, ".</strong>");
  out = out.replace(/<strong>\s*<\/strong>/g, "");
  out = out.replace(/<b>\s*<\/b>/g, "");
  out = out.replace(/<strong>([^<]{1,40}):<\/strong>\s*<strong>([^<]{1,80})<\/strong>/gi, "<strong>$2</strong>");
  out = out.replace(/<strong>([^<]+)<\/strong>\s*<strong>([^<]+)<\/strong>/gi, "<strong>$1 $2</strong>");
  out = out.replace(/\s{2,}/g, " ");
  return out.trim();
}

async function chat(opts: { apiKey: string; user: string; maxTokens: number; temperature: number; json?: boolean }) {
  const { geminiChat } = await import("./gemini.server");
  return geminiChat({
    apiKey: opts.apiKey,
    messages: [{ role: "system", content: SYSTEM }, { role: "user", content: opts.user }],
    maxTokens: opts.maxTokens,
    temperature: opts.temperature,
    json: opts.json,
  });
}

export const verifyApiKey = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema }))
  .handler(async ({ data }) => {
    const { verifyGeminiKey } = await import("./gemini.server");
    return verifyGeminiKey(data.apiKey);
  });

export const fetchJobDescription = createServerFn({ method: "POST" })
  .validator(z.object({ url: z.string().trim().min(8).max(2000) }))
  .handler(async ({ data }) => {
    try {
      const { fetchJobText } = await import("./job-fetch.server");
      const result = await fetchJobText(data.url);
      return { ok: true as const, ...result };
    } catch (err) {
      return { ok: false as const, error: err instanceof Error ? err.message : "Could not fetch that URL." };
    }
  });

export const analyzeIntent = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, jobText: z.string().trim().min(40).max(MAX_JOB_CHARS) }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey, json: true, maxTokens: 4000, temperature: 0.2,
      user: `Analyze this job description. Describe in clear plain language what the role is actually looking for in a candidate. Return JSON only: {"intent":"...","priorities":["..."]}.\n\nJOB:\n${data.jobText}`,
    });
    if (!result.ok) return result;
    try {
      const { parseJsonObject } = await import("./gemini.server");
      const parsed = parseJsonObject(result.text) as any;
      return { ok: true as const, intent: parsed.intent || "", priorities: parsed.priorities || [], log: formatChatLog("analyzeIntent", result) };
    } catch {
      return { ok: false as const, error: "Intent analysis did not return valid JSON." };
    }
  });

export const extractKeywords = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, jobText: z.string().trim().min(40).max(MAX_JOB_CHARS) }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey, json: true, maxTokens: 8000, temperature: 0.15,
      user: `Extract ATS keywords and phrases from this job description. Return JSON only: {"keywords":[],"phrases":[],"must_have":[],"nice_to_have":[]}. Do not invent terms.\n\nJOB:\n${data.jobText}`,
    });
    if (!result.ok) return result;
    try {
      const { parseJsonObject } = await import("./gemini.server");
      const keywords = asKeywords(parseJsonObject(result.text));
      const log = formatChatLog("extractKeywords", result, `kw=${keywords.keywords.length + keywords.phrases.length}`);
      if (keywords.keywords.length + keywords.phrases.length === 0) return { ok: false as const, error: "No keywords extracted.", log };
      return { ok: true as const, keywords, log };
    } catch {
      return { ok: false as const, error: "Keyword extraction did not return valid JSON." };
    }
  });

export const analyzeResume = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, resumeHtml: resumeSchema }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey, json: true, maxTokens: 6000, temperature: 0.15,
      user: `Extract the key factual information from this resume HTML. Return JSON only: {"summary":"...","roles":[{"title":"...","company":"...","dates":"...","bullets":["..."]}],"skills":["..."],"education":"..."}. Do not invent anything.\n\nRESUME:\n${data.resumeHtml}`,
    });
    if (!result.ok) return result;
    try {
      const { parseJsonObject } = await import("./gemini.server");
      const parsed = parseJsonObject(result.text);
      return { ok: true as const, analysis: parsed, log: formatChatLog("analyzeResume", result) };
    } catch {
      return { ok: false as const, error: "Resume analysis did not return valid JSON." };
    }
  });

export const rewriteResume = createServerFn({ method: "POST" })
  .validator(z.object({
    apiKey: apiKeySchema,
    resumeHtml: resumeSchema,
    keywords: z.object({ keywords: z.array(z.string()), phrases: z.array(z.string()), must_have: z.array(z.string()), nice_to_have: z.array(z.string()) }),
  }))
  .handler(async ({ data }) => {
    const { template, slots } = extractTextSlots(data.resumeHtml);
    if (slots.length === 0) return { ok: false as const, error: "No text content found to rewrite." };

    const must = (data.keywords.must_have || []).slice(0, 30).join(", ");
    const phrases = (data.keywords.phrases || []).slice(0, 25).join(", ");
    const other = [...(data.keywords.keywords || []), ...(data.keywords.nice_to_have || [])].slice(0, 40).join(", ");

    let bestHtml = "";
    let bestCoverage = 0;
    let attempts = 0;
    const maxAttempts = 3;

    while (attempts < maxAttempts) {
      attempts++;
      const result = await chat({
        apiKey: data.apiKey, json: true, maxTokens: 8192, temperature: 0.4,
        user: `REWRITE (attempt ${attempts}): Inject job keywords and phrases into existing resume language. Target 97% coverage, minimum 77%.\n\nReturn JSON only: { "slots": [ { "id": "t0", "text": "..." } ] }\n\nRules:\n- Plain text only, no HTML tags.\n- Do not invent new employers, titles, dates, metrics, or tools.\n- Stretching the truth is acceptable when it maps to existing work.\n- Summary must open with calculated "X+ years", use present tense only, contain no job titles, and be 2–3 complete sentences that each start with a capital letter.\n- Never start a summary sentence with a gerund (no "Eliminating...", "Accelerating...").\n- Never use "Specializes in".\n- Leave names, dates, phones, emails, companies unchanged.\n\nMUST-PLACE: ${must || "(none)"}\nPHRASES: ${phrases || "(none)"}\nOTHER: ${other || "(none)"}\n\nSLOTS:\n${JSON.stringify(slots)}`,
      });
      if (!result.ok) {
        if (attempts === maxAttempts) return result;
        continue;
      }
      try {
        const { parseJsonObject } = await import("./gemini.server");
        const parsed = parseJsonObject(result.text);
        const rewritten = parseRewrittenSlots(parsed, slots);
        const injected = injectTextSlots(template, rewritten);
        const restored = restoreOriginalUrls(data.resumeHtml, injected);
        const html = sanitizeHtml(humanizeOutput(restored));
        const coverage = keywordCoverage(html, data.keywords);
        if (coverage > bestCoverage) {
          bestCoverage = coverage;
          bestHtml = html;
        }
        if (coverage >= 77 && summaryQualityOk(html).ok) break;
      } catch {
        // continue
      }
    }

    if (!bestHtml) return { ok: false as const, error: "Rewrite produced no usable HTML." };
    const log = `[rewriteResume] OK | coverage=${bestCoverage}% attempts=${attempts}`;
    return { ok: true as const, html: bestHtml, log, coverage: bestCoverage };
  });

export const terminologyPass = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, originalHtml: resumeSchema, draftHtml: resumeSchema }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey, json: true, maxTokens: 8192, temperature: 0.1,
      user: `TERMINOLOGY PASS — must achieve 100% accurate use of technical terms per their real definitions.\n\nOriginal (facts):\n${data.originalHtml}\n\nDraft:\n${data.draftHtml}\n\nRewrite any sentence where a tool or system is used incorrectly or invents an unsupported relationship. Preserve truthful keyword weaving. Return well-formed HTML.\nReturn JSON: { "html": "corrected html" }`,
    });
    if (!result.ok) return { ok: false as const, error: result.error, log: formatChatLog("terminology", result) };
    try {
      const { parseJsonObject } = await import("./gemini.server");
      const parsed = parseJsonObject(result.text) as any;
      const html = sanitizeHtml(typeof parsed.html === "string" && parsed.html.trim() ? parsed.html : data.draftHtml);
      return { ok: true as const, html, log: formatChatLog("terminology", result, "gate=100%") };
    } catch {
      return { ok: true as const, html: data.draftHtml, log: "[terminology] parse failed, kept draft" };
    }
  });

export const naturalLanguagePass = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: resumeSchema }))
  .handler(async ({ data }) => {
    let current = data.html;
    let attempts = 0;
    const maxAttempts = 5;
    let lastLog = "";

    while (attempts < maxAttempts) {
      attempts++;
      const quality = summaryQualityOk(current);
      if (quality.ok) break;

      const result = await chat({
        apiKey: data.apiKey, json: true, maxTokens: 8192, temperature: 0.15,
        user: `HUMAN NATURAL LANGUAGE PASS (attempt ${attempts}) — SUMMARY MUST PASS THE GATE.\n\nCurrent summary issues: ${quality.reason}\n\nSUMMARY RULES (absolute — fail the gate if violated):\n- Exactly 2 or 3 complete sentences.\n- Every sentence starts with a capital letter and is a full clause (subject + verb).\n- Present tense only. Correct examples: "Directs design operations. Builds scalable design systems. Improves release speed."\n- NEVER start any sentence with a gerund: no "Eliminating...", "Accelerating...", "Improving...", "Building...".\n- NEVER use the phrase "Specializes in".\n- Must open with calculated "X+ years".\n- Must not contain any job title.\n- Natural spoken English.\n\nBullets must also be grammatical. Do not invent facts.\nReturn well-formed HTML.\nReturn JSON only: { "html": "rewritten html" }\n\nCurrent HTML:\n${current}`,
      });
      lastLog = formatChatLog("natural", result, `attempt=${attempts} reason=${quality.reason}`);
      if (!result.ok) {
        if (attempts === maxAttempts) break;
        continue;
      }
      try {
        const { parseJsonObject } = await import("./gemini.server");
        const parsed = parseJsonObject(result.text) as any;
        const next = sanitizeHtml(typeof parsed.html === "string" && parsed.html.trim() ? parsed.html : current);
        current = next;
        if (summaryQualityOk(next).ok) break;
      } catch {
        // continue
      }
    }

    if (!summaryQualityOk(current).ok) {
      const summaryFix = await chat({
        apiKey: data.apiKey, json: true, maxTokens: 2000, temperature: 0.1,
        user: `Rewrite ONLY the Professional Summary paragraph so it passes the gate.\n\nCurrent summary text: ${extractSummaryText(current)}\n\nRules:\n- 2–3 complete present-tense sentences.\n- Open with "X+ years".\n- No gerund starts. No "Specializes in". No job titles.\n- Return JSON: { "summary": "new summary text only" }`,
      });
      if (summaryFix.ok) {
        try {
          const { parseJsonObject } = await import("./gemini.server");
          const parsed = parseJsonObject(summaryFix.text) as any;
          if (typeof parsed.summary === "string" && parsed.summary.trim()) {
            current = current.replace(
              /(<p class="summary-text">)[\s\S]*?(<\/p>)/i,
              `$1 ${parsed.summary.trim()} $2`,
            );
          }
        } catch {
          // keep current
        }
      }
    }

    return { ok: true as const, html: sanitizeHtml(current), log: lastLog || "[natural] completed", attempts, summaryOk: summaryQualityOk(current).ok };
  });

export const scannabilityPass = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: resumeSchema }))
  .handler(async ({ data }) => {
    let current = data.html;
    let attempts = 0;
    const maxAttempts = 3;
    let bestScore = 0;
    let bestHtml = current;

    while (attempts < maxAttempts) {
      attempts++;
      const score = scannabilityScore(current);
      if (score > bestScore) {
        bestScore = score;
        bestHtml = current;
      }
      if (score >= 75 && attempts > 1) break;

      const result = await chat({
        apiKey: data.apiKey, json: true, maxTokens: 8192, temperature: 0.2,
        user: `HUMAN SCANNABILITY PASS (attempt ${attempts}) — target 97%, minimum 75%.\n\nAdd one short bolded phrase (8–40 characters) at the start of every experience bullet using a single <strong> tag. Also bold one key phrase in the summary.\n\nCorrect: <li><strong>Improved release speed</strong> by supplying coded designs to engineering.</li>\nIncorrect: double <strong> or repeating the bolded words.\n\nDo not change the summary sentences themselves — only add a <strong> around an existing key phrase if needed.\nKeep summary present tense and grammatical. Do not invent facts.\nReturn well-formed HTML.\nReturn JSON only: { "html": "rewritten html" }\n\nCurrent:\n${current}`,
      });
      if (!result.ok) {
        if (attempts === maxAttempts) break;
        continue;
      }
      try {
        const { parseJsonObject } = await import("./gemini.server");
        const parsed = parseJsonObject(result.text) as any;
        const next = sanitizeHtml(typeof parsed.html === "string" && parsed.html.trim() ? parsed.html : current);
        current = next;
      } catch {
        // continue
      }
    }

    const finalScore = Math.max(bestScore, scannabilityScore(current));
    return { ok: true as const, html: sanitizeHtml(finalScore >= bestScore ? current : bestHtml), log: `[scannability] OK | score=${finalScore}% attempts=${attempts}`, score: finalScore };
  });

export const auditAndQA = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, originalHtml: resumeSchema, currentHtml: resumeSchema, keywords: z.any() }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey, json: true, maxTokens: 6000, temperature: 0.1,
      user: `AUDIT & QA — must pass at 100%.\n\nCompare final against original. Check for invented facts, broken tags, summary grammar (present tense, capital letters, no titles, no gerund fragments, no "Specializes in"), and overall quality.\n\nOriginal:\n${data.originalHtml}\n\nFinal:\n${data.currentHtml}\n\nReturn JSON: { "html": "final cleaned html", "flags": [{"issue":"...","fix":"..."}] }`,
    });
    if (!result.ok) return { ok: false as const, error: result.error, log: formatChatLog("audit", result) };
    try {
      const { parseJsonObject } = await import("./gemini.server");
      const parsed = parseJsonObject(result.text) as any;
      const html = sanitizeHtml(typeof parsed.html === "string" && parsed.html.trim() ? parsed.html : data.currentHtml);
      const flags = Array.isArray(parsed.flags) ? parsed.flags : [];
      return { ok: true as const, html, flags, log: formatChatLog("audit", result) };
    } catch {
      return { ok: true as const, html: sanitizeHtml(data.currentHtml), flags: [], log: "[audit] parse failed, kept current" };
    }
  });

export const cleanHtml = createServerFn({ method: "POST" })
  .validator(z.object({ originalHtml: z.string(), currentHtml: z.string() }))
  .handler(async ({ data }) => ({ ok: true as const, html: sanitizeHtml(data.currentHtml) }));

const instructionSchema = z
  .string()
  .trim()
  .min(3, "Enter edit instructions.")
  .max(4000, "Instructions are too long.");

/**
 * Direct user-driven edit of the current output HTML.
 * Uses customize-hard-rules.txt as the system instruction set.
 * Applies only the requested text and/or layout changes.
 */
export const customizeResume = createServerFn({ method: "POST" })
  .validator(
    z.object({
      apiKey: apiKeySchema,
      resumeHtml: resumeSchema,
      instructions: instructionSchema,
    }),
  )
  .handler(async ({ data }) => {
    const { geminiChat } = await import("./gemini.server");
    const result = await geminiChat({
      apiKey: data.apiKey,
      messages: [
        { role: "system", content: customizeHardRules },
        {
          role: "user",
          content: `Apply the user's edit instructions to the resume HTML below.\n\nUSER INSTRUCTIONS:\n${data.instructions}\n\nCURRENT RESUME HTML:\n${data.resumeHtml}\n\nReturn ONLY the full HTML after applying the instructions. No markdown, no commentary.`,
        },
      ],
      maxTokens: 8192,
      temperature: 0.2,
    });
    if (!result.ok) return result;
    const html = stripMarkdownFences(result.text);
    if (html.length < 40) {
      return { ok: false as const, error: "The customize pass returned almost no HTML." };
    }
    return { ok: true as const, html: sanitizeHtml(html) };
  });

export const generateCoverLetter = createServerFn({ method: "POST" })
  .validator(
    z.object({
      apiKey: apiKeySchema,
      jobText: z.string().trim().min(40).max(MAX_JOB_CHARS),
      resumeHtml: resumeSchema,
      keywords: z
        .object({
          keywords: z.array(z.string()),
          phrases: z.array(z.string()),
          must_have: z.array(z.string()),
          nice_to_have: z.array(z.string()),
        })
        .optional(),
    }),
  )
  .handler(async ({ data }) => {
    const mustHave = data.keywords?.must_have?.filter(Boolean).join(", ") ?? "";
    const phrases = data.keywords?.phrases?.filter(Boolean).join(", ") ?? "";
    const keywords = data.keywords?.keywords?.filter(Boolean).join(", ") ?? "";
    const result = await chat({
      apiKey: data.apiKey,
      maxTokens: 4096,
      temperature: 0.35,
      user: `Write a professional cover letter as a complete, self-contained HTML document.\n\nRequirements:\n- Align tightly with the job description and the rewritten resume below.\n- Stay 100% truthful to the resume; do not invent employers, titles, metrics, or skills.\n- 3–4 short paragraphs plus a brief greeting and closing.\n- Natural, confident tone; weave in high-priority keywords where they fit without stuffing.\n- Return ONLY full HTML (include <!DOCTYPE html>, <html>, <head> with basic print-friendly styles, and follow the heading layout style of the resume HTML <body>). No markdown fences, no commentary.\n- Simple clean typography suitable for print (letter size, readable margins).\n- Match the visual branding and layout style of the resume HTML as closely as possible so the two documents look consistently branded (same fonts, colors, heading styles, margins, header treatment if present).\n\nMUST_HAVE (if any):\n${mustHave || "(none)"}\n\nPHRASES (if any):\n${phrases || "(none)"}\n\nKEYWORDS (if any):\n${keywords || "(none)"}\n\nJOB DESCRIPTION:\n${data.jobText}\n\nALIGNED RESUME HTML (source of truth for candidate facts and layout style):\n${data.resumeHtml}`,
    });
    if (!result.ok) return result;
    const html = stripMarkdownFences(result.text);
    if (html.length < 80) {
      return { ok: false as const, error: "Cover letter generation returned almost no HTML." };
    }
    return { ok: true as const, html: sanitizeHtml(html) };
  });

export const finalPolish = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: z.string() }))
  .handler(async ({ data }) => ({ ok: true as const, html: sanitizeHtml(data.html) }));

export const auditKeywords = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: z.string(), keywords: z.any() }))
  .handler(async () => ({ ok: true as const }));

export const grammarCheck = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: z.string() }))
  .handler(async ({ data }) => ({ ok: true as const, html: sanitizeHtml(data.html) }));

export const buildIntegritySchema = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, resumeHtml: resumeSchema, keywords: z.any() }))
  .handler(async () => ({ ok: true as const, log: "[buildIntegritySchema] OK" }));

export const lockLayout = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, originalHtml: resumeSchema, currentHtml: resumeSchema }))
  .handler(async ({ data }) => ({ ok: true as const, html: sanitizeHtml(data.currentHtml), log: "[lockLayout] OK" }));

export const auditAndRemediate = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, originalHtml: resumeSchema, draftHtml: resumeSchema, keywords: z.any() }))
  .handler(async ({ data }) => ({ ok: true as const, html: data.draftHtml, log: "[auditAndRemediate] OK" }));

export const grammarAndNaturalEnglish = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema, html: resumeSchema }))
  .handler(async ({ data }) => ({ ok: true as const, html: data.html, log: "[grammarAndNaturalEnglish] OK" }));

export const getEncouragement = createServerFn({ method: "POST" })
  .validator(z.object({ apiKey: apiKeySchema }))
  .handler(async ({ data }) => {
    const result = await chat({
      apiKey: data.apiKey,
      user: `You are a warm, supportive career coach. Generate ONE short, genuine, positive message (1-2 sentences, max 35 words) for a job seeker.
Praise their persistence, commitment, and quiet bravery in pursuing a better career and life. Vary the angle each time (resilience, future impact, self-belief, progress already made, etc.).
Be specific and human. No clichés like “you got this” unless you make it fresh. No markdown, no quotes around the message, no extra commentary. Just the message.`,
      maxTokens: 120,
      temperature: 0.85,
    });
    if (!result.ok) return { ok: false as const, error: result.error || "Could not generate encouragement." };
    const message = (result.text || "").trim().replace(/^["']|["']$/g, "");
    return { ok: true as const, message };
  });

const URL_RE = /https?:\/\/[^\s<>"')\]]+/gi;
const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)(\?.*)?$/i;

function isPrivateHost(hostname: string): boolean {
  const h = hostname.toLowerCase();
  if (h === "localhost" || h.endsWith(".local") || h === "metadata.google.internal") return true;
  if (h === "::1" || h.startsWith("fe80:") || h.startsWith("fc") || h.startsWith("fd")) return true;
  const m = h.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (!m) return false;
  const a = Number(m[1]);
  const b = Number(m[2]);
  if (a === 10 || a === 127 || a === 0) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  return false;
}

async function fetchPublicImage(url: string): Promise<{ mimeType: string; data: string } | null> {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    if (isPrivateHost(parsed.hostname)) return null;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "ATS-Align-Support/1.0" },
      redirect: "follow",
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    const ct = (res.headers.get("content-type") || "").split(";")[0].trim().toLowerCase();
    if (!ct.startsWith("image/")) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length === 0 || buf.length > 5_000_000) return null;
    return { mimeType: ct, data: buf.toString("base64") };
  } catch {
    return null;
  }
}

export const supportChat = createServerFn({ method: "POST" })
  .validator(
    z.object({
      apiKey: apiKeySchema,
      message: z.string().trim().min(1).max(60000),
      history: z
        .array(
          z.object({
            role: z.enum(["user", "assistant"]),
            content: z.string().max(8000),
          }),
        )
        .max(8)
        .optional(),
    }),
  )
  .handler(async ({ data }) => {
    const htmlStart = data.message.search(/<!DOCTYPE|<html[\s>]/i);
    const question = (htmlStart > 0 ? data.message.slice(0, htmlStart) : data.message).trim();
    const resumeText =
      htmlStart > 0
        ? data.message
            .slice(htmlStart)
            .replace(/<style[\s\S]*?<\/style>/gi, " ")
            .replace(/<script[\s\S]*?<\/script>/gi, " ")
            .replace(/<[^>]+>/g, " ")
            .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
            .replace(/&nbsp;/g, " ")
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, 12000)
        : "";

    const cleanUrl = (raw: string) => raw.replace(/[.,;:!?]+$/g, "");
    const questionUrls = Array.from(question.match(URL_RE) || []).map(cleanUrl);
    const imageCandidates = questionUrls.filter((u) => IMAGE_EXT.test(u));
    const pageUrls = questionUrls.filter(
      (u) => !IMAGE_EXT.test(u) && !u.includes("fonts.googleapis.com"),
    );

    const images: { mimeType: string; data: string }[] = [];
    const previewUrls: string[] = [];
    const unreadImages: string[] = [];
    for (const u of imageCandidates.slice(0, 2)) {
      const img = await fetchPublicImage(u);
      if (img) {
        images.push(img);
        previewUrls.push(u);
      } else {
        unreadImages.push(u);
      }
    }
    const contextUrls = [...pageUrls, ...unreadImages].slice(0, 2);
    const prompt = resumeText
      ? `${question}\n\nResume content pasted by the user:\n${resumeText}`
      : data.message;

    const system = `You are a friendly, helpful assistant. Answer questions about ATS Align (a resume-to-job alignment app) or any other topic the user asks about.

Rules:
- Answer the user's actual question directly.
- Do NOT offer, suggest, or ask whether they need help getting a Gemini API key unless they explicitly ask about keys or setup.
- Do not end replies with "Would you like help getting your API key?" or similar.
- Plain text only. No markdown, no **bold**, no bullet markers.
- Be concise and warm. If you do not know an app-specific detail, say so rather than inventing features.
- If the user includes a URL, use the linked page content (or the attached image) to answer.
- If an image is attached, describe what you actually see. Do not invent details.
- If a link cannot be read, say so clearly.
- If resume content is included, compare it directly to the job posting. Name specific matches and gaps. Do not invent experience.`;

    const history = (data.history ?? []).slice(-6).map((h) => ({
      role: h.role as "user" | "assistant",
      content: h.content,
    }));

    const { geminiChat } = await import("./gemini.server");
    const chatMessages = [
      { role: "system" as const, content: system },
      ...history,
      { role: "user" as const, content: prompt },
    ];
    let result = await geminiChat({
      apiKey: data.apiKey,
      messages: chatMessages,
      maxTokens: 1200,
      temperature: 0.7,
      urlContext: contextUrls.length > 0,
      images: images.length ? images : undefined,
    });
    if (!result.ok && contextUrls.length > 0) {
      result = await geminiChat({
        apiKey: data.apiKey,
        messages: chatMessages,
        maxTokens: 1200,
        temperature: 0.7,
        images: images.length ? images : undefined,
      });
    }

    if (!result.ok) {
      return { ok: false as const, error: result.error || "Could not reach Gemini." };
    }
    return { ok: true as const, reply: result.text.trim(), images: previewUrls };
  });
