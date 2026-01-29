/**
 * Moonshot (Kimi) OpenAI-Compatible Transformer
 *
 * Purpose:
 * - Accept internal router canonical Chat Completions (OpenAI-like) requests.
 * - Send them to Moonshot/Kimi OpenAI-compatible endpoint.
 * - Convert upstream JSON/SSE to OpenAI-like JSON/SSE expected by the router.
 *
 * Notes:
 * - Moonshot officially supports `/v1/chat/completions` with OpenAI-like schema.
 * - The router does NOT auto-append `/v1/chat/completions` unless a transformer sets config.url.
 */

function contentToText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((c) => {
      if (!c) return "";
      if (typeof c === "string") return c;
      if (c.type === "text" && typeof c.text === "string") return c.text;
      return "";
    })
    .join("");
}

function normalizeChatMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map((m) => {
    if (!m || typeof m !== "object") return m;
    if (typeof m.content === "string") return m;
    if (Array.isArray(m.content)) {
      // Keep multimodal blocks as-is (Moonshot supports vision models);
      // but ensure we don’t accidentally pass `media_type` fields used by some clients.
      const content = m.content.map((b) => {
        if (!b || typeof b !== "object") return b;
        if (b.type === "image_url" && b.image_url && typeof b.image_url === "object") {
          const copy = { ...b, image_url: { ...b.image_url } };
          if (copy.media_type) delete copy.media_type;
          return copy;
        }
        return b;
      });
      return { ...m, content };
    }
    return { ...m, content: contentToText(m.content) };
  });
}

class MoonshotOpenAITransformer {
  name = "moonshot-openai";

  constructor(options) {
    this.options = options || {};
    this.baseUrl = this.options.baseUrl || "https://api.moonshot.ai/v1";
    this.apiKey = this.options.apiKey || process.env.MOONSHOT_API_KEY;
  }

  async transformRequestIn(request, provider) {
    const baseUrl = provider?.api_base_url || this.options.baseUrl || this.baseUrl;

    // If baseUrl already includes /chat/completions, preserve it.
    // Otherwise assume it's the /v1 base and append /chat/completions.
    const trimmed = typeof baseUrl === "string" ? baseUrl.replace(/\/+$/, "") : baseUrl;
    const url =
      typeof trimmed === "string" && /\/chat\/completions$/.test(trimmed)
        ? new URL(trimmed)
        : new URL("./chat/completions", trimmed.endsWith("/v1") ? `${trimmed}/` : `${trimmed}/`);

    const apiKey = provider?.api_key || this.options.apiKey || this.apiKey;

    const body = {
      ...request,
      messages: normalizeChatMessages(request.messages),
    };

    const headers = {
      "Content-Type": "application/json",
    };

    // Router will also add Authorization: Bearer ${provider.apiKey} by default.
    // Set explicitly here for clarity and to support cases where provider.api_key is used.
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    if (body.stream) headers.Accept = "text/event-stream";

    return {
      body,
      config: {
        url,
        headers,
      },
    };
  }

  async transformResponseOut(response) {
    // Upstream is already OpenAI-like JSON or SSE. Pass through.
    return response;
  }
}

module.exports = MoonshotOpenAITransformer;
