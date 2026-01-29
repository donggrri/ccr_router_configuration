/**
 * Kimi (Moonshot) Anthropic-Compatible Transformer
 *
 * Bridges OpenAI-style Chat Completions requests (internal router format)
 * to Kimi's Anthropic-compatible Messages API, and converts responses back
 * into OpenAI-like chat.completion / chat.completion.chunk.
 */

function convertRole(role) {
  if (role === "system") return "user";
  if (role === "tool") return "user";
  if (role === "function") return "user";
  return role;
}

function convertMessages(messages) {
  let systemPrompt = null;
  const anthropicMessages = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      if (typeof msg.content === "string") {
        systemPrompt = systemPrompt ? `${systemPrompt}\n\n${msg.content}` : msg.content;
      } else if (Array.isArray(msg.content)) {
        const text = msg.content
          .filter((c) => c.type === "text")
          .map((c) => c.text)
          .join("\n");
        systemPrompt = systemPrompt ? `${systemPrompt}\n\n${text}` : text;
      }
      continue;
    }

    if (msg.role === "tool" || msg.role === "function") {
      anthropicMessages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.tool_call_id || msg.name,
            content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
          },
        ],
      });
      continue;
    }

    if (msg.role === "assistant" && msg.tool_calls) {
      const content = [];
      if (msg.content) {
        content.push({
          type: "text",
          text: typeof msg.content === "string" ? msg.content : msg.content,
        });
      }
      for (const toolCall of msg.tool_calls) {
        content.push({
          type: "tool_use",
          id: toolCall.id,
          name: toolCall.function?.name || toolCall.name,
          input:
            typeof toolCall.function?.arguments === "string"
              ? JSON.parse(toolCall.function.arguments || "{}")
              : toolCall.function?.arguments || {},
        });
      }
      anthropicMessages.push({ role: "assistant", content });
      continue;
    }

    let content;
    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      content = msg.content.map((block) => {
        if (block.type === "text") return { type: "text", text: block.text };
        if (block.type === "image_url") {
          const url = block.image_url?.url || block.image_url;
          if (typeof url === "string" && url.startsWith("data:")) {
            const [meta, data] = url.split(",");
            const mimeMatch = meta.match(/data:([^;]+)/);
            return {
              type: "image",
              source: {
                type: "base64",
                media_type: mimeMatch ? mimeMatch[1] : "image/png",
                data,
              },
            };
          }
          return {
            type: "image",
            source: {
              type: "url",
              url,
            },
          };
        }
        if (block.type === "thinking") return block;
        return block;
      });
    } else {
      content = msg.content;
    }

    anthropicMessages.push({
      role: convertRole(msg.role),
      content,
    });
  }

  return { system: systemPrompt, messages: anthropicMessages };
}

function convertTools(tools) {
  if (!tools || tools.length === 0) return undefined;

  return tools.map((tool) => {
    const fn = tool.function || tool;
    return {
      name: fn.name,
      description: fn.description || "",
      input_schema: fn.parameters || { type: "object" },
    };
  });
}

function parseThinkingConfig(request) {
  const model = request.model || "";
  const isThinkingModel = model.includes("thinking") || model.includes("k2");
  if (!isThinkingModel) return undefined;

  const thinkingConfig = {
    type: "enabled",
    budget_tokens: 10000,
  };

  if (request.thinking && typeof request.thinking === "object") {
    thinkingConfig.budget_tokens = request.thinking.budget_tokens || thinkingConfig.budget_tokens;
  }

  return thinkingConfig;
}

class KimiAnthropicTransformer {
  name = "kimi-anthropic";

  constructor(options) {
    this.options = options || {};
    this.baseUrl = this.options.baseUrl || "https://api.kimi.com/coding/";
    this.apiKey = this.options.apiKey || process.env.ANTHROPIC_API_KEY;
  }

  async transformRequestIn(request, provider) {
    const { system, messages } = convertMessages(request.messages || []);
    const tools = convertTools(request.tools);
    const thinking = parseThinkingConfig(request);

    const resolvedModel = request.model || provider?.models?.[0] || "kimi-for-coding";

    const body = {
      model: resolvedModel,
      messages,
      max_tokens: request.max_tokens || 4096,
      stream: request.stream || false,
    };

    if (system) body.system = system;
    if (tools) body.tools = tools;
    if (thinking) body.thinking = thinking;
    if (request.temperature !== undefined) body.temperature = request.temperature;
    if (request.top_p !== undefined) body.top_p = request.top_p;
    if (request.top_k !== undefined) body.top_k = request.top_k;
    if (request.stop) body.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];

    const baseUrl = provider?.api_base_url || this.options.baseUrl || this.baseUrl;
    const cleanBaseUrl = baseUrl.replace(/\/v1\/messages\/?$/, "/");
    const url = new URL("./v1/messages", cleanBaseUrl);

    const headers = {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
    };
    if (body.stream) {
      headers.Accept = "text/event-stream";
    }

    const apiKey = provider?.api_key || this.options.apiKey || this.apiKey;
    if (apiKey) headers["x-api-key"] = apiKey;

    return {
      body,
      config: {
        url,
        headers,
      },
    };
  }

  async transformResponseOut(response) {
    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) return this._transformJsonResponse(response);
    if (contentType.includes("text/event-stream")) return this._transformStreamResponse(response);
    return response;
  }

  async _transformJsonResponse(response) {
    const data = await response.json();

    const openAIResponse = {
      id: data.id || `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: data.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "",
            tool_calls: undefined,
          },
          finish_reason: this._mapStopReason(data.stop_reason),
        },
      ],
      usage: {
        prompt_tokens: data.usage?.input_tokens || 0,
        completion_tokens: data.usage?.output_tokens || 0,
        total_tokens: (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0),
      },
    };

    const textParts = [];
    const toolCalls = [];
    for (const block of data.content || []) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "thinking") {
        if (!openAIResponse.choices[0].message.reasoning_content) {
          openAIResponse.choices[0].message.reasoning_content = "";
        }
        openAIResponse.choices[0].message.reasoning_content += block.thinking;
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input || {}),
          },
        });
      }
    }

    openAIResponse.choices[0].message.content = textParts.join("");
    if (toolCalls.length > 0) openAIResponse.choices[0].message.tool_calls = toolCalls;

    return new Response(JSON.stringify(openAIResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: new Headers({ "Content-Type": "application/json" }),
    });
  }

  _transformStreamResponse(response) {
    if (!response.body) return response;

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const self = this;

    let messageId = "";
    let model = "";
    let currentBlockIndex = 0;
    let currentBlockType = null;
    let toolCallsBuffer = [];
    let inputTokens = 0;
    let outputTokens = 0;
    let hasEmittedRole = false;

    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body.getReader();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("event:")) continue;
              if (!line.startsWith("data:")) continue;

              const jsonStr = line.slice(5).trim();
              if (!jsonStr) continue;

              try {
                const event = JSON.parse(jsonStr);
                const chunks = self._processSSEEvent(event, {
                  messageId,
                  model,
                  currentBlockIndex,
                  currentBlockType,
                  toolCallsBuffer,
                  inputTokens,
                  outputTokens,
                  hasEmittedRole,
                });

                for (const chunk of chunks) {
                  if (chunk._state) {
                    messageId = chunk._state.messageId || messageId;
                    model = chunk._state.model || model;
                    currentBlockIndex = chunk._state.currentBlockIndex ?? currentBlockIndex;
                    currentBlockType = chunk._state.currentBlockType ?? currentBlockType;
                    toolCallsBuffer = chunk._state.toolCallsBuffer || toolCallsBuffer;
                    inputTokens = chunk._state.inputTokens ?? inputTokens;
                    outputTokens = chunk._state.outputTokens ?? outputTokens;
                    hasEmittedRole = chunk._state.hasEmittedRole ?? hasEmittedRole;
                    continue;
                  }

                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
                }
              } catch {
                // ignore invalid JSON
              }
            }
          }
        } catch (err) {
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: new Headers({
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      }),
    });
  }

  _processSSEEvent(event, state) {
    const chunks = [];
    switch (event.type) {
      case "message_start": {
        const message = event.message || {};
        state.messageId = message.id || `chatcmpl-${Date.now()}`;
        state.model = message.model || "";
        state.inputTokens = message.usage?.input_tokens || 0;

        chunks.push({
          id: state.messageId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: state.model,
          choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
        });
        state.hasEmittedRole = true;
        chunks.push({ _state: state });
        break;
      }

      case "content_block_start": {
        const block = event.content_block || {};
        state.currentBlockIndex = event.index ?? state.currentBlockIndex;
        state.currentBlockType = block.type;

        if (block.type === "tool_use") {
          state.toolCallsBuffer.push({
            id: block.id,
            type: "function",
            function: { name: block.name, arguments: "" },
          });

          chunks.push({
            id: state.messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: state.model,
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index: state.toolCallsBuffer.length - 1,
                      id: block.id,
                      type: "function",
                      function: { name: block.name, arguments: "" },
                    },
                  ],
                },
                finish_reason: null,
              },
            ],
          });
        }

        chunks.push({ _state: state });
        break;
      }

      case "content_block_delta": {
        const delta = event.delta || {};

        if (delta.type === "text_delta") {
          chunks.push({
            id: state.messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: state.model,
            choices: [{ index: 0, delta: { content: delta.text || "" }, finish_reason: null }],
          });
        } else if (delta.type === "thinking_delta") {
          chunks.push({
            id: state.messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: state.model,
            choices: [{ index: 0, delta: { reasoning_content: delta.thinking || "" }, finish_reason: null }],
          });
        } else if (delta.type === "input_json_delta") {
          const toolIndex = state.toolCallsBuffer.length - 1;
          if (toolIndex >= 0) {
            state.toolCallsBuffer[toolIndex].function.arguments += delta.partial_json || "";
            chunks.push({
              id: state.messageId,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: state.model,
              choices: [
                {
                  index: 0,
                  delta: { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json || "" } }] },
                  finish_reason: null,
                },
              ],
            });
          }
        }

        chunks.push({ _state: state });
        break;
      }

      case "content_block_stop": {
        state.currentBlockType = null;
        chunks.push({ _state: state });
        break;
      }

      case "message_delta": {
        const messageDelta = event.delta || {};
        state.outputTokens = event.usage?.output_tokens || state.outputTokens;

        chunks.push({
          id: state.messageId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: state.model,
          choices: [{ index: 0, delta: {}, finish_reason: this._mapStopReason(messageDelta.stop_reason) }],
          usage: {
            prompt_tokens: state.inputTokens,
            completion_tokens: state.outputTokens,
            total_tokens: state.inputTokens + state.outputTokens,
          },
        });
        chunks.push({ _state: state });
        break;
      }

      case "message_stop":
      case "ping":
      case "error":
      default:
        break;
    }
    return chunks;
  }

  _mapStopReason(stopReason) {
    switch (stopReason) {
      case "end_turn":
        return "stop";
      case "max_tokens":
        return "length";
      case "tool_use":
        return "tool_calls";
      case "stop_sequence":
        return "stop";
      default:
        return stopReason || null;
    }
  }
}

module.exports = KimiAnthropicTransformer;
