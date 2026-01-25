/**
 * Claude Anthropic Transformer
 * 
 * Transforms requests/responses between OpenAI format and Anthropic Claude API format.
 * Supports:
 * - Extended thinking (claude-opus-4-5-thinking, claude-sonnet-4-5-thinking)
 * - Tool/Function calling
 * - Streaming SSE responses
 * 
 * Based on: https://github.com/badrisnarayanan/antigravity-claude-proxy
 */

/**
 * Convert OpenAI role to Anthropic role
 * @param {string} role - OpenAI role (system, user, assistant, tool)
 * @returns {string} Anthropic role
 */
function convertRole(role) {
  if (role === "system") return "user"; // System handled separately
  if (role === "tool") return "user";
  if (role === "function") return "user";
  return role;
}

/**
 * Convert OpenAI messages to Anthropic format
 * @param {Array} messages - OpenAI format messages
 * @returns {{system: string|null, messages: Array}} Anthropic format
 */
function convertMessages(messages) {
  let systemPrompt = null;
  const anthropicMessages = [];

  for (const msg of messages) {
    // Extract system message
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

    // Convert tool/function results
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

    // Convert assistant messages with tool_calls
    if (msg.role === "assistant" && msg.tool_calls) {
      const content = [];
      
      // Add text content if present
      if (msg.content) {
        content.push({
          type: "text",
          text: typeof msg.content === "string" ? msg.content : msg.content,
        });
      }

      // Add tool_use blocks
      for (const toolCall of msg.tool_calls) {
        content.push({
          type: "tool_use",
          id: toolCall.id,
          name: toolCall.function?.name || toolCall.name,
          input: typeof toolCall.function?.arguments === "string"
            ? JSON.parse(toolCall.function.arguments || "{}")
            : toolCall.function?.arguments || {},
        });
      }

      anthropicMessages.push({
        role: "assistant",
        content,
      });
      continue;
    }

    // Convert regular messages
    let content;
    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      // Convert content blocks
      content = msg.content.map((block) => {
        if (block.type === "text") {
          return { type: "text", text: block.text };
        }
        if (block.type === "image_url") {
          const url = block.image_url?.url || block.image_url;
          if (url.startsWith("data:")) {
            const [meta, data] = url.split(",");
            const mimeMatch = meta.match(/data:([^;]+)/);
            return {
              type: "image",
              source: {
                type: "base64",
                media_type: mimeMatch ? mimeMatch[1] : "image/png",
                data: data,
              },
            };
          }
          return {
            type: "image",
            source: {
              type: "url",
              url: url,
            },
          };
        }
        // Pass through thinking blocks
        if (block.type === "thinking") {
          return block;
        }
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

/**
 * Convert OpenAI tools to Anthropic format
 * @param {Array} tools - OpenAI format tools
 * @returns {Array} Anthropic format tools
 */
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

/**
 * Parse thinking configuration
 * @param {Object} request - Original request
 * @returns {Object|undefined} Anthropic thinking config
 */
function parseThinkingConfig(request) {
  // Check if model supports thinking
  const model = request.model || "";
  const isThinkingModel = model.includes("thinking");

  if (!isThinkingModel) return undefined;

  // Default thinking config for thinking models
  const thinkingConfig = {
    type: "enabled",
    budget_tokens: 10000, // Default budget
  };

  // Override with explicit config if provided
  if (request.thinking) {
    if (typeof request.thinking === "object") {
      thinkingConfig.budget_tokens = request.thinking.budget_tokens || thinkingConfig.budget_tokens;
    }
  }

  return thinkingConfig;
}

class ClaudeAnthropicTransformer {
  name = "claude-anthropic";

  constructor(options) {
    this.options = options || {};
    this.baseUrl = this.options.baseUrl || "https://api.anthropic.com";
    this.apiKey = this.options.apiKey || process.env.ANTHROPIC_API_KEY;
  }

  /**
   * Transform incoming request from OpenAI format to Anthropic format
   * @param {Object} request - OpenAI format request
   * @param {Object} provider - Provider configuration
   * @returns {Object} Transformed request with body and config
   */
  async transformRequestIn(request, provider) {
    const { system, messages } = convertMessages(request.messages || []);
    const tools = convertTools(request.tools);
    const thinking = parseThinkingConfig(request);

    // Build Anthropic request body
    const body = {
      model: request.model || "claude-sonnet-4-20250514",
      messages,
      max_tokens: request.max_tokens || 4096,
      stream: request.stream || false,
    };

    // Add optional fields
    if (system) {
      body.system = system;
    }
    if (tools) {
      body.tools = tools;
    }
    if (thinking) {
      body.thinking = thinking;
    }
    if (request.temperature !== undefined) {
      body.temperature = request.temperature;
    }
    if (request.top_p !== undefined) {
      body.top_p = request.top_p;
    }
    if (request.top_k !== undefined) {
      body.top_k = request.top_k;
    }
    if (request.stop) {
      body.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];
    }

    // Build headers and URL
    console.log('[claude-anthropic] provider:', JSON.stringify(provider, null, 2));
    const baseUrl = provider?.api_base_url || this.options.baseUrl || this.baseUrl;
    console.log('[claude-anthropic] baseUrl:', baseUrl);
    // Remove trailing /v1/messages if present, then add it back
    const cleanBaseUrl = baseUrl.replace(/\/v1\/messages\/?$/, '');
    const url = new URL(`${cleanBaseUrl}/v1/messages`);
    console.log('[claude-anthropic] final url:', url.toString());
    
    const headers = {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
    };

    // Add API key if available
    const apiKey = provider?.api_key || this.options.apiKey || this.apiKey;
    if (apiKey) {
      headers["x-api-key"] = apiKey;
    }

    return {
      body,
      config: {
        url,
        headers,
      },
    };
  }

  /**
   * Transform outgoing response from Anthropic format to OpenAI format
   * @param {Response} response - Fetch Response object
   * @returns {Response} Transformed response
   */
  async transformResponseOut(response) {
    const contentType = response.headers.get("Content-Type") || "";

    // Handle JSON (non-streaming) response
    if (contentType.includes("application/json")) {
      return this._transformJsonResponse(response);
    }

    // Handle SSE streaming response
    if (contentType.includes("text/event-stream")) {
      return this._transformStreamResponse(response);
    }

    // Pass through unknown content types
    return response;
  }

  /**
   * Transform non-streaming JSON response
   * @param {Response} response - Original response
   * @returns {Response} Transformed response
   */
  async _transformJsonResponse(response) {
    const data = await response.json();

    // Build OpenAI format response
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

    // Process content blocks
    const textParts = [];
    const toolCalls = [];
    let toolCallIndex = 0;

    for (const block of data.content || []) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "thinking") {
        // Include thinking in a special format (OpenAI doesn't have native thinking)
        // Some clients expect reasoning_content
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
        toolCallIndex++;
      }
    }

    openAIResponse.choices[0].message.content = textParts.join("");
    if (toolCalls.length > 0) {
      openAIResponse.choices[0].message.tool_calls = toolCalls;
    }

    return new Response(JSON.stringify(openAIResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: new Headers({
        "Content-Type": "application/json",
      }),
    });
  }

  /**
   * Transform streaming SSE response
   * @param {Response} response - Original SSE response
   * @returns {Response} Transformed SSE response
   */
  _transformStreamResponse(response) {
    if (!response.body) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const self = this;

    // State for tracking stream
    let messageId = "";
    let model = "";
    let currentBlockIndex = 0;
    let currentBlockType = null;
    let toolCallsBuffer = []; // Buffer for tool calls
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
              // Emit final [DONE] message
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("event:")) {
                // Skip event type line, we process data lines
                continue;
              }

              if (!line.startsWith("data:")) {
                continue;
              }

              const jsonStr = line.slice(5).trim();
              if (!jsonStr) continue;

              try {
                const event = JSON.parse(jsonStr);
                const chunks = self._processSSEEvent(
                  event,
                  {
                    messageId,
                    model,
                    currentBlockIndex,
                    currentBlockType,
                    toolCallsBuffer,
                    inputTokens,
                    outputTokens,
                    hasEmittedRole,
                  }
                );

                // Update state from returned chunks
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
              } catch (parseError) {
                // Skip invalid JSON
                if (self.options.debug) {
                  console.error("[ClaudeAnthropic] Parse error:", parseError.message, jsonStr);
                }
              }
            }
          }
        } catch (error) {
          controller.error(error);
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

  /**
   * Process a single SSE event and return OpenAI format chunks
   * @param {Object} event - Anthropic SSE event
   * @param {Object} state - Current stream state
   * @returns {Array} Array of OpenAI chunks (may include state updates)
   */
  _processSSEEvent(event, state) {
    const chunks = [];
    const eventType = event.type;

    switch (eventType) {
      case "message_start": {
        const message = event.message || {};
        state.messageId = message.id || `chatcmpl-${Date.now()}`;
        state.model = message.model || "";
        state.inputTokens = message.usage?.input_tokens || 0;

        // Emit initial chunk with role
        chunks.push({
          id: state.messageId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: state.model,
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "" },
              finish_reason: null,
            },
          ],
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
          // Start a new tool call
          state.toolCallsBuffer.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments: "",
            },
          });

          // Emit tool call start
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
                      function: {
                        name: block.name,
                        arguments: "",
                      },
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
          // Regular text content
          chunks.push({
            id: state.messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: state.model,
            choices: [
              {
                index: 0,
                delta: { content: delta.text || "" },
                finish_reason: null,
              },
            ],
          });
        } else if (delta.type === "thinking_delta") {
          // Thinking content - emit as reasoning_content for compatible clients
          chunks.push({
            id: state.messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: state.model,
            choices: [
              {
                index: 0,
                delta: { reasoning_content: delta.thinking || "" },
                finish_reason: null,
              },
            ],
          });
        } else if (delta.type === "input_json_delta") {
          // Tool call arguments
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
                  delta: {
                    tool_calls: [
                      {
                        index: toolIndex,
                        function: {
                          arguments: delta.partial_json || "",
                        },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            });
          }
        } else if (delta.type === "signature_delta") {
          // Thinking signature - skip for OpenAI format (not supported)
          // Could store for future use if needed
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

        // Emit final chunk with finish_reason
        chunks.push({
          id: state.messageId,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: state.model,
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: this._mapStopReason(messageDelta.stop_reason),
            },
          ],
          usage: {
            prompt_tokens: state.inputTokens,
            completion_tokens: state.outputTokens,
            total_tokens: state.inputTokens + state.outputTokens,
          },
        });
        chunks.push({ _state: state });
        break;
      }

      case "message_stop": {
        // Stream complete, [DONE] will be emitted by the stream handler
        break;
      }

      case "error": {
        // Pass through error
        console.error("[ClaudeAnthropic] Stream error:", event.error);
        break;
      }

      case "ping": {
        // Ignore ping events
        break;
      }

      default:
        // Unknown event type, skip
        break;
    }

    return chunks;
  }

  /**
   * Map Anthropic stop_reason to OpenAI finish_reason
   * @param {string} stopReason - Anthropic stop reason
   * @returns {string} OpenAI finish reason
   */
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

module.exports = ClaudeAnthropicTransformer;
