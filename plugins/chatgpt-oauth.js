const os = require("os");
const path = require("path");
const fs = require("fs/promises");

const OAUTH_FILE = path.join(os.homedir(), ".codex", "auth.json");
const CHATGPT_CODEX_API = "https://chatgpt.com/backend-api/codex/responses";

class ChatGPTOAuthTransformer {
  name = "chatgpt-oauth";

  constructor(options) {
    this.options = options;
    try {
      this.oauth_creds = require(OAUTH_FILE);
    } catch (e) {
      console.error("Failed to load ChatGPT OAuth credentials:", e.message);
    }
  }

  /**
   * Transform incoming request from OpenAI Chat Completion format to ChatGPT Responses API format
   */
  async transformRequestIn(request, provider, context) {
    // Check if we need to refresh the token
    if (this.oauth_creds?.last_refresh) {
      const lastRefresh = new Date(this.oauth_creds.last_refresh);
      const hoursSinceRefresh = (Date.now() - lastRefresh.getTime()) / (1000 * 60 * 60);
      if (hoursSinceRefresh > 23) {
        await this.refreshToken(this.oauth_creds.tokens.refresh_token);
      }
    }

    // Convert OpenAI Chat Completion format to Responses API format
    const input = [];
    let instructions = "";

    // Process messages
    if (request.messages) {
      for (const msg of request.messages) {
        if (msg.role === "system") {
          const content = typeof msg.content === "string" 
            ? msg.content 
            : (Array.isArray(msg.content) 
                ? msg.content.map(c => c.text || "").join("\n") 
                : "");
          instructions += (instructions ? "\n" : "") + content;
        } else if (msg.role === "user" || msg.role === "assistant") {
          const content = typeof msg.content === "string"
            ? msg.content
            : (Array.isArray(msg.content)
                ? msg.content.map(c => {
                    if (c.type === "text") return c.text;
                    return "";
                  }).join("\n")
                : "");
          
          if (content) {
            input.push({ role: msg.role, content });
          }

          if (msg.role === "assistant" && Array.isArray(msg.tool_calls)) {
            for (const tool of msg.tool_calls) {
              input.push({
                type: "function_call",
                call_id: tool.id,
                name: tool.function.name,
                arguments: tool.function.arguments,
              });
            }
          }
        } else if (msg.role === "tool") {
          input.push({
            type: "function_call_output",
            call_id: msg.tool_call_id,
            output: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
          });
        }
      }
    }

    // Convert tools to Responses API format
    let tools = [];
    if (Array.isArray(request.tools)) {
      tools = request.tools.map(tool => ({
        type: "function",
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      }));
    }

    const transformedRequest = {
      model: request.model,
      instructions: instructions || "You are a helpful assistant.",
      input: input,
      store: false,
      stream: true,
    };

    if (tools.length > 0) {
      transformedRequest.tools = tools;
    }

    if (this.logger) {
      this.logger.info({ 
        type: "chatgpt-oauth transformRequestIn",
        model: transformedRequest.model,
        inputLength: input.length,
        toolsCount: tools.length
      });
    }

    return {
      body: transformedRequest,
      config: {
        url: new URL(CHATGPT_CODEX_API),
        headers: {
          Authorization: `Bearer ${this.oauth_creds.tokens.access_token}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
      },
    };
  }

  /**
   * Transform ChatGPT Responses API SSE stream to OpenAI Chat Completion SSE stream format
   * This format is then converted to Anthropic format by the built-in Anthropic transformer
   */
  async transformResponseOut(response) {
    const contentType = response.headers.get("Content-Type") || "";

    if (!response.body) {
      return response;
    }

    if (contentType.includes("application/json")) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let buffer = "";
    let messageId = "chatcmpl-" + Date.now();
    let model = "gpt-5.2-codex";
    let hasStarted = false;
    let toolCallIndex = -1;

    const self = this;
    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body.getReader();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              // Send final chunk with finish_reason
              const finalChunk = {
                id: messageId,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: [{
                  index: 0,
                  delta: {},
                  finish_reason: "stop"
                }]
              };
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            let lines = buffer.split(/\r?\n/);
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;

              if (line.startsWith("event: ")) {
                continue;
              } else if (line.startsWith("data: ")) {
                const dataStr = line.slice(5).trim();
                if (dataStr === "[DONE]") {
                  continue;
                }

                try {
                  const data = JSON.parse(dataStr);
                  const openAIChunks = self.convertToOpenAIChunks(
                    data, 
                    messageId, 
                    model,
                    hasStarted,
                    toolCallIndex
                  );
                  
                  for (const chunkData of openAIChunks) {
                    if (chunkData._internal) {
                      if (chunkData.hasStarted !== undefined) hasStarted = chunkData.hasStarted;
                      if (chunkData.model) model = chunkData.model;
                      if (chunkData.toolCallIndex !== undefined) toolCallIndex = chunkData.toolCallIndex;
                      continue;
                    }
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunkData)}\n\n`));
                  }
                } catch (e) {
                  // Skip unparseable lines
                }
              }
            }
          }
        } catch (error) {
          console.error("Stream error:", error);
          controller.error(error);
        } finally {
          try {
            reader.releaseLock();
          } catch (e) {}
          controller.close();
        }
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  }

  /**
   * Convert a ChatGPT Responses API event to OpenAI Chat Completion chunk format
   */
  convertToOpenAIChunks(data, messageId, model, hasStarted, toolCallIndex) {
    const chunks = [];
    const type = data.type;

    // Extract model from response
    if ((type === "response.created" || type === "response.in_progress") && data.response?.model) {
      chunks.push({ _internal: true, model: data.response.model, hasStarted: true });
    }

    // Initial role chunk
    if (!hasStarted && (type === "response.created" || type === "response.in_progress")) {
      chunks.push({
        id: messageId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model || model,
        choices: [{
          index: 0,
          delta: { role: "assistant", content: "" },
          finish_reason: null
        }]
      });
      chunks.push({ _internal: true, hasStarted: true });
    }

    // Text content delta
    if (type === "response.output_text.delta") {
      chunks.push({
        id: messageId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          delta: { content: data.delta || "" },
          finish_reason: null
        }]
      });
    }

    // Reasoning/thinking delta - send as reasoning_content for compatible clients
    if (type === "response.reasoning_summary_text.delta") {
      chunks.push({
        id: messageId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          delta: { reasoning_content: data.delta || "" },
          finish_reason: null
        }]
      });
    }

    // Tool call start
    if (type === "response.output_item.added" && data.item?.type === "function_call") {
      const newToolCallIndex = toolCallIndex + 1;
      chunks.push({
        id: messageId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          delta: {
            tool_calls: [{
              index: newToolCallIndex,
              id: data.item.call_id || data.item.id || `call_${Date.now()}`,
              type: "function",
              function: {
                name: data.item.name || "",
                arguments: ""
              }
            }]
          },
          finish_reason: null
        }]
      });
      chunks.push({ _internal: true, toolCallIndex: newToolCallIndex });
    }

    // Tool call arguments delta
    if (type === "response.function_call_arguments.delta" && toolCallIndex >= 0) {
      chunks.push({
        id: messageId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          delta: {
            tool_calls: [{
              index: toolCallIndex,
              function: {
                arguments: data.delta || ""
              }
            }]
          },
          finish_reason: null
        }]
      });
    }

    return chunks;
  }

  async refreshToken(refresh_token) {
    try {
      const response = await fetch("https://auth.openai.com/oauth/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
          grant_type: "refresh_token",
          refresh_token: refresh_token,
        }),
      });

      const data = await response.json();

      if (data.access_token) {
        this.oauth_creds.tokens.access_token = data.access_token;
        this.oauth_creds.tokens.id_token = data.id_token;
        if (data.refresh_token) {
          this.oauth_creds.tokens.refresh_token = data.refresh_token;
        }
        this.oauth_creds.last_refresh = new Date().toISOString();
        await fs.writeFile(OAUTH_FILE, JSON.stringify(this.oauth_creds, null, 2));
        console.log("ChatGPT OAuth token refreshed successfully");
      } else {
        console.error("Failed to refresh ChatGPT OAuth token:", data);
      }
    } catch (error) {
      console.error("Error refreshing ChatGPT OAuth token:", error);
    }
  }
}

module.exports = ChatGPTOAuthTransformer;
