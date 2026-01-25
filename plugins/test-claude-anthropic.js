/**
 * Test script for claude-anthropic.js transformer
 */

const ClaudeAnthropicTransformer = require('./claude-anthropic.js');

// Mock SSE response data (from user's example)
const mockSSEData = `event: message_start
data: {"type":"message_start","message":{"id":"msg_6f3735ee05ca75c34aa45da8bcce1467","type":"message","role":"assistant","content":[],"model":"claude-opus-4-5-thinking","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":193,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"안녕하세요!"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" 👋\\n\\n저"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"는 Antigravity입니다."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":108,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}

event: message_stop
data: {"type":"message_stop"}`;

// Test request transformation
async function testRequestTransform() {
  console.log("=== Testing Request Transformation ===\n");
  
  const transformer = new ClaudeAnthropicTransformer({
    baseUrl: "http://localhost:8080",
    apiKey: "test-key"
  });

  const openAIRequest = {
    model: "claude-opus-4-5-thinking",
    max_tokens: 1024,
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "안녕?" }
    ],
    stream: true
  };

  const result = await transformer.transformRequestIn(openAIRequest);
  
  console.log("Input (OpenAI format):");
  console.log(JSON.stringify(openAIRequest, null, 2));
  console.log("\nOutput (Anthropic format):");
  console.log(JSON.stringify(result.body, null, 2));
  console.log("\nConfig:");
  console.log("URL:", result.config.url.toString());
  console.log("Headers:", result.config.headers);
  console.log("\n✅ Request transformation passed!\n");
}

// Test SSE event processing
function testSSEProcessing() {
  console.log("=== Testing SSE Event Processing ===\n");
  
  const transformer = new ClaudeAnthropicTransformer();
  
  const events = [
    {
      type: "message_start",
      message: {
        id: "msg_123",
        model: "claude-opus-4-5-thinking",
        usage: { input_tokens: 100 }
      }
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "" }
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: "Let me think about this..." }
    },
    {
      type: "content_block_stop",
      index: 0
    },
    {
      type: "content_block_start",
      index: 1,
      content_block: { type: "text", text: "" }
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "text_delta", text: "안녕하세요!" }
    },
    {
      type: "content_block_stop",
      index: 1
    },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 50 }
    },
    {
      type: "message_stop"
    }
  ];

  const state = {
    messageId: "",
    model: "",
    currentBlockIndex: 0,
    currentBlockType: null,
    toolCallsBuffer: [],
    inputTokens: 0,
    outputTokens: 0,
    hasEmittedRole: false
  };

  console.log("Processing Claude SSE events:\n");
  
  for (const event of events) {
    console.log(`Input event: ${event.type}`);
    const chunks = transformer._processSSEEvent(event, state);
    
    for (const chunk of chunks) {
      if (chunk._state) {
        // Update state
        Object.assign(state, chunk._state);
        continue;
      }
      console.log("  → OpenAI chunk:", JSON.stringify(chunk, null, 2).substring(0, 200) + "...");
    }
    console.log("");
  }

  console.log("✅ SSE processing passed!\n");
}

// Test tool use transformation
function testToolUseTransform() {
  console.log("=== Testing Tool Use Transformation ===\n");
  
  const transformer = new ClaudeAnthropicTransformer();

  const state = {
    messageId: "msg_456",
    model: "claude-sonnet-4-5-thinking",
    currentBlockIndex: 0,
    currentBlockType: null,
    toolCallsBuffer: [],
    inputTokens: 0,
    outputTokens: 0,
    hasEmittedRole: true
  };

  const toolEvents = [
    {
      type: "content_block_start",
      index: 0,
      content_block: {
        type: "tool_use",
        id: "toolu_abc123",
        name: "get_weather"
      }
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: {
        type: "input_json_delta",
        partial_json: '{"location":'
      }
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: {
        type: "input_json_delta",
        partial_json: '"Seoul"}'
      }
    },
    {
      type: "content_block_stop",
      index: 0
    }
  ];

  console.log("Processing tool_use events:\n");

  for (const event of toolEvents) {
    console.log(`Input event: ${event.type}`);
    const chunks = transformer._processSSEEvent(event, state);
    
    for (const chunk of chunks) {
      if (chunk._state) {
        Object.assign(state, chunk._state);
        continue;
      }
      if (chunk.choices?.[0]?.delta?.tool_calls) {
        console.log("  → Tool call chunk:", JSON.stringify(chunk.choices[0].delta.tool_calls, null, 2));
      }
    }
    console.log("");
  }

  console.log("Final tool calls buffer:", JSON.stringify(state.toolCallsBuffer, null, 2));
  console.log("\n✅ Tool use transformation passed!\n");
}

// Run all tests
async function runTests() {
  console.log("╔════════════════════════════════════════════╗");
  console.log("║   Claude Anthropic Transformer Tests       ║");
  console.log("╚════════════════════════════════════════════╝\n");

  try {
    await testRequestTransform();
    testSSEProcessing();
    testToolUseTransform();
    
    console.log("═══════════════════════════════════════════");
    console.log("✅ All tests passed!");
    console.log("═══════════════════════════════════════════");
  } catch (error) {
    console.error("❌ Test failed:", error);
    process.exit(1);
  }
}

runTests();
