# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

No build, lint, or test commands are defined in this repository (no README or package/project manifests found). If you add tooling, document the commands here.

## Architecture overview

This repository is a configuration-driven request router with pluggable transformers:

- **Config**: `/home/dongjin/.claude-code-router/config.json` defines logging settings, default provider behavior, and a list of providers. Each provider specifies an API endpoint and a pipeline of transformer plugins.
- **Transformers**: `/home/dongjin/.claude-code-router/plugins/` contains transformer modules (e.g., `claude-anthropic.js`, `gemini-cli.js`, `chatgpt-oauth.js`). These modules translate OpenAI-style messages/tools into the target provider’s request format and convert responses/streams back into OpenAI-like completions.
- **Runtime outputs**: `/home/dongjin/.claude-code-router/logs/` holds router logs.

Typical flow: a request is routed based on provider config → transformer converts messages/tools/schema → request forwarded to provider endpoint → response (JSON or SSE stream) is decoded and re-encoded into OpenAI-style `choices` with tool calls/content blocks.