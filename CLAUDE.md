# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python 3.10 educational Agent framework implementing a modular pipeline: B1 (orchestrator) → B3 (tool execution) → B4 (LLM inference) → B5 (memory). Designed to run locally with Qwen3.5-4B. All modules are CLI entry points with a shared `common/` utilities layer.

Primary reference: `README.md` (Chinese). Server execution environment: `202.199.13.141:20356`, workdir `/root/siton-tmp/assignment_B`.

## Setup

```bash
conda create -n your_env python=3.10 -y
conda activate your_env
export PYTHONNOUSERSITE=1
pip install -r requirements.txt
```

Before running with real LLM, edit `configs/model.yaml` to set `model_name_or_path` and `tokenizer_name_or_path` to the actual Qwen3.5-4B weights location.

## Common Commands (run from `agent/code/`)

```bash
cd agent/code

# Standalone skill
python b2_run_skill.py --skill calculator --input ../data/tool_inputs/tool_input_calculator.json --outdir ../outputs/B2_skills

# Tool schema export
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --export_schema --outdir ../outputs/B3_tools
# Tool execution
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/ai_message_with_tool_calls.json --execute --outdir ../outputs/B3_tools

# LLM call (mock = no GPU; prompt_json = real model)
python b4_local_agent_llm.py --mode mock --model_config ../configs/model.yaml
python b4_local_agent_llm.py --mode prompt_json --model_config ../configs/model.yaml

# Memory load/save
python b5_memory.py load --memory_config ../configs/memory.yaml --memory_ids conv_000 --include_global
python b5_memory.py save --memory_config ../configs/memory.yaml --conversation_id conv_xxx --type conversation

# Agent runtime
python b1_agent_runtime.py --input ../data/runtime_input.json --tools_config ../configs/tools.yaml --memory_config ../configs/memory.yaml --model_config ../configs/model.yaml --llm_mode mock --outdir ../outputs/B1_runtime

# Full end-to-end demo
python run_full_demo.py --input ../data/runtime_input.json --tools_config ../configs/tools.yaml --memory_config ../configs/memory.yaml --model_config ../configs/model.yaml --llm_mode prompt_json --outdir ../outputs/full_demo
```

Exit codes: 0 = success, 1 = fatal error, 2 = argparse usage error.

## Architecture

```
B1 (Agent Runtime / orchestrator)
  ├── B5 (Memory: load & save)   → injects selected_memory into system prompt
  ├── B3 (Tool Layer: schema + exec) → executes tool_calls from AIMessage
  │     └── skills/ (5 tools)    → calculator, file_reader, local_file_search, table_analyzer, format_converter
  └── B4 (LLM: mock or local)    → generates AIMessage (content XOR tool_calls)
```

**Critical invariant:** `AIMessage` has `content` XOR `tool_calls` — never both. Enforced by `common/schemas.validate_ai_message` and `b4_local_agent_llm._candidate_to_message`. Violating this is a hard error.

**Message flow:** `system → user → assistant(tool_calls) → tool → assistant(final)`.

**Two B5 implementations exist:**
- `b5_memory.py` (simple, used by B1 in integrated mode) — file-based with char limit truncation
- `b5_test.py` (advanced, not wired into B1) — vector + LLM graph memory with `sentence-transformers` embeddings

B4 has a `_MODEL_CACHE` keyed on model/tokenizer paths and dtype — avoids reloading across multiple calls.

## Conventions

- **Structured JSON artifacts everywhere.** Skills return `SkillResult`; messages use `AIMessage`/`ToolMessage` shapes defined in `common/schemas.py`.
- **Atomic writes.** All output goes through `common/io_utils._atomic_write_text` (temp file + `os.replace`).
- **YAML config paths are relative to the config file**, resolved via `common/path_utils.resolve_from_file`.
- **Skills' `data_root`** defaults to `<project>/data`, bounded via `require_within` anti-traversal check.
- **Mock mode** for B4/B1 is first-class — enables cross-module integration testing without GPU.
- **Logs accumulate** (`*_log.jsonl`); result files are overwritten per run.
- **Output language** is Chinese throughout (prompts, fixtures, sample data).
- **Conversation IDs** restricted to `[A-Za-z0-9_.-]+` (enforced in `_safe_conversation_id`).

## Key Files

| File | Role |
|------|------|
| `code/b1_agent_runtime.py` | Orchestrator: fixture vs integrated mode, main LLM/tool loop with `max_turns` guard |
| `code/b3_tool_layer.py` | Tool schema generation (`get_tools_schema`) + execution (`execute_tool_calls`) |
| `code/b4_local_agent_llm.py` | LLM adapter: `_build_prompt_messages`, `_parse_model_output` (3 fallback strategies), `_MODEL_CACHE` |
| `code/b5_memory.py` | Simple memory: `load_memory` / `save_memory` |
| `code/b5_test.py` | Advanced memory: `LocalLLMEngine`, `MemoryStore` (vector + LLM graph) |
| `code/run_full_demo.py` | One-command end-to-end demo |
| `code/common/schemas.py` | Message/result type factories + validators (`make_ai_message`, `make_tool_message`, etc.) |
| `code/common/{io_utils,logging_utils,path_utils}.py` | Atomic IO, timestamps, path resolution |
| `skills/*.py` | 5 tool implementations (calculator, file_reader, local_file_search, table_analyzer, format_converter) |
| `configs/tools.yaml` | Tool name → module/function mapping with parameter/return schemas |
| `configs/model.yaml` | Qwen3.5-4B backend config (bfloat16, device_map=auto, local_files_only) |
| `configs/memory.yaml` | Memory directories, max_memory_chars, and (b5_test.py) graph/embedding paths |
