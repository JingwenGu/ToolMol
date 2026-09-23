#!/usr/bin/env bash
# Serves a local model as an OpenAI-compatible chat-completions endpoint via vLLM, with
# tool calling enabled - a drop-in local replacement for a remote endpoint (Cerebras,
# DeepInfra, etc.) that ToolMolAgent (--llm_backend api) can point at through --llm_base_url.
#
# Requires `pip install vllm` and a GPU with enough VRAM for the chosen model - see
# per-model notes below for how many GPUs / how much memory that means in practice.
#
# Usage:
#   ./serve_local_vllm.sh [model_id] [port] [tensor_parallel_size] [extra vllm args...]
#
#   Qwen3-4B-Instruct-2507 (Hermes-style <tool_call> tags, fits on 1 GPU):
#     ./serve_local_vllm.sh Qwen/Qwen3-4B-Instruct-2507 8000 1
#
#   gpt-oss-120b (OpenAI's Harmony response format; ~60-80GB VRAM floor per vLLM's own
#   docs, but that's tight - TP=2 on 2x A100-40GB OOM'd during weight loading here, TP=4
#   on 4x A100-40GB (160GB combined) loaded fine, ~15-40min cold start depending on the
#   backing storage's read speed):
#     ./serve_local_vllm.sh openai/gpt-oss-120b 8000 4
#
#   IMPORTANT for gpt-oss: use --llm_backend responses, not the default "api". Confirmed
#   against vLLM 0.28.0 that gpt-oss's tool_calls never come back populated over
#   /v1/chat/completions - the model's intent to call a tool only shows up in an
#   unstructured `reasoning` string, with content=null and finish_reason="stop" (matches
#   known vLLM issues #22578, #22337). /v1/responses, vLLM's native Harmony-format
#   endpoint, is what actually surfaces tool_calls - see ToolMolAgent's "responses" backend
#   in agent.py (_create_responses/_messages_to_responses_input/_parse_responses_output).
#
# Then, in another shell:
#   export OPENAI_API_KEY=not-needed   # vLLM ignores the key value unless --api-key was passed above
#   python run.py toolmol --mol_lm ToolMol \
#       --llm_model openai/gpt-oss-120b \
#       --llm_backend responses \
#       --llm_base_url http://localhost:8000/v1 \
#       ...
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${2:-8000}"
TP="${3:-1}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA_ARGS=("$@")

# Per-model-family tool-calling flags. Qwen models emit plain <tool_call> tags and need
# vLLM's Hermes parser turned on explicitly for /v1/chat/completions. gpt-oss emits
# OpenAI's Harmony format instead - passing --tool-call-parser hermes there would be
# wrong, and no chat-completions parser fixes it anyway (see the top-of-file note: use
# --llm_backend responses for gpt-oss, not a --tool-call-parser flag here).
if [[ "$MODEL" == Qwen/* ]]; then
    FAMILY_ARGS=(--enable-auto-tool-choice --tool-call-parser hermes)
else
    FAMILY_ARGS=()
fi

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    "${FAMILY_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
