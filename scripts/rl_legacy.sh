#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QWEN_PATH="${QWEN_PATH:-${PROJECT_ROOT}/exported_models/qwen2_5-omni-post-func-ext-fewshot-de}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/formatted/+post-func/train/de.jsonl,${PROJECT_ROOT}/data/formatted/+post-func/train/115.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5-omni-post-func-ext-fewshot-de-RL-Full}"
export WANDB_PROJECT="${WANDB_PROJECT:-MST2I}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$SCRIPT_DIR"
torchrun --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="${MASTER_PORT:-12345}" \
  step_grop.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$QWEN_PATH" \
  --dataset_name "$DATASET_DIR" \
  --config rl.yml
