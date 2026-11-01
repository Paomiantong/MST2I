#!/usr/bin/bash

cwd=$(
  cd $(dirname $0)
  pwd
)

# 解析 --dry-run 参数
DRY_RUN=false
args=()
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=true
  else
    args+=("$arg")
  fi
done
set -- "${args[@]}"

if [ $# -lt 1 ]; then
  echo Invalid Params
  exit 1
fi

export WANDB_MODE=offline

if [ "$DRY_RUN" = true ]; then
  echo "============ DRY RUN MODE ============"
fi

for conf_glob in $@; do
  for conf in "${cwd}/config/train_"${conf_glob}".yaml"; do
    echo ${conf}
    if [ ! -e ${conf} ]; then
      exit 1
    fi

    # 从 YAML 配置中提取 output_dir
    output_dir=$(grep -E '^output_dir\s*:' "${conf}" | sed 's/^output_dir\s*:\s*//' | tr -d '[:space:]"'"'")

    # 如果 output_dir 下已存在 training_loss.png，跳过训练
    if [ -n "${output_dir}" ] && [ -f "${output_dir}/training_loss.png" ]; then
      echo "Skipping ${conf}: training_loss.png already exists in ${output_dir}"
      continue
    fi

    if [ "$DRY_RUN" = true ]; then
      echo "[DRY RUN] Would run: FORCE_TORCHRUN=1 llamafactory-cli train ${conf}"
      echo "          output_dir: ${output_dir}"
    else
      FORCE_TORCHRUN=1 llamafactory-cli train "${conf}"
    fi
  done
done
