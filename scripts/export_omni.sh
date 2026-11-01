#!/usr/bin/bash

size=${MODEL_SIZE:-7B}

echo MODEL SIZE: ${size}

cwd=$(
  cd $(dirname $0)
  pwd
)

if [ $# -lt 1 ]; then
  echo $#
  exit 1
fi

for ckpt_glob in $@; do
  for ckpt in ${cwd}"/../checkpoints/"${ckpt_glob}; do
    echo exporting ${ckpt}
    stem=${ckpt##*/}
    python ${cwd}/../deps/LlamaFactory/scripts/qwen_omni_merge.py merge_lora \
      --model_path="${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-Omni base model directory}" \
      --lora_path=${ckpt} \
      --save_path=${cwd}"/../exported_models/"${stem}
  done
done
