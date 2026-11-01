#!/usr/bin/bash

src=$(realpath $1)
adapter=$(realpath $2)

echo src: ${src}
echo adapter: ${adapter}

cwd=$(
  cd $(dirname $0)
  pwd
)

if [ $# -lt 1 ]; then
  echo $#
  exit 1
fi

echo exporting ${adapter}
stem=${adapter#*checkpoints/}
stem=$(echo $stem | sed 's/\//#/g')
python ${cwd}/../deps/LlamaFactory/scripts/qwen_omni_merge.py merge_lora \
  --model_path="${src}" \
  --lora_path=${adapter} \
  --save_path=${cwd}"/../exported_models/"${stem}
