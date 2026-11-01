#!/usr/bin/bash

if [ $# -lt 1 ]; then
  echo Invalid Params
  exit 1
fi

cwd=$(cd $(dirname $0); pwd)

for ckpt_glob in $@; do
  for ckpt in "${cwd}/../exported_models/"${ckpt_glob}; do
    echo Infer ... ${ckpt}

    if [ ! -e ${ckpt} ]; then
      exit 1
    fi

    true && {
      python ${cwd}/infer.py \
        --template qwen2_omni \
        --model_path ${ckpt} \
        --output_dir ${cwd}/../infer_result \
        --dataset test \
        --batch_size 1
    }


  done
done

exit 0
