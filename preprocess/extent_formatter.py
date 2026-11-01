import os
from formatter_v2 import _combine as _combine_or_none, get_fixed_target
from func_util import gen_func_call
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from prompts import get_prompt

PROMPT = "pure-text-post-func"

prompt_tpl, answer_tpl = get_prompt(PROMPT)

LANGS = [
    "ar-SA",
    "de-DE",
    "es-ES",
    "fr-FR",
    "hu-HU",
    "ko-KR",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ru-RU",
    "tr-TR",
    "vi-VN",
]


def _format(ds):
    result = []
    for sample in ds:
        meta = sample.get("metadata", {}) or {}
        result.append(
            {
                "instruction": prompt_tpl.safe_substitute(**sample).strip() + "\n",
                "input": "",
                "output": answer_tpl.safe_substitute(**sample).strip(),
                "audios": [],
                "locale": sample["src_locale"],
                "src_utt": sample["src_utt"],
                **meta,
            }
        )
    return result


def get_dataset(lang):
    ds = (
        load_dataset(
            "csv",
            data_files={
                "train": f"../data/sampled_massive/{lang}/train_full.csv",
                "validation": f"../data/sampled_massive/{lang}/validation.csv",
            },
        )
        .map(lambda x: {"slot_method": eval(x["slot_method"])})
        .map(
            lambda x: get_fixed_target(
                x["src_annot_utt"],
                x["tgt_annot_utt"],
                x["slot_method"],
            ),
        )
    )
    return ds


def post_proc(data, path):
    data = data.map(
        gen_func_call,
        batched=True,
        num_proc=16,
    )
    pd.DataFrame(_format(data)).to_json(
        str(path), lines=True, orient="records", index=False
    )


target_root = Path("../data/formatted/+post-func")
eval_root = target_root / "eval"
train_root = target_root / "train"
os.makedirs(str(eval_root), exist_ok=True)
os.makedirs(str(train_root), exist_ok=True)

# --- 按 locale 收集各 split 的数据集 ---
dev_parts = []
train_patrs = []

for locale in LANGS:
    # 这里直接保存为 Dataset 列表，稍后一次性 concatenate
    ds_dict = get_dataset(locale)
    train_patrs.append(ds_dict["train"])
    dev_parts.append(ds_dict["validation"])

devset = _combine_or_none(dev_parts)
trainset = _combine_or_none(train_patrs)

post_proc(trainset, train_root / "extent-train.jsonl")
