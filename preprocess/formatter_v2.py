import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import pandas as pd
from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from func_util import gen_func_call, gen_slots_str
from prompts import get_prompt

# ── 常量 ──────────────────────────────────────────────
DATA_BUILDER = "../data/preprocessed/builder.py"

# 数据集布局配置
DATASET_SPEC = [
    # ── train ──
    {
        "key": "train-fr",
        "locale": ["fr-FR"],
        "split": "train",
        "subdir": "train",
        "filename": "fr.jsonl",
    },
    {
        "key": "train-de",
        "locale": ["de-DE"],
        "split": "train",
        "subdir": "train",
        "filename": "de.jsonl",
    },
    {
        "key": "train-115",
        "locale": "ALL",
        "split": "train_115",
        "subdir": "train",
        "filename": "115.jsonl",
    },
    # ── eval ──
    {
        "key": "test",
        "locale": "ALL",
        "split": "test",
        "subdir": "eval",
        "filename": "test.jsonl",
    },
    {
        "key": "dev",
        "locale": "ALL",
        "split": "validation",
        "subdir": "eval",
        "filename": "dev.jsonl",
    },
]

# dataset_info.json 的列映射（所有数据集共用）
COLUMN_MAP = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "audios": "audios",
}

slot_pat = re.compile(r"\[(?:[^\]]+?) : (.+?)\]")


# ── 核心处理函数 ──────────────────────────────────────


def get_fixed_target(src_annot_utt: str, tgt_annot_utt: str, slot_method: dict) -> dict:
    """
    让 'localization' 类型的 slot 支持在一句话中出现多次。
    """
    result_annot = tgt_annot_utt

    for slot, method in zip(slot_method.get("slot", []), slot_method.get("method", [])):
        if method != "localization":
            continue

        pat = re.compile(rf"\[{re.escape(slot)} : ([^\]]+?)\]")
        src_entities = [m.group(1) for m in pat.finditer(src_annot_utt)]
        if not src_entities:
            continue

        it = iter(src_entities)

        def _repl(m: re.Match, _it=it) -> str:
            entity = next(_it, None)
            return f"[{slot} : {entity}]" if entity is not None else m.group(0)

        result_annot = pat.sub(_repl, result_annot)

    return {
        "fixed_tgt_annot_utt": result_annot,
        "fixed_tgt_utt": slot_pat.sub(r"\1", result_annot),
    }


def _format(ds: Dataset, prompt_tpl, answer_tpl) -> list[dict]:
    """将 Dataset 格式化为训练用的 dict 列表。"""
    result = []
    for sample in ds:
        meta = sample.get("metadata") or {}
        result.append(
            {
                "instruction": prompt_tpl.safe_substitute(**sample).strip() + "\n",
                "input": "",
                "output": answer_tpl.safe_substitute(**sample).strip(),
                "audios": [sample["src_audio"]] if "src_audio" in sample else [],
                "locale": sample["src_locale"],
                "src_utt": sample["src_utt"],
                **meta,
            }
        )
    return result


def process_and_save(data: Dataset, path: Path, prompt_tpl, answer_tpl) -> None:
    """对数据集做后处理，格式化并写出 JSONL。"""
    data = data.map(gen_func_call, batched=True, num_proc=min(16, os.cpu_count() or 1))
    data = data.map(gen_slots_str, batched=True, num_proc=min(16, os.cpu_count() or 1))
    data = data.map(lambda x: {"metadata": {"slot_method": x["slot_method"]}})

    df = pd.DataFrame(_format(data, prompt_tpl, answer_tpl))
    df.to_json(str(path), lines=True, orient="records", index=False)
    print(f"  ✓ {path}  ({len(df)} 条)")


def _combine(parts: list[Dataset]) -> Optional[Dataset]:
    """将若干 Dataset 拼接；为空则返回 None。"""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def generate_dataset_info(
    prompt_name: str, target_root: Path, specs: list[dict]
) -> Path:
    """
    根据实际生成的文件，在 target_root 下写出 dataset_info.json。

    命名规则:
      - key == "test" → 保持 "test"
      - 其余          → "{prompt_name}.{key}"
    """
    info = {}
    for spec in specs:
        out_path = target_root / spec["subdir"] / spec["filename"]
        if not out_path.exists():
            continue

        raw_key = spec["key"]
        ds_key = raw_key if raw_key == "test" else f"{prompt_name}.{raw_key}"

        info[ds_key] = {
            "file_name": f"{spec['subdir']}/{spec['filename']}",
            "columns": dict(COLUMN_MAP),
        }

    out = target_root / "dataset_info.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {out}  ({len(info)} 个数据集)")
    return out


# ── 主流程 ────────────────────────────────────────────


def main(
    prompt: str = "baseline",
    cache_dir: Optional[Union[str, Path]] = None,
    use_temp_cache: bool = True,
    extra_specs: Optional[list[dict]] = None,
):
    prompt_tpl, answer_tpl = get_prompt(prompt)
    all_locales = get_dataset_config_names(DATA_BUILDER, trust_remote_code=True)

    specs = list(DATASET_SPEC)
    if extra_specs:
        specs.extend(extra_specs)

    # ── 缓存策略 ──
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_ctx = nullcontext(str(cache_path))
    elif use_temp_cache:
        cache_ctx = TemporaryDirectory()
    else:
        cache_ctx = nullcontext(None)

    with cache_ctx as resolved_cache:

        def load(locale: str, split: str) -> Dataset:
            ds = load_dataset(
                DATA_BUILDER,
                locale,
                trust_remote_code=True,
                split=split,
                cache_dir=resolved_cache,
            )
            return ds.map(
                lambda x: get_fixed_target(
                    x["src_annot_utt"], x["tgt_annot_utt"], x["slot_method"]
                )
            )

        def load_combined(locales: list[str], split: str) -> Optional[Dataset]:
            parts = []
            for loc in locales:
                try:
                    parts.append(load(loc, split))
                except Exception as e:
                    print(f"  ⚠ 跳过 {loc}/{split}: {e}")
            return _combine(parts)

        # ── 输出目录 ──
        target_root = Path(f"../data/formatted/{prompt}")
        (target_root / "train").mkdir(parents=True, exist_ok=True)
        (target_root / "eval").mkdir(parents=True, exist_ok=True)

        # ── 逐个 spec 处理 ──
        for spec in specs:
            out_path = target_root / spec["subdir"] / spec["filename"]
            print(f"[{spec['key']}] → {out_path}")

            locale_cfg = spec["locale"]
            if locale_cfg == "ALL":
                ds = load_combined(all_locales, spec["split"])
            else:
                ds = load_combined(locale_cfg, spec["split"])

            if ds is None:
                print("  ⚠ 无数据，跳过")
                continue

            process_and_save(ds, out_path, prompt_tpl, answer_tpl)

        # ── 生成 dataset_info.json ──
        generate_dataset_info(prompt, target_root, specs)

    print("\n全部完成。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format datasets and generate dataset_info.json"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="baseline",
        help="Prompt 模板名称 (default: baseline)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="数据集缓存目录。若指定则不会自动清理。",
    )
    parser.add_argument(
        "--no-temp-cache",
        action="store_true",
        help="不使用临时缓存目录（当 --cache-dir 未设置时）",
    )
    args = parser.parse_args()

    main(
        prompt=args.prompt,
        cache_dir=args.cache_dir,
        use_temp_cache=not args.no_temp_cache,
    )
