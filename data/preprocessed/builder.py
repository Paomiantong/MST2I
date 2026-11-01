# -*- coding: utf-8 -*-
"""
平行数据（x -> en-US）加载脚本（config 名使用安全格式：<src_lang>__to__en-US）
- 目录：data/preprocessed/<locale>/<split>
- 仅支持：name="<src_lang>__to__en-US"（如 "fr-FR__to__en-US"）
- 对齐：严格按 id 一一对齐（若缺失/重复会抛异常）
- 特征：除 id 与 intent_str 外，其余列在保留原名基础上加 src_/tgt_ 前缀
- 音频：将 audio/audio_path 统一为**绝对路径字符串**输出（不解码波形）
"""

import os
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import datasets

_VERSION = datasets.Version("3.2.0")

_DATA_ROOT = Path(os.environ.get("MST2I_PREPROCESSED_DIR", str(Path(__file__).resolve().parent)))
_TGT_LANG = "en-US"  # 目标语言固定 en-US
_NAME_DELIM = "__to__"  # 安全分隔符（避免 "->" 非法字符）


# ------------------ 工具 ------------------


def _lang_dir(lang: str) -> str:
    return os.path.join(_DATA_ROOT, lang)


def _list_splits(lang: str) -> List[str]:
    d = _lang_dir(lang)
    if not os.path.isdir(d):
        return []
    return sorted([s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))])


def _abs_path(split_dir: str, p: str) -> str:
    p = p or ""
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(split_dir, p))


def _std_split_name(raw: str):
    mapping = {
        "train": datasets.Split.TRAIN,
        "validation": datasets.Split.VALIDATION,
        "test": datasets.Split.TEST,
        "train_115": datasets.Split("train_115"),
    }
    return mapping.get(raw, datasets.Split(raw))


def _parse_pair_name(name: str) -> Tuple[str, str]:
    """解析 <src>__to__en-US；并拒绝包含非法箭头形式。"""
    if "->" in name:
        raise ValueError(
            f"Config name contains illegal characters: '->'. "
            f"Use '<src_lang>{_NAME_DELIM}{_TGT_LANG}' instead."
        )
    if _NAME_DELIM not in name:
        # 兼容：若只给了 src（如 "fr-FR"），默认配对到 en-US
        src = name.strip()
        tgt = _TGT_LANG
        return src, tgt
    src, tgt = [x.strip() for x in name.split(_NAME_DELIM, 1)]
    if tgt != _TGT_LANG:
        raise ValueError(f"Only '{_TGT_LANG}' is supported as target, got: {tgt}")
    return src, tgt


def _ensure_audio_and_paths_as_strings(row: Dict, split_dir: str) -> Dict:
    """将 audio/audio_path 归一化为绝对路径字符串；其余值保持原值（后续统一转为字符串写出）。"""
    split_dir = _DATA_ROOT / "audio" / split_dir  # ty:ignore[invalid-assignment]

    out = {}
    for k, v in row.items():
        if k == "audio" and isinstance(v, dict) and "path" in v:
            out[k] = _abs_path(split_dir, v["path"])
        elif k == "audio_path":
            out[k] = _abs_path(split_dir, str(v or ""))
        else:
            out[k] = v
    return out


# ------------------ 配置：仅 <src>__to__en-US ------------------


class PairConfig(datasets.BuilderConfig):
    def __init__(self, name: str, src_lang: str, tgt_lang: str, **kwargs):
        super().__init__(name=name, version=_VERSION, **kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


def _build_pair_configs() -> List[PairConfig]:
    langs = [d for d in os.listdir(_DATA_ROOT) if os.path.isdir(_lang_dir(d))]
    langs.sort()
    langs.remove("audio")
    cfgs: List[PairConfig] = []
    for src in langs:
        if src == _TGT_LANG:
            continue
        if os.path.isdir(_lang_dir(src)) and os.path.isdir(_lang_dir(_TGT_LANG)):
            cfgs.append(
                PairConfig(
                    name=f"{src}",
                    src_lang=src,
                    tgt_lang=_TGT_LANG,
                    description=f"{src} -> {_TGT_LANG} (id-aligned, prefixed features)",
                )
            )
    if not cfgs:
        # 占位项，便于首次填充目录前本地调试
        cfgs = [
            PairConfig(
                name=f"xx-XX{_NAME_DELIM}{_TGT_LANG}",
                src_lang="xx-XX",
                tgt_lang=_TGT_LANG,
                description="placeholder",
            )
        ]
    return cfgs


# ------------------ 构建器 ------------------


class ExportedParallelX2EN(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PairConfig
    BUILDER_CONFIGS = _build_pair_configs()
    DEFAULT_CONFIG_NAME = (
        BUILDER_CONFIGS[0].name if BUILDER_CONFIGS else f"xx-XX{_NAME_DELIM}{_TGT_LANG}"
    )

    def _info(self) -> datasets.DatasetInfo:
        # 动态推断 schema：探测一个公共 split 的列集合（不含 id/intent_str），并生成 src_/tgt_ 前缀列
        src_lang, tgt_lang = self.config.src_lang, self.config.tgt_lang  # ty:ignore[possibly-missing-attribute]
        src_dir, tgt_dir = _lang_dir(src_lang), _lang_dir(tgt_lang)
        src_splits = _list_splits(src_lang)
        tgt_splits = _list_splits(tgt_lang)
        common = sorted(set(src_splits) & set(tgt_splits))
        if not common:
            features = datasets.Features(
                {"id": datasets.Value("string"), "intent_str": datasets.Value("string")}
            )
            return datasets.DatasetInfo(
                description="Parallel x->en-US (id-aligned).",
                features=features,
                version=_VERSION,
            )

        probe = common[0]
        src_cols = list(
            datasets.load_from_disk(os.path.join(src_dir, probe)).column_names
        )
        tgt_cols = list(
            datasets.load_from_disk(os.path.join(tgt_dir, probe)).column_names
        )

        feat = {
            "id": datasets.Value("string"),
            "intent_str": datasets.Value("string"),
            "slot_method": {
                "slot": datasets.Sequence(datasets.Value("string")),
                "method": datasets.Sequence(datasets.Value("string")),
            },
        }
        for c in src_cols:
            if c in ("id", "intent_str", "slot_method"):
                continue
            feat[f"src_{c}"] = datasets.Value("string")
        for c in tgt_cols:
            if c in ("id", "intent_str", "intent"):
                continue
            feat[f"tgt_{c}"] = datasets.Value("string")

        return datasets.DatasetInfo(
            description="Parallel x->en-US (id-aligned) with prefixed features.",
            features=datasets.Features(feat),
            version=_VERSION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        # 从 config.name 解析 src/tgt（支持 "<src>" 简写为 "<src>__to__en-US"）
        src_lang, tgt_lang = _parse_pair_name(self.config.name)
        src_dir, tgt_dir = _lang_dir(src_lang), _lang_dir(tgt_lang)

        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Source language directory not found: {src_dir}")
        if not os.path.isdir(tgt_dir):
            raise FileNotFoundError(f"Target language directory not found: {tgt_dir}")

        common = sorted(set(_list_splits(src_lang)) & set(_list_splits(tgt_lang)))
        if not common:
            raise ValueError(f"No common splits between {src_lang} and {tgt_lang}.")

        gens = []
        for s in common:
            gens.append(
                datasets.SplitGenerator(
                    name=_std_split_name(s),
                    gen_kwargs={
                        "src_split_dir": os.path.join(src_lang, s),
                        "tgt_split_dir": os.path.join(tgt_lang, s),
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                    },
                )
            )
        return gens

    def _generate_examples(
        self,
        src_split_dir: str,
        tgt_split_dir: str,
        src_lang: str,
        tgt_lang: str,
    ) -> Iterable[Tuple[int, Dict]]:  # ty:ignore[invalid-method-override]
        # 加载（不解码音频）
        src = datasets.load_from_disk(_DATA_ROOT / src_split_dir)
        tgt = datasets.load_from_disk(_DATA_ROOT / tgt_split_dir)
        if "audio" in src.column_names:
            src = src.cast_column("audio", datasets.Audio(decode=False))
        if "audio" in tgt.column_names:
            tgt = tgt.cast_column("audio", datasets.Audio(decode=False))

        # 目标侧索引
        id2tgt: Dict[str, Dict] = {}
        for row in tgt:
            row = _ensure_audio_and_paths_as_strings(row, tgt_split_dir)  # ty:ignore[invalid-argument-type]
            _id = str(row["id"])
            if not _id:
                continue
            if _id in id2tgt:
                raise ValueError(f"Duplicated id in target: {_id}")
            id2tgt[_id] = row

        # 源侧遍历并对齐
        seen = set()
        out_idx = 0
        for srow in src:
            srow = _ensure_audio_and_paths_as_strings(srow, src_split_dir)  # ty:ignore[invalid-argument-type]
            _id = str(srow["id"])
            if not _id:
                raise ValueError("Source row without valid 'id'.")
            if _id in seen:
                raise ValueError(f"Duplicated id in source: {_id}")
            seen.add(_id)

            if _id not in id2tgt:
                raise ValueError(f"ID not found in target ({tgt_lang}): {_id}")
            trow = id2tgt[_id]

            example: Dict[str, str] = {
                "id": _id,
                "intent_str": srow["intent_str"],
                "slot_method": srow["slot_method"],
            }

            for k, v in srow.items():
                if k in ("id", "intent_str", "slot_method"):
                    continue
                example[f"src_{k}"] = "" if v is None else str(v)
            for k, v in trow.items():
                if k in ("id", "intent_str", "intent"):
                    continue
                example[f"tgt_{k}"] = "" if v is None else str(v)

            yield out_idx, example
            out_idx += 1
