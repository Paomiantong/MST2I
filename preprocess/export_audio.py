#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遍历 ../data/preprocessed 下的所有语种与 split，
将数据集中 audio 列解码后另存为音频文件到 ../data/preprocessed/audio/{lang}/{split} 下。

要求：
- 不用命令行参数，直接运行脚本即可。
- 统一采用“解码后另存”的方式，不去复制源文件。
- 导出文件名与样本中 audio['path'] 的文件名一致（若冲突则自动添加 _1, _2 后缀）。
- 若原始扩展名格式不被 soundfile 支持，则回退写出为同名 .wav（并打印提示）。

依赖：
pip install datasets soundfile
"""

import os
from typing import List, Optional, Tuple, Set, Dict

from datasets import load_from_disk, Audio, Dataset
import soundfile as sf


# 根目录（输入和输出）
PREPROCESSED_ROOT = "../data/preprocessed"
OUT_ROOT = "../data/preprocessed/audio"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_languages(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"未找到预处理根目录：{root}")
    langs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    langs.sort()
    return langs


def list_splits(lang_dir: str) -> List[str]:
    if not os.path.isdir(lang_dir):
        return []
    splits = [
        d for d in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, d))
    ]
    ok: List[str] = []
    for s in splits:
        split_dir = os.path.join(lang_dir, s)
        try:
            _ = load_from_disk(split_dir)
            ok.append(s)
        except Exception as e:
            print("[Warn]" + str(e))
            # 非 datasets 的 split 目录，忽略
            pass
    ok.sort()
    return ok


def unique_filename(dirpath: str, basename: str, used: Set[str]) -> str:
    """
    返回在 dirpath 下不冲突的文件名（含扩展名）。
    若存在重名，自动添加 _1, _2 后缀。
    used 用于本次导出的去重集合，避免同一次运行内的重名。
    """
    name, ext = os.path.splitext(basename)
    cand = basename
    k = 1
    while os.path.exists(os.path.join(dirpath, cand)) or cand in used:
        cand = f"{name}_{k}{ext}"
        k += 1
    used.add(cand)
    return cand


def safe_basename_from_path(path_or_empty: Optional[str], idx: int) -> str:
    """
    从 audio['path'] 提取原始文件名（含扩展名）。
    若为空或无法解析，则退化为 {idx:08d}.wav
    """
    if path_or_empty:
        base = os.path.basename(path_or_empty)
        if base:
            return base
    return f"{idx:08d}.wav"


def write_audio_with_fallback(
    dest_dir: str, base_name: str, arr, sr: int, used: Set[str]
) -> str:
    """
    按 base_name 的扩展名写出音频；若该扩展名不被 soundfile 支持，则回退为 .wav。
    返回实际写出的文件名。
    """
    # 首选使用原始扩展名
    dest_name = unique_filename(dest_dir, base_name, used)
    dest_path = os.path.join(dest_dir, dest_name)
    try:
        sf.write(dest_path, arr, sr)
        return dest_name
    except Exception as e:
        # 回退到 .wav
        name, _ext = os.path.splitext(base_name)
        wav_name = unique_filename(dest_dir, f"{name}.wav", used)
        wav_path = os.path.join(dest_dir, wav_name)
        sf.write(wav_path, arr, sr)
        print(
            f"[提示] 扩展名不被支持或写入失败，已回退为 WAV：{base_name} -> {wav_name}，原因：{e}"
        )
        return wav_name


def export_split_decoded(ds: Dataset, out_dir: str) -> Tuple[int, int]:
    """
    将一个 split 的所有样本解码并写出。
    返回 (成功写出的文件数, 回退为 .wav 的文件数)
    """
    ensure_dir(out_dir)
    written = 0
    fallback_wav = 0
    used_names: Set[str] = set()

    # 解码音频，保持原采样率
    ds_decoded = ds.cast_column("audio", Audio())

    n = len(ds_decoded)
    print(f"  - 待导出样本数：{n}")
    for i in range(n):
        ex = ds_decoded[i]
        if "audio" not in ex or ex["audio"] is None:
            print(f"  [跳过] 索引 {i} 无 audio 内容")
            continue
        aud = ex["audio"]
        arr = aud.get("array", None)
        sr = aud.get("sampling_rate", None)
        if arr is None or sr is None:
            print(f"  [跳过] 索引 {i} 音频缺少 array 或 sampling_rate")
            continue

        base_name = safe_basename_from_path(aud.get("path"), i)
        # 尝试使用原扩展名写出
        name_before = base_name
        try:
            actual_name = write_audio_with_fallback(
                out_dir, base_name, arr, sr, used_names
            )
            if (
                os.path.splitext(actual_name)[1].lower()
                != os.path.splitext(name_before)[1].lower()
            ):
                fallback_wav += 1
            written += 1
        except Exception as e:
            print(f"  [失败] 写出 {base_name} 失败：{e}")

    return written, fallback_wav


def main():
    print(f"输入根目录：{PREPROCESSED_ROOT}")
    print(f"输出根目录：{OUT_ROOT}")

    langs = list_languages(PREPROCESSED_ROOT)
    if not langs:
        print("未发现任何语种目录，结束。")
        return

    print(f"发现语种：{langs}")

    total_written = 0
    total_fallback = 0

    for lang in langs:
        lang_dir = os.path.join(PREPROCESSED_ROOT, lang)
        splits = list_splits(lang_dir)
        if not splits:
            print(f"[跳过] 语种 {lang} 未发现任何可用 split")
            continue

        print(f"\n=== 处理语种: {lang} | splits: {splits} ===")
        for split in splits:
            split_dir = os.path.join(lang_dir, split)
            out_dir = os.path.join(OUT_ROOT, lang, split)

            try:
                ds = load_from_disk(split_dir)
            except Exception as e:
                print(f"[跳过] 加载失败 {lang}/{split}: {e}")
                continue

            if "audio" not in ds.column_names:
                print(f"[跳过] {lang}/{split} 不含 audio 列")
                continue

            print(f"- 导出 {lang}/{split} -> {out_dir}")
            written, fallback = export_split_decoded(ds, out_dir)
            print(f"  [完成] 写出 {written} 个文件，其中回退为 .wav 的 {fallback} 个")
            total_written += written
            total_fallback += fallback

    print(
        f"\n全部完成。总计写出 {total_written} 个文件（回退为 .wav：{total_fallback} 个）。"
    )
    print(f"输出位于：{OUT_ROOT}")


if __name__ == "__main__":
    main()
