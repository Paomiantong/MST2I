# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jiwer
import sacrebleu
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from logutil import get_logger

# 导入自定义的 Omni Audio Trainer
from omni_grpo_trainer import Qwen2_5OmniGRPOTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

logger = get_logger()

# ==============================================================================
# Global Reward Configuration (全局权重配置)
# ==============================================================================

# fmt:off
reward_config = {
    # 1. 顶层奖励权重 (Top-level Reward Weights)
    "W_FINAL": 1.0,       # 最终答案 (Final Answer) 的权重
    "W_FUNC": 0.6,        # 函数调用 (Function Call) 的权重
    "W_ASR": 0.3,         # 语音转写 (ASR) 的权重
    "W_TRANS": 0.3,       # 中间翻译 (Intermediate Translation) 的权重
    "W_STEP": 0.0,        # 关键步骤 (Key Steps) 的权重

    # 2. Final Answer 内部评分权重 (Sub-weights for Final Answer)
    "W_SUB_INTENT": 1.0,  # 意图识别权重
    "W_SUB_SLOT": 1.0,    # [重点] 槽位填充权重
    "W_SUB_CONTENT": 1.0, # 文本内容 BLEU 权重

    # 3. 幻觉惩罚因子 (Hallucination Penalty Factors)
    "PENALTY_SLOT_FP": 0.00, # Final Answer 中每个错误槽位的扣分
    "PENALTY_FUNC_FP": 0.00, # Function Call 中每个错误参数的扣分
}
# fmt:on

# 1. 顶层奖励权重 (Top-level Reward Weights)
# ------------------------------------------------------------------------------
# 决定各部分奖励在总分中的占比
W_FINAL = reward_config["W_FINAL"]
W_FUNC = reward_config["W_FUNC"]
W_ASR = reward_config["W_ASR"]
W_TRANS = reward_config["W_TRANS"]
W_STEP = reward_config["W_STEP"]

# 2. Final Answer 内部评分权重 (Sub-weights for Final Answer)
# ------------------------------------------------------------------------------
# 决定 Intent, Slot, Content 三者在 W_FINAL 中的相对重要性
W_SUB_INTENT = reward_config["W_SUB_INTENT"]
W_SUB_SLOT = reward_config["W_SUB_SLOT"]
W_SUB_CONTENT = reward_config["W_SUB_CONTENT"]

# 3. 幻觉惩罚因子 (Hallucination Penalty Factors)
# ------------------------------------------------------------------------------
# 每个 False Positive (错误生成) 扣除的分数
PENALTY_SLOT_FP = reward_config["PENALTY_SLOT_FP"]
PENALTY_FUNC_FP = reward_config["PENALTY_FUNC_FP"]


# ==============================================================================
# Script Arguments
# ==============================================================================
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script (Audio version).
    """

    # 支持多个数据集路径
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset path(s).  Can be a single path or multiple paths separated by comma. "
            "Example: '/path/to/data1. json,/path/to/data2.json,/path/to/data3. json'"
        },
    )

    reward_funcs: list[str] = field(
        default_factory=lambda: [
            "accuracy_final",
            "accuracy_asr",
            "accuracy_trans",
            "accuracy_func",
            # "accuracy_step", # 数据缺失，即使加上权重也是0，建议注释掉
            "validity",
        ],
        metadata={
            "help": "List of reward functions.  Possible values: 'accuracy_final', 'accuracy_asr', "
            "'accuracy_trans', 'accuracy_func', 'accuracy_step', 'validity'"
        },
    )

    audio_sample_rate: int = field(
        default=16000,
        metadata={"help": "Sample rate for loading audio files. "},
    )

    # 是否打乱合并后的数据集
    shuffle_merged_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the merged dataset. "},
    )

    # 打乱时的随机种子
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed for shuffling the merged dataset."},
    )


# ==============================================================================
# Reward Function Utilities
# ==============================================================================


def safe_lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else s


def extract_block(text: str, header: str) -> Optional[str]:
    """
    Extract the content following a given header line until the next header or end.
    """
    if not isinstance(text, str):
        return None
    pattern = rf"{re.escape(header)}.*?\n(.*?)(?=\n\s*\n|$)"
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def normalize_func_call(s: str) -> tuple:
    """
    Normalize a function call string into (intent, args_tuple) for comparison.
    args_tuple format: ((arg_name, arg_value), ...)
    """
    if not isinstance(s, str) or not s.strip():
        return "unknown", ()
    s = s.strip()

    call_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$", s)
    if call_match:
        intent = safe_lower(call_match.group(1))
        args_src = call_match.group(2).strip()
        args_list = []
        if args_src:
            parts = [p.strip() for p in args_src.split(",")]
            for p in parts:
                kv = p.split("=", 1)
                if len(kv) == 2:
                    k = safe_lower(kv[0].strip())
                    v = kv[1].strip().strip('"').strip("'")
                    args_list.append((k, v))
        # Sort arguments to ensure consistent order for comparison
        return intent, tuple(sorted(args_list))

    return safe_lower(s), ()


# ==============================================================================
# Validity Reward (StepRVR)
# ==============================================================================


def validity_reward(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    StepRVR: structural validity reward.
    """
    rewards = []

    p_transcription = re.compile(
        r"^The transcription of the audio in (?P<locale>\S+) is:$", re.MULTILINE
    )
    p_translation = re.compile(
        r"^The translation from (?P<locale>\S+) to English request sentence is:$",
        re.MULTILINE,
    )
    p_func_call = re.compile(r"^The function call is:$", re.MULTILINE)
    p_final = re.compile(
        r"^The final translation from (?P<locale>\S+) to English request sentence is:$",
        re.MULTILINE,
    )

    for completion in completions:
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )

        m1 = p_transcription.search(content)
        m2 = p_translation.search(content)
        m3 = p_func_call.search(content)
        m4 = p_final.search(content)

        if not (m1 and m2 and m3 and m4):
            rewards.append(0.0)
            continue

        if not (
            m1.end() < m2.start() and m2.end() < m3.start() and m3.end() < m4.start()
        ):
            rewards.append(0.0)
            continue

        l1 = m1.group("locale")
        l2 = m2.group("locale")
        l4 = m4.group("locale")

        if l1 != l2 or l1 != l4:
            rewards.append(0.0)
            continue

        rewards.append(0.4)

    return rewards


# ==============================================================================
# Accuracy Rewards (Split for detailed metrics)
# ==============================================================================


def key_step_single(content: str, key_step: Dict[str, Any]) -> float:
    if not isinstance(content, str):
        return 0.0

    translation_block = extract_block(content, "The translation from")
    func_call_block = extract_block(content, "The function call is:")
    final_block = extract_block(content, "The final translation from")

    combined = " ".join(
        [
            x
            for x in [translation_block, func_call_block, final_block]
            if isinstance(x, str)
        ]
    ).lower()

    steps_keywords = key_step.get("Steps", [])
    total_keywords = len(steps_keywords)
    if total_keywords == 0:
        return 0.0

    matched_keywords = 0
    for group in steps_keywords:
        if isinstance(group, str):
            group = [group]
        if any(alt.lower() in combined for alt in group if isinstance(alt, str)):
            matched_keywords += 1

    return matched_keywords / total_keywords


def _extract_final_answer(text: str) -> Optional[str]:
    return extract_block(text, "The final translation from")


def _extract_transcription(text: str) -> Optional[str]:
    return extract_block(text, "The transcription of the audio")


def _extract_translation(text: str) -> Optional[str]:
    return extract_block(text, "The translation from")


def _extract_func_call(text: str) -> Optional[str]:
    return extract_block(text, "The function call is:")


slot_pat = re.compile(r"\[([^\]:]+?) : (.*?)\]")


def _compare_final_annotated(pred: str, gt: str) -> float:
    """
    Compares final answers using GLOBAL weights and Penalty for Hallucinations.
    """
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0

    def parse_final(s: str):
        if "|" not in s:
            return "unknown", s.strip(), []

        left, right = s.split("|", 1)
        intent = left.strip()
        raw_right = right.strip()

        slots = slot_pat.findall(raw_right)
        slots = [(k.strip(), v.strip()) for k, v in slots]

        content_clean = slot_pat.sub(lambda x: x.group(2), raw_right).strip()
        content_clean = " ".join(content_clean.split())
        return intent, content_clean, slots

    pi, ps, pslots = parse_final(pred)
    gi, gs, gslots = parse_final(gt)

    # 1. Intent Score
    intent_reward = 1.0 if pi == gi else 0.0

    # 2. Slot Score (F1 with Penalty)
    g_set = set(gslots)
    p_set = set(pslots)

    if len(g_set) == 0 and len(p_set) == 0:
        slot_f1 = 1.0
    elif len(g_set) == 0 or len(p_set) == 0:
        slot_f1 = 0.0
    else:
        tp = len(g_set & p_set)
        fp_count = len(p_set) - tp  # False Positives

        precision = tp / len(p_set)
        recall = tp / len(g_set)

        if (precision + recall) == 0:
            slot_f1 = 0.0
        else:
            slot_f1 = 2 * (precision * recall) / (precision + recall)

        # [Penalty Logic] 使用全局定义的 PENALTY_SLOT_FP
        fp_penalty = fp_count * PENALTY_SLOT_FP
        slot_f1 = max(0.0, slot_f1 - fp_penalty)

    # 3. Content Score
    if not gs and not ps:
        content_reward = 1.0
    elif not gs:
        content_reward = 0.0
    elif not ps:
        content_reward = 0.0
    else:
        content_reward = sacrebleu.sentence_bleu(ps, [gs]).score / 100.0

    # 4. Final Weighted Calculation (Using Global Sub-Weights)
    w_sum = W_SUB_INTENT + W_SUB_SLOT + W_SUB_CONTENT
    final_score = (
        intent_reward * W_SUB_INTENT
        + slot_f1 * W_SUB_SLOT
        + content_reward * W_SUB_CONTENT
    ) / w_sum

    return final_score


# 1. Final Answer Reward
def accuracy_reward_final(
    completions: List[List[Dict[str, str]]], solution: List[str], **kwargs
) -> List[float]:
    rewards = []
    # 使用全局变量 W_FINAL
    for idx, completion in enumerate(completions):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        sol = solution[idx] if idx < len(solution) else ""
        pred_final = _extract_final_answer(content)
        gt_final = _extract_final_answer(sol) or sol.strip()
        r = _compare_final_annotated(pred_final or "", gt_final or "")
        rewards.append(float(r * W_FINAL))
    return rewards


# 2. ASR Reward
def accuracy_reward_asr(
    completions: List[List[Dict[str, str]]],
    src_utt: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    rewards = []
    # 使用全局变量 W_ASR
    src_transcriptions = src_utt
    for idx, completion in enumerate(completions):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        r = 0.0
        if src_transcriptions is not None and idx < len(src_transcriptions):
            pred_src = _extract_transcription(content) or ""
            gt_src = src_transcriptions[idx] or ""
            if not gt_src and not pred_src:
                r = 1.0
            elif not gt_src:
                r = 0.0
            else:
                try:
                    error_rate = jiwer.wer(gt_src, pred_src)
                except ValueError:
                    error_rate = 1.0 if gt_src else 0.0
                r = max(0.0, 1.0 - error_rate)
        rewards.append(float(r * W_ASR))
    return rewards


# 3. Translation Reward
def accuracy_reward_trans(
    completions: List[List[Dict[str, str]]],
    fixed_tgt_utt: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    rewards = []
    # 使用全局变量 W_TRANS
    tgt_translations = fixed_tgt_utt
    for idx, completion in enumerate(completions):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        r = 0.0
        if tgt_translations is not None and idx < len(tgt_translations):
            pred_tgt = _extract_translation(content) or ""
            gt_tgt = tgt_translations[idx] or ""
            if not gt_tgt and not pred_tgt:
                r = 1.0
            elif not gt_tgt:
                r = 0.0
            else:
                r = sacrebleu.sentence_bleu(pred_tgt, [gt_tgt]).score / 100.0
        rewards.append(float(r * W_TRANS))
    return rewards


# 4. Function Call Reward (F1 Score with Penalty)
def accuracy_reward_func(
    completions: List[List[Dict[str, str]]],
    func_call: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    rewards = []
    # 使用全局变量 W_FUNC
    func_calls = func_call
    for idx, completion in enumerate(completions):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        if func_calls is not None and idx < len(func_calls):
            pred_call = _extract_func_call(content) or ""
            gt_call = func_calls[idx] or ""
            pi, pargs = normalize_func_call(pred_call)
            gi, gargs = normalize_func_call(gt_call)

            if pi != gi:
                ir = 0.0
            else:
                ir = 1.0

            g_set = set(gargs)
            p_set = set(pargs)

            if not g_set and not p_set:
                sr = 1.0
            elif not g_set or not p_set:
                sr = 0.0
            else:
                tp = len(g_set & p_set)
                fp_count = len(p_set) - tp

                precision = tp / len(p_set)
                recall = tp / len(g_set)
                if (precision + recall) == 0:
                    sr = 0.0
                else:
                    sr = 2 * (precision * recall) / (precision + recall)

                # [Penalty Logic] 使用全局定义的 PENALTY_FUNC_FP
                fp_penalty = fp_count * PENALTY_FUNC_FP
                sr = max(0.0, sr - fp_penalty)

        rewards.append(float((sr + ir) * W_FUNC))
    return rewards


# 5. Key Step Reward
def accuracy_reward_step(
    completions: List[List[Dict[str, str]]],
    key_steps: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    rewards = []
    # 使用全局变量 W_STEP
    for idx, completion in enumerate(completions):
        content = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        r = 0.0
        if (
            key_steps is not None
            and idx < len(key_steps)
            and isinstance(key_steps[idx], dict)
        ):
            r = key_step_single(content, key_steps[idx])
        rewards.append(float(r * W_STEP))
    return rewards


# ==============================================================================
# Reward Function Wrappers (for Trainer interface)
# ==============================================================================


def validity_reward_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return validity_reward(completions, **kwargs)


def accuracy_reward_final_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return accuracy_reward_final(completions, **kwargs)


def accuracy_reward_asr_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return accuracy_reward_asr(completions, **kwargs)


def accuracy_reward_trans_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return accuracy_reward_trans(completions, **kwargs)


def accuracy_reward_func_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return accuracy_reward_func(completions, **kwargs)


def accuracy_reward_step_wrapper(
    prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    return accuracy_reward_step(completions, **kwargs)


reward_funcs_registry: Dict[str, Callable[..., List[float]]] = {
    "validity": validity_reward_wrapper,
    "accuracy_final": accuracy_reward_final_wrapper,
    "accuracy_asr": accuracy_reward_asr_wrapper,
    "accuracy_trans": accuracy_reward_trans_wrapper,
    "accuracy_func": accuracy_reward_func_wrapper,
    "accuracy_step": accuracy_reward_step_wrapper,
}


# ==============================================================================
# Dataset Loading Utilities
# ==============================================================================


def parse_dataset_paths(dataset_name: str) -> List[str]:
    """
    解析数据集路径字符串，支持以下格式：
    1. 单个路径: "/path/to/data.json"
    2. 逗号分隔的多个路径: "/path/to/data1.json,/path/to/data2.json"
    3. 带空格的逗号分隔: "/path/to/data1. json, /path/to/data2.json"
    """
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")

    # 按逗号分割并去除空格
    paths = [p.strip() for p in dataset_name.split(",")]
    # 过滤空字符串
    paths = [p for p in paths if p]

    if not paths:
        raise ValueError("No valid dataset paths found")

    return paths


def load_and_merge_datasets(
    dataset_paths: List[str],
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
) -> DatasetDict:
    """
    加载多个数据集并合并。

    Args:
        dataset_paths: 数据集文件路径列表
        split: 数据集分割名称
        shuffle: 是否打乱合并后的数据集
        seed: 随机种子

    Returns:
        合并后的 DatasetDict
    """
    datasets_list = []

    for idx, path in enumerate(dataset_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        logger.info_rank0(f"Loading dataset [{idx + 1}/{len(dataset_paths)}]: {path}")

        # 加载单个数据集
        ds = load_dataset("json", data_files={split: path})
        datasets_list.append(ds[split])

        logger.info_rank0(f"  - Loaded {len(ds[split])} examples")

    # 合并所有数据集
    logger.info_rank0(f"\nMerging {len(datasets_list)} datasets...")
    merged_dataset = concatenate_datasets(datasets_list)
    logger.info_rank0(f"Total examples after merging: {len(merged_dataset)}")

    # 可选：打乱数据集
    if shuffle:
        logger.info_rank0(f"Shuffling merged dataset with seed={seed}...")
        merged_dataset = merged_dataset.shuffle(seed=seed)

    # 返回 DatasetDict 格式以保持兼容性
    return DatasetDict({split: merged_dataset})


# ==============================================================================
# Main Function
# ==============================================================================


def main(
    script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig
):
    # 抑制 Qwen Processor 的特定警告
    def filter_qwen_warning(record):
        return (
            "System prompt modified, audio output may not work as expected"
            not in record.getMessage()
        )

    logging.getLogger().addFilter(filter_qwen_warning)

    # 1. Get reward functions from registry
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # 2. 解析并加载多个数据集
    dataset_paths = parse_dataset_paths(script_args.dataset_name)
    logger.info_rank0(f"{'=' * 60}")
    logger.info_rank0(f"Found {len(dataset_paths)} dataset(s) to load:")
    for i, p in enumerate(dataset_paths):
        logger.info_rank0(f"  [{i + 1}] {p}")
    logger.info_rank0(f"{'=' * 60}")

    # 加载并合并数据集
    dataset = load_and_merge_datasets(
        dataset_paths=dataset_paths,
        split=script_args.dataset_train_split,
        shuffle=script_args.shuffle_merged_dataset,
        seed=script_args.shuffle_seed,
    )

    # 4. Format into conversation (Audio + Text)
    def make_conversation_audio(example):
        """
        构建适用于 Qwen2. 5-Omni 的对话格式。
        """
        user_content = []
        # 添加文本 prompt
        user_content.append(
            {
                "type": "text",
                "text": example.get("instruction", "").replace("<audio>", ""),
            }
        )
        user_content.append({"type": "audio", "audio": example.get("audios")[0]})
        solution = example.get("output", "")

        return {
            "prompt": [
                {"role": "user", "content": user_content},
            ],
            "solution": solution,
            "key_steps": example.get("key_steps", {}),
            "src_utt": _extract_transcription(solution),
            "fixed_tgt_utt": _extract_translation(solution),
            "func_call": _extract_func_call(solution),
        }

    logger.info_rank0("Formatting conversations...")
    dataset = dataset.map(make_conversation_audio)

    trainer_cls = Qwen2_5OmniGRPOTrainer
    logger.info_rank0(f"\nUsing trainer: {trainer_cls.__name__}")
    logger.info_rank0(
        f"Training on {len(dataset[script_args.dataset_train_split])} examples\n"
    )

    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    training_args.model_init_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    # 6.  Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            and script_args.dataset_test_split in dataset
            else None
        ),
        peft_config=get_peft_config(model_args),
    )

    # 7. Train
    trainer.train()

    # 8. Save model
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
