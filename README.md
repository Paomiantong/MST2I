# MST2I

多语言 Speech-to-Instruction 主实验代码，包含数据预处理、SFT、StepGRPO、推理和评测。仓库不附带数据、音频、模型权重或预测结果。

## 代码结构

| 位置 | 用途 |
| --- | --- |
| `preprocess/` | 语料整理、指令格式化与辅助 T2I 数据准备 |
| `preprocess/prompts/` | Baseline、结构化推理与文本监督提示模板 |
| `data/preprocessed/builder.py` | 源语言与英语数据加载器 |
| `data/formatted/*/dataset_info.json` | 文件名与字段注册表，不包含样本 |
| `scripts/config/` | 10 份主实验 SFT 配置 |
| `scripts/rl_legacy.sh` | 主实验 RL 启动脚本 |
| `scripts/step_grop.py` | StepGRPO 入口及分阶段奖励 |
| `scripts/omni_grpo_trainer.py` | 音频 GRPO 训练器 |
| `scripts/rl.yml` | RL 训练参数 |
| `scripts/infer.py`, `infer.sh` | 推理入口 |
| `scripts/eval.py` | 主表意图、槽位及翻译评测 |
| `scripts/api_infer.py`, `api_input_helper.py` | API 基线入口及请求构建 |

## 主实验配置

| 系统 | 配置名中的标识 | 数据目录 |
| --- | --- | --- |
| Baseline | `baseline` | `baseline` |
| + Reasoning | `post-func` | `+post-func` |
| + Reasoning + T2I | `post-func-ext` | `+post-func`，附加 `ext-train` / `ext-dev` |
| + StepGRPO | 对应 SFT 模型后运行 `rl_legacy.sh` | `+post-func` |

SFT 配置覆盖 `fewshot-de`、`fewshot-fr`、`zeroshot-de`、`zeroshot-fr`；T2I 对应两个 few-shot 配置。

## 依赖与输入

使用 Linux、Python 3.10+、CUDA / PyTorch、LLaMA-Factory、Transformers、TRL、DeepSpeed 和 qwen-omni-utils。推理另需 vLLM。常规数据处理和评测依赖：

```bash
pip install -r requirements.txt
```

LLaMA-Factory 参考版本为 `958b9c3468f0aa1b60d80d8914682156458cd893`；仓库只保留其 LoRA 合并工具及许可证。

请自行准备 Speech-MASSIVE、MASSIVE、Qwen2.5-Omni-7B，以及格式化数据：

```text
data/formatted/baseline/train/       # fr.jsonl、de.jsonl、115.jsonl
data/formatted/baseline/eval/        # dev.jsonl、test.jsonl
data/formatted/+post-func/train/     # 同上；T2I 另需 extent-train.jsonl
data/formatted/+post-func/eval/      # 同上；T2I 另需 extent-dev.jsonl
```

预处理脚本从 `preprocess/` 目录运行。辅助文本采样通过 `MASSIVE_PATH` 指定外部 MASSIVE 语料路径。`formatter_v2.py` 显式传入 `--prompt baseline` 或 `--prompt +post-func`。辅助 T2I 使用 `pure-text-post-func` 模板，写入 `+post-func` 目录，并准备训练集与验证集。

## SFT

在仓库根目录生成本机配置并训练：

```bash
python scripts/configure_paths.py \
  --model-path /path/to/Qwen2.5-Omni-7B \
  --data-root data/formatted

WANDB_MODE=offline FORCE_TORCHRUN=1 llamafactory-cli train \
  scripts/config_local/train_qwen2_5-omni-baseline-fewshot-fr.yaml
```

替换配置名即可运行其他系统与设置。配置模板位于 `config/`，本机配置位于 `config_local/`。

## StepGRPO

主入口为 `rl_legacy.sh → step_grop.py → omni_grpo_trainer.py`，训练参数来自 `rl.yml`。通过 `QWEN_PATH`、`DATASET_DIR`、`OUTPUT_DIR`、`CUDA_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 设置模型、训练数据、输出位置及 GPU：

```bash
cd scripts
bash rl_legacy.sh
```

`rl.yml` 使用 6 个 rollout、1 epoch、学习率 `1e-4`、KL 系数 `0.05`。奖励权重位于 `step_grop.py` 的 `reward_config`。

## 推理与评测

在仓库根目录合并 SFT LoRA：

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-7B \
  bash scripts/export_omni.sh qwen2_5-omni-baseline-fewshot-fr
```

运行推理与评测：

```bash
DS_CONFIG=baseline bash scripts/infer.sh qwen2_5-omni-baseline-fewshot-fr

MST2I_EVAL_DATASET=data/formatted/baseline/eval/test.jsonl \
  python scripts/eval.py \
  infer_result/qwen2_5-omni-baseline-fewshot-fr/generated_predictions.jsonl \
  --compute_comet
```

结构化系统使用 `DS_CONFIG=+post-func`，并替换模型名和评测数据路径。推理入口调用 `scripts/vllm_infer.py`，当前保留双 GPU 配置。COMET 需另行安装 `unbabel-comet` 并准备评测模型。

API 基线直接使用 Python 标准库请求 Chat Completions 接口，通过 `OPENAI_API_KEY` 和可选的 `OPENAI_BASE_URL` 配置。显式指定服务端支持的音频模型：

```bash
python scripts/api_infer.py \
  --model YOUR_AUDIO_MODEL_ID \
  --input data/formatted/+post-func/eval/test.jsonl \
  --output infer_result/api/generated_predictions.jsonl
```

输出保持输入顺序，包含 `predict`、`label`、`locale`、`src_utt`，可直接传入 `eval.py`。输出路径必须尚不存在。

LoRA 合并和 vLLM 推理代码基于 LLaMA-Factory，相关 Apache 2.0 许可证位于 `deps/LlamaFactory/LICENSE`；训练器源码保留 HuggingFace 的原始版权声明。
