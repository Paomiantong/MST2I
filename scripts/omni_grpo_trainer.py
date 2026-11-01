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

import copy
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import transformers
from datasets import Dataset, IterableDataset
from logutil import get_logger
from packaging import version
from qwen_omni_utils import process_mm_info
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.utils import is_peft_available
from trl.data_utils import is_conversational
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    prepare_peft_model,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax

if is_peft_available():
    from peft import PeftConfig

logger = get_logger()

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2_5OmniGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with Qwen2.5Omni model support.
    This extends the GRPO algorithm to handle multimodal inputs including audio and video.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        use_audio_in_video: bool = False,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs: dict = args.model_init_kwargs or {}  # ty:ignore[invalid-assignment]

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            # 在 Qwen2_5OmniGRPOTrainer.__init__ 方法中的模型初始化部分
            if "omni" in model_id.lower():
                # 为 Qwen2.5-Omni 模型创建一个新的 model_init_kwargs 字典
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]

                # 加载配置并修改
                config = Qwen2_5OmniThinkerConfig.from_pretrained(model_id)

                # 使用修改后的配置加载模型
                model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                    model_id, config=config, **omni_model_init_kwargs
                )

                # 确保 talker 被禁用
                # model.disable_talker()
            else:
                raise ValueError(
                    f"Unsupported model ID for Qwen2.5OmniGRPOTrainer: {model_id}"
                )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        logger.info_rank0(f"Model load with dtype:{model.dtype}")

        if peft_config is not None:
            model = prepare_peft_model(model, peft_config, args)

        # 在创建参考模型部分
        if is_deepspeed_zero3_enabled():
            if "omni" in model_id.lower():
                # 同样处理参考模型
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]

                config = Qwen2_5OmniThinkerConfig.from_pretrained(model_id)

                self.ref_model = (
                    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        model_id, config=config, **omni_model_init_kwargs
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported model ID for reference model: {model_id}"
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)  # ty:ignore[invalid-argument-type]
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        if processing_class is None:
            if "omni" in model_id.lower():
                processing_class = Qwen2_5OmniProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                raise ValueError(f"Unsupported model ID for processor: {model_id}")

        # The rest of the initialization is the same as in Qwen2VLGRPOTrainer
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.weighted_reward = False
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                print(self.accelerator.state.deepspeed_plugin.deepspeed_config)
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, use_cache=False, **kwargs).logits

        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, 1:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        return selective_log_softmax(
            logits, input_ids
        )  # use memory-efficient implementation of the common log softmax

    def remove_none_from_data(self, data):
        for samples in data:
            for turn in samples:
                if "content" in turn and isinstance(turn["content"], list):
                    for sub_entry in turn["content"]:
                        if isinstance(sub_entry, dict):
                            keys_to_remove = [
                                k for k, v in sub_entry.items() if v is None
                            ]
                            for k in keys_to_remove:
                                del sub_entry[k]
        return data

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]

        prompts = copy.deepcopy(prompts)
        prompts = self.remove_none_from_data(prompts)

        audios, images, videos = process_mm_info(prompts, use_audio_in_video=False)

        text = self.processing_class.apply_chat_template(
            prompts,  # 对话格式数据
            add_generation_prompt=True,
            tokenize=False,
        )

        prompt_inputs = self.processing_class(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                :, -self.max_prompt_length :
            ]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                :, -self.max_prompt_length :
            ]

        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 为Qwen2.5Omni模型生成文本和音频
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
            )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 处理输入，准备计算logps
        for key in ["input_ids", "attention_mask"]:
            if key in prompt_inputs:
                prompt_inputs.pop(key)
        prompt_inputs["attention_mask"] = torch.cat(
            [prompt_mask, completion_mask], dim=1
        )  # (B*G, P+C)

        # 处理音频特征
        prompt_inputs["input_features"] = prompt_inputs[
            "input_features"
        ].repeat_interleave(self.num_generations, dim=0)
        prompt_inputs["feature_attention_mask"] = prompt_inputs[
            "feature_attention_mask"
        ].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, **prompt_inputs
        )
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, **prompt_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, **prompt_inputs
                    )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # 计算KL散度
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        # 解码生成的完成
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if self.accelerator.is_main_process:
            print(f"\n{'=' * 20} Generated Sample {'=' * 20}")
            # Print the first prompt in the batch
            print(f"[Prompt]: {prompts[0]}\n")
            # Print the first generated completion
            print(f"[Completion]: \n{completions[0]}\n")
            print(f"[Solution]: \n{inputs[0]['solution']}\n")
            print(f"{'=' * 56}\n")

        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # 计算奖励
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {
                key: []
                for key in inputs[0].keys()
                if key not in ["prompt", "completion"]
            }
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )

            # === Handling tuple return (rewards, metrics) from reward functions ===
            if isinstance(output_reward_func, tuple):
                reward_values, reward_metrics = output_reward_func

                # Log detailed metrics (average per batch)
                for k, v in reward_metrics.items():
                    if v:
                        self._metrics[k].append(sum(v) / len(v))

                output_reward_func = reward_values
            # ====================================================================

            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # === 总奖励 START ===
        if not self.weighted_reward:
            rewards = rewards_per_func.sum(dim=1)
        else:
            # [accuracy reward, format reward]
            reward_weights = torch.tensor(
                [2.0, 1.0], dtype=torch.float32, device=device
            )
            rewards = (rewards_per_func * reward_weights).sum(dim=1)

        # 分组奖励计算
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 奖励归一化
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 损失计算
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)

        # With KL loss
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # 记录指标
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
