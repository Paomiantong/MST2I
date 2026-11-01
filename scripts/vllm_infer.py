# Copyright 2025 the LlamaFactory team.
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

import gc
import json

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str | None = None,
    dataset: str = "test",
    dataset_dir: str = "data/formatted/baseline",
    template: str = "qwen2_omni",
    cutoff_len: int = 2048,
    max_samples: int | None = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: str | None = None,
    enable_thinking: bool = True,
    seed: int | None = None,
    pipeline_parallel_size: int = 1,
    batch_size: int = 1024,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python scripts/vllm_infer.py --model_name_or_path /path/to/model --dataset test
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError(
            "Pipeline parallel size should be smaller than the number of gpus."
        )

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    max_prompt_len = int(engine_args["max_model_len"]) - int(
        generating_args.max_new_tokens
    )
    if max_prompt_len <= 0:
        raise ValueError(
            "`max_model_len` should be larger than `max_new_tokens` when using vLLM."
        )

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(
        template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module
    )
    train_dataset = dataset_module["train_dataset"]


    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty
        or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(
        range(0, len(train_dataset), batch_size), desc="Processing batched inference"
    ):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        for j in range(len(batch["input_ids"])):
            if batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {
                    "audio": zip(audio_data["audios"], audio_data["sampling_rates"])
                }
            else:
                multi_modal_data = None

            label_text = tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                skip_special_tokens=skip_special_tokens,
            )
            labels.append(label_text)

            prompt_token_ids = batch["input_ids"][j]

            vllm_input_data = {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": multi_modal_data,
            }


            vllm_inputs.append(vllm_input_data)
            prompts.append(
                tokenizer.decode(
                    prompt_token_ids, skip_special_tokens=skip_special_tokens
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        # Accumulate results
        all_prompts.extend(prompts)
        all_preds.extend(preds)
        all_labels.extend(labels)
        gc.collect()

    # Write all results at once outside the loop
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(
                json.dumps(
                    {"prompt": text, "predict": pred, "label": label},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)

if __name__ == "__main__":
    fire.Fire(vllm_infer)
