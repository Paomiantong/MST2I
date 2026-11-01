import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from fire import Fire

dataset_config = os.environ["DS_CONFIG"]
assert dataset_config is not None, "Dataset Config Can not be None"

workspace = Path(__file__).parent.parent


def run_evaluation(
    template: str,
    model_path: str,
    output_dir: str,
    dataset: str,
    batch_size: int,
    checkpoint: Optional[str] = None,
) -> str:
    """
    Generate test predictions from a merged model or a LoRA checkpoint.

    Args:
        checkpoint: Optional LoRA checkpoint directory

    Returns:
        Path to the generated jsonl output file
    """
    output_dir = os.path.join(
        output_dir, os.path.basename(checkpoint if checkpoint else model_path)
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result_files = list(Path(output_dir).glob("generated_predictions.jsonl"))
    if result_files:
        return str(result_files[0])

    cmd = [
        "WANDB_DISABLED=true",
        "python",
        str(workspace / "scripts" / "vllm_infer.py"),
        "--model_name_or_path",
        model_path,
        "--dataset_dir",
        str(workspace / f"data/formatted/{dataset_config}"),
        "--dataset",
        dataset,
        "--template",
        template,
        "--cutoff_len",
        "2048",
        "--max_samples",
        "1000000",
        "--temperature",
        "0",
        "--vllm_config",
        shlex.quote(
            json.dumps(
                {
                    "gpu_memory_utilization": 0.9,
                    "dtype": "float16",
                    "enforce_eager": True,
                    "tensor_parallel_size": 2,
                    "disable_mm_preprocessor_cache": True,
                    "limit_mm_per_prompt": {"video": 0, "image": 0, "audio": 1},
                }
            )
        ),
        "--save_name",
        os.path.join(output_dir, "generated_predictions.jsonl"),
    ]

    if checkpoint:
        print(f"Evaluating checkpoint: {checkpoint}")
        cmd += ["--adapter_name_or_path", checkpoint]

    print(cmd)

    subprocess.run(" ".join(cmd), shell=True)
    time.sleep(2)

    # Find the generated jsonl file
    result_files = list(Path(output_dir).glob("generated_predictions.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No jsonl file found in {output_dir}")

    return str(result_files[0])


if __name__ == "__main__":
    Fire(run_evaluation)
