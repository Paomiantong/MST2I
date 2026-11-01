"""Run audio baselines through a Chat Completions-compatible endpoint."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from api_input_helper import build_openai_request

WORKSPACE = Path(__file__).resolve().parent.parent


def request_prediction(item, model, api_key, base_url, temperature, timeout, retries):
    payload = {
        "model": model,
        "messages": build_openai_request(item["audios"][0], item["locale"]),
        "temperature": temperature,
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            status = error.code
            error.close()
            if status not in {408, 429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"API request failed (HTTP {status}).") from None
        except (URLError, TimeoutError):
            if attempt == retries:
                raise RuntimeError("API connection failed or timed out.") from None
        time.sleep(min(2 ** attempt, 30))

    try:
        prediction = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("API response is missing choices[0].message.content.") from None
    if not isinstance(prediction, str) or not prediction.strip():
        raise RuntimeError("API returned no text prediction.")
    return {
        "predict": prediction,
        "label": item["output"],
        "locale": item["locale"],
        "src_utt": item["src_utt"],
    }


def run(input_file, output_file, model, workers=8, temperature=0.3, timeout=120, retries=3):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running API inference.")
    if workers < 1 or retries < 0 or timeout <= 0:
        raise ValueError(
            "workers and timeout must be positive; retries must be nonnegative."
        )
    input_file, output_file = Path(input_file), Path(output_file)
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output files must differ.")
    items = []
    with input_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not all(
                    key in item for key in ("audios", "locale", "output", "src_utt")
                ):
                    raise ValueError
                if (
                    not isinstance(item["audios"], list)
                    or not item["audios"]
                    or not isinstance(item["audios"][0], str)
                    or not all(
                        isinstance(item[key], str)
                        for key in ("locale", "output", "src_utt")
                    )
                ):
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(f"Invalid input record at line {line_number}.") from None
            items.append(item)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def predict(item):
        return request_prediction(
            item, model, api_key, base_url, temperature, timeout, retries
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as destination:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(predict, items):
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=WORKSPACE / "data/formatted/+post-func/eval/test.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", required=True,
        help="Audio model ID accepted by the configured endpoint.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    try:
        count = run(
            args.input, args.output, args.model, args.workers,
            args.temperature, args.timeout, args.retries,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
