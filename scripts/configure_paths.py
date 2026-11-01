"""Generate SFT configurations for local model and data paths."""

import argparse
from pathlib import Path
import re
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="data/formatted")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).expanduser().resolve()
    model = str(Path(args.model_path).expanduser().resolve())
    destination = root / "scripts/config_local"
    destination.mkdir(exist_ok=True)
    for source in sorted((root / "scripts/config").glob("*.yaml")):
        text = source.read_text()
        family = "baseline" if "-baseline-" in source.name else "+post-func"
        updates = {
            "model_name_or_path": model,
            "dataset_dir": str(data_root / family),
            "deepspeed": str(root / "scripts/ds_config.json"),
            "output_dir": str(root / "checkpoints" / source.stem.removeprefix("train_")),
        }
        for key, value in updates.items():
            text, count = re.subn(
                rf"^{key}:.*$", lambda _: f"{key}: {json.dumps(value)}", text,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError(f"Expected one {key} in {source}")
        target = destination / source.name
        target.write_text(text)
        print(target)


if __name__ == "__main__":
    main()
