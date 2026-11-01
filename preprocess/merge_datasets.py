from typing import Dict, List
import os
from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
)

"""
目标：
- 在保留 Speech-MASSIVE 原有多语种的前提下，新增英语（来自 ../data/speech-massive-en）。
- Speech-MASSIVE 原本没有 test split，需要把 Speech-MASSIVE-test 中按语种的 test split 合并进来。
- 注意：英语语言的 test split 来自 ../data/speech-massive-en（Speech-MASSIVE-test 中没有英语 test）。
- 最终将每个语种（config）对应的 DatasetDict 分别保存到 ../data/preprocessed/{lang} 下。

约定：
- 语言 config 代码优先使用数据集自身提供的命名（例如 en-US、fr-FR 等）。
- 英语 config 名称自动从 (Speech-MASSIVE ∪ Speech-MASSIVE-test) 中挑选第一个以 "en" 开头的 config；若都不存在，则回退为 "en-US"。
- ../data/speech-massive-en 是已准备好的英文 DatasetDict，包含 train_115 / train / validation / test。
"""

BASE_PATH = "../../datasets/Speech-MASSIVE"
TEST_PATH = "../../datasets/Speech-MASSIVE-test"
EN_DISK_PATH = "../data/speech-massive-en"
OUT_DIR = "../data/preprocessed"

# 需要从英文数据中写入到最终结果的训练相关 splits
TRAIN_SPLITS = ["train_115", "train", "validation"]
# 测试 split 名称
TEST_SPLIT = "test"


def pick_english_config(configs: List[str]) -> str:
    # 优先选择以 en 开头的 config（如 en-US、en-GB）
    for c in configs:
        if c.lower().startswith("en"):
            return c
    return "en-US"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


FEATURES = set(["id", "utt", "annot_utt", "locale", "intent_str", "audio"])


def main():
    # 枚举可用的语种（config）
    base_configs = get_dataset_config_names(BASE_PATH)
    test_configs = get_dataset_config_names(TEST_PATH)

    # 合并语种列表（可能 test 有一些 base 没有的）
    all_configs = sorted(set(base_configs) | set(test_configs))
    all_configs.remove("all")

    # 自动选择英语 config 名称
    english_config = pick_english_config(all_configs)
    if english_config not in all_configs:
        # 如果两边都没有 en*，也要把英语加入到需要处理的 config 集合中
        all_configs.append(english_config)
        all_configs = sorted(set(all_configs))

    # 加载已经准备好的英文数据
    if not os.path.isdir(EN_DISK_PATH):
        raise FileNotFoundError(
            f"未找到英文数据目录：{EN_DISK_PATH}。请先生成并保存英文数据。"
        )
    english_ddict: DatasetDict = load_from_disk(EN_DISK_PATH)  # ty:ignore[invalid-assignment]

    print(f"将处理以下语种（configs）：{all_configs}")
    ensure_dir(OUT_DIR)

    for lang in all_configs:
        print(f"\n=== 处理语种: {lang} ===")

        # 1) 基础集（train_115/train/validation等）来自 Speech-MASSIVE
        ds_dict: Dict[str, Dataset] = {}
        if lang in base_configs:
            base_ddict: DatasetDict = load_dataset(
                BASE_PATH, lang, trust_remote_code=True
            )  # ty:ignore[invalid-assignment]
            for split in base_ddict.keys():
                ds = base_ddict[split]
                cols = set(ds.column_names) - FEATURES - set(["slot_method"])
                ds_dict[split] = ds.remove_columns(cols)  # ty:ignore[invalid-assignment]
            print(f"- 载入基础集 splits: {list(base_ddict.keys())}")
        else:
            print(f"- 基础集缺失：{lang}，将尝试仅填充可用的 split。")

        # 2) 如果是英语语种，用 ../data/speech-massive-en 覆盖/补充训练相关 splits
        if lang == english_config:
            for split in TRAIN_SPLITS:
                if split in english_ddict:
                    ds_dict[split] = english_ddict[split].rename_columns(
                        {
                            "slurp_id": "id",
                            "sentence": "utt",
                            "annotation": "annot_utt",
                        }
                    )
                    intent_feat = ds_dict[split].features["intent"]
                    ds_dict[split] = ds_dict[split].map(
                        lambda x: {
                            "intent_str": [
                                intent_feat.int2str(i) for i in x["intent"]
                            ]
                        },
                        batched=True,
                    )
                    print(
                        f"- 英语 {split}: {len(ds_dict[split])} 条（来自 speech-massive-en）"
                    )
                else:
                    print(f"- 英语数据缺少 split: {split}，跳过。")

            # 英语的 test split 来自 speech-massive-en（因为 Speech-MASSIVE-test 没有英语）
            if TEST_SPLIT in english_ddict:
                ds_dict[TEST_SPLIT] = english_ddict[TEST_SPLIT].rename_columns(
                    {
                        "slurp_id": "id",
                        "sentence": "utt",
                        "annotation": "annot_utt",
                        "intent": "intent",
                    }
                )
                intent_feat = ds_dict[TEST_SPLIT].features["intent"]
                ds_dict[TEST_SPLIT] = ds_dict[TEST_SPLIT].map(
                    lambda x: {
                        "intent_str": [intent_feat.int2str(i) for i in x["intent"]]
                    },
                    batched=True,
                )
                print(
                    f"- 英语 {TEST_SPLIT}: {len(ds_dict[TEST_SPLIT])} 条（来自 speech-massive-en）"
                )

        else:
            # 3) 非英语语种：测试集来自 Speech-MASSIVE-test（按语种）
            if lang in test_configs:
                test_ddict: DatasetDict = load_dataset(
                    TEST_PATH, lang, trust_remote_code=True
                )  # ty:ignore[invalid-assignment]
                if TEST_SPLIT in test_ddict:
                    ds = test_ddict[TEST_SPLIT]
                    cols = set(ds.column_names) - FEATURES - set(["slot_method"])
                    ds_dict[TEST_SPLIT] = ds.remove_columns(cols)
                    print(
                        f"- 测试集 {TEST_SPLIT}: {len(ds_dict[TEST_SPLIT])} 条（来自 Speech-MASSIVE-test）"
                    )
                else:
                    print(f"- 语种 {lang} 在 test 数据集中不含 {TEST_SPLIT}，跳过。")
            else:
                print(f"- 测试数据缺失：{lang}。")

        if not ds_dict:
            print(f"- 语种 {lang} 无任何可保存的 split，跳过。")
            continue

        # 4) 保存到 ../data/preprocessed/{lang}
        out_path = os.path.join(OUT_DIR, lang)
        ensure_dir(out_path)
        out_ddict = DatasetDict(ds_dict)  # ty:ignore[no-matching-overload]
        out_ddict.save_to_disk(out_path)
        print(f"- 已保存到: {out_path}")
        print(f"- 最终包含 splits: {list(out_ddict.keys())}")

    print("\n全部完成。输出位于目录：", OUT_DIR)


if __name__ == "__main__":
    main()
