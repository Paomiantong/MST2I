import os
from datasets import load_dataset, load_from_disk
from formatter_v2 import get_fixed_target
import pandas as pd

MASSIVE_PATH = os.environ["MASSIVE_PATH"]
LANGS = [
    "ar-SA",
    "de-DE",
    "en-US",
    "es-ES",
    "fr-FR",
    "hu-HU",
    "ko-KR",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ru-RU",
    "tr-TR",
    "vi-VN",
]


def load(lang, split=None):
    return load_dataset(MASSIVE_PATH, lang, split=split)


massive_en = load("en-US")
cl = massive_en["train"].features["intent"]


def get_dataset(lang, split):
    ret = []
    for src, tgt in zip(load(lang, split), massive_en[split]):
        ret.append(
            {
                "id": src["id"],
                "src_utt": src["utt"],
                "src_annot_utt": src["annot_utt"],
                "tgt_utt": tgt["utt"],
                "tgt_annot_utt": tgt["annot_utt"],
                "fixed_tgt_annot_utt": get_fixed_target(
                    src["annot_utt"], tgt["annot_utt"], src["slot_method"]
                ),
                "slot_method": src["slot_method"],
                "intent_str": cl.int2str(src["intent"]),
                "src_locale": lang,
            }
        )
    return pd.DataFrame(ret)


smassive_de = load_from_disk("../data/preprocessed/de-DE")
sampled_train = smassive_de["train"]["id"]

for lang in LANGS:
    os.makedirs("../data/sampled_massive/" + lang, exist_ok=True)
    df = get_dataset(lang, "train")
    df = df[df["id"].apply(lambda x: x in sampled_train)]
    df.to_csv("../data/sampled_massive/" + lang + "/train_full.csv", index=False)
