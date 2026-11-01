import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fire import Fire
from sklearn.metrics import accuracy_score
from table import dataframe_to_markdown

# Optional deps
try:
    import sacrebleu
except Exception:
    sacrebleu = None  # type: ignore
try:
    from comet import download_model, load_from_checkpoint
except Exception:
    download_model = None  # type: ignore
    load_from_checkpoint = None  # type: ignore

# Hard-coded config
DATASET_PATH = os.environ.get("MST2I_EVAL_DATASET", "data/formatted/baseline/eval/test.jsonl")
GROUP_COL = "locale"
SOURCE_COL = "src_utt"  # source text column

# Regex
slot_pat = re.compile(r"\[([^\]:]+?) : (.*?)\]")
intent_pat = re.compile(r"^([\w_]+) \|")


# ---- Text helpers ----
def _intent(text: Any) -> str:
    if not isinstance(text, str):
        return "Unk"
    m = intent_pat.search(text)
    return m.group(1).strip() if m else "Unk"


def _clean_for_gen(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = intent_pat.sub("", text)
    s = slot_pat.sub(lambda m: m.group(2), s)  # slot -> entity
    return " ".join(s.split())


# ---- Slot/Intent ----
def _get_slots(text: Any):
    if not isinstance(text, str):
        return set()
    return set(slot_pat.findall(text))


def eval_slot(df: pd.DataFrame) -> Dict[str, float]:
    TP = FP = FN = 0
    for item in df.itertuples():
        pred, gold = (
            _get_slots(getattr(item, "predict", None)),
            _get_slots(getattr(item, "label", None)),
        )
        TP += len(pred & gold)
        FP += len(pred - gold)
        FN += len(gold - pred)
    p = TP / (TP + FP) if TP + FP > 0 else 0.0
    r = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def eval_intent(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "intent_accuracy": accuracy_score(
            df.get("label", pd.Series()).map(_intent),
            df.get("predict", pd.Series()).map(_intent),
        )
    }


# ---- BLEU ----
def eval_bleu(df: pd.DataFrame) -> Dict[str, float]:
    if sacrebleu is None:
        return {"bleu": float("nan")}
    preds = df.get("predict", pd.Series(dtype=object)).map(_clean_for_gen).tolist()
    refs = df.get("label", pd.Series(dtype=object)).map(_clean_for_gen).tolist()
    return {"bleu": float(sacrebleu.corpus_bleu(preds, [refs]).score)}


# ---- COMET (single pass) ----
_COMET_CACHE: Dict[str, Any] = {}


def _load_comet(model_name: str):
    # 1) env ckpt
    env = os.getenv("COMET_PATH")
    if env:
        key = f"env:{env}"
        if key not in _COMET_CACHE:
            try:
                _COMET_CACHE[key] = (
                    load_from_checkpoint(env, reload_hparams=True)
                    if load_from_checkpoint
                    else None
                )
            except Exception:
                _COMET_CACHE[key] = None
        return _COMET_CACHE[key]
    # 2) local ckpt path
    if (isinstance(model_name, str) and model_name.endswith(".ckpt")) or os.path.exists(
        str(model_name)
    ):
        key = f"ckpt:{model_name}"
        if key not in _COMET_CACHE:
            try:
                _COMET_CACHE[key] = (
                    load_from_checkpoint(model_name, reload_hparams=True)
                    if load_from_checkpoint
                    else None
                )
            except Exception:
                _COMET_CACHE[key] = None
        return _COMET_CACHE[key]
    # 3) download by name
    key = f"name:{model_name}"
    if key not in _COMET_CACHE:
        try:
            if download_model and load_from_checkpoint:
                ckpt = download_model(model_name)
                _COMET_CACHE[key] = load_from_checkpoint(ckpt)
            else:
                _COMET_CACHE[key] = None
        except Exception:
            _COMET_CACHE[key] = None
    return _COMET_CACHE[key]


def comet_segment_scores(
    df: pd.DataFrame,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    source_col: str = SOURCE_COL,
) -> Tuple[float, List[float]]:
    model = _load_comet(model_name)
    if model is None or source_col not in df.columns:
        return float("nan"), [float("nan")] * len(df)
    preds = df["predict"].map(_clean_for_gen).tolist()
    refs = df["label"].map(_clean_for_gen).tolist()
    srcs = df[source_col].map(_clean_for_gen).tolist()
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    try:
        out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=True)
        scores = None
        system = float("nan")
        if isinstance(out, dict):
            scores = out.get("scores")
            system = float(out.get("system_score", float("nan")))
        if scores is None:
            if isinstance(out, (list, tuple)):
                scores = list(out)
            if scores is not None and (pd.isna(system) or system is None):
                system = (
                    float(sum(scores) / len(scores))
                    if len(scores) > 0
                    else float("nan")
                )
        if scores is None or len(scores) != len(df):
            scores = [float("nan")] * len(df)
        return float(system), [
            float(x) if x is not None else float("nan") for x in scores
        ]
    except Exception:
        return float("nan"), [float("nan")] * len(df)


# ---- Group (no COMET here to avoid recompute) ----
def eval_by_group(
    df: pd.DataFrame, group_col: str = GROUP_COL, compute_bleu: bool = True
) -> Optional[Dict[Any, Dict[str, float]]]:
    if group_col not in df.columns:
        return None
    res: Dict[Any, Dict[str, float]] = {}
    for key, sub in df.groupby(group_col, dropna=False):
        m = eval_slot(sub)
        m.update(eval_intent(sub))
        if compute_bleu:
            m.update(eval_bleu(sub))
        res[key] = m
    return res


# ---- Utilities ----
def _attach_col(
    df: pd.DataFrame, dataset_df: Optional[pd.DataFrame], col: str, notes: List[str]
):
    if col in df.columns or dataset_df is None:
        return
    if col in dataset_df.columns and len(dataset_df) == len(df):
        df[col] = dataset_df[col]
    else:
        notes.append(
            f"Unable to attach '{col}' from dataset (missing column or length mismatch)."
        )


# ---- CLI ----
def eval_files(
    *files,
    compute_bleu: bool = True,
    compute_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
    comet_batch_size: int = 8,
    source_col: Optional[str] = SOURCE_COL,
):
    if not files:
        print("No input files provided.")
        return

    try:
        dataset_df = pd.read_json(DATASET_PATH, lines=True, orient="records")
    except Exception:
        dataset_df = None

    results_sections: List[str] = []
    for file in files:
        notes: List[str] = []
        df = pd.read_json(file, lines=True, orient="records")
        if any(mask := df["predict"].isna()):
            for item in df[mask].itertuples():
                print(item)
            continue

        df["predict"] = df["predict"].apply(lambda x: x.splitlines()[-1].lower())
        df["label"] = df["label"].apply(lambda x: x.splitlines()[-1].lower())

        print(df["predict"])
        print(df["label"])

        _attach_col(df, dataset_df, GROUP_COL, notes)
        _attach_col(df, dataset_df, source_col or SOURCE_COL, notes)

        # Compute totals
        by: Dict[Any, Dict[str, float]] = {"total": {}}
        by["total"].update(eval_intent(df))
        by["total"].update(eval_slot(df))
        if compute_bleu:
            by["total"].update(eval_bleu(df))
            if sacrebleu is None:
                notes.append("sacrebleu not installed; BLEU=NaN.")

        comet_scores: Optional[List[float]] = None
        if compute_comet:
            sys_score, seg_scores = comet_segment_scores(
                df,
                model_name=comet_model,
                batch_size=comet_batch_size,
                source_col=source_col or SOURCE_COL,
            )
            by["total"]["comet"] = sys_score
            comet_scores = seg_scores
            if load_from_checkpoint is None:
                notes.append("COMET not installed; COMET=NaN.")
            elif (
                all(pd.isna(seg_scores))
                and (source_col or SOURCE_COL) not in df.columns
            ):
                notes.append(
                    f"No source column for COMET; expected '{source_col or SOURCE_COL}'."
                )

        # Per-group (no COMET recompute)
        grouped = eval_by_group(df, group_col=GROUP_COL, compute_bleu=compute_bleu)
        if grouped:
            by.update(grouped)
            if compute_comet and comet_scores is not None:
                df["_comet"] = comet_scores
                for k, sub in df.groupby(GROUP_COL, dropna=False):
                    by.setdefault(k, {})["comet"] = float(
                        pd.to_numeric(sub["_comet"], errors="coerce").mean()
                    )
                del df["_comet"]

        # Notes: print in place
        for n in notes:
            print(f"Note: {n}")

        # Collect results for final once-for-all output
        table = dataframe_to_markdown(
            pd.DataFrame(by).T, index=True, float_format="%.4f"
        )
        results_sections.append("\n".join([f"File: {file}", table, ""]))

    # Final output: only results
    print("\n".join(results_sections))


if __name__ == "__main__":
    Fire(eval_files)
