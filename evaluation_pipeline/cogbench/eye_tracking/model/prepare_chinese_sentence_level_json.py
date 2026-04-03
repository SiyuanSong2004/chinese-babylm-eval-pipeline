#!/usr/bin/env python3
"""Build Chinese sentence-level eye-tracking JSON from bundled XLSX files.

The output schema matches what EyeTrackingFeatures.get_sentence_level_info expects:
  evaluation_data/cogbench/eye_tracking/eye_features_sentence_level.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eye_tracking_features import EyeTrackingFeatures


OUTPUT_PATH = Path(
    "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/eye_tracking/eye_features_sentence_level.json"
)


# The runtime code expects these 9 keys.
FEATURE_KEYS = ["FFD", "GD", "FPF", "FN", "RI", "RO", "LI_left", "LI_right", "TT"]


# Mean-measure column name candidates in Main/Supplementary Measures sheets.
FEATURE_COLUMN_CANDIDATES = {
    "FFD": ["Mean_FFD", "FFD"],
    "GD": ["Mean_GD", "GD"],
    "FPF": ["Mean_FPF", "FPF"],
    "FN": ["Mean_FN", "FN"],
    "RI": ["Mean_RI", "RI"],
    "RO": ["Mean_RO", "RO"],
    "LI_left": ["Mean_LI_left", "LI_left"],
    # Historical naming differs in some files: LO_right vs LI_right.
    "LI_right": ["Mean_LI_right", "LI_right", "Mean_LO_right", "LO_right"],
    "TT": ["Mean_TT", "TT"],
}


def _safe_float(v: object) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_word_feature_cache(et: EyeTrackingFeatures) -> dict[str, dict[str, float]]:
    cache: dict[str, dict[str, float]] = {}
    mean_df = et.mean_features

    for word in mean_df.index.tolist():
        row = mean_df.loc[word]
        one = {}
        for f_key in FEATURE_KEYS:
            value = 0.0
            for col in FEATURE_COLUMN_CANDIDATES[f_key]:
                if col in row.index:
                    value = _safe_float(row[col])
                    break
            one[f_key] = value
        cache[str(word)] = one
    return cache


def build_dataset() -> dict[str, dict]:
    et = EyeTrackingFeatures()
    sentence_dict = et.get_sentence_dict()
    word_feature_cache = _build_word_feature_cache(et)

    data: dict[str, dict] = {}

    for sent_id, row in sentence_dict.items():
        sent_id_str = str(sent_id)
        content = str(row["Sentence"])

        num_split, split_words_list = et.get_sentence_splited(int(sent_id))

        split_features: dict[str, list[dict[str, float]]] = {}
        all_split: list[list[str]] = []

        for split_idx, split_words in enumerate(split_words_list):
            words = [str(w) for w in split_words]
            all_split.append(words)

            per_word = []
            for w in words:
                per_word.append(word_feature_cache.get(w, {k: 0.0 for k in FEATURE_KEYS}))
            split_features[str(split_idx)] = per_word

        data[sent_id_str] = {
            "content": content,
            "num": int(num_split),
            "all_split": all_split,
            "split_features": split_features,
        }

    return data


def main() -> None:
    data = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Sentences: {len(data)}")


if __name__ == "__main__":
    main()
