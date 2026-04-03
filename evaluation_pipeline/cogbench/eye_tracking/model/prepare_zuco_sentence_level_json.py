#!/usr/bin/env python3
"""Build ZuCo sentence-level eye-tracking JSON from ZuCo benchmark MAT files.

The output schema matches what `run_english.py` expects at
`--zuco_sentence_data_path`.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

import h5py
from tqdm import tqdm


_FEATURE_FIELDS = ["FFD", "TRT", "GD", "GPT", "SFD", "nFixations", "FFD_pupilsize"]
_INPUT_DIR = "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/zuco-benchmark/data/train"
_OUTPUT_PATH = "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/eye_features_sentence_level.zuco1_2.json"


def _decode_matlab_string(h5_file: h5py.File, ref: Any) -> str:
    data = h5_file[ref][:].squeeze()
    if getattr(data, "ndim", 0) == 0:
        return chr(int(data))
    return "".join(chr(int(c)) for c in data)


def _to_scalar(h5_file: h5py.File, ref: Any) -> float:
    arr = h5_file[ref][:]
    if arr.size == 0:
        return float("nan")
    value = float(arr.reshape(-1)[0])
    if math.isnan(value):
        return float("nan")
    return value


def _extract_word_list(h5_file: h5py.File, word_group: h5py.Group) -> list[str]:
    content_refs = word_group["content"]
    words: list[str] = []
    for i in range(len(content_refs)):
        words.append(_decode_matlab_string(h5_file, content_refs[i][0]))
    return words


def _extract_word_features(h5_file: h5py.File, word_group: h5py.Group) -> list[list[float]]:
    num_words = len(word_group["content"])
    field_to_refs = {}
    for field in _FEATURE_FIELDS:
        field_to_refs[field] = word_group[field] if field in word_group else None

    word_features: list[list[float]] = []
    for i in range(num_words):
        features_i: list[float] = []
        for field in _FEATURE_FIELDS:
            refs = field_to_refs[field]
            if refs is None:
                features_i.append(float("nan"))
                continue
            try:
                features_i.append(_to_scalar(h5_file, refs[i][0]))
            except Exception:
                features_i.append(float("nan"))
        word_features.append(features_i)
    return word_features


def build_dataset(input_dir: str) -> tuple[dict[str, Any], dict[str, int]]:
    pattern = os.path.join(input_dir, "resultsY*_*.mat")
    mat_files = sorted(glob.glob(pattern))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found under: {input_dir}")

    data: dict[str, Any] = {}
    skipped = defaultdict(int)

    for mat_path in tqdm(mat_files, desc="MAT files", unit="file"):
        base = os.path.basename(mat_path)
        experiment = base.rsplit("_", 1)[-1].split(".")[0]  # NR / TSR

        with h5py.File(mat_path, "r") as h5_file:
            sentence_data = h5_file["sentenceData"]
            sentence_contents = sentence_data["content"]
            sentence_words = sentence_data["word"]

            for sent_idx in tqdm(
                range(len(sentence_contents)),
                desc=f"{base} sentences",
                unit="sent",
                leave=False,
            ):
                sent_key = f"{experiment}-{sent_idx}"

                try:
                    sentence_text = _decode_matlab_string(h5_file, sentence_contents[sent_idx][0])
                except Exception:
                    skipped["invalid_sentence_text"] += 1
                    continue

                try:
                    word_group = h5_file[sentence_words[sent_idx][0]]
                except Exception:
                    skipped["invalid_word_reference"] += 1
                    continue

                if not isinstance(word_group, h5py.Group):
                    skipped["missing_word_group"] += 1
                    continue

                split_words = _extract_word_list(h5_file, word_group)
                split_feature_vectors = _extract_word_features(h5_file, word_group)

                if sent_key not in data:
                    data[sent_key] = {
                        "content": sentence_text,
                        "split_experiments": experiment,
                        "num": 1,
                        "all_split": [split_words],
                        "split_features": {"0": [[vec] for vec in split_feature_vectors]},
                        "subject_count": 1,
                    }
                    continue

                existing = data[sent_key]
                if sentence_text != existing["content"]:
                    skipped["content_mismatch"] += 1
                    continue

                existing_words = existing["all_split"][0]
                if len(existing_words) != len(split_words) or existing_words != split_words:
                    skipped["tokenization_mismatch"] += 1
                    continue

                for w_idx, vec in enumerate(split_feature_vectors):
                    existing["split_features"]["0"][w_idx].append(vec)
                existing["subject_count"] += 1

    return data, dict(skipped)


def main() -> None:
    dataset, skipped = build_dataset(_INPUT_DIR)

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    nr_count = sum(1 for k in dataset if k.startswith("NR-"))
    tsr_count = sum(1 for k in dataset if k.startswith("TSR-"))
    print(f"Wrote: {_OUTPUT_PATH}")
    print(f"Sentences: total={len(dataset)} NR={nr_count} TSR={tsr_count}")
    print(f"Skipped due to mismatch: {skipped}")


if __name__ == "__main__":
    main()
