"""
Download and convert Chinese evaluation datasets to JSONL format.

Zero-shot:
  - ZhoBLiMP (Junrui1202/zhoblimp): Chinese minimal pairs

Fine-tuning (CLUE):
  - AFQMC: sentence similarity
  - OCNLI: natural language inference
  - TNEWS: news topic classification
  - CLUEWSC2020: pronoun disambiguation

Usage:
    python prepare_chinese_data.py [--output_dir evaluation_data]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from datasets import load_dataset, get_dataset_config_names


def write_jsonl(data: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data)} examples to {path}")


# ──────────────────────────────────────────────
# ZhoBLiMP
# ──────────────────────────────────────────────

def prepare_zhoblimp(output_dir: pathlib.Path) -> None:
    """Download ZhoBLiMP and convert each paradigm to a JSONL file."""
    print("=== ZhoBLiMP ===")
    full_dir = output_dir / "full_eval" / "zhoblimp"
    fast_dir = output_dir / "fast_eval" / "zhoblimp"

    configs = get_dataset_config_names("Junrui1202/zhoblimp")
    print(f"  Found {len(configs)} paradigms")

    for config in configs:
        ds = load_dataset("Junrui1202/zhoblimp", config, split="train")
        rows = []
        for item in ds:
            rows.append({
                "sentence_good": item["sentence_good"],
                "sentence_bad": item["sentence_bad"],
                "UID": item.get("UID", config),
                "phenomenon": item.get("phenomenon", config),
            })

        # Full eval: all examples
        write_jsonl(rows, full_dir / f"{config}.jsonl")

        # Fast eval: subsample to 100 examples per paradigm
        fast_rows = rows[:100]
        write_jsonl(fast_rows, fast_dir / f"{config}.jsonl")


# ──────────────────────────────────────────────
# CLUE fine-tuning tasks
# ──────────────────────────────────────────────

def prepare_afqmc(output_dir: pathlib.Path) -> None:
    """AFQMC: sentence similarity, 2 labels."""
    print("=== AFQMC ===")
    clue_dir = output_dir / "full_eval" / "clue"

    for split_name, out_name in [("train", "afqmc.train"), ("validation", "afqmc.valid")]:
        ds = load_dataset("clue", "afqmc", split=split_name)
        rows = []
        for item in ds:
            rows.append({
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "label": item["label"],
            })
        write_jsonl(rows, clue_dir / f"{out_name}.jsonl")


def prepare_ocnli(output_dir: pathlib.Path) -> None:
    """OCNLI: NLI, 3 labels (0=neutral, 1=entailment, 2=contradiction)."""
    print("=== OCNLI ===")
    clue_dir = output_dir / "full_eval" / "clue"

    for split_name, out_name in [("train", "ocnli.train"), ("validation", "ocnli.valid")]:
        ds = load_dataset("clue", "ocnli", split=split_name)
        rows = []
        for item in ds:
            # Skip examples with label -1 (unlabeled)
            if item["label"] == -1:
                continue
            rows.append({
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "label": item["label"],
            })
        write_jsonl(rows, clue_dir / f"{out_name}.jsonl")


def prepare_tnews(output_dir: pathlib.Path) -> None:
    """TNEWS: news topic classification, 15 labels."""
    print("=== TNEWS ===")
    clue_dir = output_dir / "full_eval" / "clue"

    for split_name, out_name in [("train", "tnews.train"), ("validation", "tnews.valid")]:
        ds = load_dataset("clue", "tnews", split=split_name)
        # Build a mapping from original label codes to 0-indexed labels
        label_set = sorted(set(item["label"] for item in ds))
        label_map = {orig: idx for idx, orig in enumerate(label_set)}
        rows = []
        for item in ds:
            rows.append({
                "sentence": item["sentence"],
                "label": label_map[item["label"]],
            })
        write_jsonl(rows, clue_dir / f"{out_name}.jsonl")


def prepare_cluewsc2020(output_dir: pathlib.Path) -> None:
    """CLUEWSC2020: pronoun disambiguation, 2 labels.
    Flattens nested target dict to top-level span fields.
    """
    print("=== CLUEWSC2020 ===")
    clue_dir = output_dir / "full_eval" / "clue"

    for split_name, out_name in [("train", "cluewsc2020.train"), ("validation", "cluewsc2020.valid")]:
        ds = load_dataset("clue", "cluewsc2020", split=split_name)
        rows = []
        for item in ds:
            target = item["target"]
            rows.append({
                "text": item["text"],
                "span1_text": target["span1_text"],
                "span2_text": target["span2_text"],
                "label": item["label"],
            })
        write_jsonl(rows, clue_dir / f"{out_name}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Prepare Chinese evaluation data")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("evaluation_data"))
    args = parser.parse_args()

    prepare_zhoblimp(args.output_dir)
    prepare_afqmc(args.output_dir)
    prepare_ocnli(args.output_dir)
    prepare_tnews(args.output_dir)
    prepare_cluewsc2020(args.output_dir)

    print("\nDone! All Chinese evaluation data has been prepared.")


if __name__ == "__main__":
    main()
