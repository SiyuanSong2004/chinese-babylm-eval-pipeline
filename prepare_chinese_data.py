"""
Download and convert Chinese evaluation datasets to JSONL format.

Zero-shot (NLU Track):
  - ZhoBLiMP (chinese-babylm-org/zhoblimp): Chinese minimal pairs

Zero-shot (Hanzi Track):
  - hanzi-structure (chinese-babylm-org/hanzi-structure): character structure minimal pairs
  - hanzi-pinyin (chinese-babylm-org/hanzi-pinyin): character phonology minimal pairs

Cog Track:
  - CogBench fMRI (zhiheng-qian/cogbench): fMRI brain data for ridge regression

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
import tarfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


def write_jsonl(data: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data)} examples to {path}")


# ──────────────────────────────────────────────
# ZhoBLiMP  (NLU Track)
# ──────────────────────────────────────────────

def prepare_zhoblimp(output_dir: pathlib.Path) -> None:
    """Download ZhoBLiMP from HuggingFace and extract each paradigm as a JSONL file."""
    print("=== ZhoBLiMP ===")
    full_dir = output_dir / "full_eval" / "zhoblimp"
    fast_dir = output_dir / "fast_eval" / "zhoblimp"

    repo_id = "chinese-babylm-org/zhoblimp"
    print(f"  Loading from {repo_id} ...")
    paradigm_count = 0

    for filename in list_repo_files(repo_id, repo_type="dataset"):
        if not filename.endswith(".jsonl"):
            continue
        paradigm_name = pathlib.Path(filename).stem
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

        rows = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                rows.append({
                    "sentence_good": item["sentence_good"],
                    "sentence_bad": item["sentence_bad"],
                    "UID": item.get("UID", paradigm_name),
                    "phenomenon": item.get("phenomenon", paradigm_name),
                })

        write_jsonl(rows, full_dir / f"{paradigm_name}.jsonl")
        write_jsonl(rows[:100], fast_dir / f"{paradigm_name}.jsonl")
        paradigm_count += 1

    print(f"  Processed {paradigm_count} paradigms")


# ──────────────────────────────────────────────
# Hanzi Track
# ──────────────────────────────────────────────

def prepare_hanzi_structure(output_dir: pathlib.Path) -> None:
    """Download hanzi-structure from HuggingFace and write as JSONL."""
    print("=== Hanzi Structure ===")
    full_dir = output_dir / "full_eval" / "hanzi_structure"
    fast_dir = output_dir / "fast_eval" / "hanzi_structure"

    repo_id = "chinese-babylm-org/hanzi-structure"
    print(f"  Loading from {repo_id} ...")
    ds = load_dataset(repo_id, split="train")

    rows = []
    for item in ds:
        rows.append({
            "sentence_good": item["sent_good"],
            "sentence_bad": item["sent_bad"],
            "UID": item["condition"],
            "phenomenon": item["Structure"],
        })

    write_jsonl(rows, full_dir / "hanzi_structure.jsonl")
    write_jsonl(rows[:100], fast_dir / "hanzi_structure.jsonl")


def prepare_hanzi_pinyin(output_dir: pathlib.Path) -> None:
    """Download hanzi-pinyin from HuggingFace and write as JSONL."""
    print("=== Hanzi Pinyin ===")
    full_dir = output_dir / "full_eval" / "hanzi_pinyin"
    fast_dir = output_dir / "fast_eval" / "hanzi_pinyin"

    repo_id = "chinese-babylm-org/hanzi-pinyin"
    print(f"  Loading from {repo_id} ...")
    ds = load_dataset(repo_id, split="train")

    rows = []
    for item in ds:
        rows.append({
            "sentence_good": item["sentence_good"],
            "sentence_bad": item["sentence_bad"],
            "UID": item["condition"],
            "phenomenon": item["condition"],
        })

    write_jsonl(rows, full_dir / "hanzi_pinyin.jsonl")
    write_jsonl(rows[:100], fast_dir / "hanzi_pinyin.jsonl")


# ──────────────────────────────────────────────
# CogBench  (Cog Track)
# ──────────────────────────────────────────────

COGBENCH_REPO = "zhiheng-qian/cogbench"
COGBENCH_TAR = "cogbench-fmri-0415.tar.gz"
COGBENCH_DIR = "cogbench-fmri-0415"


def prepare_cogbench(output_dir: pathlib.Path) -> None:
    """Download and extract CogBench fMRI data from HuggingFace."""
    print("=== CogBench ===")
    dest_dir = output_dir / COGBENCH_DIR

    if dest_dir.exists():
        print(f"  {dest_dir} already exists, skipping download.")
        return

    print(f"  Downloading {COGBENCH_TAR} from {COGBENCH_REPO} ...")
    local_path = hf_hub_download(repo_id=COGBENCH_REPO, filename=COGBENCH_TAR, repo_type="dataset")

    print(f"  Extracting to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_path, "r") as tar:
        tar.extractall(path=output_dir)

    print(f"  Done: {dest_dir}")


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
    prepare_hanzi_structure(args.output_dir)
    prepare_hanzi_pinyin(args.output_dir)
    prepare_cogbench(args.output_dir)
    prepare_afqmc(args.output_dir)
    prepare_ocnli(args.output_dir)
    prepare_tnews(args.output_dir)
    prepare_cluewsc2020(args.output_dir)

    print("\nDone! All Chinese evaluation data has been prepared.")


if __name__ == "__main__":
    main()
