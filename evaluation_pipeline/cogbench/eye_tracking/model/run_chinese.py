import json
import pathlib
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utills import (
    calculate_word_output_sent,
    find_valid_words,
    find_vocab_word,
    get_layer_similarity,
    get_num_layers,
    get_zuco_eye_features_matrix,
    merge_eye_matrix,
    merge_layer_output,
    pearson_sim,
)


# ============================
# Constants (edit these only)
# ============================
MODEL_PATH_OR_NAME = "/mnt/models/Qwen3-1.7B"
ZUCO_JSON_PATH = "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/eye_features_sentence_level.zuco1_2.json"
OUTPUT_DIR = "results"
MIN_WORDS = 80000
VALID_MIN = 3
REMOVE_EDGE_CHARS = True
USE_STANDARDIZATION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXPERIMENTS = {"NR", "SR", "ZUCO2-NR"}


def _word_spans(sentence: str, words: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for word in words:
        start = sentence.find(word, cursor)
        if start == -1:
            raise ValueError(f"Cannot align word '{word}' in sentence: {sentence}")
        end = start + len(word)
        spans.append((start, end))
        cursor = end
    return spans


def _map_words_to_tokens(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[list[int]]:
    token_indices = []
    for ws, we in spans:
        hits = []
        for tok_i, (ts, te) in enumerate(offsets):
            if te <= ts:
                continue
            if ts < we and te > ws:
                hits.append(tok_i)
        token_indices.append(hits)
    return token_indices


def _load_zuco_sentences(path: str) -> list[tuple[str, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for sid, entry in data.items():
        if entry.get("split_experiments") in VALID_EXPERIMENTS:
            items.append((sid, entry))
    return items


def _sentence_features(entry: dict, tokenizer, model, n_layer: int):
    sentence = entry["content"]
    all_split = entry["all_split"]
    split_features = entry["split_features"]

    encoded = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [tuple(x) for x in encoded.pop("offset_mapping")[0].tolist()]
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    model_outputs = model(**encoded, output_hidden_states=True).hidden_states

    layer_word_outputs = None
    eye_matrix_merged = None

    for split_idx, split_words in enumerate(all_split):
        valid_words = find_valid_words(split_words) if REMOVE_EDGE_CHARS else None
        valid_num, valid_index = find_vocab_word(split_words, vocab=None, valid_index=valid_words)
        if valid_num < VALID_MIN:
            continue

        eye_matrix = get_zuco_eye_features_matrix(
            split_feature=split_features[str(split_idx)],
            valid_num=valid_num,
            valid_index=valid_index,
        )

        if np.any(np.sum(eye_matrix, axis=0) == 0):
            continue

        spans = _word_spans(sentence, split_words)
        word_to_token = _map_words_to_tokens(offsets, spans)

        layer_dict = {}
        for layer_idx in range(n_layer):
            _, word_outputs = calculate_word_output_sent(
                model_outputs=model_outputs[layer_idx + 1],
                split_words_list=split_words,
                output_index=word_to_token,
                valid_index=valid_index,
            )
            layer_dict[layer_idx] = word_outputs

        layer_word_outputs = merge_layer_output(layer_dict, layer_word_outputs)
        eye_matrix_merged = merge_eye_matrix(eye_matrix, eye_matrix_merged)

    return layer_word_outputs, eye_matrix_merged


def run_alignment():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_OR_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_PATH_OR_NAME, ignore_mismatched_sizes=True)
    model = model.to(DEVICE)
    model.eval()

    entries = _load_zuco_sentences(ZUCO_JSON_PATH)

    n_layer = get_num_layers(model)
    layer_similarity = {i: [] for i in range(n_layer)}

    total_content_words = 0
    merged_layers = None
    merged_eye = None

    start = time.time()

    for _, entry in tqdm(entries, desc="eye_tracking inference", unit="sent"):
        layer_dict, eye_matrix = _sentence_features(
            entry=entry,
            tokenizer=tokenizer,
            model=model,
            n_layer=n_layer,
        )

        if layer_dict is None:
            continue

        merged_layers = merge_layer_output(layer_dict, merged_layers)
        merged_eye = merge_eye_matrix(eye_matrix, merged_eye)

        if merged_eye is not None and merged_eye.shape[0] >= MIN_WORDS:
            for layer_idx in tqdm(
                range(n_layer),
                desc="layer regression",
                unit="layer",
                leave=False,
            ):
                avg_sim, sims = get_layer_similarity(
                    merged_layers[layer_idx],
                    merged_eye,
                    similarity_metric=pearson_sim,
                    standardize=USE_STANDARDIZATION,
                )
                layer_similarity[layer_idx].append(sims)
                print(f"layer={layer_idx + 1} avg={avg_sim:.6f} sims={sims}")

            total_content_words += int(merged_eye.shape[0])
            merged_layers = None
            merged_eye = None

    if merged_eye is not None:
        for layer_idx in tqdm(
            range(n_layer),
            desc="final layer regression",
            unit="layer",
            leave=False,
        ):
            avg_sim, sims = get_layer_similarity(
                merged_layers[layer_idx],
                merged_eye,
                similarity_metric=pearson_sim,
                standardize=USE_STANDARDIZATION,
            )
            layer_similarity[layer_idx].append(sims)
            print(f"layer={layer_idx + 1} avg={avg_sim:.6f} sims={sims}")
        total_content_words += int(merged_eye.shape[0])

    layer_mean_similarity = []
    for layer_idx in range(n_layer):
        scores = np.array(layer_similarity[layer_idx], dtype=float)
        if scores.size == 0:
            layer_mean_similarity.append(float("nan"))
            continue
        layer_mean_similarity.append(float(np.nanmean(np.nanmean(scores, axis=0))))

    elapsed = time.time() - start

    result_dir = pathlib.Path(OUTPUT_DIR) / pathlib.Path(MODEL_PATH_OR_NAME).name / "results" / "eye_tracking"
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / f"cogbench_eye_tracking_{pathlib.Path(MODEL_PATH_OR_NAME).name}_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "eye_tracking",
                "model_path_or_name": MODEL_PATH_OR_NAME,
                "zuco_json_path": ZUCO_JSON_PATH,
                "device": DEVICE,
                "min_words": MIN_WORDS,
                "valid_min": VALID_MIN,
                "remove_edge_chars": REMOVE_EDGE_CHARS,
                "standardize": USE_STANDARDIZATION,
                "num_layers": n_layer,
                "total_content_words": total_content_words,
                "layer_mean_similarity": layer_mean_similarity,
                "elapsed_seconds": elapsed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved report: {report_path}")


if __name__ == "__main__":
    run_alignment()
