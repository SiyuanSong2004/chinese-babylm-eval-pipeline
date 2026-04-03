import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from ..utils.utils import get_model_and_tokenizer


MIN_WORDS = 80000
VALID_MIN = 3
REMOVE_EDGE_CHARS = True
USE_STANDARDIZATION = True
CHINESE_JSON_REL_PATH = os.path.join("eye_tracking", "eye_features_sentence_level.json")
INFER_CACHE_FILENAME = "eye_tracking_infer_cache.npz"
FAST_MAX_SENTENCES = 10
FAST_MIN_WORDS = 500


def merge_layer_output(sub_dict, total_dict=None):
	if total_dict is None:
		return sub_dict

	for k in total_dict:
		total_dict[k].extend(sub_dict[k])
	return total_dict


def merge_eye_matrix(sub_matrix, total_matrix=None):
	if total_matrix is None:
		return sub_matrix
	return np.concatenate([total_matrix, sub_matrix], axis=0)


def get_eye_features_matrix(split_feature, valid_num, valid_index, features=None):
	if features is None:
		features = ["FFD", "GD", "FPF", "FN", "RI", "RO", "LI_left", "LI_right", "TT"]

	feature_num = len(features)
	eye_matrix = np.zeros((valid_num, feature_num))
	current_row = 0
	for w_i in range(len(valid_index)):
		if valid_index[w_i]:
			for f_idx, f_n in enumerate(features):
				eye_matrix[current_row][f_idx] = split_feature[w_i][f_n]
			current_row += 1
	return eye_matrix


def find_valid_words(sentence_split):
	num_words = len(sentence_split)
	valid_index = [True for _ in range(num_words)]

	left_ignore_char = 0
	idx = 0
	while left_ignore_char < 3:
		valid_index[idx] = False
		left_ignore_char += len(sentence_split[idx])
		idx += 1

	right_ignore_char = 0
	idx = num_words - 1
	while right_ignore_char < 3:
		if sentence_split[idx] == "。":
			valid_index[idx] = False
			idx -= 1

		valid_index[idx] = False
		right_ignore_char += len(sentence_split[idx])
		idx -= 1

	return valid_index


def find_vocab_word(sentence_split, valid_index=None):
	num_words = len(sentence_split)
	valid_index = [True for _ in range(num_words)] if valid_index is None else valid_index
	return sum(valid_index), valid_index


def calculate_word_output_sent(model_outputs: torch.Tensor, split_words_list: list, output_index, valid_index):
	word_outputs = []
	num_words = 0

	for w_idx, _ in enumerate(split_words_list):
		if not valid_index[w_idx]:
			continue

		output_shape = model_outputs.shape
		if len(output_shape) == 3:
			word_average_output = torch.mean(model_outputs[0][output_index[w_idx]], dim=0).detach().cpu().numpy()
		elif len(output_shape) == 2:
			word_average_output = torch.mean(model_outputs[output_index[w_idx]], dim=0).detach().cpu().numpy()
		else:
			raise ValueError(f"invalid hidden shape: {output_shape}")

		word_outputs.append(word_average_output)
		num_words += 1

	return num_words, word_outputs


def get_num_layers(model):
	if not model.config.is_encoder_decoder:
		return model.config.num_hidden_layers
	return model.config.encoder_layers + model.config.decoder_layers


def _resolve_eye_tracking_json(data_path: str) -> str:
	return os.path.join(os.path.normpath(data_path), CHINESE_JSON_REL_PATH)


def _load_entries(json_path: str) -> list[dict]:
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	return list(data.values())


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
		valid_num, valid_index = find_vocab_word(split_words, valid_index=valid_words)
		if valid_num < VALID_MIN:
			continue

		split_feature = split_features[str(split_idx)]
		eye_matrix = get_eye_features_matrix(
			split_feature=split_feature,
			valid_num=valid_num,
			valid_index=valid_index,
		)

		if eye_matrix.size == 0 or np.any(np.sum(eye_matrix, axis=0) == 0):
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


def infer_eye_tracking(
	model_path_or_name: str,
	datapath: str,
	output_dir: str | None = None,
	save_predictions: bool = True,
	revision_name: str | None = None,
	fast: bool = False,
):
	model_name = os.path.basename(os.path.normpath(model_path_or_name))
	if output_dir is None:
		output_dir = os.path.join("results", model_name)

	json_path = _resolve_eye_tracking_json(datapath)
	entries = _load_entries(json_path)
	if not entries:
		raise ValueError(f"No valid eye-tracking entries found in: {json_path}")

	min_words = MIN_WORDS
	if fast:
		entries = entries[:FAST_MAX_SENTENCES]
		min_words = FAST_MIN_WORDS

	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name)
	n_layer = get_num_layers(model)

	merged_layers = None
	merged_eye = None

	start = time.time()

	for entry in tqdm(entries, desc="eye_tracking inference", unit="sent"):
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

		if merged_eye is not None and merged_eye.shape[0] >= min_words:
			break

	if merged_layers is None or merged_eye is None:
		raise ValueError("No valid eye-tracking samples were extracted.")

	layer_arrays = {
		f"layer_{layer_idx}": np.asarray(merged_layers[layer_idx], dtype=np.float32)
		for layer_idx in range(n_layer)
	}
	eye_matrix = np.asarray(merged_eye, dtype=np.float32)
	total_content_words = int(eye_matrix.shape[0])

	elapsed = time.time() - start

	report = {
		"task": "eye_tracking_infer",
		"fast": bool(fast),
		"model_path_or_name": model_path_or_name,
		"revision_name": revision_name,
		"eye_tracking_json_path": json_path,
		"num_input_sentences": int(len(entries)),
		"min_words": int(min_words),
		"valid_min": VALID_MIN,
		"remove_edge_chars": REMOVE_EDGE_CHARS,
		"num_layers": n_layer,
		"total_content_words": total_content_words,
		"elapsed_seconds": elapsed,
	}

	if save_predictions:
		result_dir = os.path.join(output_dir, "results", "eye_tracking")
		os.makedirs(result_dir, exist_ok=True)
		cache_path = os.path.join(result_dir, INFER_CACHE_FILENAME)
		np.savez_compressed(cache_path, eye_matrix=eye_matrix, **layer_arrays)

		report_path = os.path.join(result_dir, f"cogbench_eye_tracking_{model_name}_infer_report.json")
		with open(report_path, "w", encoding="utf-8") as f:
			json.dump(report, f, ensure_ascii=False, indent=2)
		print(f"saved cache: {cache_path}")
		print(f"saved infer report: {report_path}")
		return cache_path

	return {
		"eye_matrix": eye_matrix,
		"layers": layer_arrays,
		"report": report,
	}
