import argparse
import glob
import os
import re
from typing import List

import numpy as np
import scipy.io as scio
import torch
from transformers import AutoModel, AutoTokenizer


BATCH_SIZE = 64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model_and_tokenizer(model_path_or_name: str, revision_name: str | None = None):
	model = AutoModel.from_pretrained(
		model_path_or_name,
		trust_remote_code=True,
		revision=revision_name,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_path_or_name,
		trust_remote_code=True,
		revision=revision_name,
		use_fast=True,
	)
	model = model.to(DEVICE)
	model.eval()

	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token

	return model, tokenizer


def parse_story_id(path: str) -> int:
	name = os.path.basename(path)
	match = re.search(r"story_(\d+)\.txt$", name)
	if not match:
		raise ValueError(f"Invalid story filename: {path}")
	return int(match.group(1))


def read_words_per_line(path: str) -> List[List[str]]:
	lines = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			words = line.split()
			if words:
				lines.append(words)
	return lines


def split_words_to_fit_model(words: List[str], tokenizer, max_content_tokens: int) -> List[List[str]]:
	if not words:
		return []

	chunks = []
	current = []
	for word in words:
		trial = current + [word]
		n_tokens = len(
			tokenizer(trial, is_split_into_words=True, add_special_tokens=False)["input_ids"]
		)
		if n_tokens <= max_content_tokens:
			current = trial
		else:
			if current:
				chunks.append(current)
			current = [word]
	if current:
		chunks.append(current)
	return chunks


def encode_words_mean_pool(
	words_per_line: List[List[str]],
	tokenizer,
	model,
	layer_index: int,
) -> np.ndarray:
	all_word_reprs = []
	special = tokenizer.num_special_tokens_to_add(pair=False)
	model_max_len = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
	max_len = int(min(tokenizer.model_max_length, model_max_len))
	max_content_tokens = max(1, max_len - special)

	for line_words in words_per_line:
		for words in split_words_to_fit_model(line_words, tokenizer, max_content_tokens):
			encoded_cpu = tokenizer(
				words,
				is_split_into_words=True,
				return_tensors="pt",
				truncation=False,
			)
			word_ids = encoded_cpu.word_ids(batch_index=0)
			encoded = {key: value.to(DEVICE) for key, value in encoded_cpu.items()}

			with torch.inference_mode():
				outputs = model(**encoded, output_hidden_states=True)
			hidden = outputs.hidden_states[layer_index][0]

			for word_idx in range(len(words)):
				token_positions = [idx for idx, wid in enumerate(word_ids) if wid == word_idx]
				if not token_positions:
					all_word_reprs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
					continue

				token_vecs = hidden[token_positions]
				word_vec = token_vecs.mean(dim=0)
				all_word_reprs.append(word_vec.detach().cpu().numpy().astype(np.float32))

	if not all_word_reprs:
		return np.zeros((0, model.config.hidden_size), dtype=np.float32)
	return np.stack(all_word_reprs, axis=0)


def infer_sentence(
	model_path_or_name: str,
	datapath: str,
	output_dir: str | None = None,
	revision_name: str | None = None,
	layer_index: int = -1,
):
	if output_dir is None:
		output_dir = os.path.join(datapath, "word_features", os.path.basename(model_path_or_name))

	os.makedirs(output_dir, exist_ok=True)

	script_dir = os.path.join(datapath, "story")
	story_files = sorted(glob.glob(os.path.join(script_dir, "story_*.txt")), key=parse_story_id)
	if not story_files:
		raise FileNotFoundError(f"No story files found in: {script_dir}")

	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name)

	for story_file in story_files:
		story_id = parse_story_id(story_file)
		words_per_line = read_words_per_line(story_file)
		data = encode_words_mean_pool(
			words_per_line=words_per_line,
			tokenizer=tokenizer,
			model=model,
			layer_index=layer_index,
		)

		save_path = os.path.join(output_dir, f"story_{story_id}.mat")
		scio.savemat(save_path, {"data": data})
		print(f"Saved {save_path}: data shape = {data.shape}")

	return output_dir


def main(args):
	infer_sentence(
		model_path_or_name=args.model_name,
		datapath=args.data_path,
		output_dir=args.output_dir,
		layer_index=args.layer,
		revision_name=args.revision_name,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Extract story-based word features for sentence-level cogbench tasks."
	)
	parser.add_argument(
		"--data_path",
		type=str,
		required=True,
		help="Cogbench data root containing the story directory.",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="Output directory for story_*.mat feature files.",
	)
	parser.add_argument(
		"--model_name",
		type=str,
		default="bert-base-chinese",
		help="Hugging Face model name or local model path.",
	)
	parser.add_argument(
		"--revision_name",
		type=str,
		default=None,
		help="Optional Hugging Face revision.",
	)
	parser.add_argument(
		"--layer",
		type=int,
		default=-1,
		help="Hidden layer index to extract.",
	)

	main(parser.parse_args())
