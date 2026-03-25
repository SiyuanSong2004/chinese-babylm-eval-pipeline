import argparse
import json
import os

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

try:
	from inference.infer_sentence import DEVICE, get_model_and_tokenizer
except ImportError:
	from infer_sentence import DEVICE, get_model_and_tokenizer


SAVE_PREDICTIONS = True
BATCH_SIZE = 64


def standardize_matrix(matrix):
	row_means = np.mean(matrix, axis=1, keepdims=True)
	row_stds = np.std(matrix, axis=1, keepdims=True)
	row_stds[row_stds == 0] = 1.0
	return (matrix - row_means) / row_stds


def ridge_prediction(X_train, X_test, y_train):
	model = Ridge()
	alphas = np.logspace(-4, 4, 10)
	param_grid = {"alpha": alphas}
	kf = KFold(n_splits=5, shuffle=False)

	X_train = np.nan_to_num(X_train, 0.0)
	X_test = np.nan_to_num(X_test, 0.0)

	grid = GridSearchCV(
		model,
		param_grid,
		scoring="neg_mean_squared_error",
		cv=kf,
		n_jobs=-1,
		verbose=0,
	)
	grid.fit(X_train, y_train)
	return grid.predict(X_test)


def run_prediction(feature, fmri, save_path):
	X = standardize_matrix(np.asarray(feature))
	Y = np.asarray(fmri)

	kf = KFold(n_splits=5, shuffle=False)
	all_corrs = []

	for train_idx, test_idx in kf.split(X):
		X_tr, X_te = X[train_idx], X[test_idx]
		Y_tr, Y_te = Y[train_idx], Y[test_idx]

		Y_pred = ridge_prediction(X_tr, X_te, Y_tr)

		valid_cols = ~np.all(Y_te == 0, axis=0)
		if not np.any(valid_cols):
			all_corrs.append(0.0)
			continue

		Y_pred = Y_pred[:, valid_cols]
		Y_te = Y_te[:, valid_cols]

		trial_corrs = np.array([
			pearsonr(Y_pred[i], Y_te[i])[0] for i in range(Y_te.shape[0])
		])
		trial_corrs[np.isnan(trial_corrs)] = 0.0

		k = max(1, int(len(trial_corrs) * 0.1))
		top_mean = np.mean(np.sort(trial_corrs)[-k:]) if k > 0 else 0.0
		all_corrs.append(top_mean)

	final_score = float(np.mean(all_corrs))

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	sio.savemat(save_path, {"score": final_score})
	print(f"Saved: {save_path} | overall top-10% mean pearson r = {final_score:.4f}")
	return final_score


def _mean_pool_last_hidden(last_hidden_state, attention_mask=None):
	if attention_mask is None:
		return last_hidden_state.mean(dim=1)

	mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
	masked_hidden = last_hidden_state * mask
	token_count = mask.sum(dim=1).clamp(min=1e-9)
	return masked_hidden.sum(dim=1) / token_count


def extract_word_features(words, model, tokenizer, batch_size=BATCH_SIZE):
	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token

	word_features = {}
	for start in range(0, len(words), batch_size):
		batch_words = words[start:start + batch_size]
		inputs = tokenizer(
			batch_words,
			return_tensors="pt",
			padding=True,
			truncation=True,
		)
		inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

		with torch.inference_mode():
			outputs = model(**inputs)

		pooled = _mean_pool_last_hidden(outputs.last_hidden_state, inputs.get("attention_mask"))
		pooled = pooled.detach().cpu().numpy().astype(np.float32)

		for index, word in enumerate(batch_words):
			word_features[word] = pooled[index]

	return word_features


def evaluate_word_fmri(word_features, words, datapath):
	if not word_features:
		print("No word features found, skip evaluation.")
		return

	script_path = os.path.join(datapath, "word", "script.txt")
	if os.path.exists(script_path):
		with open(script_path, "r", encoding="utf-8") as f:
			stimuli_list = [line.strip() for line in f if line.strip()]
	else:
		stimuli_list = words

	hidden_size = len(next(iter(word_features.values())))
	missing_words = []
	feature_matrix = []
	for stimulus in stimuli_list:
		feature = word_features.get(stimulus)
		if feature is None:
			missing_words.append(stimulus)
			feature = np.zeros(hidden_size, dtype=np.float32)
		feature_matrix.append(feature)
	feature_matrix = np.asarray(feature_matrix)

	if missing_words:
		print(f"Warning: {len(missing_words)} stimuli missing in word_features, filled with zeros.")

	fmri_dir = os.path.join(datapath, "word_fmri")
	fmri_files = [
		os.path.join(fmri_dir, name)
		for name in sorted(os.listdir(fmri_dir))
		if name.endswith("_selected.mat")
	] if os.path.isdir(fmri_dir) else []
	if not fmri_files:
		print(f"No *_selected.mat files found under {datapath}, skip word_fmri evaluation.")
		return

	for fmri_path in fmri_files:
		subject = os.path.basename(fmri_path).replace("_selected.mat", "")
		mat = sio.loadmat(fmri_path)
		if "examples" not in mat:
			print(f"Skip {fmri_path}: key 'examples' not found.")
			continue

		fmri_data = mat["examples"]
		n_trials = min(feature_matrix.shape[0], fmri_data.shape[0])
		if n_trials < 5:
			print(f"Skip {fmri_path}: not enough aligned trials ({n_trials}).")
			continue

		if feature_matrix.shape[0] != fmri_data.shape[0]:
			print(
				f"Warning: trial count mismatch for {subject} "
				f"(feature={feature_matrix.shape[0]}, fmri={fmri_data.shape[0]}), "
				f"truncate to {n_trials}."
			)

		save_path = os.path.join(datapath, "word", "results", "word_features", f"{subject}_score.mat")
		run_prediction(feature_matrix[:n_trials], fmri_data[:n_trials], save_path)


def infer_word(
	model_path_or_name: str,
	datapath: str,
	save_predictions: bool = SAVE_PREDICTIONS,
	revision_name: str | None = None,
):
	data_path = os.path.join(datapath, "word", "word.txt")
	with open(data_path, "r", encoding="utf-8") as f:
		words = f.read().splitlines()

	model, tokenizer = get_model_and_tokenizer(model_path_or_name, revision_name=revision_name)
	word_features = extract_word_features(words, model, tokenizer)

	if save_predictions:
		save_path = os.path.join(datapath, os.path.basename(model_path_or_name), "word_feature.json")
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		serializable_features = {word: feature.tolist() for word, feature in word_features.items()}
		with open(save_path, "w", encoding="utf-8") as f:
			json.dump(serializable_features, f, ensure_ascii=False)

	evaluate_word_fmri(word_features, words, datapath)
	return word_features


def main(args):
	infer_word(
		model_path_or_name=args.model_name,
		datapath=args.data_path,
		save_predictions=not args.no_save_predictions,
		revision_name=args.revision_name,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extract and evaluate word features for word_fmri.")
	parser.add_argument("--data_path", type=str, required=True, help="Cogbench data root.")
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
		"--no_save_predictions",
		action="store_true",
		help="Disable saving word_feature.json.",
	)

	main(parser.parse_args())
