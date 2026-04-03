import json
import os

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm


USE_STANDARDIZATION = True
INFER_CACHE_FILENAME = "eye_tracking_infer_cache.npz"


def pearson_sim(v_i, v_j):
	return pearsonr(v_i, v_j)[0]


def standardize_matrix(feature_matrix, mean=None, std=None):
	mean = np.nanmean(feature_matrix, axis=0) if mean is None else mean
	std = np.nanstd(feature_matrix, axis=0) if std is None else std
	return (feature_matrix - mean) / std


def calculate_rsm_list(w_vectors, similarity_metric=pearson_sim):
	num_words = len(w_vectors)
	rsm_sent = np.zeros((num_words, num_words), dtype=np.float32)

	for i in range(num_words):
		feature_i = w_vectors[i]
		for j in range(i, num_words):
			feature_j = w_vectors[j]
			similarity_ij = similarity_metric(feature_i, feature_j)
			rsm_sent[i][j] = similarity_ij
			rsm_sent[j][i] = similarity_ij

	return rsm_sent


def calculate_similarity(model_rsm, feature_matrix, standardize=True, similarity_metric=pearson_sim):
	if standardize:
		feature_matrix = standardize_matrix(feature_matrix)

	w_num = model_rsm.shape[0]
	model_matrix = np.matmul((model_rsm - np.identity(w_num)), feature_matrix)

	similarity_sum = 0.0
	similarities = []
	feature_num = feature_matrix.shape[1]
	for feature_i in range(feature_num):
		model_features = model_matrix[:, feature_i]
		eye_features = feature_matrix[:, feature_i]
		similarity_i = similarity_metric(model_features, eye_features)
		similarity_sum += similarity_i
		similarities.append(float(similarity_i))

	return similarity_sum / feature_num, similarities


def get_layer_similarity(word_vectors, eye_matrix, similarity_metric=pearson_sim, standardize=True):
	layer_rsm = calculate_rsm_list(word_vectors, similarity_metric=similarity_metric)
	return calculate_similarity(layer_rsm, eye_matrix, standardize=standardize, similarity_metric=similarity_metric)


def eval_eye_tracking(args):
	output_root = str(args.output_dir)
	model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
	result_dir = os.path.join(output_root, model_name, "results", "eye_tracking")

	cache_path = os.path.join(result_dir, INFER_CACHE_FILENAME)
	if not os.path.exists(cache_path):
		raise FileNotFoundError(f"Eye-tracking inference cache not found: {cache_path}")

	cache = np.load(cache_path)
	eye_matrix = np.asarray(cache["eye_matrix"], dtype=np.float32)
	layer_keys = sorted([key for key in cache.files if key.startswith("layer_")], key=lambda x: int(x.split("_")[1]))

	if eye_matrix.shape[0] == 0 or not layer_keys:
		raise ValueError(f"Invalid eye-tracking cache content: {cache_path}")

	layer_similarity = []
	layer_feature_similarity = {}

	for layer_key in tqdm(layer_keys, desc="eye_tracking regression", unit="layer"):
		layer_idx = int(layer_key.split("_")[1])
		word_vectors = np.asarray(cache[layer_key], dtype=np.float32)

		avg_sim, sims = get_layer_similarity(
			word_vectors=word_vectors,
			eye_matrix=eye_matrix,
			similarity_metric=pearson_sim,
			standardize=USE_STANDARDIZATION,
		)

		print(f"layer={layer_idx + 1} avg={avg_sim:.6f} sims={sims}")
		layer_similarity.append(float(avg_sim))
		layer_feature_similarity[str(layer_idx)] = sims

	report = {
		"task": "eye_tracking",
		"model_name": model_name,
		"cache_path": cache_path,
		"num_layers": len(layer_keys),
		"total_content_words": int(eye_matrix.shape[0]),
		"layer_mean_similarity": layer_similarity,
		"layer_feature_similarity": layer_feature_similarity,
		"standardize": USE_STANDARDIZATION,
	}

	os.makedirs(result_dir, exist_ok=True)
	report_path = os.path.join(result_dir, f"cogbench_eye_tracking_{model_name}_report.json")
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	print(f"Saved eye-tracking report: {report_path}")
	return report_path
