import os
import glob
import json
import numpy as np
import scipy.io as sio
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr


def standardize_matrix(matrix):
    row_means = np.mean(matrix, axis=1, keepdims=True)
    row_stds = np.std(matrix, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0  # avoid divide by zero
    return (matrix - row_means) / row_stds


def ridge_prediction(X_train, X_test, y_train):
    model = Ridge()
    alphas = np.logspace(-4, 4, 10)  # tunable alpha search range
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
    """
    feature: (n_trials, n_features)
    fmri:    (n_trials, n_voxels_selected)
    """
    X = standardize_matrix(np.asarray(feature))
    Y = np.asarray(fmri)

    kf = KFold(n_splits=5, shuffle=False)
    all_corrs = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        Y_pred = ridge_prediction(X_tr, X_te, Y_tr)

        # remove all-zero columns if there are any
        valid_cols = ~np.all(Y_te == 0, axis=0)
        if not np.any(valid_cols):
            all_corrs.append(0.0)
            continue

        Y_pred = Y_pred[:, valid_cols]
        Y_te = Y_te[:, valid_cols]

        # pearson r for each trial
        trial_corrs = np.array([
            pearsonr(Y_pred[i], Y_te[i])[0] for i in range(Y_te.shape[0])
        ])
        trial_corrs[np.isnan(trial_corrs)] = 0.0

        # top 10% mean
        k = max(1, int(len(trial_corrs) * 0.1))
        top_mean = np.mean(np.sort(trial_corrs)[-k:]) if k > 0 else 0.0
        all_corrs.append(top_mean)

    # average across 5 folds
    final_score = float(np.mean(all_corrs))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {"score": final_score})
    print(f"Saved: {save_path} | overall top-10% mean pearson r = {final_score:.4f}")


def _resolve_cogbench_root(data_path):
    if os.path.isdir(os.path.join(data_path, "word")) and os.path.isdir(os.path.join(data_path, "word_fmri")):
        return data_path

    if os.path.basename(os.path.normpath(data_path)) == "word_fmri":
        candidate_root = os.path.dirname(os.path.normpath(data_path))
        if os.path.isdir(os.path.join(candidate_root, "word")):
            return candidate_root

    return data_path


def _load_feature_matrix(feature_json_path, stimuli_list):
    with open(feature_json_path, "r", encoding="utf-8") as f:
        feature_dict = json.load(f)

    if not feature_dict:
        raise ValueError(f"Feature file is empty: {feature_json_path}")

    hidden_size = len(next(iter(feature_dict.values())))
    missing = []
    feature_rows = []

    for stimulus in stimuli_list:
        feature = feature_dict.get(stimulus)
        if feature is None:
            missing.append(stimulus)
            feature = [0.0] * hidden_size
        feature_rows.append(feature)

    if missing:
        print(f"Warning: {len(missing)} stimuli missing in feature json, filled with zeros.")

    return np.asarray(feature_rows, dtype=np.float32)


def eval_word_fmri(args):
    data_path = _resolve_cogbench_root(str(args.data_path))
    output_root = str(args.output_dir)
    words_path = os.path.join(data_path, "word", "word.txt")
    with open(words_path, encoding="utf-8") as f:
        stimuli_list = [line.strip() for line in f if line.strip()]

    model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
    feature_json_path = os.path.join(output_root, model_name, "word_feature.json")
    if not os.path.exists(feature_json_path):
        raise FileNotFoundError(
            f"Feature file not found: {feature_json_path}. Please run inference first."
        )

    feature_matrix = _load_feature_matrix(feature_json_path, stimuli_list)

    fmri_dir = os.path.join(data_path, "word_fmri")
    fmri_files = sorted(glob.glob(os.path.join(fmri_dir, "*_selected.mat")))
    out_dir = os.path.join(output_root, model_name, "results", "word_fmri")

    if args.fast:
        fmri_files = fmri_files[:1]
        print("[FAST] Running sanity check only: first subject, small subset.")

    if not fmri_files:
        raise FileNotFoundError(f"No *_selected.mat files found under {fmri_dir}")

    for fmri_path in fmri_files:
        subject = os.path.basename(fmri_path).replace("_selected.mat", "")
        print(f"\nProcessing {subject}")

        mat = sio.loadmat(fmri_path)
        fmri_data = mat["examples"]  # (672, n_selected)

        n_trials = min(feature_matrix.shape[0], fmri_data.shape[0])
        if n_trials < 5:
            print(f"Skip {subject}: not enough aligned trials ({n_trials}).")
            continue

        if args.fast:
            n_trials = max(5, min(50, n_trials))
            n_voxels = min(128, fmri_data.shape[1])
            fmri_for_eval = fmri_data[:n_trials, :n_voxels]
            feat_for_eval = feature_matrix[:n_trials]
            save_path = os.path.join(out_dir, f"{subject}_sanity_score.mat")
        else:
            fmri_for_eval = fmri_data[:n_trials]
            feat_for_eval = feature_matrix[:n_trials]
            save_path = os.path.join(out_dir, f"{subject}_score.mat")

        run_prediction(feat_for_eval, fmri_for_eval, save_path)

