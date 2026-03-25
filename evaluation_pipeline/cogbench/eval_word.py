import os
import glob
import argparse
import numpy as np
import pandas as pd
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


def main(args):
    # load stimuli order and align feature rows with fmri trials
    with open("./script.txt", encoding="utf-8") as f:
        stimuli_list = [line.strip() for line in f if line.strip()]

    fmri_files = glob.glob(os.path.join(args.path_fmri, "*_selected.mat"))

    for fmri_path in fmri_files:
        subject = os.path.basename(fmri_path).replace("_selected.mat", "")
        print(f"\nProcessing {subject}")

        mat = sio.loadmat(fmri_path)
        fmri_data = mat["examples"]  # (672, n_selected)

        for feat_path in glob.glob(args.path_feature):
            feat_name = os.path.splitext(os.path.basename(feat_path))[0]
            out_dir = os.path.join(args.path_result, feat_name)
            save_path = os.path.join(out_dir, f"{subject}_score.mat")

            if os.path.exists(save_path):
                print(f"Already exists, skip -> {save_path}")
                continue

            df_feat = pd.read_csv(feat_path, header=None, index_col=0)
            df_feat = df_feat.loc[stimuli_list]

            print(f"  -> {feat_name}")
            run_prediction(df_feat, fmri_data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI encoding - whole selected voxels")
    parser.add_argument("--path_fmri", type=str, required=True, help="")
    parser.add_argument("--path_result", type=str, required=True, help="output folder")
    parser.add_argument(
        "--path_feature",
        type=str,
        default="/emb_layer_zh/g*",
        help="glob pattern for feature files",
    )

    args = parser.parse_args()
    main(args)
