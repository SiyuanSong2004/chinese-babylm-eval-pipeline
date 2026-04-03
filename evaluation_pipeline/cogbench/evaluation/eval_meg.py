import os
from argparse import ArgumentParser

import numpy as np
import scipy.io as scio

from ..utils.meg_data_utils import load_meg, load_feature, ridge_nested_cv
from ..utils.meg_selection import sensor_selection


def eval_meg(args: ArgumentParser):
    data_path = str(args.data_path)
    output_root = str(args.output_dir)
    model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
    model_root = os.path.join(output_root, model_name)

    feature_root = model_root
    pu_root = data_path + "/"
    result_root = os.path.join(model_root, "results", "meg")

    starts = [6, 7]
    if args.fast:
        starts = starts[:1]

    sessions = [
        [1, 11, 31, 41, 56, 46, 36, 26, 16, 6],
        [21, 51, 2, 12, 32, 42, 47, 37, 7, 17],
        [22, 52, 53, 33, 57, 27, 48, 38, 18, 8],
        [13, 23, 43, 3, 4, 34, 58, 28, 39, 9],
        [14, 24, 44, 54, 59, 49, 29, 19, 40, 10],
        [15, 5, 25, 35, 45, 55, 60, 50, 30, 20],
    ]

    subs = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    if args.fast:
        subs = subs[:1]

    meg_root = os.path.join(data_path, "encoding_meg_100ms", "sub-")

    os.makedirs(result_root, exist_ok=True)
    is_zs = False

    feat_key = os.path.basename(feature_root)

    for start in starts:
        end = start + 1
        score_path = os.path.join(result_root, f"{model_name}_rsa_{start}.mat")
        mask_path = os.path.join(result_root, f"{model_name}_masks_{start}.mat")

        corrs_sess = {feat_key: [[] for _ in range(len(subs))]}
        masks = {feat_key: [[] for _ in range(len(subs))]}

        for nsub, sub in enumerate(subs):
            for sess_idx, sess in enumerate(sessions):
                if args.fast and sess_idx > 0:
                    break

                meg_path = meg_root + sub
                meg_response, noexist = load_meg(meg_path, sess, is_zs)
                meg_response = meg_response[:, :, start:end].mean(2)

                word_feature, _ = load_feature(feature_root, pu_root, sess, noexist, is_zs)
                meg_tmp, mask = sensor_selection(meg_response, word_feature, 0.05)
                corr = ridge_nested_cv(meg_tmp, word_feature)

                masks[feat_key][nsub].append(mask)
                corrs_sess[feat_key][nsub].append(corr)
                print(corr)

        corrs_sess[feat_key] = np.array(corrs_sess[feat_key])
        masks[feat_key] = np.stack(masks[feat_key])

        scio.savemat(score_path, {"sess_avg": corrs_sess})
        scio.savemat(mask_path, {"masks": masks})
