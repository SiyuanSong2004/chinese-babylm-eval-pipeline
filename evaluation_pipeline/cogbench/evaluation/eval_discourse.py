import os
from argparse import ArgumentParser
import re

import h5py
import numpy as np
import scipy.io as scio
import torch
from scipy.stats import gamma

from ..utils.data_utils import ridge_nested_cv, ridge_train_dev_test


TR_SECONDS = 0.71
TR_OVERSAMPLING = 71
HRF_OFFSET_TRS = 19
FAST_STORY_COUNT = 5


def _available_story_ids(data_path: str) -> list[int]:
    ref_root = os.path.join(data_path, "node_count_bu")
    story_ids = []
    for name in os.listdir(ref_root):
        m = re.match(r"story_(\d+)\.mat$", name)
        if m:
            story_ids.append(int(m.group(1)))
    return sorted(story_ids)


def _spm_hrf(tr: float, oversampling: int, time_length: float = 32.0) -> np.ndarray:
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)
    peak = gamma.pdf(time_stamps, 6)
    undershoot = gamma.pdf(time_stamps, 16)
    hrf = peak - undershoot / 6
    hrf_sum = hrf.sum()
    if hrf_sum != 0:
        hrf = hrf / hrf_sum
    return hrf.astype(np.float32)


def _zs(array: np.ndarray) -> np.ndarray:
    std = array.std()
    if std == 0:
        return array - array.mean()
    return (array - array.mean()) / std


def _load_ref_tr_lengths(data_path: str, story_ids: list[int]) -> list[int]:
    ref_root = os.path.join(data_path, "node_count_bu")
    lengths = []
    for story_id in story_ids:
        file_path = os.path.join(ref_root, f"story_{story_id}.mat")
        with h5py.File(file_path, "r") as handle:
            lengths.append(handle["word_feature"].shape[1])
    return lengths


def _postprocess_story_feature(
    feature_root: str,
    data_path: str,
    story_id: int,
    hrf: np.ndarray,
    ref_length: int,
) -> np.ndarray:
    feature_path = os.path.join(feature_root, f"sentence_feature_story_{story_id}.mat")
    if not os.path.exists(feature_path):
        # Backward compatibility with previously generated sentence feature names.
        feature_path = os.path.join(feature_root, f"story_{story_id}.mat")
    raw_feature = scio.loadmat(feature_path)["data"].astype(np.float32)

    time_info = scio.loadmat(
        os.path.join(data_path, "word_time_features_postprocess", f"story_{story_id}_word_time.mat")
    )
    word_end = time_info["end"]

    notpu = scio.loadmat(os.path.join(data_path, "notPU", f"story_{story_id}.mat"))
    valid_indices = np.where(notpu["isvalid"] == 1)[0]

    raw_feature = raw_feature[valid_indices]
    word_end = word_end[0, valid_indices]

    if raw_feature.shape[0] == 0:
        return np.zeros((ref_length, 0), dtype=np.float32)

    timeline_length = int(word_end[-1] * 100)
    time_series = np.zeros((timeline_length, raw_feature.shape[1]), dtype=np.float32)

    token_index = 0
    for frame_index in range(timeline_length):
        if token_index >= len(word_end):
            break
        if frame_index == int(word_end[token_index] * 100):
            time_series[frame_index] = raw_feature[token_index]
            while token_index < len(word_end) and frame_index == int(word_end[token_index] * 100):
                token_index += 1

    conv_series = []
    for feature_index in range(raw_feature.shape[1]):
        conv_series.append(np.convolve(hrf, time_series[:, feature_index]))
    conv_series = np.stack(conv_series, axis=1)[:timeline_length]

    downsampled = conv_series[::TR_OVERSAMPLING]
    processed = downsampled[HRF_OFFSET_TRS : ref_length + HRF_OFFSET_TRS]
    return _zs(processed).astype(np.float32)


def _load_feature_matrix(feature_root: str, data_path: str, story_ids: list[int]) -> torch.FloatTensor:
    if not os.path.isdir(feature_root):
        raise FileNotFoundError(f"Feature directory not found: {feature_root}. Please run inference first.")

    hrf = _spm_hrf(TR_SECONDS, TR_OVERSAMPLING)
    ref_lengths = _load_ref_tr_lengths(data_path, story_ids)

    all_features = []
    for idx, story_id in enumerate(story_ids):
        processed = _postprocess_story_feature(feature_root, data_path, story_id, hrf, ref_lengths[idx])
        all_features.append(torch.from_numpy(processed))

    return torch.cat(all_features, dim=0)


def _resolve_split_dirs(data_path: str) -> dict[str, str] | None:
    train_dir = os.path.join(data_path, "train")
    dev_dir = os.path.join(data_path, "dev")
    test_dir = os.path.join(data_path, "test")
    if os.path.isdir(train_dir) and os.path.isdir(dev_dir):
        return {
            "train": train_dir,
            "dev": dev_dir,
            "test": test_dir if os.path.isdir(test_dir) else None,
        }
    return None


def _load_split_fmri_response(split_path: str, roi: str, sub: str, story_ids: list[int]) -> torch.FloatTensor | None:
    fmri_path = os.path.join(split_path, "fmri", roi, sub)
    fmri_story_blocks = []
    for sid in story_ids:
        story_file = os.path.join(fmri_path, f"story_{sid}.mat")
        if not os.path.exists(story_file):
            continue
        mat = scio.loadmat(story_file)
        fmri_story_blocks.append(np.array(mat["fmri_response"].T))

    if not fmri_story_blocks:
        return None

    fmri_response = np.concatenate(fmri_story_blocks, axis=0)
    return torch.from_numpy(fmri_response)


def _detect_subjects_for_roi(data_path: str, roi: str, split_dirs: dict[str, str] | None) -> list[str]:
    if split_dirs is None:
        roi_dir = os.path.join(data_path, "fmri", roi)
        if not os.path.isdir(roi_dir):
            return []
        subjects = [name for name in os.listdir(roi_dir) if os.path.isdir(os.path.join(roi_dir, name))]
        return sorted(subjects)

    available_splits = ["train", "dev"]
    if split_dirs["test"] is not None:
        available_splits.append("test")

    common_subjects = None
    for split_name in available_splits:
        roi_dir = os.path.join(split_dirs[split_name], "fmri", roi)
        if not os.path.isdir(roi_dir):
            return []
        subjects = {name for name in os.listdir(roi_dir) if os.path.isdir(os.path.join(roi_dir, name))}
        common_subjects = subjects if common_subjects is None else common_subjects & subjects

    return sorted(common_subjects) if common_subjects else []


def eval_fmri(args: ArgumentParser):
    data_path = str(args.data_path)
    output_root = str(args.output_dir)
    model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
    model_root = os.path.join(output_root, model_name)

    split_dirs = _resolve_split_dirs(data_path)

    if split_dirs is None:
        all_story_ids = _available_story_ids(data_path)
        if not all_story_ids:
            raise FileNotFoundError(f"No story_*.mat files found under: {os.path.join(data_path, 'node_count_bu')}")

        story_ids = all_story_ids[:FAST_STORY_COUNT] if args.fast else all_story_ids
        feature_matrix = _load_feature_matrix(model_root, data_path, story_ids)
    else:
        split_story_ids = {}
        split_features = {}
        available_splits = ["train", "dev"]
        if split_dirs["test"] is not None:
            available_splits.append("test")
        for split_name in available_splits:
            split_path = split_dirs[split_name]
            ids = _available_story_ids(split_path)
            if not ids:
                raise FileNotFoundError(
                    f"No story_*.mat files found under: {os.path.join(split_path, 'node_count_bu')}"
                )
            if args.fast:
                ids = ids[:FAST_STORY_COUNT]
            split_story_ids[split_name] = ids
            split_features[split_name] = _load_feature_matrix(model_root, split_path, ids)

    roi_types = ["Cognition", "Language", "Manipulation", "Memory", "Reward", "Vision"]

    if args.fast:
        roi_types = roi_types[:1]

    fmri_root = os.path.join(data_path, "fmri")
    result_root = os.path.join(model_root, "results", "fmri")

    for roi in roi_types:
        roi_result_dir = os.path.join(result_root, roi)
        os.makedirs(roi_result_dir, exist_ok=True)

        subs = _detect_subjects_for_roi(data_path, roi, split_dirs)
        if not subs:
            print(f"Skip ROI={roi}: no subject directories found in fmri.")
            continue
        if args.fast:
            subs = subs[:1]

        for sub in subs:
            if split_dirs is None:
                fmri_path = os.path.join(fmri_root, roi, sub)
                fmri_story_blocks = []
                for sid in story_ids:
                    story_file = os.path.join(fmri_path, f"story_{sid}.mat")
                    if not os.path.exists(story_file):
                        continue
                    mat = scio.loadmat(story_file)
                    fmri_story_blocks.append(np.array(mat["fmri_response"].T))

                if not fmri_story_blocks:
                    print(f"Skip ROI={roi}, sub={sub}: no selected fMRI story files found.")
                    continue

                fmri_response = np.concatenate(fmri_story_blocks, axis=0)
                fmri_response = torch.from_numpy(fmri_response)
                ridge_nested_cv(fmri_response, feature_matrix, roi_result_dir + "/", sub)
                continue

            train_fmri = _load_split_fmri_response(split_dirs["train"], roi, sub, split_story_ids["train"])
            dev_fmri = _load_split_fmri_response(split_dirs["dev"], roi, sub, split_story_ids["dev"])
            has_test = split_dirs["test"] is not None
            if has_test:
                test_fmri = _load_split_fmri_response(split_dirs["test"], roi, sub, split_story_ids["test"])
            else:
                test_fmri = dev_fmri

            if train_fmri is None or dev_fmri is None or test_fmri is None:
                print(f"Skip ROI={roi}, sub={sub}: missing train/dev/test fMRI split files.")
                continue

            feat_train = split_features["train"]
            feat_dev = split_features["dev"]
            feat_test = split_features["test"] if has_test else feat_dev

            n_train = min(train_fmri.shape[0], feat_train.shape[0])
            n_dev = min(dev_fmri.shape[0], feat_dev.shape[0])
            n_test = min(test_fmri.shape[0], feat_test.shape[0])

            train_fmri = train_fmri[:n_train]
            dev_fmri = dev_fmri[:n_dev]
            test_fmri = test_fmri[:n_test]
            feat_train = feat_train[:n_train]
            feat_dev = feat_dev[:n_dev]
            feat_test = feat_test[:n_test]

            if args.fast:
                train_fmri = train_fmri[: min(100, train_fmri.shape[0])]
                dev_fmri = dev_fmri[: min(50, dev_fmri.shape[0])]
                test_fmri = test_fmri[: min(50, test_fmri.shape[0])]
                feat_train = feat_train[: train_fmri.shape[0]]
                feat_dev = feat_dev[: dev_fmri.shape[0]]
                feat_test = feat_test[: test_fmri.shape[0]]

            if not has_test:
                print(f"ROI={roi}, sub={sub}: test split missing, reporting dev-set performance.")

            ridge_train_dev_test(
                train_fmri,
                feat_train,
                dev_fmri,
                feat_dev,
                test_fmri,
                feat_test,
                roi_result_dir + "/",
                sub,
            )
