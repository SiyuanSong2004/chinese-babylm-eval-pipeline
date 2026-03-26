import os
from argparse import ArgumentParser

import h5py
import numpy as np
import scipy.io as scio
import torch
from scipy.stats import gamma

from ..utils.data_utils import load_fmri, ridge_nested_cv


TR_SECONDS = 0.71
TR_OVERSAMPLING = 71
HRF_OFFSET_TRS = 19
FAST_STORY_COUNT = 5


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


def _load_ref_tr_lengths(data_path: str, story_amount: int) -> list[int]:
    ref_root = os.path.join(data_path, "node_count_bu")
    lengths = []
    for story_id in range(1, story_amount + 1):
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


def _load_feature_matrix(feature_root: str, data_path: str, story_amount: int) -> torch.FloatTensor:
    if not os.path.isdir(feature_root):
        raise FileNotFoundError(f"Feature directory not found: {feature_root}. Please run inference first.")

    hrf = _spm_hrf(TR_SECONDS, TR_OVERSAMPLING)
    ref_lengths = _load_ref_tr_lengths(data_path, story_amount)

    all_features = []
    for story_id in range(1, story_amount + 1):
        processed = _postprocess_story_feature(feature_root, data_path, story_id, hrf, ref_lengths[story_id - 1])
        all_features.append(torch.from_numpy(processed))

    return torch.cat(all_features, dim=0)


def _resolve_mask_path(mask_root: str, roi: str, sub: str, model_name: str) -> str:
    candidate = os.path.join(mask_root, roi, f"sub_{sub}_gpt2_layer12_mask.mat")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"No voxel mask found for ROI={roi}, subject={sub}, expected=sub_{sub}_gpt2_layer12_mask.mat"
    )


def eval_fmri(args: ArgumentParser):
    data_path = str(args.data_path)
    output_root = str(args.output_dir)
    model_name = os.path.basename(os.path.normpath(str(args.model_path_or_name)))
    model_root = os.path.join(output_root, model_name)

    story_amount = FAST_STORY_COUNT if args.fast else 60
    feature_matrix = _load_feature_matrix(model_root, data_path, story_amount)

    roi_types = ["Cognition", "Language", "Manipulation", "Memory", "Reward", "Vision"]
    subs = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    if args.fast:
        roi_types = roi_types[:1]
        subs = subs[:1]

    fmri_root = os.path.join(data_path, "fmri")
    mask_root = os.path.join(data_path, "mask", "vox_select_RSA")
    result_root = os.path.join(model_root, "results", "fmri")

    for roi in roi_types:
        roi_result_dir = os.path.join(result_root, roi)
        os.makedirs(roi_result_dir, exist_ok=True)

        for sub in subs:
            save_path = os.path.join(roi_result_dir, f"{sub}_average.mat")

            fmri_path = os.path.join(fmri_root, roi, sub)
            fmri_response, _ = load_fmri(fmri_path, story_amount=story_amount)

            mask_path = _resolve_mask_path(mask_root, roi, sub, model_name)
            mask = scio.loadmat(mask_path)
            fmri_response = np.array(fmri_response)
            fmri_response = fmri_response[:, np.where(mask["mask"] == 1)[1]]
            fmri_response = torch.from_numpy(fmri_response)

            ridge_nested_cv(fmri_response, feature_matrix, roi_result_dir + "/", sub)
