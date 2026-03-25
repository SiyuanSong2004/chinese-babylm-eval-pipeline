# 63
import numpy as np
import scipy.io as scio
from meg_data_utils import load_meg, load_feature,ridge_nested_cv
from meg_selection import sensor_selection, mix_selection
from argparse import ArgumentParser
import os
import glob

def eval_meg(args: ArgumentParser):
    for start in range(6,8):
        end = start + 1

        sessions = [[1, 11, 31, 41, 56, 46, 36, 26, 16, 6], \
            [21, 51, 2, 12, 32, 42, 47, 37, 7, 17], \
            [22, 52, 53, 33, 57, 27, 48, 38, 18, 8], \
            [13, 23, 43, 3, 4, 34, 58, 28, 39, 9], \
            [14, 24, 44, 54, 59, 49, 29, 19, 40, 10], \
            [15, 5, 25, 35, 45, 55, 60, 50, 30, 20]]

        subs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        meg_root = '/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/encoding_meg_100ms/sub-'
        features = ['transformer']
        feature_root = 'word_features/'
        pu_root = '../discourse_fmri/'
        result_root = 'meg/'
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        is_zs = False
        # corrs_sess: dict, corrs_sess[i]: (n_sub, n_sess)=(12, 6)
        corrs_run, corrs_sess, masks = {}, {}, {}
        for feat in features:
            feat = feat.split('/')
            feat_label = feat[-1].split('_')[0]
            if len(feat)>2:
                feat = feat[2]
            elif len(feat)>1:
                feat = feat[1]
            else:
                feat = feat[0]

            corrs_run[feat] = []
            corrs_sess[feat] = [[] for i in range(12)]
            masks[feat] = [[] for i in range(12)]
        nsub = 0
        for sub in subs: 
            count = 1
            for sess in sessions:
                meg_path = meg_root+sub
                meg_response, noexist = load_meg(meg_path, sess, is_zs)
                # import pdb
                # pdb.set_trace()
                n_words, n_sensor, n_epoch = meg_response.shape
                meg_response = meg_response[:, :, start:end].mean(2)
                for feat in features:
                    feature_path = feature_root+feat+'/'
                    word_feature, run_starts = load_feature(feature_path, pu_root, sess, noexist, is_zs)
                    meg_tmp, mask = sensor_selection(meg_response, word_feature, 0.05)
                    feat = feat.split('/')
                    if len(feat)>2:
                        feat = feat[2]
                    elif len(feat)>1:
                        feat = feat[1]
                    else:
                        feat = feat[0]
                    masks[feat][nsub].append(mask)
                    corr = ridge_nested_cv(meg_tmp, word_feature)
                    corrs_sess[feat][nsub].append(corr)
                    print(corr)
                count += 1
            nsub += 1
        for feat in features:
            feat = feat.split('/')
            if len(feat)>2:
                feat = feat[2]
            elif len(feat)>1:
                feat = feat[1]
            else:
                feat = feat[0]
            corrs_sess[feat] = np.array(corrs_sess[feat])
            masks[feat] = np.stack(masks[feat])
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        scio.savemat(result_root+'/'+'%s_rsa_%d.mat'%(feat_label, start), {'sess_avg':corrs_sess})
        scio.savemat(result_root+'/'+'%s_masks_%d.mat'%(feat_label, start), {'masks':masks})

        

                