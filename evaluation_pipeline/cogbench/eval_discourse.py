# 63
import numpy as np
import scipy.io as scio
from data_utils import load_fmri, load_feature, ridge_multidim, ridge_nested_cv
from argparse import ArgumentParser
import os
import torch
        

if __name__ == "__main__":
    subs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    roi_types = ['Cognition', 'Language', 'Manipulation', 'Memory', 'Reward', 'Vision']
    fmri_root = 'fmri/'
    result_root = 'result_fmri/'
    mask_root = 'mask/vox_select_RSA/'
    features = ['glove_after_hrf']
    
    for roi in roi_types:
        for feat in features:
            if not os.path.exists(result_root+roi+'/'+feat):
                os.makedirs(result_root+roi+'/'+feat)
            feature_path = feat+'/'
            word_feature, run_starts = load_feature(feature_path)
            corrs = [[], [], []]
            for sub in subs:
                if(os.path.exists(result_root+roi+'/'+feat+'/'+sub+'_average.mat')):
                    continue
                fmri_path = fmri_root+roi+'/'+sub
                fmri_response, _ = load_fmri(fmri_path)

                mask_path = mask_root + roi + '/' + 'sub_%s_%s_mask.mat'%(sub, feat.split('/')[-1])
                if os.path.exists(mask_path):
                    mask_path = mask_root + roi + '/' + 'sub_%s_%s_mask.mat'%(sub, feat.split('/')[-1])
                    print(feat.split('/')[-1])
                else:
                    mask_path = mask_root + roi + '/' + 'sub_%s_gpt2_layer%s_mask.mat'%(sub,feat.split('/')[-1].split('_')[0][7:])
                    print(feat.split('/')[-1],feat.split('/')[-1].split('_')[0][7:])
                mask = scio.loadmat(mask_path)
                fmri_response = np.array(fmri_response)
                fmri_response = fmri_response[:, np.where(mask['mask']==1)[1]]
                fmri_response = torch.from_numpy(fmri_response)

                ridge_nested_cv(fmri_response, word_feature, result_root+roi+'/'+feat+'/',sub)

