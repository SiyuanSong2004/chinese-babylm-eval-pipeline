# 要先将词向量进行hrf卷积降采样后和fmri采样率保持一致
import scipy.io as scio
import h5py
import hdf5storage as hdf5
import numpy as np
from nilearn import glm
import os
import glob
zs = lambda v: (v-v.mean())/v.std()

def convolve_offset(in_path, time_root, out_path, pu_root, hrf, ref_length):
    for i in range(1, 61):
        data = scio.loadmat(in_path+'story_%d.mat'%i)
        data = data['data']
        word_time = scio.loadmat(time_root+'story_%d_word_time.mat'%i)
        word_time = word_time['end']
        isvalid = scio.loadmat(pu_root+'story_%d.mat'%i)
        inds = np.where(isvalid['isvalid']==1)[0]
        word_time = word_time[0, inds].reshape(1, len(inds))
        length = int(word_time[0][-1]*100)
        # 如果是单个特征，就np.zeros([length])；如果是多个特征，np.zeros([length,data.shape[1]])
        time_series = np.zeros([length,data.shape[1]])
        t = 0
        for j in range(length):
            if j == int(word_time[0][t]*100):
                time_series[j] = data[t]
                while(j == int(word_time[0][t]*100)):
                    t += 1
                    if t == data.shape[0]:
                        break
        conv_series = []
        for j in range(data.shape[1]):
            conv_series.append(np.convolve(hrf, time_series[:,j]))
        conv_series = np.stack(conv_series).T
        conv_series = conv_series[:length]
        conv_series_ds = [conv_series[j] for j in range(0, length, 71)]
        conv_series_ds = np.array(conv_series_ds)
        
        word_feature = zs(conv_series_ds[19:ref_length[i-1]+19])
        tmp = {'word_feature':word_feature.astype('float32')}
        hdf5.writes(tmp, out_path+'/story_%d.mat'%i, matlab_compatible=True)

def load_ref_TRs():
    file_root = 'node_count_bu/'
    res = []
    for i in range(1, 61):
        data = h5py.File(file_root+'story_%d.mat'%i, 'r')
        res.append(data['word_feature'].shape[1])
    return res

if __name__ == "__main__":
    pu_root = 'notPU/'
    time_root = '/data/word_time_features_postprocess/'

    # feature_content_avg convolve
    in_root = '/data/feature_word2vec_word/word_feature/'
    out_root = '/data/feature_word2vec_word/word_feature_convolved/'
    hrf = glm.first_level.spm_hrf(0.71, 71)
    ref_length = load_ref_TRs()
    time_type = glob.glob('/data/feature_word2vec_word/word_feature/*')
    time_type = [i.split('/')[-1] for i in time_type]
    for ti in time_type:
        if not os.path.exists(out_root + ti):
            os.makedirs(out_root + ti)
        in_path = in_root + ti + '/'
        out_path = out_root + ti + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(out_path)
        convolve_offset(in_path, time_root, out_path, pu_root, hrf, ref_length)
    
    exit()

