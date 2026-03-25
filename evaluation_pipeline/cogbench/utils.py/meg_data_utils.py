from tqdm import tqdm
import torch
import numpy as np
import scipy.io as scio
from scipy.stats import pearsonr
import os
import itertools as itools
from functools import reduce
zs = lambda v: (v-v.mean(0))/v.std(0)

def mult_diag(d, mtx, left=True):
    """
    the code is adapted from https://github.com/HuthLab/speechmodeltutorial
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
        d -- 1D (N,) array (contains the diagonal elements)
        mtx -- 2D (N,N) array

    Output:
        mult_diag(d, mts, left=True) == dot(diag(d), mtx)
        mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        # return (d*mtx.T).T
        return (d*mtx.transpose()).transpose()
    else:
        return d*mtx

def ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, alphas,
            cuda0=0, cuda1=1, use_cuda=False, singcutoff=1e-10):
    """
    this function can be used on features with more than 1 dimension, 
    such as word embeddings (BERT, elmo, etc.), pos tags, 
    or other semantic features
    """
    
    U,S,V = torch.svd(train_feature) #cuda 1
    
    ngoodS = torch.sum(S>singcutoff)
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    V = V[:,:ngoodS]

    if use_cuda:
        alphas = torch.tensor(alphas).cuda(cuda0)
    else:
        alphas = torch.tensor(alphas)

    if use_cuda:
        UR = torch.matmul(U.transpose(0, 1).cuda(cuda1), train_fmri).cuda(cuda0)
        PVh = torch.matmul(valid_feature, V)
    else:
        UR = torch.matmul(U.transpose(0, 1), train_fmri)
        PVh = torch.matmul(valid_feature, V)

    zvalid_fmri = zs(valid_fmri)
    Rcorrs = [] ## Holds training correlations for each alpha
    for a in alphas:
        D = S/(S**2+a**2) ## Reweight singular vectors by the ridge parameter
        if use_cuda:
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
        else:
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR)
        Rcorr = (zvalid_fmri*zs(pred)).mean(0)                
        Rcorr[torch.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
    
    return Rcorrs

def load_meg(meg_path, sess, is_zs=True):
    '''
    Return:
        meg: (n_words, n_sensors, n_epochs); 
        starts: the start index of each story in the meg_path;
    '''
    meg = torch.tensor([])
    starts = [0]
    noexist = []
    for i in tqdm(sess, desc='loading meg from '+meg_path+' ...'):
        meg_file = meg_path+'/story_'+str(i)+'.mat'
        if not os.path.exists(meg_file):
            noexist.append(i)
            continue
        data =scio.loadmat(meg_file)
        n_word, n_sensor, n_epoch = data['meg'].shape
        if is_zs:
            single_meg = np.zeros(data['meg'].shape)
            for i in range(n_sensor):
                single_meg[:,i,:] = zs(data['meg'][:,i,:]) 
        else:
            single_meg = data['meg']
        meg = torch.cat([meg, torch.FloatTensor(single_meg)])
        starts.append(meg.shape[0])
    return meg, noexist

def load_feature(feature_path, PU_path, sess, noexist=[], is_zs=True):
    '''
    Return:
        train_feature: (n_TRs, feature_dim)
    '''
    train_feature = torch.FloatTensor([])
    starts = [0]
    for i in tqdm(sess, desc='loading stimulus from '+feature_path+' ...'):
        feature_file = feature_path + '/story_'+str(i)+'.mat'
        if i in noexist:
            continue
        data = scio.loadmat(feature_file)
        notPU = scio.loadmat(PU_path+'notPU/story_'+str(i)+'.mat')
        inds = np.where(notPU['isvalid']==1)[0]
        single_feature = torch.FloatTensor(data['data'][inds])
        if is_zs:
            single_feature = zs(single_feature)
        train_feature = torch.cat([train_feature, single_feature])
        starts.append(train_feature.shape[0])
        
    return train_feature, starts

def ridge_nested_cv(total_fmri, total_feature):
    """ 
    nested cross-validation, which is applicable to situations without
    designated test set
    Return: corrs on all out test set
    """
    alphas = np.logspace(-3, 3, 10)
    nTR, n_dim = total_feature.shape
    foldlen = int(nTR/5)
    test_corrs = []
    weights = []
    inds = list(range(nTR))
    # import pdb;pdb.set_trace()
    for fold in tqdm(range(5), desc='doing nested ridge regression...'):
        test_inds = inds[fold*foldlen:(fold+1)*foldlen]
        inner_inds = inds[0:fold*foldlen]+inds[(fold+1)*foldlen:]
        infoldlen = int(len(inner_inds)/5)
        test_fmri = total_fmri[test_inds]
        test_feature = total_feature[test_inds]

        val_corrs = []
        for infold in range(5):
            val_inds = inner_inds[infold*infoldlen:(infold+1)*infoldlen]
            train_inds = inner_inds[0:infold*infoldlen]+inner_inds[(infold+1)*infoldlen:]
            train_fmri = total_fmri[train_inds]
            valid_fmri = total_fmri[val_inds]
            train_feature = total_feature[train_inds]
            valid_feature = total_feature[val_inds]
            Rcorrs = ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, alphas)
            '''if n_dim == 1:
                Rcorrs = self.ridge_1dim(train_fmri, train_feature, valid_fmri, valid_feature, self.alphas)
            else:
                Rcorrs = self.ridge_multidim(train_fmri, train_feature, valid_fmri, valid_feature, \
                    self.alphas, self.cuda0, self.cuda1, self.use_cuda)'''
            val_corrs.append(torch.stack(Rcorrs))
        val_corrs = torch.stack(val_corrs)  
        max_ind = torch.argmax(val_corrs.mean(2).mean(0))
        bestalpha = alphas[max_ind]  
        U,S,V = torch.svd(total_feature[inner_inds])
        UR = torch.matmul(U.transpose(0, 1), total_fmri[inner_inds])
        wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha**2)), UR])
        pred = torch.matmul(test_feature, wt)
        pred = pred.numpy()
        test_fmri = test_fmri.numpy()
        a,_ = test_fmri.shape
        corr = []
        for i in range(a):
            corr.append(pearsonr(pred[i], test_fmri[i])[0])
        test_corrs.append(corr)
    test_corrs = np.array(test_corrs)
    test_corrs = test_corrs.mean()
    # import pdb;pdb.set_trace()
    return test_corrs
