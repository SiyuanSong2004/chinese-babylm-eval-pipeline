from tqdm import tqdm
import h5py
import torch
import numpy as np
import scipy.io as scio
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

def load_fmri(fmri_path, story_amount=60, language='zh'):
    '''
    Return:
        train_fmri: (n_TRs, n_voxels); 
        starts: the start index of each story in the fmri_path;
    '''
    train_fmri = torch.tensor([])
    starts = [0]
    if language == 'zh':
        for i in tqdm(range(1, story_amount+1), desc='loading fmri from '+fmri_path+' ...'):
            fmri_file = fmri_path+'/story_'+str(i)+'.mat'
            data =scio.loadmat(fmri_file)
            single_fmri = np.array(data['fmri_response'].T)
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])
    
    elif language == 'en':
        for i in tqdm(range(1, story_amount+1), desc='loading fmri from...'+fmri_path+' ...'):
            filepath = fmri_path+'/story_'+str(i)+'.mat'
            data = scio.loadmat(filepath)
            single_fmri = np.array(data['fmri_response'].T)
            train_fmri = torch.cat([train_fmri, torch.FloatTensor(single_fmri)])
            starts.append(train_fmri.shape[0])

    else:
        raise('Unknown language!')
        
    return train_fmri, starts

def load_feature(feature_path, story_amount=60, language='zh'):
    '''
    Return:
        train_feature: (n_TRs, feature_dim)
    '''
    train_feature = torch.tensor([])
    starts = [0]
    if language == 'zh':
        for i in tqdm(range(1, story_amount+1), desc='loading stimulus from '+feature_path+' ...'):
            feature_file = feature_path + '/story_'+str(i)+'.mat'
            data = h5py.File(feature_file, 'r')
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
    elif language == 'en':
        for i in tqdm(range(1, story_amount+1), desc='loading stimulus from '+feature_path+' ...'):
            filepath = feature_path+'/story_%d.mat'%i
            data = h5py.File(filepath, 'r')
            single_feature = torch.tensor(np.array(data['word_feature'])).transpose(0, 1)
            train_feature = torch.cat([train_feature, single_feature])
            starts.append(train_feature.shape[0])
    else:
        raise('Unknown language!')
        
    return train_feature, starts

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

def ridge_nested_cv(total_fmri, total_feature, result_dir, sub):
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
        corrs = (zs(pred)*zs(test_fmri)).mean(0)
        test_corrs.append(corrs)
        weights.append(wt)
    test_corrs = torch.stack(test_corrs)
    weights = torch.stack(weights)
    savefile = result_dir+sub+'_average.mat'
    scio.savemat(savefile, {'test_corrs':np.array(test_corrs.mean(0).cpu())})
    return np.array(test_corrs.mean(0).cpu())