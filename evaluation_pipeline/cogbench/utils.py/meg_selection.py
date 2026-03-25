import numpy as np
import torch
import random
import itertools as itools
from functools import reduce

zs = lambda v: (v-v.mean(0))/v.std(0)

def mult_diag(d, mtx, left=True):
    if left:
        # return (d*mtx.T).T
        return (d*mtx.transpose()).transpose()
    else:
        return d*mtx

def ridge_corr(train_fmri, train_feature, valid_fmri, valid_feature, alphas, cuda0=0, cuda1=1, 
               singcutoff=1e-10, use_corr=True, use_cuda=False):
    U,S,V = torch.svd(train_feature) #cuda 1
    
    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = torch.sum(S>singcutoff)
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    V = V[:,:ngoodS]

    if use_cuda:
        nalphas = torch.tensor(alphas).cuda(cuda0)
    else:
        nalphas = torch.tensor(alphas)

    ## Precompute some products for speed
    if use_cuda:
        UR = torch.matmul(U.transpose(0, 1).cuda(cuda1), train_fmri).cuda(cuda0)
        PVh = torch.matmul(valid_feature, V)
    else:
        UR = torch.matmul(U.transpose(0, 1), train_fmri)
        PVh = torch.matmul(valid_feature, V)

    zvalid_fmri = zs(valid_fmri)
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        D = S/(S**2+na**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        if use_cuda:
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        else:
            pred = torch.matmul(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            Rcorr = (zvalid_fmri*zs(pred)).mean(0)
        else:
            ## Compute variance explained
            Prespvar = valid_fmri.var(0)
            resvar = (valid_fmri-pred).var(0)
            Rcorr = torch.clamp(1-(resvar/Prespvar), 0, 1)
            
        Rcorr[torch.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
    
    return Rcorrs

def encoding(total_fmri, total_feature):
    nresp, nvox = total_fmri.shape
    allinds = range(nresp)
    blocklen = 100
    indblocks = list(zip(*[iter(allinds)]*blocklen))
    if nresp%blocklen != 0:
        indblocks.append(range(len(indblocks)*blocklen, nresp))
    random.shuffle(indblocks)
    nblocks = len(indblocks)

    train_inds = list(itools.chain(*indblocks[:int(nblocks*0.8)]))
    valid_inds = list(itools.chain(*indblocks[int(nblocks*0.8):int(nblocks*0.9)]))
    test_inds = list(itools.chain(*indblocks[int(nblocks*0.9):]))
    train_fmri = zs(total_fmri[train_inds, :])
    valid_fmri = zs(total_fmri[valid_inds, :])
    test_fmri = zs(total_fmri[test_inds, :])
    train_feature = zs(total_feature[train_inds, :])
    valid_feature = zs(total_feature[valid_inds, :])
    test_feature = zs(total_feature[test_inds, :])
    alphas = np.logspace(-3, 3, 20)
    Rcorrs = ridge_corr(train_fmri, train_feature, valid_fmri, valid_feature, alphas)
    Rcorrs = torch.stack(Rcorrs)
    meanbootcorr = Rcorrs.mean(1)
    bestalphaind = torch.argmax(meanbootcorr)
    bestalpha = alphas[bestalphaind]
    
    U,S,V = torch.svd(train_feature)
    UR = torch.matmul(U.transpose(0, 1), train_fmri)
    wt = reduce(torch.matmul, [V, torch.diag(S/(S**2+bestalpha**2)), UR])
    pred = torch.matmul(test_feature, wt)
    corrs = (zs(pred)*zs(test_fmri)).mean(0)
    return corrs

def encoding_cv(total_fmri, total_feature, nfold):
    nresp, nvox = total_fmri.shape
    allinds = range(nresp)
    blocklen = 100
    indblocks = list(zip(*[iter(allinds)]*blocklen))
    if nresp%blocklen != 0:
        indblocks.append(range(len(indblocks)*blocklen, nresp))
    random.shuffle(indblocks)
    nblocks = len(indblocks)

    test_corrs = []
    foldlen = int(nblocks/nfold)
    for fold in range(nfold):
        test_inds = list(itools.chain(*indblocks[fold*foldlen:(fold+1)*foldlen]))
        inner_inds = list(itools.chain(*indblocks[:fold*foldlen]))+list(itools.chain(*indblocks[(fold+1)*foldlen:]))
        infoldlen = int(len(inner_inds)/fold)
        test_fmri = total_fmri[test_inds]
        test_feature = total_feature[test_inds]

        val_corrs = []
        for infold in range(nfold):
            val_inds = inner_inds[infold*infoldlen:(infold+1)*infoldlen]
            train_inds = inner_inds[0:infold*infoldlen]+inner_inds[(infold+1)*infoldlen:]
            train_fmri = zs(total_fmri[train_inds])
            valid_fmri = zs(total_fmri[val_inds])
            train_feature = zs(total_feature[train_inds])
            valid_feature = zs(total_feature[val_inds])
            alphas = np.logspace(-3, 3, 10)
            Rcorrs = ridge_corr(train_fmri, train_feature, valid_fmri, valid_feature, alphas)
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
    
    test_corrs = torch.stack(test_corrs).mean(0)
    return test_corrs

def voxel_selection(fmri, feature, percent):
    _, n_vox = fmri.shape
    corrs = encoding(fmri, feature)
    print(corrs.mean())
    scorrs, idxs = torch.sort(corrs)
    fstart = int(n_vox*percent)
    mask = np.zeros([n_vox])
    mask[idxs[-fstart:]] = 1
    return fmri[:, idxs[-fstart:]], mask

def sensor_selection_old(meg, feature, save_ratio):
    """
    select the most predictive epoch for each sensor
    meg: (n_words, n_sensor, n_epoch)
    feature: (n_words, n_dim)
    save_ratio: the save ratio of top predictive data
    """
    n_words, n_sensor, n_epoch = meg.shape
    fstart = int(n_epoch*save_ratio)
    res_meg = np.zeros([n_words, n_sensor, fstart])
    mask = np.zeros([n_sensor, n_epoch])
    for sen in range(n_sensor):
        corrs = encoding(meg[:,sen,:], feature)
        scorrs, idxs = torch.sort(corrs)
        mask[sen,idxs[-fstart:]] = 1
        res_meg[:,sen,:] = meg[:,sen,idxs[-fstart:]]
    return res_meg.reshape(n_words, n_sensor*fstart), mask

def sensor_selection(meg, feature, save_ratio):
    """
    select the most predictive sensor, the input meg only have one epoch
    meg: (n_words, n_sensor)
    feature: (n_words, n_dim)
    save_ratio: the save ratio of top predictive sensor
    """
    n_words, n_sensor = meg.shape
    fstart = int(n_sensor*save_ratio)
    res_meg = np.zeros([n_words, fstart])
    mask = np.zeros([n_sensor])
    corrs = encoding(meg, feature)
    scorrs, idxs = torch.sort(corrs)
    mask[idxs[-fstart:]] = 1
    res_meg = meg[:,idxs[-fstart:]]
    return res_meg, mask

def mix_selection(meg, feature, save_ratio):
    """
    meg: (n_words, n_data), n_data = n_sensor*n_epoch
    feature: (n_words, n_dim)
    save_ratio: the save ratio of top predictive data
    """
    _, n_data = meg.shape
    corrs = encoding(meg, feature)
    print(corrs.mean())
    scorrs, idxs = torch.sort(corrs)
    fstart = int(n_data*save_ratio)
    mask = np.zeros([n_data])
    mask[idxs[-fstart:]] = 1
    return meg[:, idxs[-fstart:]], mask