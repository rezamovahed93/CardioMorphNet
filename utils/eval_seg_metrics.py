# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:06:35 2024

@author: reza
"""

import numpy as np
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import cdist

def dice_score(binary_map1, binary_map2):
    binary_map1 = binary_map1.astype(bool) 
    binary_map2 = binary_map2.astype(bool) 
    intersection = np.sum(binary_map1 & binary_map2)
    dice = (2. * intersection) / (np.sum(binary_map1) + np.sum(binary_map2)) 
    return dice

def jaccard_score(binary_map1, binary_map2):
    binary_map1 = binary_map1.astype(bool) 
    binary_map2 = binary_map2.astype(bool) 
    intersection = np.sum(binary_map1 & binary_map2)
    union = np.sum(binary_map1 | binary_map2)
    jaccard = intersection / (union)
    return jaccard

def hausdorff_metrice_cal(Masks, Segs, ES_ED_Idxs):
    
    hausd_metrics_RV = []
    hausd_metrics_MYO = []
    
    for each_sample in range(len(Masks)): 
        T = Masks[each_sample]
        Y = Segs[each_sample]
        Y = (Y > 0.5).astype('uint8') * 1
        Y = Y[ES_ED_Idxs[each_sample][0], :,:,:,:]
        
        hausd_metrics_RV.append(hausdorff_distance(T[:,1,:,:,:], Y[:,1,:,:,:])) 
        hausd_metrics_MYO.append(hausdorff_distance(T[:,2,:,:,:], Y[:,2,:,:,:]))
    return hausd_metrics_RV, hausd_metrics_MYO

def compute_sens_fdr(ground_truth, prediction):
    # Convert arrays to boolean where 'True' is a positive (1)
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

    # True Positives (TP): Both ground truth and prediction are positive
    TP = np.logical_and(ground_truth, prediction).sum()

    # False Negatives (FN): Ground truth is positive but prediction is not
    FN = np.logical_and(ground_truth, np.logical_not(prediction)).sum()
    
    # False Positives (FP): Prediction is positive but ground truth is not
    FP = np.logical_and(np.logical_not(ground_truth), prediction).sum()

    # Compute sensitivity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero
    false_discovery_rate = FP / (FP + TP) if (FP + TP) > 0 else 0  # Avoid division by zero

    return sensitivity, false_discovery_rate
    

def dice_metric_cal(Masks, Segs, ES_ED_Idxs): 
    
    dice_metrics_RV = []
    dice_metrics_MYO = []
    
    for each_sample in range(len(Masks)): 
        T = Masks[each_sample]
        Y = Segs[each_sample]
        Y = (Y > 0.5).astype('uint8') * 1
        Y = Y[ES_ED_Idxs[each_sample][0], :,:,:,:]
        
        dice_metrics_RV.append(dice_score(T[:,1,:,:,:], Y[:,1,:,:,:]))
        dice_metrics_MYO.append(dice_score(T[:,2,:,:,:], Y[:,2,:,:,:]))
        
    return dice_metrics_RV, dice_metrics_MYO

def jacc_metric_cal(Masks, Segs, ES_ED_Idxs): 
    
    jacc_metrics_RV = []
    jacc_metrics_MYO = []
    
    for each_sample in range(len(Masks)): 
        T = Masks[each_sample]
        Y = Segs[each_sample]
        Y = (Y > 0.5).astype('uint8') * 1
        Y = Y[ES_ED_Idxs[each_sample][0], :,:,:,:]
        
        jacc_metrics_RV.append(jaccard_score(T[:,1,:,:,:], Y[:,1,:,:,:]))
        jacc_metrics_MYO.append(jaccard_score(T[:,2,:,:,:], Y[:,2,:,:,:]))
        
    return jacc_metrics_RV, jacc_metrics_MYO

def fdr_sen_cal(Masks, Segs, ES_ED_Idxs):
    
    sen_metrics_RV = []
    sen_metrics_MYO = []
    
    fdr_metrics_RV = []
    fdr_metrics_MYO = []
    
    for each_sample in range(len(Masks)): 
        T = Masks[each_sample]
        Y = Segs[each_sample]
        Y = (Y > 0.5).astype('uint8') * 1
        Y = Y[ES_ED_Idxs[each_sample][0], :,:,:,:]
        
        sen_rv, fdr_rv = compute_sens_fdr(T[:,1,:,:,:], Y[:,1,:,:,:])
        sen_myo, fdr_myo = compute_sens_fdr(T[:,2,:,:,:], Y[:,2,:,:,:])
        
        sen_metrics_RV.append(sen_rv)
        sen_metrics_MYO.append(sen_myo)
        
        fdr_metrics_RV.append(fdr_rv)
        fdr_metrics_MYO.append(fdr_myo)
        
    return sen_metrics_RV, sen_metrics_MYO, fdr_metrics_RV, fdr_metrics_MYO