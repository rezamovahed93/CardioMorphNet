# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:10:14 2024

@author: reza
"""

import os 
import pickle
import numpy as np
from skimage.metrics import hausdorff_distance
from utils.eval_metrics import dice_cal, compute_sensitivity, jaccard_score, compute_false_discovery_rate

def dice_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    dice_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]
        dice_all.append(dice_cal(x_tar, x_pre)) 
    dice_all = np.asarray(dice_all)
    return dice_all

def hausdorff_distance_all(Mask_target, Mask_predicted, label, thresholding=False, mode='standard'):
    hd_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]
        hd_all.append(hausdorff_distance(x_tar, x_pre, mode)) 
    hd_all = np.asarray(hd_all)
    return hd_all 

def jaccarod_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    jac_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]
        jac_all.append(jaccard_score(x_tar, x_pre)) 
    jac_all = np.asarray(jac_all)
    return jac_all

def sensitivity_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    se_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]
        se_all.append(compute_sensitivity(x_tar, x_pre)) 
    se_all = np.asarray(se_all)
    return se_all

def fdr_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    fdr_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]
        fdr_all.append(compute_false_discovery_rate(x_tar, x_pre)) 
    fdr_all = np.asarray(fdr_all)
    return fdr_all
    
results_link = os.path.join(os.getcwd(), "results_seg_only")
results_filenames = os.listdir(results_link)

Segs_refs = []
Segs_outs = []

for each_filename in results_filenames:
    patient_file = os.path.join(results_link, each_filename)
    
    with open(patient_file, 'rb') as file:
        patient_result = pickle.load(file)
        
    seg_ref = patient_result['Segs_in']
    seg_out = patient_result['Segs_out']

    Segs_refs.append(seg_ref)
    Segs_outs.append(seg_out)

dice_all_LV_seg = dice_for_all(Segs_refs, Segs_outs, thresholding=True, label=1)
dice_all_Myo_seg  = dice_for_all(Segs_refs, Segs_outs, thresholding=True, label=2)
dice_all_RV_seg = dice_for_all(Segs_refs, Segs_outs, thresholding=True, label=3)

jac_all_LV_seg = jaccarod_for_all(Segs_refs, Segs_outs, thresholding=True, label=1)
jac_all_Myo_seg  = jaccarod_for_all(Segs_refs, Segs_outs, thresholding=True, label=2)
jac_all_RV_seg = jaccarod_for_all(Segs_refs, Segs_outs, thresholding=True, label=3)

se_all_LV_seg = sensitivity_for_all(Segs_refs, Segs_outs, thresholding=True, label=1)
se_all_Myo_seg  = sensitivity_for_all(Segs_refs, Segs_outs, thresholding=True, label=2)
se_all_RV_seg = sensitivity_for_all(Segs_refs, Segs_outs, thresholding=True, label=3)

fdr_all_LV_seg = fdr_for_all(Segs_refs, Segs_outs, thresholding=True, label=1)
fdr_all_Myo_seg  = fdr_for_all(Segs_refs, Segs_outs, thresholding=True, label=2)
fdr_all_RV_seg = fdr_for_all(Segs_refs, Segs_outs, thresholding=True, label=3)

hd_all_LV_seg = hausdorff_distance_all(Segs_refs, Segs_outs, thresholding=True,  label=1, mode='standard')
hd_all_Myo_seg  = hausdorff_distance_all(Segs_refs, Segs_outs, thresholding=True, label=2, mode='standard')
hd_all_RV_seg = hausdorff_distance_all(Segs_refs, Segs_outs, thresholding=True, label=3, mode='standard')


print("Mean Dice LV Seg : ", np.mean(dice_all_LV_seg)*100)
print("Std Dice LV Seg : ", np.std(dice_all_LV_seg)*100)
print("Mean Jaccard LV Seg : ", np.mean(jac_all_LV_seg)*100)
print("Std Jaccard LV Seg : ", np.std(jac_all_LV_seg)*100)
print("Mean Sensitivity LV Seg : ", np.mean(se_all_LV_seg)*100)
print("Std Sensitivity LV  : ", np.std(se_all_LV_seg)*100)
print("Mean FDR LV Seg : ", np.mean(fdr_all_LV_seg)*100)
print("Std FDR LV Seg: ", np.std(fdr_all_LV_seg)*100)

print("Mean Dice Myo Seg : ", np.mean(dice_all_Myo_seg)*100)
print("Std Dice Myo Seg : ", np.std(dice_all_Myo_seg)*100)
print("Mean Jaccard Myo Seg : ", np.mean(jac_all_Myo_seg)*100)
print("Std Jaccard Myo Seg : ", np.std(jac_all_Myo_seg)*100)
print("Mean Sensitivity Myo Seg : ", np.mean(se_all_Myo_seg)*100)
print("Std Sensitivity Myo Seg : ", np.std(se_all_Myo_seg)*100)
print("Mean FDR Myo Seg : ", np.mean(fdr_all_Myo_seg)*100)
print("Std FDR Myo Seg : ", np.std(fdr_all_Myo_seg)*100)

print("Mean Dice RV Seg : ", np.mean(dice_all_RV_seg)*100)
print("Std Dice RV Seg : ", np.std(dice_all_RV_seg)*100)
print("Mean Jaccard RV Seg : ", np.mean(jac_all_RV_seg)*100)
print("Std Jaccard RV Seg : ", np.std(jac_all_RV_seg)*100)
print("Mean Sensitivity RV Seg : ", np.mean(se_all_RV_seg)*100)
print("Std Sensitivity RV Seg : ", np.std(se_all_RV_seg)*100)
print("Mean FDR RV Seg : ", np.mean(fdr_all_RV_seg)*100)
print("Std FDR RV Seg : ", np.std(fdr_all_RV_seg)*100)

print("Mean Dice LV Seg: ", np.mean(dice_all_LV_seg)*100)
print("Std Dice LV Seg: ", np.std(dice_all_LV_seg)*100)
print("Mean Dice Myo Seg: ", np.mean(dice_all_Myo_seg)*100)
print("Std Dice Myo Seg: ", np.std(dice_all_Myo_seg)*100)
print("Mean Dice RV Seg: ", np.mean(dice_all_RV_seg)*100)
print("Std Dice RV Seg: ", np.std(dice_all_RV_seg)*100)

print("Mean Hausdorff Distance LV Seg: ", np.mean(hd_all_LV_seg))
print("Std Hausdorff Distance LV Seg: ", np.std(hd_all_LV_seg))
print("Mean Hausdorff Distance Myo Seg: ", np.mean(hd_all_Myo_seg))
print("Std Hausdorff Distance Myo Seg: ", np.std(hd_all_Myo_seg))
print("Mean Hausdorff Distance RV Seg: ", np.mean(hd_all_RV_seg))
print("Std Hausdorff Distance RV Seg: ", np.std(hd_all_RV_seg))