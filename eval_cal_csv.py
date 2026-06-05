# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:10:14 2024

@author: reza
"""

import os 
import pickle
import numpy as np
from skimage.metrics import hausdorff_distance
from utils.eval_metrics import rmse_cal, ssim_cal, dice_cal, compute_sensitivity, compute_msd, jaccard_score, compute_false_discovery_rate
from utils.jac_det import Fn_jac_all_frames
import pandas as pd
import argparse

def rmse_mse_for_all(X_target, X_predicted): 
    rmse_all = []
    mse_all = []
    for x_tar, x_pre in zip(X_target, X_predicted): 
        rmse,mse = rmse_cal(x_tar, x_pre)
        rmse_all.append(rmse)
        mse_all.append(mse)
    rmse_all = np.asarray(rmse_all)
    mse_all = np.asarray(mse_all)
    return rmse_all, mse_all

def negative_jacobian_count_for_all(DVFs):
    neg_jac_all = []
    for idx in range(len(DVFs)): 
        each_deformation_field = DVFs[idx]
        jac_matrix = Fn_jac_all_frames(each_deformation_field)
        neg_jac_counts = np.where(jac_matrix<=0, 1,0).sum()
        neg_jac_all.append(neg_jac_counts)
    neg_jac_all = np.asarray(neg_jac_all)
    return neg_jac_all

def ssim_for_all(X_target, X_predicted): 
    ssim_all = []
    for x_tar, x_pre in zip(X_target, X_predicted):
        ssim_all.append(ssim_cal(x_tar, x_pre))
    ssim_all = np.asarray(ssim_all)
    return ssim_all 

def dice_for_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False):
    dice_all = []
    for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[es_ed_idx, label, :, :, :]
        x_tar = x_tar[es_ed_idx, label, :, :, :]
        dice_all.append(dice_cal(x_tar, x_pre)) 
    dice_all = np.asarray(dice_all)
    return dice_all

# def hausdorff_distance_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False, mode='standard'):
#     hd_all = []
#     for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
#         if thresholding: 
#             x_pre = (x_pre>=0.5).astype('uint8')
#         x_pre = x_pre[es_ed_idx, label, :, :, :]
#         x_tar = x_tar[es_ed_idx, label, :, :, :]
#         hd_all.append(hausdorff_distance(x_tar, x_pre, mode)) 
#     hd_all = np.asarray(hd_all)
#     return hd_all 

def surfaces_metrics_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False, spacing = [1.25, 1.25, 2.0]):
    surface_metrics_all = []
    surface_metrics_all_dict = {'hd95':[], 'hd':[], 'msd':[]}
    for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        for frame_idx in es_ed_idx:
            x_pre_each_frame = x_pre[frame_idx, label, :, :, :]
            x_tar_each_frame = x_tar[frame_idx, label, :, :, :]
            sf_metrics = compute_msd(x_tar_each_frame, x_pre_each_frame, spacing)
            # surface_metrics_all.append(compute_msd(x_tar_each_frame, x_pre_each_frame, spacing)) 
            for key in surface_metrics_all_dict.keys(): 
                surface_metrics_all_dict[key].append(sf_metrics[key])
                
    for key in surface_metrics_all_dict.keys():
        surface_metrics_all_dict[key] = np.asarray(surface_metrics_all_dict[key])
    return surface_metrics_all_dict 

def jaccarod_for_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False):
    jac_all = []
    for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[es_ed_idx, label, :, :, :]
        x_tar = x_tar[es_ed_idx, label, :, :, :]
        jac_all.append(jaccard_score(x_tar, x_pre)) 
    jac_all = np.asarray(jac_all)
    return jac_all

def sensitivity_for_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False):
    se_all = []
    for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[es_ed_idx, label, :, :, :]
        x_tar = x_tar[es_ed_idx, label, :, :, :]
        se_all.append(compute_sensitivity(x_tar, x_pre)) 
    se_all = np.asarray(se_all)
    return se_all

def fdr_for_all(Mask_target, Mask_predicted, ES_ED_Idxs, label, thresholding=False):
    fdr_all = []
    for x_tar, x_pre, es_ed_idx in zip(Mask_target, Mask_predicted, ES_ED_Idxs):
        if thresholding: 
            x_pre = (x_pre>=0.5).astype('uint8')
        x_pre = x_pre[es_ed_idx, label, :, :, :]
        x_tar = x_tar[es_ed_idx, label, :, :, :]
        fdr_all.append(compute_false_discovery_rate(x_tar, x_pre)) 
    fdr_all = np.asarray(fdr_all)
    return fdr_all


def arg_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("suffix_name", type=str, help="The string to process")
    parser.add_argument("results_path", type=str, help="Results' path ")


    args = parser.parse_args()
    return args 
    
if __name__ == "__main__":

    args = arg_parser()
    
    results_link = args.results_path
    results_filenames = os.listdir(results_link)
    
    Data = []
    Mask = []
    ES_ED_Idxs = []
    Seqs = []
    Segs = []
    Disps = []
    Segs_refs = []
    Masks_out = [] 
    
    for each_filename in results_filenames:
        patient_file = os.path.join(results_link, each_filename)
        
        with open(patient_file, 'rb') as file:
            patient_result = pickle.load(file)
            
        img_org = patient_result['Seqs'].squeeze(1)
        img_def = patient_result['Seqs_def'].squeeze(1)
        disp = patient_result['Disps']
        seg_ref = patient_result['Segs_refs']
        seg_def = patient_result['Segs_def']
        es_ed_idxs = patient_result['ES_ED_Idxs']
        mask_org = patient_result['Segs_in']
        mask_def = patient_result['Masks_def']
        # es_ed_idxs = [0,1]
        
        Data.append(img_org)
        Seqs.append(img_def)
        ES_ED_Idxs.append(es_ed_idxs)
        Mask.append(mask_org)
        Segs.append(seg_def)
        Segs_refs.append(seg_ref)
        Masks_out.append(mask_def)
        Disps.append(disp)
        
    rmse_all, mse_all = rmse_mse_for_all(Data, Seqs)
    neg_jac_all = negative_jacobian_count_for_all(Disps)
    ssim_all = ssim_for_all(Data, Seqs)
    
    dice_all_LV = dice_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=1)
    dice_all_Myo = dice_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=2)
    dice_all_RV = dice_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=3)
    
    jac_all_LV = jaccarod_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=1)
    jac_all_Myo  = jaccarod_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=2)
    jac_all_RV = jaccarod_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=3)
    
    se_all_LV = sensitivity_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=1)
    se_all_Myo  = sensitivity_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=2)
    se_all_RV = sensitivity_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=3)
    
    fdr_all_LV = fdr_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=1)
    fdr_all_Myo  = fdr_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=2)
    fdr_all_RV = fdr_for_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=3)
    
    surface_metrics_all_LV = surfaces_metrics_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=1)
    surface_metrics_all_Myo = surfaces_metrics_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=2)
    surface_metrics_all_RV = surfaces_metrics_all(Mask, Masks_out, ES_ED_Idxs, thresholding=True, label=3)
    
    dice_all_LV_seg_mov = dice_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=1)
    dice_all_Myo_seg_mov  = dice_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=2)
    dice_all_RV_seg_mov = dice_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=3)
    
    surface_metrics_all_LV_seg_mov = surfaces_metrics_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=1)
    surface_metrics_all_Myo_seg_mov  = surfaces_metrics_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=2)
    surface_metrics_all_RV_seg_mov = surfaces_metrics_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=3)
    
    jac_all_LV_seg_mov = jaccarod_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=1)
    jac_all_Myo_seg_mov  = jaccarod_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=2)
    jac_all_RV_seg_mov = jaccarod_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=3)
    
    se_all_LV_seg_mov = sensitivity_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=1)
    se_all_Myo_seg_mov  = sensitivity_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=2)
    se_all_RV_seg_mov = sensitivity_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=3)
    
    fdr_all_LV_seg_mov = fdr_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=1)
    fdr_all_Myo_seg_mov  = fdr_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=2)
    fdr_all_RV_seg_mov = fdr_for_all(Mask, Segs, ES_ED_Idxs, thresholding=True, label=3)
    
    dice_all_LV_seg = dice_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=1)
    dice_all_Myo_seg  = dice_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=2)
    dice_all_RV_seg = dice_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=3)
    
    jac_all_LV_seg = jaccarod_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=1)
    jac_all_Myo_seg  = jaccarod_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=2)
    jac_all_RV_seg = jaccarod_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=3)
    
    se_all_LV_seg = sensitivity_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=1)
    se_all_Myo_seg  = sensitivity_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=2)
    se_all_RV_seg = sensitivity_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=3)
    
    fdr_all_LV_seg = fdr_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=1)
    fdr_all_Myo_seg  = fdr_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=2)
    fdr_all_RV_seg = fdr_for_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=3)
    
    surface_metrics_all_LV_seg = surfaces_metrics_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True,  label=1)
    surface_metrics_all_Myo_seg  = surfaces_metrics_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=2)
    surface_metrics_all_RV_seg = surfaces_metrics_all(Mask, Segs_refs, ES_ED_Idxs, thresholding=True, label=3)
    
    print("Mean ssim: ", np.mean(ssim_all)*100)
    print("Std ssim: ", np.std(ssim_all)*100)
    print("Mean rsme: ", np.mean(rmse_all)*100)
    print("Std rsme: ", np.std(rmse_all)*100)
    print("Mean mse: ", np.mean(mse_all)*100)
    print("Std mse: ", np.std(mse_all)*100)
    print("Mean NJD: ", np.mean(neg_jac_all))
    print("Std NJD: ", np.std(neg_jac_all))
    print('*******************************')
    
    print("Mean Dice LV mask: ", np.mean(dice_all_LV)*100)
    print("Std Dice LV mask: ", np.std(dice_all_LV)*100)
    
    print("Mean HD95 LV mask: ", np.mean(surface_metrics_all_LV['hd95']))
    print("Std HD95 LV mask: ", np.std(surface_metrics_all_LV['hd95']))
    print("Mean MSD LV mask: ", np.mean(surface_metrics_all_LV['msd']))
    print("Std MSD LV mask: ", np.std(surface_metrics_all_LV['msd']))
    
    print("Mean Jaccard LV mask: ", np.mean(jac_all_LV)*100)
    print("Std Jaccard LV mask: ", np.std(jac_all_LV)*100)
    print("Mean Sensitivity LV mask: ", np.mean(se_all_LV)*100)
    print("Std Sensitivity LV mask: ", np.std(se_all_LV)*100)
    print("Mean FDR LV mask: ", np.mean(fdr_all_LV)*100)
    print("Std FDR LV mask: ", np.std(fdr_all_LV)*100)
    
    print('*******************************')
    
    print("Mean Dice Myo mask: ", np.mean(dice_all_Myo)*100)
    print("Std Dice Myo mask: ", np.std(dice_all_Myo)*100)

    print("Mean HD95 Myo mask: ", np.mean(surface_metrics_all_Myo['hd95']))
    print("Std HD95 Myo mask: ", np.std(surface_metrics_all_Myo['hd95']))
    print("Mean MSD Myo mask: ", np.mean(surface_metrics_all_Myo['msd']))
    print("Std MSD Myo mask: ", np.std(surface_metrics_all_Myo['msd']))
    
    print("Mean Jaccard Myo mask: ", np.mean(jac_all_Myo)*100)
    print("Std Jaccard Myo mask: ", np.std(jac_all_Myo)*100)
    print("Mean Sensitivity Myo mask: ", np.mean(se_all_Myo)*100)
    print("Std Sensitivity Myo mask: ", np.std(se_all_Myo)*100)
    print("Mean FDR Myo mask: ", np.mean(fdr_all_Myo)*100)
    print("Std FDR Myo mask: ", np.std(fdr_all_Myo)*100)
    
    print('*******************************')
    
    print("Mean Dice RV mask: ", np.mean(dice_all_RV)*100)
    print("Std Dice RV mask: ", np.std(dice_all_RV)*100)
    
    print("Mean HD95 RV mask: ", np.mean(surface_metrics_all_RV['hd95']))
    print("Std HD95 RV mask: ", np.std(surface_metrics_all_RV['hd95']))
    print("Mean MSD RV mask: ", np.mean(surface_metrics_all_RV['msd']))
    print("Std MSD RV mask: ", np.std(surface_metrics_all_RV['msd']))
    
    print("Mean Jaccard RV mask: ", np.mean(jac_all_RV)*100)
    print("Std Jaccard RV mask: ", np.std(jac_all_RV)*100)
    print("Mean Sensitivity RV mask: ", np.mean(se_all_RV)*100)
    print("Std Sensitivity RV mask: ", np.std(se_all_RV)*100)
    print("Mean FDR RV mask: ", np.mean(fdr_all_RV)*100)
    print("Std FDR RV mask: ", np.std(fdr_all_RV)*100)
    
    print('*******************************')
    
    print("Mean Dice LV Seg Mov: ", np.mean(dice_all_LV_seg_mov)*100)
    print("Std Dice LV Seg Mov: ", np.std(dice_all_LV_seg_mov)*100)
    
    print("Mean HD95 LV Seg Mov: ", np.mean(surface_metrics_all_LV_seg_mov['hd95']))
    print("Std HD95 LV Seg Mov: ", np.std(surface_metrics_all_LV_seg_mov['hd95']))
    print("Mean MSD LV Seg Mov: ", np.mean(surface_metrics_all_LV_seg_mov['msd']))
    print("Std MSD LV Seg Mov: ", np.std(surface_metrics_all_LV_seg_mov['msd']))
    
    print("Mean Jaccard LV Seg Mov: ", np.mean(jac_all_LV_seg_mov)*100)
    print("Std Jaccard LV Seg Mov: ", np.std(jac_all_LV_seg_mov)*100)
    print("Mean Sensitivity LV Seg Mov: ", np.mean(se_all_LV_seg_mov)*100)
    print("Std Sensitivity LV Seg Mov: ", np.std(se_all_LV_seg_mov)*100)
    print("Mean FDR LV Seg Mov: ", np.mean(fdr_all_LV_seg_mov)*100)
    print("Std FDR LV Seg Mov: ", np.std(fdr_all_LV_seg_mov)*100)
    print('*******************************')
    
    print("Mean Dice Myo Seg Mov: ", np.mean(dice_all_Myo_seg_mov)*100)
    print("Std Dice Myo Seg Mov: ", np.std(dice_all_Myo_seg_mov)*100)
    
    print("Mean HD95 Myo Seg Mov: ", np.mean(surface_metrics_all_Myo_seg_mov['hd95']))
    print("Std HD95 Myo Seg Mov: ", np.std(surface_metrics_all_Myo_seg_mov['hd95']))
    print("Mean MSD Myo Seg Mov: ", np.mean(surface_metrics_all_Myo_seg_mov['msd']))
    print("Std MSD Myo Seg Mov: ", np.std(surface_metrics_all_Myo_seg_mov['msd']))
    
    print("Mean Jaccard Myo Seg Mov: ", np.mean(jac_all_Myo_seg_mov)*100)
    print("Std Jaccard Myo Seg Mov: ", np.std(jac_all_Myo_seg_mov)*100)
    print("Mean Sensitivity Myo Seg Mov: ", np.mean(se_all_Myo_seg_mov)*100)
    print("Std Sensitivity Myo Seg Mov: ", np.std(se_all_Myo_seg_mov)*100)
    print("Mean FDR Myo Seg Mov: ", np.mean(fdr_all_Myo_seg_mov)*100)
    print("Std FDR Myo Seg Mov: ", np.std(fdr_all_Myo_seg_mov)*100)
    print('*******************************')
    
    print("Mean Dice RV Seg Mov: ", np.mean(dice_all_RV_seg_mov)*100)
    print("Std Dice RV Seg Mov: ", np.std(dice_all_RV_seg_mov)*100)
    
    print("Mean HD95 RV Seg Mov: ", np.mean(surface_metrics_all_RV_seg_mov['hd95']))
    print("Std HD95 RV Seg Mov: ", np.std(surface_metrics_all_RV_seg_mov['hd95']))
    print("Mean MSD RV Seg Mov: ", np.mean(surface_metrics_all_RV_seg_mov['msd']))
    print("Std MSD RV Seg Mov: ", np.std(surface_metrics_all_RV_seg_mov['msd']))
    
    print("Mean Jaccard RV Seg Mov: ", np.mean(jac_all_RV_seg_mov)*100)
    print("Std Jaccard RV Seg Mov: ", np.std(jac_all_RV_seg_mov)*100)
    print("Mean Sensitivity RV Seg Mov: ", np.mean(se_all_RV_seg_mov)*100)
    print("Std Sensitivity RV Seg Mov: ", np.std(se_all_RV_seg_mov)*100)
    print("Mean FDR RV Seg Mov: ", np.mean(fdr_all_RV_seg_mov)*100)
    print("Std FDR RV Seg Mov: ", np.std(fdr_all_RV_seg_mov)*100)
    print('*******************************')
    

    results = {
        
        "Mean_SSIM": np.mean(ssim_for_all(Data, Seqs))*100,
        "Std_SSIM": np.std(ssim_for_all(Data, Seqs))*100,
        "Mean_RMSE": np.mean(rmse_mse_for_all(Data, Seqs)[0])*100,
        "Std_RMSE": np.std(rmse_mse_for_all(Data, Seqs)[0])*100,
        "Mean_MSE": np.mean(rmse_mse_for_all(Data, Seqs)[1])*100,
        "Std_MSE": np.std(rmse_mse_for_all(Data, Seqs)[1])*100,
        "Mean_NJD": np.mean(negative_jacobian_count_for_all(Disps)),
        "Std_NJD": np.std(negative_jacobian_count_for_all(Disps)),
    }
    
    data_mask = {
        
        "LV": {"Mean Dice":  np.mean(dice_all_LV)*100, 
               "Std Dice": np.std(dice_all_LV)*100,
               "Mean Jaccard": np.mean(jac_all_LV)*100, 
               "Std Jaccard": np.std(jac_all_LV)*100,   
               "Mean Sensitivity": np.mean(se_all_LV)*100, 
               "Std Sensitivity": np.std(se_all_LV)*100, 
               "Mean FDR": np.mean(fdr_all_LV)*100, 
               "Std FDR": np.std(fdr_all_LV)*100, 
               "Mean HD95": np.mean(surface_metrics_all_LV['hd95']), 
               "Std HD95": np.std(surface_metrics_all_LV['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_LV['msd']), 
               "Std MSD": np.std(surface_metrics_all_LV['msd'])},
        
        "Myo": {"Mean Dice":  np.mean(dice_all_Myo)*100, 
               "Std Dice": np.std(dice_all_Myo)*100,
               "Mean Jaccard": np.mean(jac_all_Myo)*100, 
               "Std Jaccard": np.std(jac_all_Myo)*100,   
               "Mean Sensitivity": np.mean(se_all_Myo)*100, 
               "Std Sensitivity": np.std(se_all_Myo)*100, 
               "Mean FDR": np.mean(fdr_all_Myo)*100, 
               "Std FDR": np.std(fdr_all_Myo)*100, 
               "Mean HD95": np.mean(surface_metrics_all_Myo['hd95']), 
               "Std HD95": np.std(surface_metrics_all_Myo['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_Myo['msd']), 
               "Std MSD": np.std(surface_metrics_all_Myo['msd'])},
        
        "RV": {"Mean Dice":  np.mean(dice_all_RV)*100, 
               "Std Dice": np.std(dice_all_RV)*100,
               "Mean Jaccard": np.mean(jac_all_RV)*100, 
               "Std Jaccard": np.std(jac_all_RV)*100,   
               "Mean Sensitivity": np.mean(se_all_RV)*100, 
               "Std Sensitivity": np.std(se_all_RV)*100, 
               "Mean FDR": np.mean(fdr_all_RV)*100, 
               "Std FDR": np.std(fdr_all_RV)*100, 
               "Mean HD95": np.mean(surface_metrics_all_RV['hd95']), 
               "Std HD95": np.std(surface_metrics_all_RV['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_RV['msd']), 
               "Std MSD": np.std(surface_metrics_all_RV['msd'])}, 
    }
    
    data_seg_mov = {
        
        "LV": {"Mean Dice":  np.mean(dice_all_LV_seg_mov)*100, 
               "Std Dice": np.std(dice_all_LV_seg_mov)*100,
               "Mean Jaccard": np.mean(jac_all_LV_seg_mov)*100, 
               "Std Jaccard": np.std(jac_all_LV_seg_mov)*100,   
               "Mean Sensitivity": np.mean(se_all_LV_seg_mov)*100, 
               "Std Sensitivity": np.std(se_all_LV_seg_mov)*100, 
               "Mean FDR": np.mean(fdr_all_LV_seg_mov)*100, 
               "Std FDR": np.std(fdr_all_LV_seg_mov)*100, 
               "Mean HD95": np.mean(surface_metrics_all_LV_seg_mov['hd95']), 
               "Std HD95": np.std(surface_metrics_all_LV_seg_mov['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_LV_seg_mov['msd']), 
               "Std MSD": np.std(surface_metrics_all_LV_seg_mov['msd'])},
        
        "Myo": {"Mean Dice":  np.mean(dice_all_Myo_seg_mov)*100, 
               "Std Dice": np.std(dice_all_Myo_seg_mov)*100,
               "Mean Jaccard": np.mean(jac_all_Myo_seg_mov)*100, 
               "Std Jaccard": np.std(jac_all_Myo_seg_mov)*100,   
               "Mean Sensitivity": np.mean(se_all_Myo_seg_mov)*100, 
               "Std Sensitivity": np.std(se_all_Myo_seg_mov)*100, 
               "Mean FDR": np.mean(fdr_all_Myo_seg_mov)*100, 
               "Std FDR": np.std(fdr_all_Myo_seg_mov)*100, 
               "Mean HD95": np.mean(surface_metrics_all_Myo_seg_mov['hd95']), 
               "Std HD95": np.std(surface_metrics_all_Myo_seg_mov['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_Myo_seg_mov['msd']), 
               "Std MSD": np.std(surface_metrics_all_Myo_seg_mov['msd'])},
        
        "RV": {"Mean Dice":  np.mean(dice_all_RV_seg_mov)*100, 
               "Std Dice": np.std(dice_all_RV_seg_mov)*100,
               "Mean Jaccard": np.mean(jac_all_RV_seg_mov)*100, 
               "Std Jaccard": np.std(jac_all_RV_seg_mov)*100,   
               "Mean Sensitivity": np.mean(se_all_RV_seg_mov)*100, 
               "Std Sensitivity": np.std(se_all_RV_seg_mov)*100, 
               "Mean FDR": np.mean(fdr_all_RV_seg_mov)*100, 
               "Std FDR": np.std(fdr_all_RV_seg_mov)*100, 
               "Mean HD95": np.mean(surface_metrics_all_RV_seg_mov['hd95']), 
               "Std HD95": np.std(surface_metrics_all_RV_seg_mov['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_RV_seg_mov['msd']), 
               "Std MSD": np.std(surface_metrics_all_RV_seg_mov['msd'])}, 
    }
    
    data_seg = {
        
        "LV": {"Mean Dice":  np.mean(dice_all_LV_seg)*100, 
               "Std Dice": np.std(dice_all_LV_seg)*100,
               "Mean Jaccard": np.mean(jac_all_LV_seg)*100, 
               "Std Jaccard": np.std(jac_all_LV_seg)*100,   
               "Mean Sensitivity": np.mean(se_all_LV_seg)*100, 
               "Std Sensitivity": np.std(se_all_LV_seg)*100, 
               "Mean FDR": np.mean(fdr_all_LV_seg)*100, 
               "Std FDR": np.std(fdr_all_LV_seg)*100, 
               "Mean HD95": np.mean(surface_metrics_all_LV_seg['hd95']), 
               "Std HD95": np.std(surface_metrics_all_LV_seg['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_LV_seg['msd']), 
               "Std MSD": np.std(surface_metrics_all_LV_seg['msd'])},
        
        "Myo": {"Mean Dice":  np.mean(dice_all_Myo_seg)*100, 
               "Std Dice": np.std(dice_all_Myo_seg)*100,
               "Mean Jaccard": np.mean(jac_all_Myo_seg)*100, 
               "Std Jaccard": np.std(jac_all_Myo_seg)*100,   
               "Mean Sensitivity": np.mean(se_all_Myo_seg)*100, 
               "Std Sensitivity": np.std(se_all_Myo_seg)*100, 
               "Mean FDR": np.mean(fdr_all_Myo_seg)*100, 
               "Std FDR": np.std(fdr_all_Myo_seg)*100, 
               "Mean HD95": np.mean(surface_metrics_all_Myo_seg['hd95']), 
               "Std HD95": np.std(surface_metrics_all_Myo_seg['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_Myo_seg['msd']), 
               "Std MSD": np.std(surface_metrics_all_Myo_seg['msd'])},
        
        "RV": {"Mean Dice":  np.mean(dice_all_RV_seg)*100, 
               "Std Dice": np.std(dice_all_RV_seg)*100,
               "Mean Jaccard": np.mean(jac_all_RV_seg)*100, 
               "Std Jaccard": np.std(jac_all_RV_seg)*100,   
               "Mean Sensitivity": np.mean(se_all_RV_seg)*100, 
               "Std Sensitivity": np.std(se_all_RV_seg)*100, 
               "Mean FDR": np.mean(fdr_all_RV_seg)*100, 
               "Std FDR": np.std(fdr_all_RV_seg)*100, 
               "Mean HD95": np.mean(surface_metrics_all_RV_seg['hd95']), 
               "Std HD95": np.std(surface_metrics_all_RV_seg['hd95']), 
               "Mean MSD": np.mean(surface_metrics_all_RV_seg['msd']), 
               "Std MSD": np.std(surface_metrics_all_RV_seg['msd'])}, 
    }
    
    data_img_reg = {
        
        "SSIM": {"Mean": np.mean(ssim_all)*100, 
               "Std": np.std(ssim_all)*100},
        
        "RMSE": {"Mean": np.mean(rmse_all)*100, 
               "Std": np.std(rmse_all)*100},
        
        "NJD": {"Mean": np.mean(neg_jac_all), 
               "Std": np.std(neg_jac_all)}, 
    }
    
    
    df_mask_reg = pd.DataFrame.from_dict(data_mask, orient="index")  
    df_seg_reg = pd.DataFrame.from_dict(data_seg_mov, orient="index")  
    df_seg = pd.DataFrame.from_dict(data_seg, orient="index")  
    df_img_reg = pd.DataFrame.from_dict(data_img_reg, orient="index")  

    
    
    df_mask_reg.to_csv(f"mask_reg_results_{args.suffix_name}.csv")
    df_seg_reg.to_csv(f"seg_reg_results_{args.suffix_name}.csv")
    df_seg.to_csv(f"seg_results_{args.suffix_name}.csv")
    df_img_reg.to_csv(f"img_reg_results_{args.suffix_name}.csv")


