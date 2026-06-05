# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:54:52 2024

@author: reza
"""

import numpy as np
import SimpleITK as sitk

def rmse_cal(predictions, targets):
    
    # Calculate squared differences between predictions and targets
    squared_diff = (predictions - targets) ** 2
    
    # Calculate mean squared error
    mse = np.mean(squared_diff)
    
    # Calculate RMSE
    rmse_score = np.sqrt(mse)
    
    return rmse_score, mse


def ssim_cal(image1, image2, L=1):

    # Define constants
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    # Calculate means
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    
    # Calculate variances
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    
    # Calculate covariance
    sigma12 = np.cov(image1.flatten(), image2.flatten())[0, 1]
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_score = numerator / denominator
    
    return ssim_score


def dice_cal(image1, image2, eps=1e-9):
    # Ensure the images are binary
    image1 = image1.astype(bool)
    image2 = image2.astype(bool)
    
    intersection = np.logical_and(image1, image2)
    dice = 2. * intersection.sum() / (image1.sum() + image2.sum() + eps)
    return dice

def jaccard_score(image1, image2, eps=1e-9):
    # Ensure the images are binary
    image1 = image1.astype(bool)
    image2 = image2.astype(bool)
    
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    
    jaccard = intersection.sum() / (union.sum() + eps)
    return jaccard

def compute_sensitivity(ground_truth, prediction):
    # Convert arrays to boolean where 'True' is a positive (1)
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

    # True Positives (TP): Both ground truth and prediction are positive
    TP = np.logical_and(ground_truth, prediction).sum()

    # False Negatives (FN): Ground truth is positive but prediction is not
    FN = np.logical_and(ground_truth, np.logical_not(prediction)).sum()

    # Compute sensitivity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero
    return sensitivity

def compute_false_discovery_rate(ground_truth, prediction):
    # Convert arrays to boolean where 'True' is a positive (1)
    ground_truth = ground_truth.astype(bool)
    prediction = prediction.astype(bool)

    # True Positives (TP): Both ground truth and prediction are positive
    TP = np.logical_and(ground_truth, prediction).sum()

    # False Positives (FP): Prediction is positive but ground truth is not
    FP = np.logical_and(np.logical_not(ground_truth), prediction).sum()

    # Compute False Discovery Rate
    false_discovery_rate = FP / (FP + TP) if (FP + TP) > 0 else 0  # Avoid division by zero
    return false_discovery_rate


def compute_msd(lT, lP, spacing, fullyConnected=True):
    
    """

    :param lP: prediction, shape (x, y, z)
    :param lT: ground truth, shape (x, y, z)
    :param spacing: shape order (x, y, z)
    :return: metrics_names: container contains metircs names
    """
    
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing(np.array(spacing).astype(np.float64))
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing(np.array(spacing).astype(np.float64))  # spacing order (x, y, z)
    
    if np.sum(lT) == 0:  # all 0, set the distance map to 0. Otherwise, SignedMaurerDistanceMap may raise exceptions.
        signed_distance_map = sitk.GetImageFromArray(lT, isVector=False)
        signed_distance_map.SetSpacing(spacing)
    else:
        signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False,
                                                       useImageSpacing=True)  # It need to be adapted.

    ref_distance_map = sitk.Abs(signed_distance_map)

    ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=fullyConnected)
    ref_surface_array = sitk.GetArrayViewFromImage(ref_surface)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface > 0.5)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())


    if np.sum(lP) == 0:  # all 0, set the distance map to 0. Otherwise, SignedMaurerDistanceMap may raise exceptions.
        signed_distance_map_pred = sitk.GetImageFromArray(lP, isVector=False)
        signed_distance_map_pred.SetSpacing(spacing)
    else:
        signed_distance_map_pred = sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False,
                                                                useImageSpacing=True)

    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    seg_surface = sitk.LabelContour(labelPred > 0.5, fullyConnected=fullyConnected)
    seg_surface_array = sitk.GetArrayViewFromImage(seg_surface)

    seg2ref_distance_map = sitk.Cast(ref_distance_map, sitk.sitkFloat32) * sitk.Cast(seg_surface, sitk.sitkFloat32)

    ref2seg_distance_map = sitk.Cast(seg_distance_map, sitk.sitkFloat32) * sitk.Cast(ref_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(seg_surface > 0.5)

    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    
    quality = dict()

    if len(all_surface_distances) == 0:  
        quality["hd95"] = 0
        quality["hd"] = 0
        quality["msd"] = 0
        quality["mdsd"] = 0
        quality["stdsd"] = 0
        quality["vs"] = 1
    else:
        quality["hd95"] = np.percentile(all_surface_distances, 95)
        quality["hd"] = np.max(all_surface_distances)
        quality["msd"] = np.mean(all_surface_distances)
        quality["mdsd"] = np.median(all_surface_distances)
        quality["stdsd"] = np.std(all_surface_distances)
    
    return quality