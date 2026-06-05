# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:21:33 2024

@author: reza
"""

import numpy as np 
import cv2 
from scipy.ndimage import label, find_objects

def get_largest_region_bbox(mask):
    """Get the bounding box of the largest connected region in the mask."""
    labeled_mask, num_features = label(mask)
    sizes = [np.sum(labeled_mask == i + 1) for i in range(num_features)]
    largest_region_label = np.argmax(sizes) + 1
    largest_region_mask = (labeled_mask == largest_region_label)
    bbox = find_objects(labeled_mask == largest_region_label)[0]
    return bbox, largest_region_mask

def crop_img(img, mask, margin=30):
    bbox, largest_region_mask = get_largest_region_bbox(mask)
    x_min, x_max = bbox[0].start, bbox[0].stop
    y_min, y_max = bbox[1].start, bbox[1].stop
    
    # Ensure margin does not go out of bounds
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, img.shape[0])
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, img.shape[1])
    
    cropped_img = img[x_min:x_max, y_min:y_max]
    cropped_mask = mask[x_min:x_max, y_min:y_max]
    return cropped_img, cropped_mask


def arange_frames(Frame_number, ED_index, ES_index, space = 2): 
    seq = np.arange(Frame_number)
    diff_ES_ED = abs(ED_index - ES_index) - 1
    step = diff_ES_ED//space
    Idx = 0
    inc_idxs = []
    frame_counter = 0
    while  frame_counter<=space:
        if Idx + ED_index >= Frame_number:
            inc_idxs.append(seq[Idx + ED_index - Frame_number])
        else: 
            inc_idxs.append(seq[Idx + ED_index])
        Idx = Idx + step
        frame_counter = frame_counter + 1
    frame_counter = 0    
    Idx = 0
    while frame_counter<=space:
        if ES_index + Idx >= Frame_number:
            inc_idxs.append(seq[ES_index + Idx - Frame_number])
        else: 
            inc_idxs.append(seq[ES_index + Idx])
        Idx = Idx + step
        frame_counter = frame_counter + 1
    return inc_idxs

def resize_4DIMG(input_img, desired_size):
    img_resized = np.zeros([desired_size[0], desired_size[1], input_img.shape[-2], input_img.shape[-1]], input_img.dtype)
    for frame_idx in range(img_resized.shape[-1]):
        for slice_idx in range(img_resized.shape[-2]):
            img_resized[:,:,slice_idx, frame_idx] = cv2.resize(input_img[:,:,slice_idx, frame_idx], desired_size)
    return img_resized

def One_hot_cat(Mask): 
    Labels = np.unique(Mask)
    One_hot_form = []
    for each_label in Labels:
        One_hot_form.append(np.where(Mask == each_label, 1, 0))
    One_hot_form = np.asarray(One_hot_form)
    return One_hot_form

def normalize(img):
    min_values = img.min(axis=(0,1), keepdims=True)
    max_values = img.max(axis=(0,1), keepdims=True)
    img_normalised = (img - min_values) / (max_values - min_values)
    # Scale to [-1, 1]
    img_scaled = 2 * img_normalised - 1
    return img_normalised, img_scaled

def zero_padding(x, type='img'):
    if type=='img':
        padding = [(0, 0), (0, 0), (0,0), (0, 22 - x.shape[-1])]
    elif type=='mask':
        padding = [(0, 0), (0,0), (0, 22 - x.shape[-2]), (0, 0)]    
    x_padded = np.pad(x, pad_width=padding, mode='constant', constant_values=0.0)
    return x_padded


def preprocess_fun(Img, Mask, ED_index, ES_index, zer_pad_bool, desired_size=(128,128)):
    ## Cropping the image 
    Img, Mask = crop_img(Img,Mask, margin=30)
    # Make the mask LV =1, Myp=2, and RV =3 
    tmp = -1              # temporary label not used in the mask
    Mask[Mask == 1] = tmp
    Mask[Mask == 3] = 1
    Mask[Mask == tmp] = 3

    ## Time sampling between ED and ES phases
    inc_idxs = arange_frames(Img.shape[-1], ED_index, ES_index, space=2)
    ED_ES_indexes = [inc_idxs.index(ED_index), inc_idxs.index(ES_index)]
    Img = Img[:,:,:,inc_idxs]
    # Mask = Mask[:,:,:,inc_idxs]
    if zer_pad_bool:
        Mask = zero_padding(Mask, type='mask')
    ## One-hot encoding for ground truth representattion
    Mask = One_hot_cat(Mask)   
    Mask = Mask.astype('uint8') ## Changing the data type of mask image 
    ## Resizing step 
    Img_resized = resize_4DIMG(Img, desired_size)
    Mask_resized = np.zeros([Mask.shape[0], desired_size[0], desired_size[1], Mask.shape[3], Mask.shape[4]], Mask.dtype)
    for each_label in range(Mask.shape[0]):
        Mask_resized[each_label, :, :, :, :] = resize_4DIMG(Mask[each_label, :, :, :, :], desired_size)        
    Img_normalized,_ = normalize(Img_resized)
    Img_normalized = np.transpose(Img_normalized, axes=(3, 0, 1, 2))
    Mask_resized = np.transpose(Mask_resized, axes=(4, 0, 1, 2, 3))
    if zer_pad_bool:
        Img_normalized = zero_padding(Img_normalized, type='img')
    return Img_normalized, Mask_resized, ED_ES_indexes