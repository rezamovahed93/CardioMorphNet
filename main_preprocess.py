#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:51:42 2025

@author: rezeakbarimovahed
"""


import numpy as np
import nibabel as nib
import os
from utils.parameter_parser import parse_args_preprocess as parse_args
from utils.preprocess_pack import preprocess_fun
import shutil
import pandas as pd
import pickle


def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        os.mkdir(directory_path)
    else:
        os.mkdir(directory_path)
        
def read_preprocess_and_save(seq_data_path, mask_data_path, ED_idx, ES_idx, zero_pad_bool, saving_path):
    
    seq_data = nib.load(seq_data_path).get_fdata()
    mask_data = nib.load(mask_data_path).get_fdata()
    
    seq_data, mask_data, es_ed_idxs = preprocess_fun(seq_data, 
                                                     mask_data, 
                                                     ED_idx, 
                                                     ES_idx, 
                                                     zero_pad_bool)

    print(es_ed_idxs)
    print(saving_path)
    print("----------")
    
    data = {'seqs':seq_data, 'masks':mask_data, 'es_ed_idxs':es_ed_idxs}
    
    pikle_data_path = os.path.join(saving_path, 'data.pkl')
    with open(pikle_data_path, 'wb') as file:
        pickle.dump(data, file)
    

    
if __name__ == "__main__":
    
    args = parse_args()
    
    # args.data_path = "/data/MyPhD/My PHD Datasets/M&M Dataset/OpenDataset"
    # args.csv_paths = '/data/MyPhD/My PHD Datasets/M&M Dataset/OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'
    # args.preprocessed_data_path = "./M&M data" 
    # args.zero_pad_flag = True
    
    df = pd.read_csv(args.csv_paths)
    
    suffixes = ['Training', 'Validation', 'Testing']
    
    create_directory(args.preprocessed_data_path)
    
    zero_pad_bool = args.zero_pad_flag

    for each_suffix_set in suffixes:
        
        if each_suffix_set=='Training':
            data_dir = os.path.join(args.data_path, each_suffix_set, 'Labeled')
            patients_folders = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
        else:
            data_dir = os.path.join(args.data_path, each_suffix_set)
            patients_folders = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
            
            
        df_data = pd.DataFrame(patients_folders, columns=['Patient_ID'])
        df_data.to_csv(f'{each_suffix_set}_patients.csv', index=False)
            
        for each_patient in patients_folders: 
            
            sa_dir = os.path.join(data_dir, each_patient, f"{each_patient}_sa.nii.gz")
            sa_mask_dir = os.path.join(data_dir, each_patient, f"{each_patient}_sa_gt.nii.gz")
            
            ED_idx = df.loc[df["External code"] == each_patient, 'ED'].iloc[0]
            ES_idx = df.loc[df["External code"] == each_patient, 'ES'].iloc[0]
            
            each_patient_dir = os.path.join(args.preprocessed_data_path, each_patient)
            
            create_directory(each_patient_dir)
            
            read_preprocess_and_save(sa_dir, 
                                     sa_mask_dir, 
                                     ED_idx, 
                                     ES_idx, 
                                     zero_pad_bool, 
                                     each_patient_dir)
            
            print(f'The process for patient {each_patient} finished....')
            
            
    