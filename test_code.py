# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:05:16 2024

@author: reza
"""

import numpy as np
import os
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from utils.parameter_parser import parse_args_test as parse_args
import shutil
from utils.model import MyModel
import pickle
import warnings 

warnings.filterwarnings("ignore")

def freez_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        os.mkdir(directory_path)
    else:
        os.mkdir(directory_path)

class Dataset(Dataset):
    def __init__(self, data_path, df):
        
        self.df_patients = df
        self.data_dirs = [os.path.join(data_path, self.df_patients['Patient_ID'][x], 'data.pkl') for x in range(len(self.df_patients))]        
        self.patient_IDs = [self.df_patients['Patient_ID'][x] for x in range(len(self.df_patients))]            
       
    def __getitem__(self, index):
        
        data_path = self.data_dirs[index]
        
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
            
        seq_data = data['seqs']
        mask_data = data['masks']
        es_ed_idxs = data['es_ed_idxs']
        
        seq_data = torch.tensor(seq_data).float()
        seq_data = seq_data.unsqueeze(1)
        mask_data = torch.tensor(mask_data).float()
        
        es_ed_idxs = torch.tensor(es_ed_idxs)

        patient_id = self.df_patients['Patient_ID'][index]

        return (seq_data, mask_data, es_ed_idxs, patient_id)

    def __len__(self):
        return len(self.df_patients)
    
    
if __name__ == "__main__":

    ### Parameters ============================================================
    args = parse_args()
    
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print('#############################')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    df_val = pd.read_csv("./Testing_patients_main.csv")
    
    # args.data_path = "../Data Preprocessed"
    # args.model_weights_path = "./model_weights/model_weights.pth"
    
    create_directory(args.saving_results_path)
    
    dataset_val = Dataset(args.data_path, df_val)  
    dataloader_val = DataLoader(dataset_val, batch_size= args.batch_size, shuffle= False)
    
    n_batch_val = len(dataloader_val)      # Total number of batches for the validation phase
    
    model = MyModel(device, 0.03, args.seg_model_path).to(device)
    
    model.load_state_dict(torch.load(args.model_weights_path, map_location=device))
    
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_val):
            seq_in = data[0]
            seg_in = data[1]
            es_ed_idx = data[2]
            patient_id = data[3]
            
            seq_in = seq_in.to(device)
            seg_in = seg_in.to(device)

            print(es_ed_idx[0])
            
            seg_outs, dis_out, seg_refs, seq_outs, masks_outs, seq_gens, unc_dis_out  = model.inference_forward(seq_in, seg_in, es_ed_idx[0])


            batch_size = seq_in.shape[0]

            # 🔁 Save results PER PATIENT
            for b in range(batch_size):
                
                result = {'Masks_def': masks_outs[b],
                          'Disps': dis_out[b], 
                          'Segs_refs': seg_refs[b],
                          'Segs_def': seg_outs[b],
                          'ES_ED_Idxs':es_ed_idx[b].cpu().numpy(), 
                          'Segs_in': seg_in[b].cpu().numpy(),
                          'Seqs': seq_in[b].cpu().numpy(), 
                          'Seqs_def': seq_outs[b], 
                          'seq_gens':seq_gens[b], 
                          'unc_dis_out': unc_dis_out[b]}
                
                result_filepath = os.path.join(args.saving_results_path, f"{patient_id[b]}_results.pkl")
                
                with open(result_filepath, 'wb') as file:
                    pickle.dump(result, file)


    print('..........finished')
                
                      

            
