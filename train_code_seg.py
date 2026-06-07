#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:39:24 2024

@author: rezeakbarimovahed
"""

import numpy as np
import nibabel as nib
import os
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from utils.parameter_parser import parse_args_train as parse_args
from utils.preprocess_pack import preprocess_fun
import shutil
from utils.model import MyModel
from torch.utils.tensorboard import SummaryWriter
from utils.plot_lib import plot_fun
import pickle
import warnings 

warnings.filterwarnings("ignore")

import random


def freez_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        os.mkdir(directory_path)
    else:
        os.mkdir(directory_path)


def random_right_angle_rotation(x, hw_dims=(-3, -2), p=0.5):
    """
    Applies a random rotation of {0, 90, 180, 270} degrees using torch.rot90.
    - x: tensor with at least 2 spatial dims
    - hw_dims: which dims correspond to (H, W)
    - p: probability to apply a non-zero rotation
    Returns: rotated tensor, k (0..3)
    """
    if random.random() > p:
        return x, 0

    k = random.choice([1, 2, 3])  # 90, 180, 270
    return torch.rot90(x, k=k, dims=hw_dims), k

class Dataset(Dataset):
    def __init__(self, data_path, df, augment=False, rot_p=0.5):
        
        self.df_patients = df
        self.data_dirs = [os.path.join(data_path, self.df_patients['Patient_ID'][x], 'data.pkl') for x in range(len(self.df_patients))]        
        self.patient_IDs = [self.df_patients['Patient_ID'][x] for x in range(len(self.df_patients))]  

        self.augment = augment
        self.rot_p = rot_p
        
       
    def __getitem__(self, index):
        
        data_path = self.data_dirs[index]
        
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
                    
        seq_data = torch.tensor(data['seqs']).float()    # [T, D, H, W] (assumed)
        mask_data = torch.tensor(data['masks']).float()          # keep as int/long if categorical
        es_ed_idxs = torch.tensor(data['es_ed_idxs'])

        # add channel: [T, 1, D, H, W]
        seq_data = seq_data.unsqueeze(1)

        # Make mask shape consistent: [T, 1, D, H, W] or [T, D, H, W] is fine
        # Here I’ll rotate it as [T, D, H, W] (no channel needed)
        # Rotate only if training
        if self.augment:
            # rotate images in (H,W) plane
            seq_data, k = random_right_angle_rotation(seq_data, hw_dims=(-3, -2), p=self.rot_p)

            # rotate masks with same k in same plane
            # mask_data is [T, D, H, W] => H,W are (-2,-1)
            if k != 0:
                mask_data = torch.rot90(mask_data, k=k, dims=(-3, -2))

        return (seq_data, mask_data, es_ed_idxs)

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
    
    # args.data_path = "ACDC_data_preprocessed"
    
    df_train = pd.read_csv("./Training_patients_init.csv")
    df_val = pd.read_csv("./Validation_patients_init.csv")
    
    dataset_train = Dataset(args.data_path, df_train, augment=True, rot_p=0.5)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    
    dataset_val = Dataset(args.data_path, df_val, augment=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    n_batch_train = len(dataloader_train)  # Total number of batches for the train phase
    n_batch_val = len(dataloader_val)      # Total number of batches for the validation phase
    
    model = MyModel(device, args.coeff_smoothness, args.seg_model_path).to(device)
    # model.load_state_dict(torch.load(args.model_weights_path, map_location=device))
    
    freez_weights(model.net_img)
    freez_weights(model.net_dt_posterior)
    freez_weights(model.net_img_est)
    freez_weights(model.net_phi_zt)
    freez_weights(model.net_infer_z)
    freez_weights(model.net_z_prior)
    freez_weights(model.ConvLSTM)        
        
    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    writer = SummaryWriter()
    
    create_directory('plots_seg_only')
    create_directory('model_weights_seg_only')
    
    train_seg_loss = []

    val_seg_loss = []
    
    ### Training the model ===================================================
    best_metric = np.inf
    val_interval = 1 
    val_iter = 0
    patience = args.patience # Number of epochs to wait if no improvement is observed
    min_delta = args.min_delta  # Minimum change in the monitored quantity to qualify as an improvement
    epochs_no_improve  = 0 
    
    for epoch in range(args.epoch_size):

        total_loss_tr = 0
        
        model.train()
        for batch_idx, data in enumerate(dataloader_train):
            seq_in = data[0]
            seg_in = data[1]
            es_ed_idx = data[2][0]
            
            
            optimizer.zero_grad()
            
            total_loss, _ = model.forward_seg(seq_in, seg_in, es_ed_idx)
                                    
            total_loss.backward()
            optimizer.step()
            
            
            # Extract loss values from tensors
            total_loss_tr = total_loss_tr + total_loss.item()
            
            # Print training progress every 5 batches
            if (batch_idx) % args.batch_size and (args.verbose):
                print(f"Epoch: {epoch} [{batch_idx}/{n_batch_train} ({batch_idx/n_batch_train*100.0 :.0f}%)]"
                      f"\t Total Segmentation Training loss: {total_loss.item() :.5f}")
        
        total_loss_tr = total_loss_tr/n_batch_train 

        train_seg_loss.append(total_loss_tr)
        
        print('################################################')
        print(f"Epoch: {epoch} is completed for training"
              f"\t Total Training loss: {total_loss_tr :.5f}")
        print('################################################')
        
        # Log loss values for training using tensorboard writer
        writer.add_scalar("train_total_loss", total_loss_tr, epoch)    # Log total loss

        if (epoch + 1) % val_interval == 0:
            total_loss_val = 0
            model.eval() 
            
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader_val):
                    seq_in = data[0]
                    seg_in = data[1]
                    es_ed_idx = data[2][0]
                    
                    seq_in = seq_in.to(device)
                    seg_in = seg_in.to(device)
                    
                    total_loss, _ = model.forward_seg(seq_in, seg_in, es_ed_idx)
                                        
                    total_loss_val = total_loss_val + total_loss.item()
                    
                    if (batch_idx) % args.batch_size and (args.verbose):
                        print(f"Epoch: {epoch} [{batch_idx}/{n_batch_val} ({batch_idx/n_batch_val*100.0 :.0f}%)]"
                              f"\t Total Segmentation Validation loss: {total_loss.item() :.5f}")
                        
            total_loss_val = total_loss_val/n_batch_val
            
            scheduler.step(total_loss_val)
                
            val_seg_loss.append(total_loss_val)
            
            print('################################################')
            print(f"Epoch: {epoch} is completed for validation"
                  f"\t Total Validation loss: {total_loss_val :.5f}")
            print('################################################')
                          
            writer.add_scalar("val_total_loss", total_loss_val, val_iter)
            val_iter = val_iter + 1 
                
            if total_loss_val < best_metric - min_delta:
                best_metric = total_loss_val
                torch.save(model.state_dict(), os.path.join(os.getcwd(),'model_weights_seg_only', 'model_weights_seg_only.pth'))
                print('################################################')
                print('The model is saved in epoch: {} with the val loss of {:.5f}'.format(epoch, best_metric))
                print('################################################')
                epochs_no_improve = 0 
            else: 
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch+1} as there has been no improvement in validation loss.')
                break
                
    plot_fun(np.asarray(train_seg_loss), x_axis_label='Epoch', y_axis_label='Dice Score', title='Training Dice Score Loss', filename=os.path.join(os.getcwd(),'plots_seg_only','train_seg_loss.png'))
         
    plot_fun(np.asarray(val_seg_loss), x_axis_label='Epoch', y_axis_label='Dice Score', title='Validation Dice Score Loss', filename=os.path.join(os.getcwd(),'plots_seg_only','val_seg_loss.png'))
    
    Results = {'loss_tot_tr':train_seg_loss,
            'loss_tot_val':val_seg_loss,
                }
    
    with open('training_results_seg_only.pkl', 'wb') as file:
        pickle.dump(Results, file)
        
    writer.close()
    print('finished....')
            
            
    
