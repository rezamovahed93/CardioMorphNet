# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:30:45 2024

@author: reza
"""
import numpy as np
import os
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from utils.parameter_parser import parse_args_train as parse_args
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
    
    df_train = pd.read_csv("./Training_patients.csv")
    df_val = pd.read_csv("./Validation_patients.csv")
    # df_train = []
    # df_val = []
    # args.batch_size = 1 
    
    # args.data_path = "ACDC_data_preprocessed"
    
    dataset_train = Dataset(args.data_path, df_train, augment=True, rot_p=0.5)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    
    dataset_val = Dataset(args.data_path, df_val, augment=False)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)

    n_batch_train = len(dataloader_train)  # Total number of batches for the train phase
    n_batch_val = len(dataloader_val)      # Total number of batches for the validation phase
    
    model = MyModel(device, args.coeff_smoothness, args.seg_model_path).to(device)
    model.load_state_dict(torch.load(args.model_weights_path, map_location=device))
    
    plots_folder_name = f'plots_seg_freezed_{args.coeff_smoothness: .1e}'
    freez_weights(model.seg_posterior.model)
    model_weights_folder_name = f"model_weights_seg_freezed_{args.coeff_smoothness: .1e}"
    model_weights_name = f"model_weights_seg_freezed_{args.coeff_smoothness: .1e}.pth"
    
    # if args.seg_freez:
        # plots_folder_name = f'plots_seg_freezed_{args.coeff_smoothness: .1e}'
        # freez_weights(model.seg_posterior)
        # model_weights_folder_name = f"model_weights_seg_freezed_{args.coeff_smoothness: .1e}"
        # model_weights_name = f"model_weights_seg_freezed_{args.coeff_smoothness: .1e}.pth"
    # else:
    #     plots_folder_name = f'plots_seg_unfreezed_{args.coeff_smoothness: .1e}'
    #     model_weights_folder_name = f"model_weights_seg_unfreezed_{args.coeff_smoothness: .1e}"
    #     model_weights_name = f"model_weights_seg_unfreezed_{args.coeff_smoothness: .1e}.pth"
        
    #     freez_weights(model.seg_posterior)

    #     for param in model.seg_posterior.decoder.conv3d1.parameters():
    #         param.requires_grad = True
        
    #     for param in model.seg_posterior.decoder.conv3d2.parameters():
    #         param.requires_grad = True
        
    #     for param in model.seg_posterior.decoder.conv3d3.parameters():
    #         param.requires_grad = True
        
    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    writer = SummaryWriter()
    
    create_directory(plots_folder_name)
    create_directory(model_weights_folder_name)
    
    train_img_sim_loss = []
    train_klz_loss = []
    train_kld_loss = []
    train_smooth_loss = []
    train_seg_loss_1 = []
    train_seg_loss_2 = []
    train_total_loss = []
    
    val_img_sim_loss = []
    val_klz_loss = []
    val_kld_loss = []
    val_smooth_loss = []
    val_seg_loss_1 = []
    val_seg_loss_2 = []
    val_total_loss = []
    
    ### Training the model ===================================================
    best_metric = np.inf
    val_interval = 1 
    val_iter = 0
    patience = args.patience # Number of epochs to wait if no improvement is observed
    min_delta = args.min_delta  # Minimum change in the monitored quantity to qualify as an improvement
    epochs_no_improve  = 0 
    for epoch in range(args.epoch_size):
        img_sim_loss_tr = 0 
        klz_loss_tr = 0 
        kld_loss_tr = 0 
        smooth_loss_tr = 0 
        seg_loss_1_tr = 0
        seg_loss_2_tr = 0 
        total_loss_tr = 0
        
        model.train()
        for batch_idx, data in enumerate(dataloader_train):
            seq_in = data[0]
            seg_in = data[1]
            es_ed_idx = data[2][0]
            
            optimizer.zero_grad()
                        
            seg_outs, dis_out, img_sim_loss, klz_loss, kld_loss, smooth_loss, seg_loss_1, seg_loss_2, total_loss = model(seq_in, seg_in, es_ed_idx)            
            
            total_loss.backward()
            optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)       # Clip gradients to prevent exploding gradients
            
            # Extract loss values from tensors
            img_sim_loss_tr = img_sim_loss_tr + img_sim_loss.item()
            klz_loss_tr = klz_loss_tr + klz_loss.item()
            kld_loss_tr = kld_loss_tr + kld_loss.item()
            smooth_loss_tr = smooth_loss_tr + smooth_loss.item()
            seg_loss_1_tr = seg_loss_1_tr + seg_loss_1.item()
            seg_loss_2_tr = seg_loss_2_tr + seg_loss_2.item()
            total_loss_tr = total_loss_tr + total_loss.item()
            
            # Print training progress every 5 batches
            if (batch_idx) % args.batch_size and (args.verbose):
                print(f"Epoch: {epoch} [{batch_idx}/{n_batch_train} ({batch_idx/n_batch_train*100.0 :.0f}%)]"
                      f"\t Image Similarity Training Loss: {img_sim_loss.item() :.5f}"
                      f"\t KL(z) Training loss: {klz_loss.item() :.5f}"
                      f"\t Smoothness Training loss: {smooth_loss.item() :.5f}"
                      f"\t KL(d) Training loss: {kld_loss.item() :.5f}"
                      f"\t Segmentation Training loss (Unsupervised): {seg_loss_1.item() :.5f}"
                      f"\t Segmentation Training loss (Supervised): {seg_loss_2.item() :.5f}"
                      f"\t Total Training loss: {total_loss.item() :.5f}")
        
        img_sim_loss_tr = img_sim_loss_tr/n_batch_train
        klz_loss_tr = klz_loss_tr/n_batch_train
        kld_loss_tr = kld_loss_tr/n_batch_train
        smooth_loss_tr = smooth_loss_tr/n_batch_train
        seg_loss_1_tr = seg_loss_1_tr/n_batch_train
        seg_loss_2_tr = seg_loss_2_tr/n_batch_train
        total_loss_tr = total_loss_tr/n_batch_train 
        
        train_img_sim_loss.append(img_sim_loss_tr)
        train_klz_loss.append(klz_loss_tr)
        train_kld_loss.append(kld_loss_tr)
        train_smooth_loss.append(smooth_loss_tr)
        train_seg_loss_1.append(seg_loss_1_tr)
        train_seg_loss_2.append(seg_loss_2_tr)
        train_total_loss.append(total_loss_tr)
        
        print('################################################')
        print(f"Epoch: {epoch} is completed for training"
              f"\t Image Similarity Training Loss: {img_sim_loss_tr :.5f}"
              f"\t KL(z) Training loss: {klz_loss_tr :.5f}"
              f"\t Smoothness Training loss: {smooth_loss_tr :.5f}"
              f"\t KL(d) Training loss: {kld_loss_tr :.5f}"
              f"\t Segmentation Training loss (Unsupervised): {seg_loss_1_tr :.5f}"
              f"\t Segmentation Training loss (Supervised): {seg_loss_2_tr :.5f}"
              f"\t Total Training loss: {total_loss_tr :.5f}")
        print('################################################')
        
        # Log loss values for training using tensorboard writer
        writer.add_scalar("train_img_sim_loss", img_sim_loss_tr, epoch)        # Image similarity loss
        writer.add_scalar("train_klz_loss", klz_loss_tr, epoch)        # Log KL divergence loss z (latent spase)
        writer.add_scalar("train_seg_loss_1", seg_loss_1_tr, epoch)        # Uupervised Segmentation loss
        writer.add_scalar("train_seg_loss_2", seg_loss_2_tr, epoch)        # Supervised Segmentation loss
        writer.add_scalar("train_smooth_loss", smooth_loss_tr, epoch)  # Log smoothness loss
        writer.add_scalar("train_kld_loss", kld_loss_tr, epoch)        # Log KL divergence loss d (displacement)
        writer.add_scalar("train_total_loss", total_loss_tr, epoch)    # Log total loss

        if (epoch + 1) % val_interval == 0:
            img_sim_loss_val = 0 
            klz_loss_val = 0 
            kld_loss_val = 0 
            smooth_loss_val = 0 
            seg_loss_1_val = 0
            seg_loss_2_val = 0 
            total_loss_val = 0
            
            model.eval() 
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader_val):
                    seq_in = data[0]
                    seg_in = data[1]
                    es_ed_idx = data[2][0]
                    
                    seg_outs, dis_out, img_sim_loss, klz_loss, kld_loss, smooth_loss, seg_loss_1, seg_loss_2, total_loss = model(seq_in, seg_in, es_ed_idx)            
                    
                    img_sim_loss_val = img_sim_loss_val + img_sim_loss.item()
                    klz_loss_val = klz_loss_val + klz_loss.item()
                    kld_loss_val = kld_loss_val + kld_loss.item()
                    seg_loss_1_val = seg_loss_1_val + seg_loss_1.item()
                    seg_loss_2_val = seg_loss_2_val + seg_loss_2.item()
                    smooth_loss_val = smooth_loss_val + smooth_loss.item()
                    total_loss_val = total_loss_val + total_loss.item()
                    
                    if (batch_idx) % args.batch_size and (args.verbose):
                        print(f"Epoch: {epoch} [{batch_idx}/{n_batch_val} ({batch_idx/n_batch_val*100.0 :.0f}%)]"
                              f"\t Image Similarity Validation Loss: {img_sim_loss.item() :.5f}"
                              f"\t KL(z) Validation loss: {klz_loss.item() :.5f}"
                              f"\t Smoothness Validation loss: {smooth_loss.item() :.5f}"
                              f"\t Segmentation Validation loss (Unsupervised): {seg_loss_1.item() :.5f}"
                              f"\t Segmentation Validation loss (Supervised): {seg_loss_2.item() :.5f}"
                              f"\t KL(d) Validation loss: {kld_loss.item() :.5f}"
                              f"\t Total Validation loss: {total_loss.item() :.5f}")
                        
            img_sim_loss_val = img_sim_loss_val/n_batch_val
            klz_loss_val = klz_loss_val/n_batch_val
            kld_loss_val = kld_loss_val/n_batch_val
            smooth_loss_val = smooth_loss_val/n_batch_val
            seg_loss_1_val = seg_loss_1_val/n_batch_val
            seg_loss_2_val = seg_loss_2_val/n_batch_val
            total_loss_val = total_loss_val/n_batch_val
            
            scheduler.step(total_loss_val)
                
            val_img_sim_loss.append(img_sim_loss_val)
            val_klz_loss.append(klz_loss_val)
            val_kld_loss.append(kld_loss_val)
            val_smooth_loss.append(smooth_loss_val)
            val_seg_loss_1.append(seg_loss_1_val)
            val_seg_loss_2.append(seg_loss_2_val)
            val_total_loss.append(total_loss_val)
            
            print('################################################')
            print(f"Epoch: {epoch} is completed for validation"
                  f"\t Image Similarity Validation Loss: {img_sim_loss_val :.5f}"
                  f"\t KL(z) Validation loss: {klz_loss_tr :.5f}"
                  f"\t Smoothness Validation loss: {smooth_loss_val :.5f}"
                  f"\t Segmentation Validation loss (Unsupervised): {seg_loss_1_val :.5f}"
                  f"\t Segmentation Validation loss (Supervised): {seg_loss_2_val :.5f}"
                  f"\t KL(d) Validation loss: {kld_loss_val :.5f}"
                  f"\t Total Validation loss: {total_loss_val :.5f}")
            print('################################################')
                    
            writer.add_scalar("val_img_sim_loss", img_sim_loss_val, val_iter)        
            writer.add_scalar("val_klz_loss", klz_loss_val, val_iter)        
            writer.add_scalar("val_smooth_loss", smooth_loss_val, val_iter)
            writer.add_scalar("val_seg_loss_1", seg_loss_1_val, val_iter)
            writer.add_scalar("val_seg_loss_2", seg_loss_2_val, val_iter)       
            writer.add_scalar("val_kld_loss", kld_loss_val, val_iter)        
            writer.add_scalar("val_total_loss", total_loss_val, val_iter)
            val_iter = val_iter + 1 
                
            if total_loss_val < best_metric - min_delta:
                best_metric = total_loss_val
                torch.save(model.state_dict(), os.path.join(os.getcwd(), model_weights_folder_name, model_weights_name))
                print('################################################')
                print('The model is saved in epoch: {} with the val loss of {:.5f}'.format(epoch, best_metric))
                print('################################################')
                epochs_no_improve = 0 
            else: 
                epochs_no_improve += 1
            
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch+1} as there has been no improvement in validation loss.')
                break
                
    plot_fun(np.asarray(train_img_sim_loss), x_axis_label='Epoch', y_axis_label='MSE Image Similarity', title='Training Image Similarity Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_img_sim_loss.png'))
    plot_fun(np.asarray(train_klz_loss), x_axis_label='Epoch', y_axis_label='KL(z) loss', title='Training KL(z) Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_klz_loss.png'))
    plot_fun(np.asarray(train_kld_loss), x_axis_label='Epoch', y_axis_label='KL(d) loss', title='Training KL(d) Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_kld_loss.png'))   
    plot_fun(np.asarray(train_smooth_loss), x_axis_label='Epoch', y_axis_label='Smoothness loss', title='Training Smoothness Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_smooth_loss.png'))
    plot_fun(np.asarray(train_seg_loss_1), x_axis_label='Epoch', y_axis_label='Unsupervised segmentation loss', title='Training Unsupervised Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_seg_loss_1.png'))
    plot_fun(np.asarray(train_seg_loss_2), x_axis_label='Epoch', y_axis_label='Supervised segmentation loss', title='Training Supervised Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_seg_loss_2.png'))
    plot_fun(np.asarray(train_seg_loss_2) + np.asarray(train_seg_loss_1), x_axis_label='Epoch', y_axis_label='Overall segmentation loss', title='Training Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_seg_loss.png'))
    plot_fun(np.asarray(train_total_loss), x_axis_label='Epoch', y_axis_label='Total loss', title='Training Total Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'train_total_loss.png'))            
         
    plot_fun(np.asarray(val_img_sim_loss), x_axis_label='Epoch', y_axis_label='MSE Image Similarity', title='Validation Image Similarity Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_img_sim_loss.png'))
    plot_fun(np.asarray(val_klz_loss), x_axis_label='Epoch', y_axis_label='KL(z) loss', title='Validation KL(z) Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_klz_loss.png'))
    plot_fun(np.asarray(val_kld_loss), x_axis_label='Epoch', y_axis_label='KL(d) loss', title='Validation KL(d) Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_kld_loss.png'))   
    plot_fun(np.asarray(val_smooth_loss), x_axis_label='Epoch', y_axis_label='Smoothness loss', title='Validation Smoothness Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_smooth_loss.png'))
    plot_fun(np.asarray(val_seg_loss_1), x_axis_label='Epoch', y_axis_label='Unsupervised segmentation loss', title='Validation Unsupervised Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_seg_loss_1.png'))
    plot_fun(np.asarray(val_seg_loss_2), x_axis_label='Epoch', y_axis_label='Supervised segmentation loss', title='Validation Supervised Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_seg_loss_2.png'))
    plot_fun(np.asarray(val_seg_loss_1) + np.asarray(val_seg_loss_2), x_axis_label='Epoch', y_axis_label='Overall segmentation loss', title='Validation Segmentation Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_seg_loss.png'))
    plot_fun(np.asarray(val_total_loss), x_axis_label='Epoch', y_axis_label='Total loss', title='Validation Total Loss', filename=os.path.join(os.getcwd(),plots_folder_name,'val_total_loss.png'))  
    
    Results = {'MSE_tr':train_img_sim_loss, 
                'loss_KL(Z)_tr':train_klz_loss,
                'loss_KL(D)_tr':train_kld_loss,
                'loss_smooth_tr':train_smooth_loss,
                'loss_seg_1_tr':train_seg_loss_1,
                'loss_seg_2_tr':train_seg_loss_2,
                'loss_tot_tr':train_total_loss,
               
                'MSE_val':val_img_sim_loss, 
            'loss_KL(Z)_val':val_klz_loss,
            'loss_KL(D)_val':val_kld_loss,
            'loss_smooth_val':val_smooth_loss,
            'loss_seg_1_val':val_seg_loss_1,
            'loss_seg_2_val':val_seg_loss_2,
            'loss_tot_val':val_total_loss,
                }
    
    with open('training_results.pkl', 'wb') as file:
        pickle.dump(Results, file)
        
    writer.close()
    print('finished....')
            
            
    