# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:13:22 2024

@author: reza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils.Conv3d_lstm import ConvLSTM
from utils.reparametr_funcs import zt_reparameterize_multidim, covariance_d, dt_reparameterize, mask_reparameterize
from utils.loss_pack import loss_fn
from monai.networks.blocks.warp import DVF2DDF, Warp

import os 
from monai.bundle import ConfigParser, download
from monai.transforms import AsDiscrete


class Net_img_enc(nn.Module):
    def __init__(self, in_channels, device, dropout_rate):
        super(Net_img_enc, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv3d_img_1 = nn.Conv3d(in_channels, 16, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
        self.conv3d_img_2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
        self.conv3d_img_3 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
        self.conv3d_img_4 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
        self.conv3d_img_5 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        # self.conv3d_img_6 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
                
    def forward(self,x):
        skips = []
        x = self.dropout(self.conv3d_img_1(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d_img_2(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d_img_3(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d_img_4(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d_img_5(x)) 
        x = self.LRelu(x)
        return x, skips
    
    
class Seg_net(nn.Module): 
    def __init__(self, seg_path, device):
        super(Seg_net, self).__init__()

        self.model_config = ConfigParser()
        self.model_config.read_config(os.path.join(seg_path, "configs", "inference.json"))
        self.model = self.model_config.get_parsed_content("network").to(device)
        self.model.load_state_dict(torch.load(os.path.join(seg_path, "models", "model.pth"), map_location=device))
        
    def eval_on(self, cond): 
        if cond:
            self.model.eval()
        
    def get_output(self, x):
        # x = x.squeeze(0)
        # x = x.permute(0, 4, 1, 2, 3)
        # print(x.shape)
        # B, T, C, H, W, D = x.shape
        
        # x = x.reshape(B * T, C, H, W, D)
        
        x = self.model(x)
        # x = x.reshape(B, T, C, H, W, D)
        # x = x.permute(0,2,3,4,1)
        # x = x.unsqueeze(0)
        return x
    
    def get_prob_output(self,x):
        return F.softmax(self.get_output(x), dim=1)
    
    def get_class_output(self,x): 
        transform = AsDiscrete(threshold=0.5)
        return transform(F.softmax(self.get_output(x), dim=1))
    
    def seg_inference_score(self, img): 

        # print(img.shape)
        img = img.permute(0, 4, 1, 2, 3)
        B, D, C, H, W = img.shape
        # print(img.shape)
        # print('==========')

        img = img.reshape(B * D, C, H, W)
        # print(img.shape)

        
        img_resized = F.interpolate(img, size=(256, 256), 
                                  mode='bilinear', align_corners=False)

        # print(img_resized.shape)
        # print('==========')

        
        # img_resized = img_resized.permute(1, 2, 3, 0).unsqueeze(0)

        seg_resized = self.get_output(img_resized)
        
        seg = F.interpolate(seg_resized, size=(128, 128), 
                                  mode='bilinear', align_corners=False)

        # print('seg: ', seg.shape)
        # seg = seg.permute(1, 2, 3, 0)
        # print('seg: ', seg.shape)
        # seg = seg.reshape(B, 4, H, W, D)
        # print('seg: ', seg.shape)
        
        return seg 

    def seg_inference_prob(self, img): 

        img = img.permute(0, 4, 1, 2, 3)
        B, D, C, H, W = img.shape
        img = img.reshape(B * D, C, H, W)


        img_resized = F.interpolate(img, size=(256, 256), 
                                  mode='bilinear', align_corners=False)

        seg_resized = self.get_output(img_resized)

        
        # img_resized = img_resized.permute(1, 2, 3, 0).unsqueeze(0)

        # seg_resized = self.get_prob_output(img_resized)

        # seg_resized = self.get_output(img_resized)
        
        seg = F.interpolate(seg_resized, size=(128, 128), 
                                  mode='bilinear', align_corners=False)

        # seg = seg.permute(1, 2, 3, 0)

        # seg = seg.reshape(B, 4, H, W, D)
        
        seg = F.softmax(seg, dim=1)
        
        return seg    
    
    def seg_inference_score_fw(self, img):
        
        img_resized = F.interpolate(
            img.permute(0, 1, 4, 2, 3),
            size=(16, 256, 256),  # (D, H, W)
            mode='trilinear',
            align_corners=False
        )
        
        img_resized = img_resized.permute(0, 1, 3, 4, 2)
        
        seg_resized = torch.empty((img_resized.shape[0], 
                                   4, 
                                   img_resized.shape[2], 
                                   img_resized.shape[3], 
                                   img_resized.shape[4]), device=img.device)
        
        for batch_idx in range(img_resized.shape[0]):
            seg_resized[batch_idx] = self.get_output(img_resized[batch_idx].permute(3, 0, 1, 2)).permute(1,2,3,0).unsqueeze(0)
            
        seg = F.interpolate(
            seg_resized.permute(0, 1, 4, 2, 3),
            size=(16, 128, 128),  # (D, H, W)
            mode='trilinear',
            align_corners=False
        )
        
        seg = seg.permute(0, 1, 3, 4, 2)
        
        return seg 
    
    def seg_inference_prob_fw(self, img):
        
        seg = self.seg_inference_score_fw(img)
        
        seg = F.softmax(seg, dim=1)
        
        return seg

    
    
class Net_phi_zt(nn.Module): 
    def __init__(self, device, dropout_rate):
        super(Net_phi_zt, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        
        self.conv3fzt_dt = nn.Conv3d(1, 32, kernel_size=(3,3,3), 
                                        stride=(1,1,1), padding=1).to(device)
        
    def forward(self, z_t):
        z_t = self.dropout(self.conv3fzt_dt(z_t))
        z_t = self.LRelu(z_t)
        return z_t

class Net_dt_posterior(nn.Module):
    def __init__(self, device, dropout_rate):
        super(Net_dt_posterior, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv3d1_dt_pos = nn.Conv3d(2, 16, kernel_size=(3,3,3), 
                                        stride=(1,1,1), padding=1).to(device)
        
        self.conv3d2_dt_pos = nn.Conv3d(16, 32, kernel_size=(3,3,3), 
                                        stride=(2, 2, 1), padding=1).to(device)
        
        self.conv3d3_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
                                        stride=(2, 2, 1), padding=1).to(device)

        self.conv3d4_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
                                        stride=(2, 2, 1), padding=1).to(device)
        
        self.conv3d5_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
                                        stride=(2, 2, 1), padding=1).to(device)
        
        self.conv3d6_dt_pos = nn.Conv3d(64, 32, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)
        
        self.convtrans1_dt_pos = nn.ConvTranspose3d(64, 32, 
                                                      kernel_size=3, 
                                                      stride=(2,2,1), 
                                                      padding=(1,1,1), 
                                                      output_padding=(1,1,0)).to(device) 
        
        self.convtrans2_dt_pos = nn.ConvTranspose3d(64, 32, 
                                                      kernel_size=3, 
                                                      stride=(2,2,1), 
                                                      padding=(1,1,1), 
                                                      output_padding=(1,1,0)).to(device)
        
        self.convtrans3_dt_pos = nn.ConvTranspose3d(64, 32, 
                                                      kernel_size=3, 
                                                      stride=(2,2,1), 
                                                      padding=(1,1,1), 
                                                      output_padding=(1,1,0)).to(device)  
        
        self.convtrans4_dt_pos = nn.ConvTranspose3d(64, 32, 
                                                      kernel_size=3, 
                                                      stride=(2,2,1), 
                                                      padding=(1,1,1), 
                                                      output_padding=(1,1,0)).to(device)
        
        self.convtrans5_dt_pos = nn.ConvTranspose3d(48, 32, kernel_size=3, stride=1, padding=1)
        
        
        self.conv3d7_dt_pos = nn.Conv3d(32, 16, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)
        
        self.conv3d8_dt_pos = nn.Conv3d(16, 16, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)
        
        self.conv3d8_dt_pos = nn.Conv3d(16, 3, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)
        
        self.conv_mu_dt = nn.Conv3d(3, 3, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)
        
        self.conv_logvar_dt = nn.Conv3d(3, 1, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device) 
        
        self.conv_v_dt =  nn.Conv3d(3, 3, kernel_size=(3,3,3), 
                                        stride=(1, 1, 1), padding=1).to(device)  
        
    
    def forward(self, x, fzt):
        skips = []
        x = self.dropout(self.conv3d1_dt_pos(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d2_dt_pos(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d3_dt_pos(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d4_dt_pos(x))
        x = self.LRelu(x)
        skips.append(x)
        x = self.dropout(self.conv3d5_dt_pos(x))
        x = self.LRelu(x)
        skips.append(x)
        x = torch.cat([x,fzt], 1)
        x = self.dropout(self.conv3d6_dt_pos(x))
        x = self.LRelu(x)
        x = torch.cat([skips[-1],x], 1)
        x = self.dropout(self.convtrans1_dt_pos(x))
        x = self.LRelu(x)
        x = torch.cat([skips[-2],x], 1)
        x = self.dropout(self.convtrans2_dt_pos(x))
        x = self.LRelu(x)
        x = torch.cat([skips[-3],x], 1)
        x = self.dropout(self.convtrans3_dt_pos(x))
        x = self.LRelu(x)
        x = torch.cat([skips[-4],x], 1)
        x = self.dropout(self.convtrans4_dt_pos(x))
        x = self.LRelu(x)
        x = torch.cat([skips[-5],x], 1)
        x = self.dropout(self.convtrans5_dt_pos(x))
        x = self.LRelu(x)
        x = self.dropout(self.conv3d7_dt_pos(x))
        x = self.LRelu(x)
        x = self.dropout(self.conv3d8_dt_pos(x))
        d_mu = self.conv_mu_dt(x)
        d_logvar = self.conv_logvar_dt(x)
        d_log_v = self.conv_v_dt(x)
        return d_mu, d_logvar, d_log_v
    
class Net_VoxelMorph(nn.Module):
    def __init__(self, half_res):
        super(Net_VoxelMorph, self).__init__() 
        
        self.half_res = half_res
        self.diffeomorphic = False
        self.warp = Warp(mode="bilinear", padding_mode="zeros")
        
    def enable_diffeomorphic(self, integration_steps):
        self.diffeomorphic = True
        self.integration_steps = integration_steps
        self.dvf2ddf = DVF2DDF(num_steps=self.integration_steps, 
                               mode="bilinear", padding_mode="zeros")
        
    def forward(self, x, img):
        if self.half_res:
            x = F.interpolate(x, scale_factor=0.5, mode="trilinear", align_corners=True) * 2.0
        if self.diffeomorphic: 
            x = self.dvf2ddf(x) 
        if self.half_res:
            x = F.interpolate(x * 0.5, scale_factor=2.0, mode="trilinear", align_corners=True)
        return self.warp(img, x), x 


    
class Net_infer_z(nn.Module): 
    def __init__(self, device, dropout_rate, in_channels=64): 
        super(Net_infer_z, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv3d_decoder = nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        self.conv3d_inf_zt_1 = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        self.conv3d_inf_zt_2 = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
             
    def forward(self, x, h_recurrent):
        x = torch.cat([x,h_recurrent], 1) 
        x = self.dropout(self.conv3d_decoder(x))
        x = self.LRelu(x)
        z_mu = self.conv3d_inf_zt_1(x)
        z_logvar = self.conv3d_inf_zt_2(x)
        return z_mu, z_logvar 
    
class Net_img_est(nn.Module):
    def __init__(self, device, dropout_rate, in_channels=1):
        super(Net_img_est, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # self.convtrans1_img_est = nn.ConvTranspose3d(in_channels, 32, kernel_size=3, 
        #                                               stride=1, padding=1).to(device)
        
        self.convtrans2_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
                                                    padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
        self.convtrans3_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
                                                    padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
        self.convtrans4_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
                                                    padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
        self.convtrans5_img_est = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(2,2,1), 
                                                    padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
        self.conv3d_img_est = nn.Conv3d(16, 1, kernel_size=(3,3,3), 
                                        stride=(1,1,1), padding=1).to(device)
          
    def forward(self, x):
        # x = self.dropout(self.convtrans1_img_est(x))
        # x = self.LRelu(x)
        x = self.dropout(self.convtrans2_img_est(x))
        x = self.LRelu(x)
        x = self.dropout(self.convtrans3_img_est(x))
        x = self.LRelu(x)
        x = self.dropout(self.convtrans4_img_est(x))
        x = self.LRelu(x)
        x = self.dropout(self.convtrans5_img_est(x))
        x = self.LRelu(x)
        x = self.dropout(self.conv3d_img_est(x))
        x = self.ReLU(x)
        return x
    
class Net_z_prior(nn.Module):
    def __init__(self, device, dropout_rate, in_channels=32):
        super(Net_z_prior, self).__init__()
        
        self.LRelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv3d1_z_prior = nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        self.conv3d2_z_prior = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        
        self.conv3d3_z_prior = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        self.conv3d4_z_prior = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)    
        
    def forward(self, x):
        x = self.dropout(self.conv3d1_z_prior(x))
        x = self.LRelu(x)
        x = self.dropout(self.conv3d2_z_prior(x))
        x = self.LRelu(x)
        z_prior_mu = self.conv3d3_z_prior(x)
        z_prior_logvar = self.conv3d4_z_prior(x)
        return z_prior_mu, z_prior_logvar


class MyModel(nn.Module):
    def __init__(self, device, coef_smooth_loss, seg_path): 
        super(MyModel, self).__init__()
        
        self.device = device
        self.net_img = Net_img_enc(in_channels=1, device=self.device, dropout_rate=0.0)
        # self.seg_posterior = Seg_Net(in_channels=1, device=self.device, dropout_rate=0.0)
        self.seg_posterior = Seg_net(seg_path, device)
        
        self.net_dt_posterior = Net_dt_posterior(self.device, dropout_rate=0.0)
        self.net_voxelmorph = Net_VoxelMorph(half_res=False)
        self.net_phi_zt = Net_phi_zt(device=self.device, dropout_rate=0.0)
        
        
        self.net_infer_z = Net_infer_z(device=self.device, dropout_rate=0.0)
        self.net_img_est = Net_img_est(device=self.device, dropout_rate=0.0)
        self.net_z_prior = Net_z_prior(device=self.device, dropout_rate=0.0)
        # self.net_dt_prior = Net_dt_prior(device=self.device, dropout_rate=0.0)
        
        self.ConvLSTM = ConvLSTM(input_size= (8, 8),
                                  input_dim= 64,
                                  hidden_dim= [64, 32],
                                  kernel_size= [3, 3],
                                  num_layers= 2,
                                  device=self.device,
                                  droput_rates = [0.0, 0.0],
                                  batch_first=True,
                                  bias=True) 
        
        
        
        ### coefficients of loss term
        self.coef_klz_loss = 2e-4
        self.coef_smooth_loss = coef_smooth_loss
        self.coef_img_sim_loss = 0.01
        self.coef_kld_loss  = 1e-4
        self.coef_seg_loss_1 = 0.8
        self.coef_seg_loss_2 = 0.8
        
    def forward_seg(self, seq_in, seg_in, es_ed_idx):
        
        batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
        loss_fn = nn.CrossEntropyLoss()
        
        seg_outs = np.zeros((batch_size, 2, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        
        total_loss = 0.0 
        
        for idx, frame_idx in enumerate(es_ed_idx.squeeze(0)):
            img = seq_in[:, int(frame_idx)].to(self.device)
            mask = seg_in[:, int(frame_idx)].to(self.device)
            
            mask_per = mask.permute(0, 4, 1, 2, 3)
            
            seg = self.seg_posterior.seg_inference_score(img)
            
            loss_val = loss_fn(seg, mask_per.reshape(batch_size * d_img, 4, h_img, w_img))
            
            total_loss = total_loss + loss_val
            
            seg_reshaped = seg.permute(1,2,3,0)
            seg_reshaped = seg_reshaped.reshape(batch_size, 4, h_img, w_img, d_img)
            
            seg_outs[:, idx, :,:,:,:] = F.softmax(seg_reshaped, dim=1).cpu().detach().numpy()
                   
        return total_loss, seg_outs
        
        
    def forward(self, seq_in, seg_in, es_ed_idx):
        
        batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
        h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
        hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
                
        seg_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        dis_out = np.zeros((batch_size, frame_num, 3, h_img, w_img, d_img), dtype=np.float32)
        
        ### Initial value of loss terms
        img_sim_loss = 0.0
        klz_loss = 0.0
        kld_loss = 0.0 
        smooth_loss = 0.0
        seg_loss_1 = 0.0 
        seg_loss_2 = 0.0 
        total_loss = 0.0
        
        for t in range(frame_num+1):
                 
            idx_cur = t % frame_num           ### current index
            idx_pas = (t-1) % frame_num       ### past index

            seg_cri = True if idx_cur in es_ed_idx else False
            
            """
            seg_cri defines whether we are in the ED or ES frames or not 
            if seg_cri==True, it means we are in one of the ED or ES frames 
            if seg_cri==False, it indicates we are not in ED and ES frames 
            """
        
            Img_cur = seq_in[:, idx_cur].to(self.device)         ### current image
            Img_pas = seq_in[:, idx_pas].to(self.device)         ### past image
            
            ### z_prior
            z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
            
            ### Feature maps extraction for Img_cur   
            fit, _ = self.net_img(Img_cur)
            
            ### zt_inference and reparameterization
            z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
            z_t = zt_reparameterize_multidim(z_mu, z_logvar)
            fzt = self.net_phi_zt(z_t)
            
            ### Dt_inference
            d_mu, d_logvar, d_log_v = self.net_dt_posterior(torch.cat([Img_pas, Img_cur], dim=1), fzt)
            d_cov = covariance_d(d_logvar, d_log_v)
            d_t = dt_reparameterize(d_mu, d_cov)
            
            if t>0:
                if not(seg_cri):
                    Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_cur)
                    # Seg_deformed = stn_3d(d_t, Seg_cur, self.device)
                else:
                    Seg_deformed, deform_field = self.net_voxelmorph(d_t, mask_reparameterize(F.softmax(Seg_cur, dim=1)))
                    # Seg_deformed = stn_3d(d_t, mask_reparameterize(Seg_cur), self.device)
            
            Img_cur_est = self.net_img_est(fzt)
            
            input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
            h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
            Seg_cur = self.seg_posterior.seg_inference_score_fw(Img_cur)
            
            # Seg_cur = torch.empty((batch_size, 4, h_img, w_img, d_img), device=self.device)
            # for b_idx in range(batch_size): 
            #     Seg_cur[b_idx,:,:,:] = self.seg_posterior.seg_inference_score(Img_cur[b_idx, :, :, :].unsqueeze(0))
                
            # Seg_cur = self.seg_posterior.seg_inference_score(Img_cur)
            # Seg_cur = Seg_cur.permute(1,2,3,0)
            # Seg_cur = Seg_cur.reshape(batch_size, 4, h_img, w_img, d_img)
                
            if seg_cri:
                cond1 = idx_cur - es_ed_idx[0]
                cond2 = idx_cur - es_ed_idx[1]
                if cond1==0:
                    Seg_ref = seg_in[:,idx_cur].to(self.device)  
                elif cond2==0:
                    Seg_ref = seg_in[:,idx_cur].to(self.device)  
                    
            else: 
                Seg_ref = Seg_cur.clone()
                   
            if t>0:
                img_sim_loss_t, klz_loss_t, kld_loss_t, smooth_loss_t, seg_loss_1_t, seg_loss_2_t = loss_fn(Img_cur, Img_cur_est, 
                                                                              z_mu, z_logvar, z_prior_mu, z_prior_logvar, 
                                                                              d_cov, d_mu, deform_field, 
                                                                              seg_prior=Seg_deformed, seg_posterior=Seg_ref, 
                                                                              seg_cri = seg_cri, device = self.device)
                
                img_sim_loss = img_sim_loss + img_sim_loss_t 
                klz_loss = klz_loss + klz_loss_t
                kld_loss = kld_loss + kld_loss_t
                smooth_loss = smooth_loss + smooth_loss_t
                seg_loss_1 = seg_loss_1 + seg_loss_1_t
                seg_loss_2 = seg_loss_2 + seg_loss_2_t
                
                seg_outs[:, idx_cur, :,:,:,:] = Seg_deformed.cpu().detach().numpy()
                dis_out[:, idx_cur, :, : ,: ,:] = deform_field.cpu().detach().numpy()

        
        total_loss = self.coef_img_sim_loss*img_sim_loss + self.coef_smooth_loss*smooth_loss + self.coef_klz_loss*klz_loss + self.coef_kld_loss*kld_loss + self.coef_seg_loss_1*seg_loss_1 + self.coef_seg_loss_2*seg_loss_2 
        
        return seg_outs, dis_out, img_sim_loss, klz_loss, kld_loss, smooth_loss, seg_loss_1, seg_loss_2, total_loss
    
    def inference_forward(self, seq_in, seg_in, es_ed_idx):
                
        batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
        h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
        hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
                
        seg_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        masks_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        seq_outs = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)
        dis_out = np.zeros((batch_size, frame_num, 3, h_img, w_img, d_img), dtype=np.float32)
        seg_refs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        seq_gens = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)
        unc_dis_out = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)    
        
        
        for t in range(frame_num+1):
            
            idx_cur = t % frame_num           ### current index
            idx_pas = (t-1) % frame_num       ### past index

            seg_cri = True if idx_cur in es_ed_idx else False
        
            Img_cur = seq_in[:, idx_cur].to(self.device)           ### current image
            Img_pas = seq_in[:, idx_pas].to(self.device)           ### past image
            
            ### z_prior
            z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
            
            ### Feature maps extraction for Img_cur   
            fit, skips_ic = self.net_img(Img_cur)
            
            ### zt_inference and reparameterization
            z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
            z_t = zt_reparameterize_multidim(z_mu, z_logvar)
            fzt = self.net_phi_zt(z_t)
            
            ### Dt_inference
            d_mu, d_logvar, d_log_v = self.net_dt_posterior(torch.cat([Img_pas, Img_cur], dim=1), fzt)
            d_cov = covariance_d(d_logvar, d_log_v)
            d_t = dt_reparameterize(d_mu, d_cov)
            
            unc_map_d_t = 0.5 * torch.log(((2*torch.pi) ** 3) * d_cov.det())
            unc_map_d_t = unc_map_d_t.unsqueeze(1)
            
            if t>0:
                Img_deformed, deform_field = self.net_voxelmorph(d_t,Img_pas)
                Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_cur)
                Mask_deformed, deform_field = self.net_voxelmorph(d_t, Mask_ref)
                Mask_ref = Mask_deformed.clone()
                seq_outs[:, idx_cur, :,:,:,:] = Img_deformed.cpu().detach().numpy()
                seg_refs[:, idx_pas, :,:,:,:] = Seg_cur.cpu().detach().numpy()
                masks_outs[:, idx_cur, :,:,:,:] = Mask_deformed.cpu().detach().numpy()
                seg_outs[:, idx_cur, :,:,:,:] = Seg_deformed.cpu().detach().numpy()
                dis_out[:, idx_cur, :, : ,: ,:] = deform_field.cpu().detach().numpy()
                seq_gens[:, idx_cur, :,:,:,:] = Img_cur_est.cpu().detach().numpy()
                unc_dis_out[:,idx_cur, :, :, :, :] = unc_map_d_t.cpu().detach().numpy()
            
            Img_cur_est = self.net_img_est(fzt)
            
            input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
            h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
            # Seg_cur = torch.empty((batch_size, 4, h_img, w_img, d_img), device=self.device)
            # for b_idx in range(batch_size): 
            #     Seg_cur[b_idx,:,:,:] = self.seg_posterior.seg_inference_prob(Img_cur[b_idx, :, :, :].unsqueeze(0))
            
            # Seg_cur = self.seg_posterior.seg_inference_prob(Img_cur)
            # Seg_cur = Seg_cur.permute(1,2,3,0)
            # Seg_cur = Seg_cur.reshape(batch_size, 4, h_img, w_img, d_img)
            
            Seg_cur = self.seg_posterior.seg_inference_prob_fw(Img_cur)
            
            if seg_cri:
                cond1 = idx_cur - es_ed_idx[0]
                cond2 = idx_cur - es_ed_idx[1]
                if cond1==0:
                    Mask_ref = seg_in[:,idx_cur].to(self.device)  
                elif cond2==0:
                    Mask_ref = seg_in[:,idx_cur].to(self.device)  
                      
        return seg_outs, dis_out, seg_refs, seq_outs, masks_outs, seq_gens, unc_dis_out

    # def gen_method_1(self, Img, M_ED, frame_size):
        
    #     batch_size, _, h_img, w_img, d_img = Img.shape

    #     ### Initialize the recurrent layer with zeros
    #     h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
    #     hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
        
    #     seq_gen = np.zeros((batch_size, frame_size, 1, h_img, w_img, d_img), dtype=np.float32)
    #     seg_outs = np.zeros((batch_size, frame_size, 4, h_img, w_img, d_img), dtype=np.float32)
    #     dis_gen = np.zeros((batch_size, frame_size, 3, h_img, w_img, d_img), dtype=np.float32)
        
    #     for t in range(frame_size+1):
            
    #         idx_cur = t % frame_size           ### current index
            
    #         if t == 0:
    #             ### phi_x -----------------------------------------------------
    #             fit = self.net_img(Img)
                                
    #             ### zt_inference and reparameterization
    #             z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
    #             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
    #             fzt = self.net_phi_zt(z_t)
                
    #             Img_pas = torch.clone(Img)
    #             Seg_pas = torch.clone(M_ED)
                                
    #         elif t > 0:
    #             ### z_prior ---------------------------------------------------
    #             z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
    #             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
    #             fzt = self.net_phi_zt(z_t)
                
    #             d_mu_prior, d_logvar_prior, d_log_v_prior = self.net_dt_prior(fzt)
    #             d_cov_prior = covariance_d(d_logvar_prior, d_log_v_prior)
    #             d_t = dt_reparameterize(d_mu_prior, d_cov_prior)
                
    #             Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_pas)
    #             Img_deformed, deform_field = self.net_voxelmorph(d_t, Img_pas)
                
    #             Seg_pas = torch.clone(Seg_deformed)
    #             Img_pas = torch.clone(Img_deformed)
                
    #             ### store results ---------------------------------------------
    #             seq_gen[:, idx_cur] = Img_deformed.cpu().detach().numpy()
    #             seg_outs[:, idx_cur] = Seg_deformed.cpu().detach().numpy()
    #             dis_gen[:, idx_cur] = deform_field.cpu().detach().numpy()

    #         ### net_recurrent ------------------------------------------------
    #         input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
    #         h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
    #     return seq_gen, seg_outs, dis_gen
    

        

        
        


        
        
        
        

        
   


# # -*- coding: utf-8 -*-
# """
# Created on Fri Feb 16 14:13:22 2024

# @author: reza
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np 
# from utils.Conv3d_lstm import ConvLSTM
# from utils.reparametr_funcs import zt_reparameterize_multidim, covariance_d, dt_reparameterize, mask_reparameterize
# from utils.loss_pack import loss_fn
# from monai.networks.blocks.warp import DVF2DDF, Warp

# import os 
# from monai.bundle import ConfigParser, download
# from monai.transforms import AsDiscrete


# class Net_img_enc(nn.Module):
#     def __init__(self, in_channels, device, dropout_rate):
#         super(Net_img_enc, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         self.conv3d_img_1 = nn.Conv3d(in_channels, 16, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
#         self.conv3d_img_2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
#         self.conv3d_img_3 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
#         self.conv3d_img_4 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
#         self.conv3d_img_5 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
#         # self.conv3d_img_6 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
                
#     def forward(self,x):
#         skips = []
#         x = self.dropout(self.conv3d_img_1(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d_img_2(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d_img_3(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d_img_4(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d_img_5(x)) 
#         x = self.LRelu(x)
#         return x, skips
    
    
# class Seg_net(nn.Module): 
#     def __init__(self, seg_path, device):
#         super(Seg_net, self).__init__()

#         self.model_config = ConfigParser()
#         self.model_config.read_config(os.path.join(seg_path, "configs", "inference.json"))
#         self.model = self.model_config.get_parsed_content("network").to(device)
#         self.model.load_state_dict(torch.load(os.path.join(seg_path, "models", "model.pth"), map_location=device))
        
#     def eval_on(self, cond): 
#         if cond:
#             self.model.eval()
        
#     def get_output(self, x):
#         x = x.squeeze(0)
#         x = x.permute(3, 0, 1, 2)
#         x = self.model(x)
#         x = x.permute(1,2,3,0)
#         x = x.unsqueeze(0)
#         return x
    
#     def get_prob_output(self,x):
#         return F.softmax(self.get_output(x), dim=1)
    
#     def get_class_output(self,x): 
#         transform = AsDiscrete(threshold=0.5)
#         return transform(F.softmax(self.get_output(x), dim=1))
    
#     def seg_inference_score(self, img): 
        
#         img_resized = F.interpolate(img.squeeze(0).permute(3, 0, 1, 2), size=(256, 256), 
#                                   mode='bilinear', align_corners=False)
        
#         img_resized = img_resized.permute(1, 2, 3, 0).unsqueeze(0)

#         seg_resized = self.get_output(img_resized)
        
#         seg = F.interpolate(seg_resized.squeeze(0).permute(3, 0, 1, 2), size=(128, 128), 
#                                   mode='bilinear', align_corners=False)
        
#         seg = seg.permute(1, 2, 3, 0).unsqueeze(0)
        
#         return seg 

#     def seg_inference_prob(self, img): 
        
#         img_resized = F.interpolate(img.squeeze(0).permute(3, 0, 1, 2), size=(256, 256), 
#                                   mode='bilinear', align_corners=False)
        
#         img_resized = img_resized.permute(1, 2, 3, 0).unsqueeze(0)

#         # seg_resized = self.get_prob_output(img_resized)

#         seg_resized = self.get_output(img_resized)
        
#         seg = F.interpolate(seg_resized.squeeze(0).permute(3, 0, 1, 2), size=(128, 128), 
#                                   mode='bilinear', align_corners=False)
        
#         seg = seg.permute(1, 2, 3, 0).unsqueeze(0)

#         seg = F.softmax(seg, dim=1)
        
#         return seg

# # class Net_img_enc_seg(nn.Module):
# #     def __init__(self, in_channels, device, dropout_rate):
# #         super(Net_img_enc_seg, self).__init__()
        
# #         self.LRelu = nn.LeakyReLU(0.2)
# #         self.dropout = nn.Dropout(p=dropout_rate)
        
# #         self.conv3d_img_1 = nn.Conv3d(in_channels, 16, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
# #         self.conv3d_img_2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
# #         self.conv3d_img_3 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
# #         self.conv3d_img_4 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(2,2,1), padding=1).to(device)
# #         self.conv3d_img_5 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
# #         # self.conv3d_img_6 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
                
# #     def forward(self,x):
# #         skips = []
# #         x = self.dropout(self.conv3d_img_1(x))
# #         x = self.LRelu(x)
# #         skips.append(x)
# #         x = self.dropout(self.conv3d_img_2(x))
# #         x = self.LRelu(x)
# #         skips.append(x)
# #         x = self.dropout(self.conv3d_img_3(x))
# #         x = self.LRelu(x)
# #         skips.append(x)
# #         x = self.dropout(self.conv3d_img_4(x))
# #         x = self.LRelu(x)
# #         skips.append(x)
# #         x = self.dropout(self.conv3d_img_5(x)) 
# #         x = self.LRelu(x)
# #         return x, skips
    
# # class Net_img_dec_seg(nn.Module):
# #     def __init__(self, device, dropout_rate):
# #         super(Net_img_dec_seg, self).__init__()
        
# #         self.LRelu = nn.LeakyReLU(0.2)
# #         self.dropout = nn.Dropout(p=dropout_rate)
        
        
# #         self.convtrans1 = nn.ConvTranspose3d(64, 32, 
# #                                                       kernel_size=3, 
# #                                                       stride=(2,2,1), 
# #                                                       padding=(1,1,1), 
# #                                                       output_padding=(1,1,0)).to(device) 
        
# #         self.convtrans2 = nn.ConvTranspose3d(64, 32, 
# #                                                       kernel_size=3, 
# #                                                       stride=(2,2,1), 
# #                                                       padding=(1,1,1), 
# #                                                       output_padding=(1,1,0)).to(device)
        
# #         self.convtrans3 = nn.ConvTranspose3d(64, 32, 
# #                                                       kernel_size=3, 
# #                                                       stride=(2,2,1), 
# #                                                       padding=(1,1,1), 
# #                                                       output_padding=(1,1,0)).to(device)  
        
# #         self.convtrans4 = nn.ConvTranspose3d(48, 32, 
# #                                                       kernel_size=3, 
# #                                                       stride=(2,2,1), 
# #                                                       padding=(1,1,1), 
# #                                                       output_padding=(1,1,0)).to(device)  
        
# #         self.conv3d1= nn.Conv3d(32, 16, kernel_size=(3,3,3), 
# #                                         stride=(1, 1, 1), padding=1).to(device)
        
# #         self.conv3d2 = nn.Conv3d(16, 16, kernel_size=(3,3,3), 
# #                                         stride=(1, 1, 1), padding=1).to(device)
        
# #         self.conv3d3 = nn.Conv3d(16, 4, kernel_size=(3,3,3), 
# #                                         stride=(1, 1, 1), padding=1).to(device)
        
        
# #     def forward(self, x, skips):
# #         x = torch.cat([skips[-1],x], 1)
# #         x = self.dropout(self.convtrans1(x))
# #         x = self.LRelu(x)
# #         x = torch.cat([skips[-2],x], 1)
# #         x = self.dropout(self.convtrans2(x))
# #         x = self.LRelu(x)
# #         x = torch.cat([skips[-3],x], 1)
# #         x = self.dropout(self.convtrans3(x))
# #         x = self.LRelu(x)
# #         x = torch.cat([skips[-4],x], 1)
# #         x = self.dropout(self.convtrans4(x))
# #         x = self.LRelu(x)
# #         x = self.dropout(self.conv3d1(x))
# #         x = self.LRelu(x)
# #         x = self.dropout(self.conv3d2(x))
# #         x = self.LRelu(x)
# #         x = self.conv3d3(x)
# #         return x
    
# # class Seg_Net(nn.Module):
# #     def __init__(self, in_channels, device, dropout_rate):
# #         super(Seg_Net, self).__init__()
        
# #         self.encoder = Net_img_enc_seg(in_channels, device, dropout_rate)
# #         self.decoder = Net_img_dec_seg(device, dropout_rate)
        
# #     def forward(self, x): 
# #         x, skips = self.encoder(x)
# #         x = self.decoder(x, skips)
# #         return x
        
    
# class Net_phi_zt(nn.Module): 
#     def __init__(self, device, dropout_rate):
#         super(Net_phi_zt, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout_rate)
        
        
#         self.conv3fzt_dt = nn.Conv3d(1, 32, kernel_size=(3,3,3), 
#                                         stride=(1,1,1), padding=1).to(device)
        
#     def forward(self, z_t):
#         z_t = self.dropout(self.conv3fzt_dt(z_t))
#         z_t = self.LRelu(z_t)
#         return z_t

# class Net_dt_posterior(nn.Module):
#     def __init__(self, device, dropout_rate):
#         super(Net_dt_posterior, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         self.conv3d1_dt_pos = nn.Conv3d(2, 16, kernel_size=(3,3,3), 
#                                         stride=(1,1,1), padding=1).to(device)
        
#         self.conv3d2_dt_pos = nn.Conv3d(16, 32, kernel_size=(3,3,3), 
#                                         stride=(2, 2, 1), padding=1).to(device)
        
#         self.conv3d3_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
#                                         stride=(2, 2, 1), padding=1).to(device)

#         self.conv3d4_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
#                                         stride=(2, 2, 1), padding=1).to(device)
        
#         self.conv3d5_dt_pos = nn.Conv3d(32, 32, kernel_size=(3,3,3), 
#                                         stride=(2, 2, 1), padding=1).to(device)
        
#         self.conv3d6_dt_pos = nn.Conv3d(64, 32, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)
        
#         self.convtrans1_dt_pos = nn.ConvTranspose3d(64, 32, 
#                                                       kernel_size=3, 
#                                                       stride=(2,2,1), 
#                                                       padding=(1,1,1), 
#                                                       output_padding=(1,1,0)).to(device) 
        
#         self.convtrans2_dt_pos = nn.ConvTranspose3d(64, 32, 
#                                                       kernel_size=3, 
#                                                       stride=(2,2,1), 
#                                                       padding=(1,1,1), 
#                                                       output_padding=(1,1,0)).to(device)
        
#         self.convtrans3_dt_pos = nn.ConvTranspose3d(64, 32, 
#                                                       kernel_size=3, 
#                                                       stride=(2,2,1), 
#                                                       padding=(1,1,1), 
#                                                       output_padding=(1,1,0)).to(device)  
        
#         self.convtrans4_dt_pos = nn.ConvTranspose3d(64, 32, 
#                                                       kernel_size=3, 
#                                                       stride=(2,2,1), 
#                                                       padding=(1,1,1), 
#                                                       output_padding=(1,1,0)).to(device)
        
#         self.convtrans5_dt_pos = nn.ConvTranspose3d(48, 32, kernel_size=3, stride=1, padding=1)
        
        
#         self.conv3d7_dt_pos = nn.Conv3d(32, 16, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)
        
#         self.conv3d8_dt_pos = nn.Conv3d(16, 16, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)
        
#         self.conv3d8_dt_pos = nn.Conv3d(16, 3, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)
        
#         self.conv_mu_dt = nn.Conv3d(3, 3, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)
        
#         self.conv_logvar_dt = nn.Conv3d(3, 1, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device) 
        
#         self.conv_v_dt =  nn.Conv3d(3, 3, kernel_size=(3,3,3), 
#                                         stride=(1, 1, 1), padding=1).to(device)  
        
    
#     def forward(self, x, fzt):
#         skips = []
#         x = self.dropout(self.conv3d1_dt_pos(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d2_dt_pos(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d3_dt_pos(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d4_dt_pos(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = self.dropout(self.conv3d5_dt_pos(x))
#         x = self.LRelu(x)
#         skips.append(x)
#         x = torch.cat([x,fzt], 1)
#         x = self.dropout(self.conv3d6_dt_pos(x))
#         x = self.LRelu(x)
#         x = torch.cat([skips[-1],x], 1)
#         x = self.dropout(self.convtrans1_dt_pos(x))
#         x = self.LRelu(x)
#         x = torch.cat([skips[-2],x], 1)
#         x = self.dropout(self.convtrans2_dt_pos(x))
#         x = self.LRelu(x)
#         x = torch.cat([skips[-3],x], 1)
#         x = self.dropout(self.convtrans3_dt_pos(x))
#         x = self.LRelu(x)
#         x = torch.cat([skips[-4],x], 1)
#         x = self.dropout(self.convtrans4_dt_pos(x))
#         x = self.LRelu(x)
#         x = torch.cat([skips[-5],x], 1)
#         x = self.dropout(self.convtrans5_dt_pos(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.conv3d7_dt_pos(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.conv3d8_dt_pos(x))
#         d_mu = self.conv_mu_dt(x)
#         d_logvar = self.conv_logvar_dt(x)
#         d_log_v = self.conv_v_dt(x)
#         return d_mu, d_logvar, d_log_v
    
# class Net_VoxelMorph(nn.Module):
#     def __init__(self, half_res):
#         super(Net_VoxelMorph, self).__init__() 
        
#         self.half_res = half_res
#         self.diffeomorphic = False
#         self.warp = Warp(mode="bilinear", padding_mode="zeros")
        
#     def enable_diffeomorphic(self, integration_steps):
#         self.diffeomorphic = True
#         self.integration_steps = integration_steps
#         self.dvf2ddf = DVF2DDF(num_steps=self.integration_steps, 
#                                mode="bilinear", padding_mode="zeros")
        
#     def forward(self, x, img):
#         if self.half_res:
#             x = F.interpolate(x, scale_factor=0.5, mode="trilinear", align_corners=True) * 2.0
#         if self.diffeomorphic: 
#             x = self.dvf2ddf(x) 
#         if self.half_res:
#             x = F.interpolate(x * 0.5, scale_factor=2.0, mode="trilinear", align_corners=True)
#         return self.warp(img, x), x 


    
# class Net_infer_z(nn.Module): 
#     def __init__(self, device, dropout_rate, in_channels=64): 
#         super(Net_infer_z, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         self.conv3d_decoder = nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
#         self.conv3d_inf_zt_1 = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
#         self.conv3d_inf_zt_2 = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
             
#     def forward(self, x, h_recurrent):
#         x = torch.cat([x,h_recurrent], 1) 
#         x = self.dropout(self.conv3d_decoder(x))
#         x = self.LRelu(x)
#         z_mu = self.conv3d_inf_zt_1(x)
#         z_logvar = self.conv3d_inf_zt_2(x)
#         return z_mu, z_logvar 
    
# class Net_img_est(nn.Module):
#     def __init__(self, device, dropout_rate, in_channels=1):
#         super(Net_img_est, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.ReLU = nn.ReLU()
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         # self.convtrans1_img_est = nn.ConvTranspose3d(in_channels, 32, kernel_size=3, 
#         #                                               stride=1, padding=1).to(device)
        
#         self.convtrans2_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
#                                                     padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
#         self.convtrans3_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
#                                                     padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
#         self.convtrans4_img_est = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2,2,1), 
#                                                     padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
#         self.convtrans5_img_est = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(2,2,1), 
#                                                     padding=(1,1,1), output_padding=(1,1,0)).to(device)
        
#         self.conv3d_img_est = nn.Conv3d(16, 1, kernel_size=(3,3,3), 
#                                         stride=(1,1,1), padding=1).to(device)
          
#     def forward(self, x):
#         # x = self.dropout(self.convtrans1_img_est(x))
#         # x = self.LRelu(x)
#         x = self.dropout(self.convtrans2_img_est(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.convtrans3_img_est(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.convtrans4_img_est(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.convtrans5_img_est(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.conv3d_img_est(x))
#         x = self.ReLU(x)
#         return x
    
# class Net_z_prior(nn.Module):
#     def __init__(self, device, dropout_rate, in_channels=32):
#         super(Net_z_prior, self).__init__()
        
#         self.LRelu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout_rate)
        
#         self.conv3d1_z_prior = nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
#         self.conv3d2_z_prior = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
        
#         self.conv3d3_z_prior = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)
#         self.conv3d4_z_prior = nn.Conv3d(32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1).to(device)    
        
#     def forward(self, x):
#         x = self.dropout(self.conv3d1_z_prior(x))
#         x = self.LRelu(x)
#         x = self.dropout(self.conv3d2_z_prior(x))
#         x = self.LRelu(x)
#         z_prior_mu = self.conv3d3_z_prior(x)
#         z_prior_logvar = self.conv3d4_z_prior(x)
#         return z_prior_mu, z_prior_logvar


# class MyModel(nn.Module):
#     def __init__(self, device, coef_smooth_loss, seg_path): 
#         super(MyModel, self).__init__()
        
#         self.device = device
#         self.net_img = Net_img_enc(in_channels=1, device=self.device, dropout_rate=0.0)
#         # self.seg_posterior = Seg_Net(in_channels=1, device=self.device, dropout_rate=0.0)
#         self.seg_posterior = Seg_net(seg_path, device)
        
#         self.net_dt_posterior = Net_dt_posterior(self.device, dropout_rate=0.0)
#         self.net_voxelmorph = Net_VoxelMorph(half_res=False)
#         self.net_phi_zt = Net_phi_zt(device=self.device, dropout_rate=0.0)
        
        
#         self.net_infer_z = Net_infer_z(device=self.device, dropout_rate=0.0)
#         self.net_img_est = Net_img_est(device=self.device, dropout_rate=0.0)
#         self.net_z_prior = Net_z_prior(device=self.device, dropout_rate=0.0)
#         # self.net_dt_prior = Net_dt_prior(device=self.device, dropout_rate=0.0)
        
#         self.ConvLSTM = ConvLSTM(input_size= (8, 8),
#                                   input_dim= 64,
#                                   hidden_dim= [64, 32],
#                                   kernel_size= [3, 3],
#                                   num_layers= 2,
#                                   device=self.device,
#                                   droput_rates = [0.0, 0.0],
#                                   batch_first=True,
#                                   bias=True) 
        
        
        
#         ### coefficients of loss term
#         self.coef_klz_loss = 2e-4
#         self.coef_smooth_loss = coef_smooth_loss
#         self.coef_img_sim_loss = 0.01
#         self.coef_kld_loss  = 1e-4
#         self.coef_seg_loss_1 = 1.0
#         self.coef_seg_loss_2 = 1.0
        
#     def forward_seg(self, seq_in, seg_in, es_ed_idx):
        
#         batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
#         loss_fn = nn.CrossEntropyLoss()
        
#         seg_outs = np.zeros((batch_size, 2, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
        
#         total_loss = 0.0 
        
#         for idx, frame_idx in enumerate(es_ed_idx.squeeze(0)):
#             img = seq_in[:, int(frame_idx)].to(self.device)
#             mask = seg_in[:, int(frame_idx)].to(self.device)
            
#             seg = self.seg_posterior.seg_inference_score(img)
            
#             loss_val = loss_fn(seg, mask)
            
#             total_loss = total_loss + loss_val
            
#             seg_outs[:, idx, :,:,:,:] = F.softmax(seg, dim=1).cpu().detach().numpy()
                   
#         return total_loss, seg_outs
        
        
#     def forward(self, seq_in, seg_in, es_ed_idx):
        
#         batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
#         h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
#         hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
                
#         seg_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
#         dis_out = np.zeros((batch_size, frame_num, 3, h_img, w_img, d_img), dtype=np.float32)
        
#         ### Initial value of loss terms
#         img_sim_loss = 0.0
#         klz_loss = 0.0
#         kld_loss = 0.0 
#         smooth_loss = 0.0
#         seg_loss_1 = 0.0 
#         seg_loss_2 = 0.0 
#         total_loss = 0.0
        
#         for t in range(frame_num+1):
                 
#             idx_cur = t % frame_num           ### current index
#             idx_pas = (t-1) % frame_num       ### past index

#             seg_cri = True if idx_cur in es_ed_idx else False
            
#             """
#             seg_cri defines whether we are in the ED or ES frames or not 
#             if seg_cri==True, it means we are in one of the ED or ES frames 
#             if seg_cri==False, it indicates we are not in ED and ES frames 
#             """
        
#             Img_cur = seq_in[:, idx_cur].to(self.device)         ### current image
#             Img_pas = seq_in[:, idx_pas].to(self.device)         ### past image
            
#             ### z_prior
#             z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
            
#             ### Feature maps extraction for Img_cur   
#             fit, _ = self.net_img(Img_cur)
            
#             ### zt_inference and reparameterization
#             z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
#             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
#             fzt = self.net_phi_zt(z_t)
            
#             ### Dt_inference
#             d_mu, d_logvar, d_log_v = self.net_dt_posterior(torch.cat([Img_pas, Img_cur], dim=1), fzt)
#             d_cov = covariance_d(d_logvar, d_log_v)
#             d_t = dt_reparameterize(d_mu, d_cov)
            
#             if t>0:
#                 if not(seg_cri):
#                     Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_cur)
#                     # Seg_deformed = stn_3d(d_t, Seg_cur, self.device)
#                 else:
#                     Seg_deformed, deform_field = self.net_voxelmorph(d_t, mask_reparameterize(F.softmax(Seg_cur, dim=1)))
#                     # Seg_deformed = stn_3d(d_t, mask_reparameterize(Seg_cur), self.device)
            
#             Img_cur_est = self.net_img_est(fzt)
            
#             input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
#             h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
#             Seg_cur = torch.empty((batch_size, 4, h_img, w_img, d_img), device=self.device)
#             for b_idx in range(batch_size): 
#                 Seg_cur[b_idx,:,:,:] = self.seg_posterior.seg_inference_score(Img_cur[b_idx, :, :, :].unsqueeze(0))
                
#             if seg_cri:
#                 cond1 = idx_cur - es_ed_idx[0]
#                 cond2 = idx_cur - es_ed_idx[1]
#                 if cond1==0:
#                     Seg_ref = seg_in[:,idx_cur].to(self.device)  
#                 elif cond2==0:
#                     Seg_ref = seg_in[:,idx_cur].to(self.device)  
                    
#             else: 
#                 Seg_ref = Seg_cur.clone()
                   
#             if t>0:
#                 img_sim_loss_t, klz_loss_t, kld_loss_t, smooth_loss_t, seg_loss_1_t, seg_loss_2_t = loss_fn(Img_cur, Img_cur_est, 
#                                                                               z_mu, z_logvar, z_prior_mu, z_prior_logvar, 
#                                                                               d_cov, d_mu, deform_field, 
#                                                                               seg_prior=Seg_deformed, seg_posterior=Seg_ref, 
#                                                                               seg_cri = seg_cri, device = self.device)
                
#                 img_sim_loss = img_sim_loss + img_sim_loss_t 
#                 klz_loss = klz_loss + klz_loss_t
#                 kld_loss = kld_loss + kld_loss_t
#                 smooth_loss = smooth_loss + smooth_loss_t
#                 seg_loss_1 = seg_loss_1 + seg_loss_1_t
#                 seg_loss_2 = seg_loss_2 + seg_loss_2_t
                
#                 seg_outs[:, idx_cur, :,:,:,:] = Seg_deformed.cpu().detach().numpy()
#                 dis_out[:, idx_cur, :, : ,: ,:] = deform_field.cpu().detach().numpy()

        
#         total_loss = self.coef_img_sim_loss*img_sim_loss + self.coef_smooth_loss*smooth_loss + self.coef_klz_loss*klz_loss + self.coef_kld_loss*kld_loss + self.coef_seg_loss_1*seg_loss_1 + self.coef_seg_loss_2*seg_loss_2 
        
#         return seg_outs, dis_out, img_sim_loss, klz_loss, kld_loss, smooth_loss, seg_loss_1, seg_loss_2, total_loss
    
#     def inference_forward(self, seq_in, seg_in, es_ed_idx):
                
#         batch_size, frame_num, _, h_img, w_img, d_img = seq_in.shape
        
#         h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
#         hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
                
#         seg_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
#         masks_outs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
#         seq_outs = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)
#         dis_out = np.zeros((batch_size, frame_num, 3, h_img, w_img, d_img), dtype=np.float32)
#         seg_refs = np.zeros((batch_size, frame_num, seg_in.shape[2], h_img, w_img, d_img), dtype=np.float32)
#         seq_gens = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)
#         unc_dis_out = np.zeros((batch_size, frame_num, 1, h_img, w_img, d_img), dtype=np.float32)    
        
        
#         for t in range(frame_num+1):
            
#             idx_cur = t % frame_num           ### current index
#             idx_pas = (t-1) % frame_num       ### past index

#             seg_cri = True if idx_cur in es_ed_idx else False
        
#             Img_cur = seq_in[:, idx_cur].to(self.device)           ### current image
#             Img_pas = seq_in[:, idx_pas].to(self.device)           ### past image
            
#             ### z_prior
#             z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
            
#             ### Feature maps extraction for Img_cur   
#             fit, skips_ic = self.net_img(Img_cur)
            
#             ### zt_inference and reparameterization
#             z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
#             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
#             fzt = self.net_phi_zt(z_t)
            
#             ### Dt_inference
#             d_mu, d_logvar, d_log_v = self.net_dt_posterior(torch.cat([Img_pas, Img_cur], dim=1), fzt)
#             d_cov = covariance_d(d_logvar, d_log_v)
#             d_t = dt_reparameterize(d_mu, d_cov)
            
#             unc_map_d_t = 0.5 * torch.log(((2*torch.pi) ** 3) * d_cov.det())
#             unc_map_d_t = unc_map_d_t.unsqueeze(1)
            
#             if t>0:
#                 Img_deformed, deform_field = self.net_voxelmorph(d_t,Img_pas)
#                 Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_cur)
#                 Mask_deformed, deform_field = self.net_voxelmorph(d_t, Mask_ref)
#                 Mask_ref = Mask_deformed.clone()
#                 seq_outs[:, idx_cur, :,:,:,:] = Img_deformed.cpu().detach().numpy()
#                 seg_refs[:, idx_pas, :,:,:,:] = Seg_cur.cpu().detach().numpy()
#                 masks_outs[:, idx_cur, :,:,:,:] = Mask_deformed.cpu().detach().numpy()
#                 seg_outs[:, idx_cur, :,:,:,:] = Seg_deformed.cpu().detach().numpy()
#                 dis_out[:, idx_cur, :, : ,: ,:] = deform_field.cpu().detach().numpy()
#                 seq_gens[:, idx_cur, :,:,:,:] = Img_cur_est.cpu().detach().numpy()
#                 unc_dis_out[:,idx_cur, :, :, :, :] = unc_map_d_t.cpu().detach().numpy()
            
#             Img_cur_est = self.net_img_est(fzt)
            
#             input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
#             h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
#             Seg_cur = torch.empty((batch_size, 4, h_img, w_img, d_img), device=self.device)
#             for b_idx in range(batch_size): 
#                 Seg_cur[b_idx,:,:,:] = self.seg_posterior.seg_inference_prob(Img_cur[b_idx, :, :, :].unsqueeze(0))
            
#             if seg_cri:
#                 cond1 = idx_cur - es_ed_idx[0]
#                 cond2 = idx_cur - es_ed_idx[1]
#                 if cond1==0:
#                     Mask_ref = seg_in[:,idx_cur].to(self.device)  
#                 elif cond2==0:
#                     Mask_ref = seg_in[:,idx_cur].to(self.device)  
                      
#         return seg_outs, dis_out, seg_refs, seq_outs, masks_outs, seq_gens, unc_dis_out

#     # def gen_method_1(self, Img, M_ED, frame_size):
        
#     #     batch_size, _, h_img, w_img, d_img = Img.shape

#     #     ### Initialize the recurrent layer with zeros
#     #     h_recurrent = torch.zeros([batch_size, 32, 8, 8, d_img], device=self.device)
#     #     hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size, depth=d_img)
        
#     #     seq_gen = np.zeros((batch_size, frame_size, 1, h_img, w_img, d_img), dtype=np.float32)
#     #     seg_outs = np.zeros((batch_size, frame_size, 4, h_img, w_img, d_img), dtype=np.float32)
#     #     dis_gen = np.zeros((batch_size, frame_size, 3, h_img, w_img, d_img), dtype=np.float32)
        
#     #     for t in range(frame_size+1):
            
#     #         idx_cur = t % frame_size           ### current index
            
#     #         if t == 0:
#     #             ### phi_x -----------------------------------------------------
#     #             fit = self.net_img(Img)
                                
#     #             ### zt_inference and reparameterization
#     #             z_mu, z_logvar = self.net_infer_z(fit, h_recurrent)
#     #             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
#     #             fzt = self.net_phi_zt(z_t)
                
#     #             Img_pas = torch.clone(Img)
#     #             Seg_pas = torch.clone(M_ED)
                                
#     #         elif t > 0:
#     #             ### z_prior ---------------------------------------------------
#     #             z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)
#     #             z_t = zt_reparameterize_multidim(z_mu, z_logvar)
#     #             fzt = self.net_phi_zt(z_t)
                
#     #             d_mu_prior, d_logvar_prior, d_log_v_prior = self.net_dt_prior(fzt)
#     #             d_cov_prior = covariance_d(d_logvar_prior, d_log_v_prior)
#     #             d_t = dt_reparameterize(d_mu_prior, d_cov_prior)
                
#     #             Seg_deformed, deform_field = self.net_voxelmorph(d_t, Seg_pas)
#     #             Img_deformed, deform_field = self.net_voxelmorph(d_t, Img_pas)
                
#     #             Seg_pas = torch.clone(Seg_deformed)
#     #             Img_pas = torch.clone(Img_deformed)
                
#     #             ### store results ---------------------------------------------
#     #             seq_gen[:, idx_cur] = Img_deformed.cpu().detach().numpy()
#     #             seg_outs[:, idx_cur] = Seg_deformed.cpu().detach().numpy()
#     #             dis_gen[:, idx_cur] = deform_field.cpu().detach().numpy()

#     #         ### net_recurrent ------------------------------------------------
#     #         input_net_conv_lstm = torch.cat([fit,fzt], dim=1)
#     #         h_recurrent, hidden_state = self.ConvLSTM(input_net_conv_lstm, hidden_state)
            
#     #     return seq_gen, seg_outs, dis_gen
    

        

        
        


        
        
        
        

        
   

