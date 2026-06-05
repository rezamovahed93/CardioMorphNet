# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:54:25 2024

@author: reza
"""

import torch
from torch.nn import functional as F
from torch.nn import NLLLoss
from torch.nn import LogSoftmax
from torch.nn import CrossEntropyLoss
from monai.losses import BendingEnergyLoss

def det_d_cov_cal(d_cov): 
    det_d_cov1 = d_cov[:,:,:,:,0,0]*d_cov[:,:,:,:,1,1]*d_cov[:,:,:,:,2,2]
    det_d_cov2 = - d_cov[:,:,:,:,0,0]*d_cov[:,:,:,:,1,2]*d_cov[:,:,:,:,2,1]
    det_d_cov3 = - d_cov[:,:,:,:,0,1]*d_cov[:,:,:,:,1,0]*d_cov[:,:,:,:,2,2]
    det_d_cov4 = d_cov[:,:,:,:,0,1]*d_cov[:,:,:,:,1,2]*d_cov[:,:,:,:,2,0]
    det_d_cov5 = d_cov[:,:,:,:,0,2]*d_cov[:,:,:,:,1,0]*d_cov[:,:,:,:,2,1]
    det_d_cov6 = - d_cov[:,:,:,:,0,2]*d_cov[:,:,:,:,1,1]*d_cov[:,:,:,:,2,0]
    det_d_cov = det_d_cov1 + det_d_cov2 + det_d_cov3 + det_d_cov4 + det_d_cov5 + det_d_cov6
    return det_d_cov

def loss_mask_1(Seg_prior, Seg_posterior):
    loss_fn = CrossEntropyLoss()
    element1 = - loss_fn(Seg_posterior, F.softmax(Seg_posterior, dim=1))
    element2 = loss_fn(Seg_prior, F.softmax(Seg_posterior, dim=1))
    loss = element1 + element2
    return loss

def loss_mask_2(Mask, Seg_prior):
    loss_fn = NLLLoss()
    log_softmax = LogSoftmax(dim=1)
    loss = loss_fn(log_softmax(Seg_prior), torch.argmax(Mask, dim=1)) 
    return loss    

def loss_fn(x_target, x_prediction, z_mu, z_logvar, z_prior_mu, z_prior_logvar, 
            d_cov, d_mu, d_t, seg_prior, seg_posterior, 
            seg_cri, device): 
    
    ### L_sim
    sigma_l_sim = 1.0
    l_sim_loss_t = 1.0 / sigma_l_sim**2 * torch.mean(torch.square(x_target - x_prediction))
    
    ### KL divergence loss z 
    klz_element =  z_prior_logvar - z_logvar - 1. + (z_logvar.exp() + (z_mu - z_prior_mu).pow(2)) / (z_prior_logvar.exp()+1e-9)
    klz_loss_t = 0.5 * torch.mean(klz_element)
    
    ### KL divergence loss D
    det_d_cov = d_cov.det()
    kld_element_1 = - torch.log(det_d_cov + 1e-9)
    kld_element_2 = d_cov[:,:,:,:,0,0] + d_cov[:,:,:,:,1,1] + d_cov[:,:,:,:,2,2]
    d_mu_expand = d_mu.permute(0,2,3,4,1).unsqueeze(-1)
    kld_element_3 = torch.matmul(d_mu_expand.permute(0,1,2,3,5,4), d_mu_expand)
    kld_element_3 = kld_element_3.squeeze(-1).squeeze(-1) 
    kld_element = kld_element_1 - 3. + kld_element_2 + kld_element_3
    kld_loss_t = 0.5 * torch.mean(kld_element)

    ### Smoothness loss
    regularization = BendingEnergyLoss()
    smooth_loss_t = regularization(d_t)
    
    ### Segmentation loss
    seg_loss_1_t = 0.0 
    seg_loss_2_t = 0.0
    if seg_cri==False:
        seg_loss_1_t = loss_mask_1(seg_prior,seg_posterior)
    elif seg_cri==True: 
        seg_loss_2_t = loss_mask_2(Mask=seg_posterior, Seg_prior=seg_prior)
    
    return l_sim_loss_t, klz_loss_t, kld_loss_t, smooth_loss_t, seg_loss_1_t, seg_loss_2_t


def diff_fun(y_in, k_dim):
    n_dim = torch.Tensor.dim(y_in)

    rp = [k_dim, *range(k_dim), *range(k_dim + 1, n_dim)]
    y = y_in.permute(rp)

    df = y[1:, ...] - y[:-1, ...]

    rn = [*range(1, k_dim+1), 0, *range(k_dim+1, n_dim)]
    df = df.permute(rn)  # permute back

    df = df.pow(2)
    return df
