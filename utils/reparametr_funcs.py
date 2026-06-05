# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:37:55 2024

@author: reza
"""
import torch
from torch.nn import functional as F 


def zt_reparameterize(mu, logvar): 
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z

def zt_reparameterize_multidim(mu, logcov): 
    cov = torch.exp(0.5 * logcov)
    eps_m = torch.randn_like(mu)
    # z = mu + torch.matmul(cov, eps_m)
    z = mu + cov * eps_m
    return z


def covariance_d(log_var, log_v):
    v_m = torch.exp(log_v).permute(0,2,3,4,1).unsqueeze(-1)
    cov_m = torch.matmul(v_m, v_m.permute(0,1,2,3,5,4))
    temp1 = torch.exp(log_var)[:,0] + 1e-6
    cov_m[:,:,:,:,0,0] = cov_m[:,:,:,:,0,0] + temp1
    cov_m[:,:,:,:,1,1] = cov_m[:,:,:,:,1,1] + temp1
    cov_m[:,:,:,:,2,2] = cov_m[:,:,:,:,2,2] + temp1
    return cov_m

def dt_reparameterize(mu, cov):
    mu_m = mu.permute(0,2,3,4,1).unsqueeze(-1)
    eps_m = torch.randn_like(mu_m)
    cov_m = 0.5 * cov
    x = mu_m + torch.matmul(cov_m, eps_m)
    x = x.squeeze(-1)
    x = x.permute(0,4,1,2,3)
    return x

def mask_reparameterize(probability_map, temperature=2/3, eps=1e-6):
    # It is based on Gumble-Softmax reparametrization trick 
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(probability_map) + eps) + eps)
    modified_logits = (torch.log(probability_map + eps) + gumbel_noise) / temperature
    softmax_probs = F.softmax(modified_logits, dim=1)
    return softmax_probs
    
    