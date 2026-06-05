# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:08:35 2024

@author: reza
"""

import torch
import torch.nn.functional as F

### generate_grid_3d ==========================================================
def generate_grid_3d(B, C, H, W, D, device):
    aff = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]])
    aff = aff.expand(B, 3, 4)
    grid = torch.nn.functional.affine_grid(aff, size=(B,C,D,H,W)).to(device)
    
    # aff = aff.unsqueeze(0)
    # aff = torch.FloatTensor([[[1, 0, 0],[0, 1, 0]]])
    # aff = aff.expand(B, 2, 3)  # expand to the number of batches you need
    # grid = torch.nn.functional.affine_grid(aff, size=(B,C,H,W)).to(device)
    return grid

### stn_3d ====================================================================
def stn_3d(flow, img, device):
    B, C, H, W, D = flow.shape
    grid = generate_grid_3d(B, C, H, W, D, device)
    factor = torch.FloatTensor([[[[3/W, 3/H, 3/D]]]]).to(device)
    # deformation = flow.permute(0, 1, 4, 2, 3)*factor + grid
    deformation = flow.permute(0, 4, 2, 3, 1)*factor + grid
    warped_img = F.grid_sample(img.permute(0, 1, 4, 2, 3), deformation, align_corners=False)
    warped_img = warped_img.permute(0, 1, 3, 4, 2)
    # warped_img = warped_img.int()
    return warped_img