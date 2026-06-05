# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:29:49 2024

@author: reza
"""

import numpy as np
import matplotlib.pyplot as plt  

def make_colorwheel():
    """
    Generates a colorwheel for optical flow visualization.

    Returns:
        np.ndarray: A colorwheel represented as an array of RGB values.
    """
    RY, YG, GC, CB, BM, MR = [15, 6, 4, 11, 13, 6]

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)  # r g b

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    Compute color image from optical flow vectors u and v.

    Parameters:
        u (np.ndarray): Optical flow vector in x-direction
        v (np.ndarray): Optical flow vector in y-direction

    Returns:
        np.ndarray: Color image with optical flow visualization
    """
    colorwheel = make_colorwheel()

    # Identify NaN values in u and v arrays
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)

    # Replace NaN values with zero in u and v arrays
    u[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.hypot(u, v)
    angle = np.arctan2(u, v) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)  # -1~1 mapped to 1~ncols
    k0 = fk.astype(np.uint8)            # 1, 2, ..., ncols
    k1 = (k0 + 1) % ncols               # Modulo operation to handle edge case
    f = fk - k0

    img = np.empty((k1.shape[0], k1.shape[1], 3), dtype=np.uint8)  # Initialize as uint8 array for better performance
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75                            # out of range
        img[:, :, 2 - i] = (255 * col).astype(np.uint8)  # Use astype() for type casting

    return img

def com_colormaps_DVFs(D): 
    Colormapped_D = np.zeros((D.shape[0], D.shape[2], D.shape[3], D.shape[1]), dtype=np.uint8)
    for i in range(D.shape[0]):
        Colormapped_D[i,:,:,:] = compute_color(D[i,0,:,:], D[i,1,:,:])
    return Colormapped_D

# def com_colormaps_DVFs_3D(D): 
#     Colormapped_D = np.zeros((D.shape[0], D.shape[2], D.shape[3], D.shape[1], D.shape[4]), dtype=np.uint8)
#     for i in range(D.shape[0]):
#         for j in range(D.shape[-1]):
#             Colormapped_D[i, :, :, :, j] = compute_color3d(D[i, :, :, :, j])
#     return Colormapped_D

# def make_colorwheel():
#     """
#     Generates a colorwheel for optical flow visualization.

#     Returns:
#         np.ndarray: A colorwheel represented as an array of RGB values.
#     """
#     RY, YG, GC, CB, BM, MR = [15, 6, 4, 11, 13, 6]

#     ncols = RY + YG + GC + CB + BM + MR
#     colorwheel = np.zeros((ncols, 3), dtype=np.uint8)  # r g b

#     col = 0

#     # RY
#     colorwheel[0:RY, 0] = 255
#     colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
#     col += RY

#     # YG
#     colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
#     colorwheel[col:YG + col, 1] = 255
#     col += YG

#     # GC
#     colorwheel[col:GC + col, 1] = 255
#     colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
#     col += GC

#     # CB
#     colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
#     colorwheel[col:CB + col, 2] = 255
#     col += CB

#     # BM
#     colorwheel[col:BM + col, 2] = 255
#     colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
#     col += BM

#     # MR
#     colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
#     colorwheel[col:MR + col, 0] = 255

#     return colorwheel


# def compute_color3d(deformation_field):
#     """
#     Compute color image from a 3-dimensional deformation field.

#     Parameters:
#         deformation_field (np.ndarray): Deformation field with shape [3, height, width, depth]

#     Returns:
#         np.ndarray: Color image with deformation field visualization
#     """
#     colorwheel = make_colorwheel()
#     ncols = colorwheel.shape[0]

#     # Decompose the deformation field
#     u = deformation_field[0]
#     v = deformation_field[1]
#     w = deformation_field[2]

#     # Identify NaN values and replace with zero
#     u[np.isnan(u)] = 0
#     v[np.isnan(v)] = 0
#     w[np.isnan(w)] = 0

#     # Compute magnitude and angles
#     radius = np.sqrt(u**2 + v**2 + w**2)
#     angle_xy = np.arctan2(v, u) / np.pi  # Angle in the xy-plane
#     angle_z = np.arctan2(w, np.sqrt(u**2 + v**2)) / np.pi  # Angle in the z-direction

#     # Compute color mapping for each angle
#     fk_xy = (angle_xy + 1) / 2 * (ncols - 1)
#     fk_z = (angle_z + 1) / 2 * (ncols - 1)

#     k0_xy = fk_xy.astype(np.uint8)
#     k1_xy = (k0_xy + 1) % ncols
#     f_xy = fk_xy - k0_xy

#     k0_z = fk_z.astype(np.uint8)
#     k1_z = (k0_z + 1) % ncols
#     f_z = fk_z - k0_z

#     img = np.empty([deformation_field.shape[1], deformation_field.shape[2], 3], dtype=np.uint8)
#     ncolors = colorwheel.shape[1]
#     for i in range(ncolors):
#         tmp = colorwheel[:, i]
#         col0_xy = tmp[k0_xy] / 255
#         col1_xy = tmp[k1_xy] / 255
#         col_xy = (1 - f_xy) * col0_xy + f_xy * col1_xy

#         col0_z = tmp[k0_z] / 255
#         col1_z = tmp[k1_z] / 255
#         col_z = (1 - f_z) * col0_z + f_z * col1_z

#         col = (col_xy + col_z) / 2  # Combine the colors

#         idx = radius <= 1
#         col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
#         col[~idx] *= 0.75  # out of range
#         img[..., 2 - i] = (255 * col).astype(np.uint8)  # Use astype() for type casting

#     return img


