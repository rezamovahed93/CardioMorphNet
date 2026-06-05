
import numpy as np
import SimpleITK as sitk

### jacobian_2d ===============================================================
def Fn_jac_det_calc_2d(dis_x, dis_y, add_identity=True):
    """Computes jacobian_2d of given deformation phi = [dis_x, dis_y]."""
    gx_y, gx_x = np.gradient(dis_x)
    gy_y, gy_x = np.gradient(dis_y)
    if add_identity:
        gx_x += 1.
        gy_y += 1.

    det = gx_x * gy_y - gy_x * gx_y
    return det

### jacobian_3d ===============================================================
def Fn_jac_det_calc_3d(dis_x, dis_y, dis_z, add_identity=True):
    """Computes jacobian_3d of given deformation phi = [dis_x, dis_y, dis_z]."""
    gx_y, gx_x, gx_z = np.gradient(dis_x)
    gy_y, gy_x, gy_z = np.gradient(dis_y)
    gz_y, gz_x, gz_z = np.gradient(dis_z)
    if add_identity:
        gx_x += 1.0
        gy_y += 1.0
        gz_z += 1.0

    temp1 = gx_x * (gy_y * gz_z - gy_z * gz_y)
    temp2 = gx_y * (gy_x * gz_z - gy_z * gz_x)
    temp3 = gx_z * (gy_x * gz_y - gy_y * gz_x)
    det = temp1 - temp2 + temp3
    return det

### jacobian_2d using sitk ====================================================
def jac_det_sitk_2d(dis_x, dis_y):
    H, W = dis_x.shape
    dis_xy = np.zeros([1, H, W, 3])
    dis_xy[0,:,:,0] = dis_x
    dis_xy[0,:,:,1] = dis_y

    dis_xy_sitk = sitk.GetImageFromArray(dis_xy, isVector=True)
    jac_det_sitk = sitk.DisplacementFieldJacobianDeterminant(dis_xy_sitk)
    jac_det_sitk_np = sitk.GetArrayViewFromImage(jac_det_sitk)[0]

    return jac_det_sitk_np

### jacobian_3d using sitk ====================================================
def jac_det_sitk_3d(dis_x, dis_y, dis_z):
    dis_xyz = np.stack([dis_x, dis_y, dis_z], axis=-1)
    dis_xyz_sitk = sitk.GetImageFromArray(dis_xyz)
    jac_det_sitk = sitk.DisplacementFieldJacobianDeterminant(dis_xyz_sitk)
    jac_det_sitk_np = sitk.GetArrayViewFromImage(jac_det_sitk)

    return jac_det_sitk_np

def Fn_jac_all_frames(dis):
    Jac_all_frames = np.zeros((dis.shape[0], dis.shape[2], dis.shape[3], dis.shape[4]), dtype=dis.dtype)
    for i in range(dis.shape[0]): 
        Jac_each_frame = Fn_jac_det_calc_3d(dis[i,0,:,:,:], dis[i,1,:,:,:], dis[i,2,:,:,:])
        Jac_all_frames[i, :, :, :] = Jac_each_frame 
    return Jac_all_frames
