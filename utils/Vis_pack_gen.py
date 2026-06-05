# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:54:40 2024

@author: reza
"""
import imageio
import io 
import numpy as np 
import time as TT
import matplotlib.pyplot as plt 
from skimage import measure
from utils.Colormap_DVFs import com_colormaps_DVFs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import os

def Visualize_SeqOrd(image_data, delay=0.5, cmap='gray'):
    for i in range(image_data.shape[2]):
        fig1, axes1 = plt.subplots(1, 1, figsize=(15, 6))
        image_slice = image_data[:, :, i]
        axes1.imshow(image_slice, cmap='gray')
        axes1.set_title('Slice {}'.format(i + 1))
        axes1.axis('off')
        plt.show()
        plt.close(fig1)
        TT.sleep(delay)
        

def Visualize_SeqOrd_BothIG(image_data, mask_data, delay=0.5, cmap='gray', Colors = {1:'red', 2:'blue', 3:'green'}):
    # Visualise the MRI scan of patient004 with its ground-truth in the sequence manner
    mask_data = mask_data.astype('uint8')
    for i in range(image_data.shape[2]):
        fig1, axes1 = plt.subplots(1, 1, figsize=(15, 6))
        image_slice = image_data[:, :, i]
        mask_slice = mask_data[:,:,i]
        axes1.imshow(image_slice, cmap='gray')
        axes1.set_title('Slice {}'.format(i + 1))
        axes1.axis('off')
        Class_labels = list(set(mask_slice.flatten()))
        for each_label in Class_labels[1:]:
            Mask_each_class = mask_slice == each_label
            Mask_each_class = Mask_each_class.astype('uint8')
            contours = measure.find_contours(Mask_each_class)
            for contour in contours:
                axes1.plot(contour[:, 1], contour[:, 0], linewidth=2, color= Colors[each_label])
        plt.show()
        plt.close(fig1)
        TT.sleep(delay)
        
        
def result_show_sep_with_DVFs(X, Y, S, M, D, slice_idx, saving_gif=True, saving_dir='subplot_animation_1.gif'): 
    if saving_gif: 
        frames = []
    frame_num = X.shape[0]
    D = com_colormaps_DVFs(D)
    for t in range(frame_num):
        I_t = X[t, :, :, slice_idx]
        I_t_prime = Y[t, :, :, slice_idx]
        M_t = M[t, [1,2,3] , :, :, slice_idx].transpose((1,2,0)) 
        S_t = S[t, [1,2,3] , :, :, slice_idx].transpose((1,2,0)) 

        D_t = D[t,:,:,:,slice_idx]
        
        fig, axs = plt.subplots(1, 5, figsize=(20, 16))
    
        axs[0].imshow(I_t, cmap='gray')
        axs[0].set_title(f'$I_{{{t+1}}}$', fontsize=18)
        axs[0].axis('off')
        
        axs[1].imshow(I_t_prime, cmap='gray')
        axs[1].set_title(f'$\hat{{I}}_{{{t+1}}}$', fontsize=18)
        axs[1].axis('off')
        
        axs[2].imshow(M_t)
        axs[2].set_title(f'$M_{{{t+1}}}$', fontsize=18)
        axs[2].axis('off')
        
        axs[3].imshow(S_t)
        axs[3].set_title(f'$\hat{{M}}_{{{t+1}}}$', fontsize=18)
        axs[3].axis('off')
        
        axs[4].imshow(D_t)
        axs[4].set_title(f'$D_{{{t+1}}}$', fontsize=18)
        axs[4].axis('off')
        
        plt.tight_layout()
        TT.sleep(0.7)
        if saving_gif: 
            # Save the current plot as a frame
            buf = io.BytesIO()
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close()
        plt.show()
    if saving_gif: 
        imageio.mimsave(saving_dir, frames, duration=500)
        
def Result_show(X, Y, S, M, D, J, slice_idx, saving_img=True, saving_dir='plot.png', format='png'):
    fig, axs = plt.subplots(7, 6, figsize=(25,16))
    D = D[:,:,:,:,slice_idx]
    D = com_colormaps_DVFs(D)
    #S = np.argmax(S, axis=1)
    #M = np.argmax(M, axis=1)
    abs_err = np.abs(X - Y)
    for x in range(7): 
        for y in range(6):
            if x==0: 
                axs[x][y].imshow(X[y, :, :, slice_idx], cmap='gray')
                axs[x][y].set_title(f'$I_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
            elif x==1: 
                axs[x][y].imshow(Y[y, :, :, slice_idx], cmap='gray')
                axs[x][y].set_title(f'$\hat{{I}}_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
            elif x==2: 
                axs[x][y].imshow(abs_err[y, :, :, slice_idx], cmap='gray')
                axs[x][y].set_title(f'|$I_{{{y+1}}} - \hat{{I}}_{{{y+1}}}$|', fontsize=14)
                axs[x][y].axis('off')
            elif x==3: 
                axs[x][y].imshow(M[y, [1,2,3] , :, :, slice_idx].transpose((1,2,0)))
                axs[x][y].set_title(f'$M_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
            elif x==4: 
                axs[x][y].imshow(S[y, [1,2,3] , :, :, slice_idx].transpose((1,2,0)))
                axs[x][y].set_title(f'$\hat{{M}}_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
            elif x==5: 
                axs[x][y].imshow(D[y,:,:,:], vmin=0, vmax=255)
                axs[x][y].set_title(f'$D_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
            else:
                axs[x][y].imshow(J[y, :, :, slice_idx], cmap='gray')
                axs[x][y].set_title(f'$J_{{{y+1}}}$', fontsize=14)
                axs[x][y].axis('off')
    
    fig.savefig(saving_dir, format=format, dpi=200)
    plt.tight_layout()
    plt.show()
    plt.close()
    
def all_visulize_gen(img_target, img_moved, predicted_mask, 
                 moved_mask, displacement, 
                 jacobian, slice_dx, saving_dir, format='png',
                 label_title= "Registration", 
                 label_out= "Registred"):
    
    img_target = img_target[:, : ,: , slice_dx]
    img_moved = img_moved[:, : ,: , slice_dx]
    predicted_mask = predicted_mask[:, :, : ,: , slice_dx]
    moved_mask = moved_mask[:, :, : ,: , slice_dx]
    jacobian = jacobian[:, :, :, slice_dx]
    displacement = displacement[:,:,:,:,slice_dx]
    
    displacement_colored = com_colormaps_DVFs(displacement)
    # displacement_colored = displacement_colored[:, :, :, :, slice_dx]
    
    fontsize = 14
    
    frame_size, w_img, h_img = img_target.shape 
    
    fig, axs = plt.subplots(7, frame_size, figsize=(19, 21))
    # fig.suptitle(label_title, fontweight="bold", fontsize= 14)
    
    
    
    abs_err = np.abs(img_target - img_moved)
    
    for i in range(frame_size):
        
        #-----------------------------------
        ax = axs[0,i]
        ax.imshow(img_target[i], cmap= "gray", vmin=0, vmax=1)
        
        if i == 0:
            ax.set_title("0 (ED)", fontsize=fontsize)
            ax.set_ylabel("Original", fontsize=fontsize)
        elif i == 3:
            ax.set_title("3 (ES)", fontsize=fontsize)
        else:
            ax.set_title(str(i), fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        ax = axs[1,i]
        ax.imshow(img_moved[i], cmap= "gray")
        
        if i == 0:
            ax.set_ylabel(label_out, fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        ax = axs[2,i]
        ax.imshow(abs_err[i], cmap= "gray", vmin=0, vmax=1)
        
        if i == 0:
            ax.set_ylabel("Absolute Error", fontsize=fontsize)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[3,i]
        ax.imshow(predicted_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        if i == 0:
            ax.set_ylabel("Predicted Mask", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[4,i]
        ax.imshow(moved_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        if i == 0:
            ax.set_ylabel("Moved Predicted Mask", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[5,i]
        ax.imshow(displacement_colored[i], vmin=0, vmax=255)
        
        if i == 0:
            ax.set_ylabel("Deformation Feild", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        colors1 = plt.cm.gray(np.linspace(1.0, 0.0, 128))
        colors2 = plt.cm.cool(np.linspace(0., 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
    
        ax = axs[6,i]
        pcm = ax.imshow(jacobian[i], cmap=mymap, vmin=-2, vmax=2)
        
        divider = make_axes_locatable(ax)
        
        if i == frame_size-1:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax= cax)
        
        if i == 0:
            ax.set_ylabel("Jac. Det.", fontsize=fontsize)
            
        
        ax.axis([0, w_img, 0, h_img])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.align_labels()
    plt.tight_layout()
    plt.show()
    fig.savefig(saving_dir, format=format, dpi=200)
    plt.close()

def all_visulize_without_abserr(img_target, img_moved, predicted_mask, 
                 moved_mask, displacement, 
                 jacobian, slice_dx, saving_dir, format='png',
                 label_title= "Registration", 
                 label_out= "Registred"):
    
    img_target = img_target[:, : ,: , slice_dx]
    img_moved = img_moved[:, : ,: , slice_dx]
    predicted_mask = predicted_mask[:, :, : ,: , slice_dx]
    moved_mask = moved_mask[:, :, : ,: , slice_dx]
    jacobian = jacobian[:, :, :, slice_dx]
    
    displacement = displacement[:,:,:,:,slice_dx]
    displacement_colored = com_colormaps_DVFs(displacement)
    
    fontsize = 14
    
    frame_size, w_img, h_img = img_target.shape 
    
    fig, axs = plt.subplots(6, frame_size, figsize=(19, 21))
    # fig.suptitle(label_title, fontweight="bold", fontsize= 14)
    
    
    
    
    for i in range(frame_size):
        
        #-----------------------------------
        ax = axs[0,i]
        ax.imshow(img_target[i], cmap= "gray", vmin=0, vmax=1)
        
        if i == 0:
            ax.set_title("0 (ED)", fontsize=fontsize)
            ax.set_ylabel("Original", fontsize=fontsize)
        elif i == 3:
            ax.set_title("3 (ES)", fontsize=fontsize)
        else:
            ax.set_title(str(i), fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        ax = axs[1,i]
        ax.imshow(img_moved[i], cmap= "gray", vmin=0, vmax=1)
        
        if i == 0:
            ax.set_ylabel(label_out, fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[2,i]
        ax.imshow(predicted_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        if i == 0:
            ax.set_ylabel("Predicted Mask", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[3,i]
        ax.imshow(moved_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        if i == 0:
            ax.set_ylabel("Moved Predicted Mask", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[4,i]
        ax.imshow(displacement_colored[i], vmin=0, vmax=255)
        
        if i == 0:
            ax.set_ylabel("Deformation Feild", fontsize=fontsize)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        colors1 = plt.cm.gray(np.linspace(1.0, 0.0, 128))
        colors2 = plt.cm.cool(np.linspace(0., 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
    
        ax = axs[5,i]
        pcm = ax.imshow(jacobian[i], cmap=mymap, vmin=-2, vmax=2)
        
        divider = make_axes_locatable(ax)
        
        if i == frame_size-1:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax= cax)
        
        if i == 0:
            ax.set_ylabel("Jac. Det.", fontsize=fontsize)
            
        
        ax.axis([0, w_img, 0, h_img])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.align_labels()
    plt.tight_layout()
    plt.show()
    fig.savefig(saving_dir, format=format, dpi=200)
    plt.close()

    
def all_visulize_seg(img_target, predicted_mask, ground_truth, 
                 moved_mask, ES_ED_idx, slice_dx, saving_dir, label_title, format='png'):
    
    img_target = img_target[ES_ED_idx, : ,: , slice_dx]
    predicted_mask = predicted_mask[ES_ED_idx, :, : ,: , slice_dx]
    moved_mask = moved_mask[ES_ED_idx, :, : ,: , slice_dx]
    ground_truth = ground_truth[:, :, : ,: , slice_dx]
    
    frame_size, w_img, h_img = img_target.shape 
    
    fig, axs = plt.subplots(4, frame_size, figsize=(19,12.5))
    fig.suptitle(label_title, fontweight="bold", fontsize= 14)
        
    for i in range(frame_size):
        
        #-----------------------------------
        ax = axs[0,i]
        ax.imshow(img_target[i], cmap= "gray", vmin=0, vmax=1)
        
        if i == 0:
            ax.set_title("ED")
            ax.set_ylabel("Original")
        else:
            ax.set_title("ES")

            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
                
        ax = axs[1,i]
        ax.imshow(ground_truth[i][[1,2,3],:,:].transpose((1,2,0))*255)
        
        if i == 0:
            ax.set_ylabel("Ground truth")
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[2,i]
        ax.imshow(predicted_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        if i == 0:
            ax.set_ylabel("Predicted Mask")
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
        
        ax = axs[3,i]
        ax.imshow(moved_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        
        
        if i == 0:
            ax.set_ylabel("Moved Predicted Mask")
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        #-----------------------------------
      
        
        
    fig.align_labels()
    plt.tight_layout()
    plt.show()
    fig.savefig(saving_dir, format=format, dpi=200)
    plt.close()
    
def saving_results_per_cat(img_target, img_moved, predicted_mask, moved_mask, 
                           displacement, jacobian, slice_dx, saving_lik):
    
    os.mkdir(os.path.join(saving_lik, 'results'))
    saving_lik = os.path.join(saving_lik, 'results')
    os.mkdir(os.path.join(saving_lik,'Jac'))
    os.mkdir(os.path.join(saving_lik,'DVF'))
    os.mkdir(os.path.join(saving_lik,'Img'))
    os.mkdir(os.path.join(saving_lik,'Pre Img'))
    os.mkdir(os.path.join(saving_lik,'Pre Mask'))
    os.mkdir(os.path.join(saving_lik,'Moved Pre Mask'))
    os.mkdir(os.path.join(saving_lik,'Abs err'))
    
    colors1 = plt.cm.gray(np.linspace(1.0, 0.0, 128))
    colors2 = plt.cm.cool(np.linspace(0., 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
    
    img_target = img_target[:, : ,: , slice_dx]
    img_moved = img_moved[:, : ,: , slice_dx]
    predicted_mask = predicted_mask[:, :, : ,: , slice_dx]
    moved_mask = moved_mask[:, :, : ,: , slice_dx]
    jacobian = jacobian[:, :, :, slice_dx]
    displacement = displacement[:,:,:,:,slice_dx]
    
    frame_size, w_img, h_img = img_target.shape 
    
    displacement_colored = com_colormaps_DVFs(displacement)
    # displacement_colored = displacement_colored[:, :, :, :, slice_dx]
    
    abs_err = np.abs(img_target - img_moved)
    
    for i in range(frame_size):
        
        plt.imshow(img_target[i], cmap= "gray", vmin=0, vmax=1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Img', f"I{i}.png"),  bbox_inches='tight')

        
        plt.close()
        #----------------------------------------------------
        
        plt.imshow(img_moved[i], cmap= "gray", vmin=0, vmax=1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Pre Img', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
        #----------------------------------------------------
        
        plt.imshow(abs_err[i], cmap= "gray", vmin=0, vmax=1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Abs err', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
        #----------------------------------------------------
        
        plt.imshow(predicted_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Pre Mask', f"I{i}.png" ),  bbox_inches='tight')
        
        plt.close()
        
        #----------------------------------------------------
        
        plt.imshow(moved_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Moved Pre Mask', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
        
        #----------------------------------------------------
        
        plt.imshow(moved_mask[i][[1,2,3],:,:].transpose((1,2,0)))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Moved Pre Mask', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
        #----------------------------------------------------
        
        plt.imshow(displacement_colored[i], vmin=0, vmax=255)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'DVF', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
        
        #----------------------------------------------------
        
        plt.imshow(jacobian[i], cmap=mymap, vmin=-2, vmax=2)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        
        plt.savefig(os.path.join(saving_lik,'Jac', f"I{i}.png"),  bbox_inches='tight')
        
        plt.close()
    
    # Create a new figure for the colorbar
    fig_colorbar, ax_colorbar = plt.subplots(figsize=(2, 5))  # Adjust size as needed
    
    # Create a ScalarMappable with the colormap and the normalization
    norm = mcolors.Normalize(vmin=-2, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=mymap, norm=norm)
    sm.set_array([])
    
    # Add the colorbar to the new figure
    fig_colorbar.colorbar(sm, cax=ax_colorbar)
    
    # Save the colorbar to a separate file
    fig_colorbar.savefig(os.path.join(saving_lik,'separate_colorbar.png'), bbox_inches='tight')