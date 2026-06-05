import os 
import pickle
import numpy as np 
from utils.Vis_pack import Result_show, Visualize_SeqOrd, result_show_sep_with_DVFs, all_visulize, all_visulize_seg, all_visulize_without_abserr, all_visulize_with_gt, saving_results_per_cat
from utils.jac_det import Fn_jac_all_frames
import argparse
import shutil

def create_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        os.mkdir(directory_path)
    else:
        os.mkdir(directory_path)

def obt_zero_slices(padded_array):    
    # Check if all values in a slice along the last axis (dim=2) are zero
    slices_with_only_zeros = np.all(padded_array == 0, axis=(0, 1, 2))
    # Count the number of slices that are all zeros
    num_zero_slices = np.sum(slices_with_only_zeros)
    return num_zero_slices
    

def arg_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--results_path", type=str, help="The results path", default="./results")
    parser.add_argument("--saving_dir", type=str, help="The results path", default="./vis results")


    args = parser.parse_args()
    return args 
    
if __name__ == "__main__":
    
    args = arg_parser()
    
    saving_dir = args.saving_dir
    create_directory(saving_dir)

    results_link = args.results_path
    results_filenames = os.listdir(results_link)
    random_patient = np.random.randint(0, len(results_filenames)-1)

    random_patient = 10

    print(random_patient)
    
    random_patient_file = os.path.join(results_link, results_filenames[random_patient])
    
    with open(random_patient_file, 'rb') as file:
        patient_result = pickle.load(file)
        
    img_org = patient_result['Seqs'].squeeze(1)
    img_def = patient_result['Seqs_def'].squeeze(1)
    disp = patient_result['Disps']
    seg_ref = patient_result['Segs_refs']
    seg_def = patient_result['Segs_def']
    es_ed_idxs = patient_result['ES_ED_Idxs']
    mask_org = patient_result['Segs_in']
    mask_def = patient_result['Masks_def']
    jac_det = Fn_jac_all_frames(disp)
    # es_ed_idxs = [0,1]
    
    slice_dx = (img_org.shape[-1]-obt_zero_slices(img_org))//2
    
    all_visulize_with_gt(img_org, img_def, seg_ref, 
                      seg_def, mask_org, mask_def, disp, 
                      jac_det, slice_dx=slice_dx, saving_dir= os.path.join(saving_dir, "output_with_gt.png"), 
                      label_title= "Proposed Method Results", 
                      label_out= "Registred")
    
    
    all_visulize(img_org, img_def, seg_ref, 
                      seg_def, disp, 
                      jac_det, slice_dx=slice_dx, saving_dir= os.path.join(saving_dir, "output_without_gt.png") , 
                      label_title= "Proposed Method Results", 
                      label_out= "Registred")
    
    all_visulize_without_abserr(img_org, img_def, seg_ref, 
                      seg_def, disp, 
                      jac_det, slice_dx=slice_dx, saving_dir= os.path.join(saving_dir, "output_without_gt_abserr.png") , 
                      label_title= "Proposed Method Results", 
                      label_out= "Registred")
    
    all_visulize_seg(img_org, 
                     seg_ref, 
                     mask_org,
                     seg_def,
                     es_ed_idxs,
                     slice_dx=slice_dx, label_title="Proposed Method Results", 
                     saving_dir= os.path.join(saving_dir, "output_seg_based.png"))
    
    saving_results_per_cat(img_org, img_def, seg_ref, 
                           seg_def, disp, jac_det, mask_org, es_ed_idxs, slice_dx=slice_dx, 
                            saving_lik = saving_dir)