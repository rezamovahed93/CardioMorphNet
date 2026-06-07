import argparse
import os
import pickle
import numpy as np
from skimage.metrics import hausdorff_distance

from utils.eval_metrics import (
    dice_cal,
    compute_sensitivity,
    jaccard_score,
    compute_false_discovery_rate,
)


def dice_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    dice_all = []
    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding:
            x_pre = (x_pre >= 0.5).astype("uint8")

        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]

        dice_all.append(dice_cal(x_tar, x_pre))

    return np.asarray(dice_all)


def hausdorff_distance_all(
    Mask_target,
    Mask_predicted,
    label,
    thresholding=False,
    mode="standard",
):
    hd_all = []

    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding:
            x_pre = (x_pre >= 0.5).astype("uint8")

        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]

        hd_all.append(hausdorff_distance(x_tar, x_pre, mode))

    return np.asarray(hd_all)


def jaccarod_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    jac_all = []

    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding:
            x_pre = (x_pre >= 0.5).astype("uint8")

        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]

        jac_all.append(jaccard_score(x_tar, x_pre))

    return np.asarray(jac_all)


def sensitivity_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    se_all = []

    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding:
            x_pre = (x_pre >= 0.5).astype("uint8")

        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]

        se_all.append(compute_sensitivity(x_tar, x_pre))

    return np.asarray(se_all)


def fdr_for_all(Mask_target, Mask_predicted, label, thresholding=False):
    fdr_all = []

    for x_tar, x_pre in zip(Mask_target, Mask_predicted):
        if thresholding:
            x_pre = (x_pre >= 0.5).astype("uint8")

        x_pre = x_pre[:, label, :, :, :]
        x_tar = x_tar[:, label, :, :, :]

        fdr_all.append(compute_false_discovery_rate(x_tar, x_pre))

    return np.asarray(fdr_all)


def main(args):

    results_filenames = os.listdir(args.results_path)

    if len(results_filenames) == 0:
        raise ValueError(f"No result files found in {args.results_path}")

    Segs_refs = []
    Segs_outs = []

    for each_filename in results_filenames:

        patient_file = os.path.join(
            args.results_path,
            each_filename,
        )

        with open(patient_file, "rb") as file:
            patient_result = pickle.load(file)

        Segs_refs.append(patient_result["Segs_in"])
        Segs_outs.append(patient_result["Segs_out"])

    # LV
    dice_all_LV_seg = dice_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=1
    )
    jac_all_LV_seg = jaccarod_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=1
    )
    se_all_LV_seg = sensitivity_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=1
    )
    fdr_all_LV_seg = fdr_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=1
    )
    hd_all_LV_seg = hausdorff_distance_all(
        Segs_refs,
        Segs_outs,
        thresholding=True,
        label=1,
        mode="standard",
    )

    # Myo
    dice_all_Myo_seg = dice_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=2
    )
    jac_all_Myo_seg = jaccarod_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=2
    )
    se_all_Myo_seg = sensitivity_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=2
    )
    fdr_all_Myo_seg = fdr_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=2
    )
    hd_all_Myo_seg = hausdorff_distance_all(
        Segs_refs,
        Segs_outs,
        thresholding=True,
        label=2,
        mode="standard",
    )

    # RV
    dice_all_RV_seg = dice_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=3
    )
    jac_all_RV_seg = jaccarod_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=3
    )
    se_all_RV_seg = sensitivity_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=3
    )
    fdr_all_RV_seg = fdr_for_all(
        Segs_refs, Segs_outs, thresholding=True, label=3
    )
    hd_all_RV_seg = hausdorff_distance_all(
        Segs_refs,
        Segs_outs,
        thresholding=True,
        label=3,
        mode="standard",
    )

    print("\n================ LV =================")
    print(f"Dice        : {np.mean(dice_all_LV_seg)*100:.2f} ± {np.std(dice_all_LV_seg)*100:.2f}")
    print(f"Jaccard     : {np.mean(jac_all_LV_seg)*100:.2f} ± {np.std(jac_all_LV_seg)*100:.2f}")
    print(f"Sensitivity : {np.mean(se_all_LV_seg)*100:.2f} ± {np.std(se_all_LV_seg)*100:.2f}")
    print(f"FDR         : {np.mean(fdr_all_LV_seg)*100:.2f} ± {np.std(fdr_all_LV_seg)*100:.2f}")
    print(f"Hausdorff   : {np.mean(hd_all_LV_seg):.4f} ± {np.std(hd_all_LV_seg):.4f}")

    print("\n================ MYO =================")
    print(f"Dice        : {np.mean(dice_all_Myo_seg)*100:.2f} ± {np.std(dice_all_Myo_seg)*100:.2f}")
    print(f"Jaccard     : {np.mean(jac_all_Myo_seg)*100:.2f} ± {np.std(jac_all_Myo_seg)*100:.2f}")
    print(f"Sensitivity : {np.mean(se_all_Myo_seg)*100:.2f} ± {np.std(se_all_Myo_seg)*100:.2f}")
    print(f"FDR         : {np.mean(fdr_all_Myo_seg)*100:.2f} ± {np.std(fdr_all_Myo_seg)*100:.2f}")
    print(f"Hausdorff   : {np.mean(hd_all_Myo_seg):.4f} ± {np.std(hd_all_Myo_seg):.4f}")

    print("\n================ RV =================")
    print(f"Dice        : {np.mean(dice_all_RV_seg)*100:.2f} ± {np.std(dice_all_RV_seg)*100:.2f}")
    print(f"Jaccard     : {np.mean(jac_all_RV_seg)*100:.2f} ± {np.std(jac_all_RV_seg)*100:.2f}")
    print(f"Sensitivity : {np.mean(se_all_RV_seg)*100:.2f} ± {np.std(se_all_RV_seg)*100:.2f}")
    print(f"FDR         : {np.mean(fdr_all_RV_seg)*100:.2f} ± {np.std(fdr_all_RV_seg)*100:.2f}")
    print(f"Hausdorff   : {np.mean(hd_all_RV_seg):.4f} ± {np.std(hd_all_RV_seg):.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate segmentation results."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Directory containing segmentation result pickle files.",
    )

    args = parser.parse_args()

    main(args)
