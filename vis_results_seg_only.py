#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np

from utils.Vis_pack import all_visulize_seg_only


def obt_zero_slices(padded_array):
    """
    Count the number of slices that contain only zeros.
    """
    slices_with_only_zeros = np.all(padded_array == 0, axis=(0, 1, 2))
    return np.sum(slices_with_only_zeros)


def main(args):

    results_filenames = os.listdir(args.results_path)

    if len(results_filenames) == 0:
        raise ValueError(f"No files found in {args.results_path}")

    random_patient = np.random.randint(0, len(results_filenames))

    random_patient_file = os.path.join(
        args.results_path,
        results_filenames[random_patient]
    )

    with open(random_patient_file, "rb") as file:
        patient_result = pickle.load(file)

    img_org = patient_result["Seqs_in"].squeeze(1)
    seg_ref = patient_result["Segs_in"]
    seg_out = patient_result["Segs_out"]

    slice_dx = (
        img_org.shape[-1] - obt_zero_slices(img_org)
    ) // 2

    os.makedirs(args.saving_dir, exist_ok=True)

    save_path = os.path.join(
        args.saving_dir,
        "seg_sample_result.png"
    )

    all_visulize_seg_only(
        img_org,
        seg_out,
        seg_ref,
        slice_dx=slice_dx,
        label_title="Proposed Method Segmentation Results",
        saving_dir=save_path,
    )

    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize segmentation results for a random patient."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Directory containing pickled result files.",
    )

    parser.add_argument(
        "--saving_dir",
        type=str,
        required=True,
        help="Directory where the visualization will be saved.",
    )

    args = parser.parse_args()

    main(args)
