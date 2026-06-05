#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.parameter_parser import parse_args_preprocess as parse_args
from utils.preprocess_pack import preprocess_fun


def create_clean_dir(path: str):
    """Create an empty directory (delete if exists). Safe when called on distinct paths."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_preprocess_and_save(seq_data_path, mask_data_path, ED_idx, ES_idx, zero_pad_bool, saving_path):
    seq_data = nib.load(seq_data_path).get_fdata()
    mask_data = nib.load(mask_data_path).get_fdata()

    seq_data, mask_data, es_ed_idxs = preprocess_fun(
        seq_data,
        mask_data,
        ED_idx,
        ES_idx,
        zero_pad_bool
    )

    data = {'seqs': seq_data, 'masks': mask_data, 'es_ed_idxs': es_ed_idxs}

    pikle_data_path = os.path.join(saving_path, 'data.pkl')
    with open(pikle_data_path, 'wb') as f:
        pickle.dump(data, f)


def process_one_patient(patient_id: str,
                        data_dir: str,
                        out_root: str,
                        ed_es_map: dict,
                        zero_pad_bool: bool):
    """
    Worker function (runs in a separate process).
    Returns (patient_id, status, message).
    """
    try:
        sa_dir = os.path.join(data_dir, patient_id, f"{patient_id}_sa.nii.gz")
        sa_mask_dir = os.path.join(data_dir, patient_id, f"{patient_id}_sa_gt.nii.gz")

        if patient_id not in ed_es_map:
            return (patient_id, "FAIL", "No ED/ES found for this External code in the CSV.")

        ED_idx, ES_idx = ed_es_map[patient_id]

        # Output directory per patient (safe: unique per task)
        patient_out_dir = os.path.join(out_root, patient_id)
        create_clean_dir(patient_out_dir)

        read_preprocess_and_save(
            sa_dir,
            sa_mask_dir,
            ED_idx,
            ES_idx,
            zero_pad_bool,
            patient_out_dir
        )

        return (patient_id, "OK", "Finished")
    except Exception as e:
        return (patient_id, "FAIL", repr(e))


if __name__ == "__main__":
    args = parse_args()

    # Load dataset info CSV ONCE
    df_info = pd.read_csv(args.csv_paths)

    # Build a fast lookup: External code -> (ED, ES)
    ed_es_map = {}
    df_tmp = df_info.dropna(subset=["External code", "ED", "ES"])
    for _, r in df_tmp.iterrows():
        ed_es_map[str(r["External code"])] = (int(r["ED"]), int(r["ES"]))

    suffixes = ["Training", "Validation", "Testing"]

    # Create root output folder fresh (serial, once)
    create_clean_dir(args.preprocessed_data_path)

    zero_pad_bool = bool(args.zero_pad_flag)

    max_workers = getattr(args, "num_workers", None)
    if not max_workers:
        max_workers = os.cpu_count() or 4

    for each_suffix_set in suffixes:
        if each_suffix_set == "Training":
            data_dir = os.path.join(args.data_path, each_suffix_set, "Labeled")
        else:
            data_dir = os.path.join(args.data_path, each_suffix_set)

        patients_folders = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]

        # Initial patients list for this split (before filtering)
        df_patients = pd.DataFrame(patients_folders, columns=["Patient_ID"])

        print(f"\n=== {each_suffix_set}: {len(patients_folders)} patients | workers={max_workers} ===")

        failures = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    process_one_patient,
                    patient_id,
                    data_dir,
                    args.preprocessed_data_path,
                    ed_es_map,
                    zero_pad_bool
                )
                for patient_id in patients_folders
            ]

            for fut in as_completed(futures):
                patient_id, status, msg = fut.result()
                if status == "OK":
                    print(f"[OK]   {patient_id}")
                else:
                    print(f"[FAIL] {patient_id} -> {msg}")
                    failures.append((patient_id, msg))

        # -------------------- ### NEW/CHANGED: remove failures from dataframes --------------------
        failed_ids = {pid for pid, _ in failures}

        # 1) Save failures (same as your code, but kept here for context)
        if failures:
            fail_csv = os.path.join(args.preprocessed_data_path, f"{each_suffix_set}_failures.csv")
            pd.DataFrame(failures, columns=["Patient_ID", "Reason"]).to_csv(fail_csv, index=False)
            print(f"Saved failures to: {fail_csv}")

        # 2) Filter df_patients and save a *clean* patients list
        df_patients_clean = df_patients[~df_patients["Patient_ID"].isin(failed_ids)].reset_index(drop=True)

        # Save both (optional): original + clean
        df_patients.to_csv(f"{each_suffix_set}_patients_all.csv", index=False)      # before filtering
        df_patients_clean.to_csv(f"{each_suffix_set}_patients.csv", index=False)   # after filtering (clean)
        print(f"Saved clean patients CSV (failures removed): {each_suffix_set}_patients.csv "
              f"({len(df_patients_clean)}/{len(df_patients)})")

        # 3) (Optional) Also filter df_info for this split and save a split-specific info CSV
        #    This assumes Patient_ID == External code.
        df_info_clean_split = df_info[~df_info["External code"].astype(str).isin(failed_ids)].reset_index(drop=True)
        info_clean_path = os.path.join(args.preprocessed_data_path, f"{each_suffix_set}_df_info_clean.csv")
        df_info_clean_split.to_csv(info_clean_path, index=False)
        print(f"Saved split-specific df_info with failures removed: {info_clean_path}")
        # -----------------------------------------------------------------------------------------
