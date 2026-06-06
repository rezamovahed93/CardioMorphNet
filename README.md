# CardioMorphNet: Cardiac motion prediction using a shape-guided Bayesian recurrent deep network

**Official implementation of CardioMorphNet published in [Medical Image Analysis, 2026](https://www.sciencedirect.com/science/article/pii/S1361841526002185)**
> Reza Akbari Movahed, Abuzar Rezaee, Arezoo Zakeri, Colin Berry, Edmond S.L. Ho, Ali Gooya

## Overview
<p align="justify">
CardioMorphNet is a recurrent Bayesian deep learning framework for 3D cardiac 
shape-guided deformable registration using short-axis (SAX) cine CMR images. 
It employs a recurrent variational autoencoder (RVAE) to model spatio-temporal 
dependencies across the cardiac cycle, coupled with two posterior models for 
bi-ventricular segmentation and motion estimation.CardioMorphNet avoids 
intensity-based image registration similarity losses by recursively registering 
segmentation maps, guiding the framework to focus on anatomical cardiac regions. 
The Bayesian formulation further enables uncertainty map computation for estimated 
motion fields, providing confidence measures for predictions. Validated on the 
UK Biobank and M&M datasets, CardioMorphNet outperforms state-of-the-art methods 
in cardiac motion estimation and clinical indices extraction.
</p>

## Data
<p align="justify">
This repository uses the M&M (Multi-Centre, Multi-Vendor & Multi-Disease) dataset (https://www.ub.edu/mnms/). 
To reproduce the experiments, please download the dataset from the link above. 
</p>

## Installation
The repository is implemented in Python 3.10. For the most compatibility, please use this version and run the commands below to install the required packages: 


```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
conda create --name CMorhNet python=3.10
conda activate CMorhNet
pip install -r requirments.txt
```

## Usage

```bash
python main_preprocess.py ...
python split_csvs.py
python train_code_seg.py
python test_code_seg.py
python vis_results_seg_only.py
python eval_cal_seg_only.py 
python train_code.py
python test_code.py
python vis_results.py
python eval_cal_csv.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{movahed2025cardiomorphnet,
  title={CardioMorphNet: Cardiac Motion Prediction Using a Shape-Guided Bayesian Recurrent Deep Network},
  author={Movahed, Reza Akbari and Rezaee, Abuzar and Zakeri, Arezoo and Berry, Colin and Ho, Edmond SL and Gooya, Ali},
  journal={Medical Image Analysis},
  year={2026}
}
```



