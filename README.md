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

