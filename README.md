# Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design

# Introduction
This repository provides the source codes associated with the paper Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design.

# Environment and External Tool Setup
- Create the primary environment: ```conda env create -f environment.yml```
- Create the environment for chemprop: ```conda env create -f chemprop.yml```
- Install QuickVina2-GPU-2.1: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1

# Data Source
Molecules (download molecular data in sdf format):
- QM9: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
- ZINC15: https://zinc15.docking.org/tranches/home/#
- PubChem: https://pubchem.ncbi.nlm.nih.gov/#query=small%20molecule&tab=compound

Protein:
- 6VHN: https://www.rcsb.org/structure/6VHN

# Uncertainty Prediction
- Install ```chemprop```: https://github.com/chemprop/chemprop
- ```cd uncertainty```
- Train surrogate models: ```./train_surrogates.sh [datasetName] [property]```

# Diffusion Models
- ```cd scripts```
- Train diffusion models: ```./a_run_zz_train.sh [dataset] [removeH] [batchsize] [n_layers] [nf] [n_epoch] [CUDA_DEVICES] [properties]```
- Optimize diffusion models: ```././a_run_zz_optimize.sh```
- Generate molecules and evaluate: ```./a_run_zz_evaluate.sh [exp_name]```




