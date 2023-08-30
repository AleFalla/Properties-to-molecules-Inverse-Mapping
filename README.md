# Properties-to-molecules-Inverse-Mapping

This repository contains the code for the paper titled: **Enabling Inverse Design in Chemical Compound Space: Mapping Quantum Properties to Structures for Small Organic Molecules**

## Description

This repository provides the code to reproduce the main results from the paper. The code is organized into various scripts and notebooks. The variable `reproduce_paper` is used in multiple scripts to automatically locate the `data_paper` folder.

## Main Packages

The main packages to run scripts and notebooks are numpy, scipy, scikit-learn, jupyter, openbabel, pandas, pytorch, pytorch-lightning and rmsd.

## Data

The data used for training and testing in the paper can be downloaded [here](https://drive.google.com/file/d/19r1UIPgTiZCVxR-o2u-EYe-H-puyggSn/view?usp=sharing) (zip folder). The relevant data is located in the `data_paper` directory. To use the data, place the `data_paper` folder in the same directory as the notebooks and scripts. New data can be prepared using the `initialize_data.py` script, which needs to be modified as required.

## Models

The model architectures are defined in the `models_old.py` script (alternatively, `models.py` for testing alternatives). The PyTorch Lightning model definition is provided in the `Model.py` file. Pre-trained models are available in the `models_saved` folder.

## Notebooks

- `testing.ipynb`: Reproduces the main results of the paper.
- `mol_gen_test.ipynb`: Demonstrates targeted molecule generation using functions from `molecular_generation_utils.py`.

## Miscellaneous

Other scripts serve as utilities for various applications. The code related to the geodesic search and interpolation for the NEB part is being prepared.

## General Remarks

While the code could be better organized and structured, the current organization serves the purpose of scientifically presenting and prototyping the novel methodology outlined in the paper.

**Disclaimer:** This repository provides the code associated with the paper, and additional steps may be required to reproduce the results fully.

## About

No description, website, or topics provided.

## Resources

- [Paper Link](#) (Provide a link to the published paper if available)
- [Download Data](https://drive.google.com/file/d/19r1UIPgTiZCVxR-o2u-EYe-H-puyggSn/view?usp=sharing)
