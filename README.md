# Properties-to-molecules-Inverse-Mapping

This repository contains the code for the paper titled: [**Inverse mapping of quantum properties to structures for chemical space of small organic molecules**](https://www.nature.com/articles/s41467-024-50401-1)

## Description

This repository provides the code to reproduce the main results from the paper. The code is organized into various scripts and notebooks. The variable `reproduce_paper` is used in multiple scripts to automatically locate the `data_paper` folder. 
To train the model using the train.py script it usually takes around 3 hours (depending on your GPU). The notebook reproducing the main results should run in a few minutes, excluding the computation of RMSDs for the test set (depending on test set size) which can take longer.

## Main Packages

The main packages to run scripts and notebooks are reported here, together with the version we tested on:

- ase                         3.22.0
- matplotlib                  3.5.0
- numpy                       1.21.4
- pandarallel                 1.6.4
- pandas                      1.3.4
- pyarrow                     7.0.0
- pytorch-lightning           1.5.10
- rmsd                        1.4
- scipy                       1.7.3
- torch                       1.12.1
- tqdm                        4.62.3
- openbabel                   3.1.1

A lot of the code can be run without openbabel, dftb+ or machine learning force fields. These packages are needed though in order to add hydrogens and relax geometries. For a simple installation and use of a force field, any force field that can be used within the ase framework will do, for the one used in the work we refer to [SpookyNet](https://github.com/OUnke/SpookyNet). For what concerns openbabel we reccomend using a conda environment.

The installation of the main packages should take a few minutes on standard hardware.

## Data

The data used for training and testing in the paper can be downloaded [here](https://drive.google.com/file/d/19r1UIPgTiZCVxR-o2u-EYe-H-puyggSn/view?usp=sharing) (zip folder). The relevant data is located in the `data_paper` directory. To use the data, place the `data_paper` folder in the same directory as the notebooks and scripts. New data can be prepared using the `initialize_data.py` script, which needs to be modified as required.

## Models

The model architectures are defined in the `models_old.py` script (alternatively, `models.py` for testing alternatives). The PyTorch Lightning model definition is provided in the `Model.py` file. Pre-trained models are available in the `models_saved` folder.

## Notebooks

- `testing.ipynb`: Reproduces the main results of the paper.
- `mol_gen_test.ipynb`: Demonstrates targeted molecule generation using functions from `molecular_generation_utils.py`.

## Miscellaneous

Other scripts serve as utilities for various applications. For the interpolation we have here a script called interpolator.py which implements the procedure used in the paper. For the NEB part there is a notebook called NEB_interp.ipynb, please change the SpookyNet chackpoint (or force field) to what you want to use.

## General Remarks

While the code could be better organized and structured, the current organization serves the purpose of scientifically presenting and prototyping the novel methodology outlined in the paper.

## Resources

- [Paper Link](#) (https://www.nature.com/articles/s41467-024-50401-1)
- [Download Data](https://www.dropbox.com/scl/fi/mnhm3fua5tl01dre0t4j0/dati.zip?rlkey=b7ah86h3jrgcshexgokxnjl3d&st=db6uk77m&dl=0)
