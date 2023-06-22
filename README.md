Public code for the paper: 'Enabling Inverse Design in Chemical Compound Space: Mapping Quantum Properties to Structures for Small Organic Molecules'

Generic description:
Here is reported most of the code to reproduce the main results from the paper. In multiple scripts it is reported a variable 'reproduce_paper' which in general is used to automatically look for the data_paper folder.

Data:
The data used for training and testing in the paper can be downloaded with the proper splittings at the following link (zip folder):
https://drive.google.com/file/d/19r1UIPgTiZCVxR-o2u-EYe-H-puyggSn/view?usp=sharing
The relevant data can be retrieve under the directory dati/data_paper/. The folder data_paper needs to be pasted in the same directory as the notebooks and scripts to properly work.
For new data, this needs to be prepared using the script initialize_data.py which will need to be modified accordingly. In particular one needs to prepare a json file and also report the right property names in the same file. Once well prepared initialize data will do the rest by creating a data folder and adjusting elements, coulomb matrices and padding.

Models:
As per architecture definition this can be found in the script models_old.py, models.py is for testing alternatives. The pytorch lightning model definition instead can be found in the Model.py file. For what concerns saved models under the models_saved folder you will find both the original model and the one trained with random masking.

Notebooks:
One notebook reproduces the main results of the paper and goes under the name testing.ipynb. As for doing targeted generation this can be found in the notebook mol_gen_test.ipynb which calls some functions from molecular_generation_utils.py

Misc:
Other scripts are intended as Utilities to be called in the rest of the applications. As per the NEB part, for now I only cleaned and reprepared the code for a generic interpolation but it should reproduce at least the methodology for the geodesic search (to be updated of course).

General remark:
As a general remark the code can be better organized and structured but it would go beyond the purpose of the paper, which is to scientifically present and prototype a novel methodology.


