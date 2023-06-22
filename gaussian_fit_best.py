import numpy as np
from matplotlib import pyplot as plt
from molecular_generation_utils import *
from invert_CM import *
import torch
from Model import Multi_VAE
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import copy

reproduce_paper = True

if reproduce_paper:
    paper_path = 'special/'
else:
    paper_path = ''


properties = torch.load('./{}data/data_train/properties.pt'.format(paper_path))
p_means = torch.load('./{}data/properties_means.pt'.format(paper_path))
p_stds = torch.load('./{}data/properties_stds.pt'.format(paper_path))
norm_props = (properties - p_means)/p_stds
gm, labels = props_fit_Gaussian_mix(
    norm_props, 
    min_components = 1,
    max_components = 100
    )