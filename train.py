import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from Model import Multi_VAE
import math
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from Data_Handler import Data_Handler

torch.set_default_dtype(torch.float32)

# property list is useful only if you didn't save data in pt
# properties = [
#  'eAT',
#  'eMBD',
#  'eNN',
#  'eKIN',
#  'eNE',
#  'eEE',
#  'eXX',
#  'EGAP',
#  'POL',
#  'DIP',
#  'HOMO_0',
#  'LUMO_0',
#  'HOMO_1',
#  'LUMO_1',
#  'HOMO_2',
#  'LUMO_2',
#  'dimension'
#     ]


reproduce_paper = True

if reproduce_paper:
    paper_path = '_paper'
else:
    paper_path = ''

data = Data_Handler(
    folder_path='./data{}/data/'.format(paper_path)
)

# if data is not initialized in other script put some arguments like these
# json_data_path = '/home/users/afallani/Generative_VAE/datasets/QM7-X_unnorm_.json',
# properties_list = properties,
# list_elements = [8, 7, 6]


# train_samples = torch.load('./{}data/data_train/CMs.pt'.format(paper_path))
# zeros = np.array([len(train_samples[:,n][train_samples[:,n]==0]) for n in range(0,len(train_samples[0,:]))])
# b = 3
# t = len(train_samples[:,0])
# def boltzmann_weight(arr, T):
#     p = np.exp(-arr/T)
#     return p/p.sum()

# w = boltzmann_weight(zeros, t)
# w = .27*w/w.max()#
# w = 0.*torch.ones_like(torch.tensor(w.tolist()))

datasets_dict, p_means, p_stds = data.load_datas_from_files()

training_set = datasets_dict['train_dataset']
validation_set = datasets_dict['val_dataset']
batch_size = 500
train_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(validation_set, batch_size = int(len(validation_set)), shuffle = True)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=2, monitor="proptomol", save_last = True)

model = Multi_VAE(
    structures_dim = len(torch.load('./data{}/data_val/CMs.pt'.format(paper_path))[0,:]),
    properties_dim = len(torch.load('./data{}/data_val/properties.pt'.format(paper_path))[0,:]),
    latent_size = 21,
    extra_dim = 32 - len(torch.load('./data{}/data_val/properties.pt'.format(paper_path))[0,:]),
    initial_lr = 1e-3,
    properties_means = p_means,
    properties_stds = p_stds,
    beta_init = 3.,
    beta_0=1,
    beta_1=1.1,
    alpha = 2,
    decay = .995,
    freq = 0,
    
)

trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=[checkpoint_callback], max_epochs = 3000, gradient_clip_val = 2, gradient_clip_algorithm = "norm")
trainer.fit(model, train_loader, valid_loader)