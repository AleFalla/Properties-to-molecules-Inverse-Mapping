import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
from Data_prep_utils import *

class Data_Handler():
    def __init__(self, 
    json_data_path = './',
    properties_list = [],
    list_elements = [],
    folder_path = './data/',
    train_fraction = 0.8,
    validation_fraction = 0.05,
    device = 'cpu',
    ) -> None:

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for key in ['train', 'val', 'test']:
            if not os.path.exists(folder_path + 'data_{}/'.format(key)):
                os.makedirs(folder_path + 'data_{}/'.format(key))
                
        self.folder_path = folder_path
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.device = device
        self.pandas_dataset = pd.DataFrame([])
        self.json_data_path = json_data_path
        self.list_elements = list_elements
        self.properties_list = properties_list
        self.properties_means = None
        self.properties_stds = None
    
    def prepare_pandas(self):
        
        print('Starting dataset preparation:')
        print('. reading dataset...')
        self.pandas_dataset = pd.read_json(self.json_data_path)
        print(len(self.pandas_dataset))
        print('. choosing molecules based on species...')
        self.pandas_dataset = pick_molecules(self.pandas_dataset, self.list_elements)
        print(len(self.pandas_dataset))
        print('. adding element count...')
        self.pandas_dataset = count_elements(self.pandas_dataset, self.list_elements)
        print('. excluding hydrogens from positions...')
        self.pandas_dataset = de_hydrogenize_positions(self.pandas_dataset)
        print('. preparing padded Coulomb matrices...')
        self.pandas_dataset = compute_standardized_CM(self.pandas_dataset, self.list_elements)
        print('. saving properties normalization...')
        self.properties_means = torch.tensor(self.pandas_dataset[self.properties_list].mean(axis = 0).values.tolist())
        self.properties_stds = torch.tensor(self.pandas_dataset[self.properties_list].std(axis = 0).values.tolist())
        print('Dataset prepared.')
        return True
    
    def prepare_TensorDatasets(self, save_to_file = True):
        
        CMs = torch.tensor(self.pandas_dataset['CM'].values.tolist())
        properties = torch.tensor(self.pandas_dataset[self.properties_list].values.tolist())
        
        #admittedly a bit stupid to save also the total dataset in .pt, so maybe modify this 
        if save_to_file == True:
            torch.save(CMs, self.folder_path + 'CMs_total.pt')
            torch.save(properties, self.folder_path + 'properties_total.pt')
            
        CMs.to(torch.float32)
        properties.to(torch.float32)
        index = torch.randperm(CMs.size(0))
        train_size = int(len(index)*self.train_fraction)
        val_size = int(train_size*self.validation_fraction)
        train_index = index[val_size:train_size]
        val_index = index[0:val_size]
        test_index = index[train_size::]
        
        total_dict = {}

        iter_dict = {
            'CMs': CMs,
            'properties': properties
        }

        for key in iter_dict.keys():
            
            train_tensor = iter_dict[key][train_index]
            val_tensor = iter_dict[key][val_index]
            test_tensor = iter_dict[key][test_index]
            
            total_dict['{}_train'.format(key)] = train_tensor
            total_dict['{}_val'.format(key)] = val_tensor
            total_dict['{}_test'.format(key)] = test_tensor

            if save_to_file:

                torch.save(train_tensor, self.folder_path + 'data_train/{}.pt'.format(key))
                torch.save(val_tensor, self.folder_path + 'data_val/{}.pt'.format(key))
                torch.save(test_tensor, self.folder_path + 'data_test/{}.pt'.format(key))
                torch.save(self.properties_means, self.folder_path + 'properties_means.pt')
                torch.save(self.properties_stds, self.folder_path + 'properties_stds.pt')
                
        tensordatasets_dict = {}

        for label in ['train', 'val', 'test']:

            tensors_list = [total_dict['{}_{}'.format(key, label)] for key in iter_dict.keys()]
            tensordata = TensorDataset(*tensors_list)
            tensordatasets_dict['{}_dataset'] = tensordata

        return tensordatasets_dict, total_dict
    
    def prepare_and_tensorize(self, save_to_file = True):
        
        self.prepare_pandas()
        tensordatasets_dict, total_dict = self.prepare_TensorDatasets(save_to_file)
        return tensordatasets_dict, total_dict, self.properties_means, self.properties_stds
        
    def load_datas_from_files(self):
        
        """
        Loads dataset from file structure
        Returns a dict of tensordatasets with keys '{train, val, test}_dataset'
        """

        tensordatasets_dict = {}

        for label in ['train', 'val', 'test']:

            tensors_list = [torch.load(self.folder_path + 'data_{}/{}.pt'.format(label, key), map_location = self.device)for key in ['CMs', 'properties']]
            tensordata = TensorDataset(*tensors_list)
            tensordatasets_dict['{}_dataset'.format(label)] = tensordata
        
        self.properties_means = torch.load(self.folder_path + 'properties_means.pt')
        self.properties_stds = torch.load(self.folder_path + 'properties_stds.pt')
        
        return tensordatasets_dict, self.properties_means, self.properties_stds
