
from Data_Handler import Data_Handler

properties = [
 'eAT',
 'eMBD',
 'eNN',
 'eKIN',
 'eNE',
 'eEE',
 'eXX',
 'EGAP',
 'POL',
 'DIP',
 'HOMO_0',
 'LUMO_0',
 'HOMO_1',
 'LUMO_1',
 'HOMO_2',
 'LUMO_2',
 'dimension'
    ]

data = Data_Handler(
    json_data_path = '/home/users/afallani/Generative_VAE/datasets/qm7x_eq_more_unnorm.json',#'/home/users/afallani/Generative_VAE/datasets/QM7-X_unnorm_.json',
    properties_list = properties,
    list_elements = [8, 7, 6],
    train_fraction = .68,
    validation_fraction = .05
)

datasets_dict, _, p_means, p_stds = data.prepare_and_tensorize(save_to_file = True)