import pandas as pd
import numpy as np
from CM_preparation import get_standardized_CM

def check_in(
    a,
    b
):
    uniques = np.unique(a)
    uniques = uniques[uniques!=1]
    uniques = uniques[uniques!=0]
    uniques_b_check = np.unique(b)
    uniques_b_check = uniques_b_check[uniques_b_check!=1]
    overlap = np.array([i for i in uniques if i in uniques_b_check])
    if len(overlap) == len(uniques):
        return True
    else:
        return False

def pick_molecules(
    df,
    species,
    name = 'atom_numbers'
):
    temp_col = df[name].apply(lambda x: check_in(x, species))
    df = df[temp_col]
    compos = np.array([df['atom_numbers'].values[i] for i in range(0,len(df['atom_numbers'].values))])
    compos[compos == 1 ] = 0
    num_heavy = np.count_nonzero(compos, axis = 1)
    
    return df[num_heavy!=1]

def count_elements(
    df,
    element_list,
    name = 'atom_numbers'
):
    for element in element_list:
        df[element] = df[name].apply(lambda x: len(np.array(x)[np.array(x) == element]))
    
    return df

def de_hydrogenize_positions(
    df,
    name_p = 'positions',
    name_a = 'atom_numbers'
):
    df[name_p] = df[name_p].combine(df[name_a], lambda x, y: [x[i] for i in range(0,len(x)) if y[i] != 1])
    return df

def compute_standardized_CM(
    df,
    element_list,
    name_p = 'positions',
    name_a = 'atom_numbers'
):
    max_n=df[element_list].max().values
    tmp=[]
    
    #build master vector
    for i in range(0,len(element_list)):
        tmp=tmp+[element_list[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)
    
    #creates a column with CM representation
    df['CM'] = df['positions'].combine(df['atom_numbers'], lambda x,y: get_standardized_CM(y, x, master_vec))
    
    return df