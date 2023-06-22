import numpy as np
import scipy.spatial

def get_standardized_CM(atoms, positions, master_vec, n = 1):
    """ does everything based on atoms and positions """
    
    #compute coulomb matrix
    coul_mat,atoms=coulombize(positions, atoms, n)
    
    #standardize it
    MCM_st=standardize_coulomb(coul_mat, atoms, master_vec, n)
    
    #if one wants can get only triupper
    MCM_st = MCM_st[np.triu_indices(len(master_vec))]
    return MCM_st


def coulombize(positions, atoms, n=1):
    """ computes coulomb matrix """
    
    #get atomic composition without zeros and hydrogens, then get sorting index by atomic number
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)>n]
    indices=np.argsort(atoms)[::-1]
    
    #get only position of the remaining atoms (heavy atoms) and compute distance matrix
    positions=positions[0:len(atoms)]
    matrix=scipy.spatial.distance_matrix(positions,positions)
    
    #compute coulomb matrix from distance and atomic composition
    for i in range(0,len(atoms)):
        
        matrix[i,i]=1
        matrix[i,:]=atoms[i]*np.multiply((matrix[i,:]**(-1)),atoms)
        matrix[i,i]=0.5*(abs(atoms[i])**2.4)
    
    #fix nans
    matrix=np.nan_to_num(matrix,posinf=0,neginf=0)
    
    #sort rows and columns in descending order of atomic number
    matrix=matrix[:,indices]
    matrix=matrix[indices,:]
    
    #return matrix sorted by diagonal element and clean atomic composition
    return matrix, atoms


def standardize_coulomb(coul_mat,atoms,master_vec, n=1):
    """ standardize coulomb matrices across dataset and adj too if needed"""
    #get atomic composition SORTED in descending order
    
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)>n]
    atoms=np.sort(atoms)[::-1]
    #print(atoms, master_vec, coul_mat)
    #get maximum dimension from the master vector of maximum atoms per type and prepare a base for the standardized coulomb matrix
    max_len=len(master_vec)
    base=np.zeros((max_len,max_len))
    
    #create a zeros matrix max x max and insert coulomb matrix in the first nxn with n dimension of coulomb matrix
    padded=np.zeros((max_len,max_len))

    if len(atoms)<=max_len:
        padded[:len(atoms),:len(atoms)]=coul_mat
    
    else:
        padded=coul_mat[:max_len,:max_len]


    #atoms without zeros
    atoms_red=[x for x in atoms if x!=0]
    
    #counter for repeated species
    count=0
    
    #indices buffer
    indices=[]
    
    #starting atom number to check (null)
    atom_number=0
    
    #loop over atoms
    for i in range(0,len(atoms_red)):
        
        #check for repeated atom species,if not repeated reset counter
        if atom_number!=atoms_red[i]:
            count=0
        
        #set current atom number as the ith atom number in the list
        atom_number=atoms_red[i]
        
        #check where that species is in the master vector and save the index, go one after the other in case of repeated species
        index_list=[idx for idx,el in enumerate(master_vec) if el==atom_number]
        j=index_list[count]
        
        #save the index in indices and increase counter
        indices.append(j)
        count=count+1
        
    #order the coulomb matrix based on the master vector using the indices saved in the previous step (works because everything is sorted)
    for i in range(0,len(indices)):
        for j in range(0,len(indices)):
            base[indices[i],indices[j]]=padded[i,j]
    
    #return the standardized matrix
    return base
