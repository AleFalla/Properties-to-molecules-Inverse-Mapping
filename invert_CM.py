import numpy as np
import pandas as pd
from tqdm import tqdm
import rmsd
from pandarallel import pandarallel
from ase.geometry.analysis import Analysis
from scipy.sparse import csgraph

def rmsd_calc(
    pos1,
    comp1,
    pos2,
    comp2
):
    
    
    try:
        rmsds = []
        for reorder in [rmsd.reorder_brute, rmsd.reorder_inertia_hungarian, rmsd.reorder_hungarian]:
            comp1_, comp2_, pos1_, pos2_ = np.array(comp1), np.array(comp2), np.array(pos1), np.array(pos2)
            pos1_ = pos1_ - rmsd.centroid(pos1_)
            pos2_ = pos2_ - rmsd.centroid(pos2_)
            
            indices = reorder(comp1_, comp2_, pos1_, pos2_)
            pos2_ = pos2_[indices]
            vrmsd_0 = float(rmsd.quaternion_rmsd(pos1_, pos2_))

            pos2_[:,0] = -pos2_[:,0]
            indices = reorder(comp1_, comp2_, pos1_, pos2_)
            pos2_ = pos2_[indices]
            vrmsd_1 = float(rmsd.quaternion_rmsd(pos1_, pos2_))

            rmsds.append(min(vrmsd_0, vrmsd_1))
        return np.min(rmsds)
    
    except:
        
        return float('inf')
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def recover_distance_mat(mcm):
    n=len(mcm)
    lun=int((-1+(1+2*4*n)**0.5)/2)
    M=np.zeros((lun,lun))
    i,j=np.triu_indices(lun)
    M[i,j]=mcm
    M[j,i]=mcm
    idx=np.argsort(np.diag(-M))
    M=M[idx,:]
    M=M[:,idx]
    tmp=lun
    for i in range(0,lun):
        if M[i,i]<=0.25*(6**2.4):
            tmp=i
            break
    
    M=M[:tmp,:tmp]
    species=np.array([17,16,9,8,7,6])
    master_vec=[]
    Z_w=[]
    for i in range(0,tmp):
        Z=(2*M[i,i])**(1/2.4)
        Z_w.append(Z)
        Z=find_nearest(species,Z)
        master_vec.append(Z)
    
    Z_w=master_vec
    for i in range(0,tmp):
        for j in range(0,tmp):
                if i==j:
                    M[i,j]=0
                else:
                    M[i,j]=(M[i,j]/(Z_w[i]*Z_w[j]))**(-1)
    M[M==np.inf]=0
    M = np.nan_to_num(M)
    if len(Z_w)==1:
        dummy = np.zeros((2,2))
        dummy[0,0] = M[0,0]
        M = dummy
    return M, master_vec

def cartesian_recovery(distance_mat):
    D=distance_mat
    M=np.zeros_like(D)
    
    for i in range(0,len(D)):
        for j in range(0,len(D)):
            M[i,j]=0.5*(D[0,i]**2+D[j,0]**2-D[i,j]**2)
            
    
    S,P=np.linalg.eig(M)
    tmp=np.flip(np.sort(S))
    S[S<0]=0
    if len(tmp)>=3:
        S[np.abs(S)<np.abs(tmp[2])]=0
    S_05=np.diag(np.sqrt(S))
    cartesian=P@S_05
    return cartesian

def get_cartesian(mcm):
    """please don't ask"""
    dist,master=recover_distance_mat(mcm)
    tempo=cartesian_recovery(dist)
    tempo[tempo>=1e10]=0
    cartesian=np.real(np.nan_to_num(tempo))
    sasso = 0
    if np.shape(cartesian)[0]<1:
        return [[0,0,0]],[0]
    else:
        for i in range(0,np.shape(dist)[0]):
            indices=np.nonzero(cartesian[i,:])
            if len(indices[0])==3:
                sasso = 1
                break
        if sasso == 0:
            for i in range(0,np.shape(dist)[0]):
                indices=np.nonzero(cartesian[i,:])
                if len(indices[0])==2:
                    sasso = 2
                    break
        if sasso == 0:
            for i in range(0,np.shape(dist)[0]):
                indices=np.nonzero(cartesian[i,:])
                if len(indices[0])==1:
                    sasso = 3
                    break
        if sasso == 1:
            cartesian=cartesian[:,indices[0]]
        if sasso == 2:
            cartesian=cartesian[:,indices[0]]
            cartesian = np.pad(cartesian, ((0,0),(0,1)))
        if sasso == 3:
            cartesian=cartesian[:,indices[0]]
            cartesian = np.pad(cartesian, ((0,0),(0,2)))
    if np.shape(cartesian)[1]!=3:
        cartesian = cartesian[:,0:3]
        
    return cartesian,master


def invert_cm_batch(CMs, keyword = 'orig'):
    
    positions = []
    compositions = []
    for i in tqdm(range(0,len(CMs))):
        pos, comp = get_cartesian(CMs[i,:].tolist())
        positions.append(pos)
        compositions.append(comp)
    return pd.concat([pd.Series(positions), pd.Series(compositions)], axis = 1).rename(columns={0:'positions_{}'.format(keyword), 1:'compositions_{}'.format(keyword)})

def get_rmsds(CMs_orig, CMs_reco):
    
    df = pd.concat([invert_cm_batch(CMs_orig, 'orig'), invert_cm_batch(CMs_reco, 'reco')], axis = 1)
    pandarallel.initialize(progress_bar=True)
    df['rmsd'] = df[df.columns.values.tolist()].parallel_apply(lambda x: rmsd_calc(*x), axis = 1)
    return df

def get_connected(mole):
    an=Analysis(mole)
    mat=an.adjacency_matrix[0].toarray()
    mat=(mat+mat.T)
    np.fill_diagonal(mat, 0)
    n_comp, labels = csgraph.connected_components(mat)
    return n_comp