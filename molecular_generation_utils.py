
import numpy as np
from sklearn.mixture import GaussianMixture
import torch 
from torch.distributions.categorical import Categorical
import copy
from tqdm import tqdm
from invert_CM import *
from ase import Atoms

def props_fit_Gaussian_mix(
    properties, 
    min_components = 70,
    max_components = 100
    ):
    
    X = properties
    lowest_bic = np.infty
    bic = []
    n_components_range = range(min_components, max_components)
    cv_type = "full"
    for n_components in tqdm(n_components_range):
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=cv_type, init_params='random', random_state=0 
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    bic = np.array(bic)
    idx_min = np.argmin(bic)
    print('using {} components'.format(n_components_range[idx_min]))
    components = n_components_range[idx_min]
    gm = GaussianMixture(n_components = components, init_params='random', covariance_type='full', random_state=0).fit(properties.numpy())
    labels = gm.predict(properties.numpy())
    return gm, labels



def boltzmann_weight(arr, T):
    p = np.exp(-arr/T)
    return p/p.sum()

def best_fitted_conditional_distributions(
    target_properties_indices,
    nontarget_properties_indices,
    properties_target_values,
    fitted_means,
    fitted_stds
):    
    resorting_idx = np.argsort(target_properties_indices)
    target_properties_indices = np.array(target_properties_indices)[resorting_idx]
    properties_target_values = np.array(properties_target_values)[resorting_idx]
    
    n_components = len(np.array(fitted_means)[:, 0])
    
    neg_log_likelyoods = []
    for component in range(0, n_components):
        mean = torch.Tensor(fitted_means[component])[target_properties_indices]
        cov = torch.Tensor(fitted_stds[component])[np.ix_(target_properties_indices, target_properties_indices)]
        dist=torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
        log_pxz=dist.log_prob(torch.Tensor(properties_target_values).view(1,-1))
        reco=log_pxz.sum(-1)
        neg_log_likelyoods.append(-reco.item())
    
    min_idxs = np.argsort(np.array(neg_log_likelyoods))
    weights = np.array(neg_log_likelyoods)[min_idxs] - np.array(neg_log_likelyoods)[min_idxs].min()
    weights = boltzmann_weight(weights, 1.5)
    mus, Es = [], []
    for min_idx in min_idxs:
        E_12 = fitted_stds[min_idx][np.ix_(nontarget_properties_indices, target_properties_indices)]
        E_22 = fitted_stds[min_idx][np.ix_(target_properties_indices, target_properties_indices)]
        invE_22 = np.linalg.inv(E_22)
        E = fitted_stds[min_idx][np.ix_(nontarget_properties_indices, nontarget_properties_indices)]- E_12@invE_22@E_12.T
        mu_1 = fitted_means[min_idx][nontarget_properties_indices]
        mu_2 = fitted_means[min_idx][target_properties_indices]
        mu = mu_1 + E_12@invE_22@(properties_target_values-mu_2).reshape(-1)
        mus.append(mu)
        Es.append(E)

    return mus, Es, weights


def sample_conditional_properties_vec(
    target_properties_indices,
    properties_target_values,
    mu,
    E
):
    resorting_idx = np.argsort(target_properties_indices)
    target_properties_indices = np.array(target_properties_indices)[resorting_idx]
    properties_target_values = np.array(properties_target_values)[resorting_idx]
    
    properties = np.random.multivariate_normal(mu, E)
    properties = properties.tolist()
    
    for k in range(0,len(properties_target_values)):
        properties.insert(target_properties_indices[k], properties_target_values[k])

    properties = torch.Tensor(properties)
    return properties

def start_generation(
    modello,
    targets_n_values,
    all_props,
    N_samples,
    tolerance,
    means,
    covariances,
    deltaz = .4,
    cm_diff = 5,
    check_new_comp = True,
    verbose = False
):

    target_props = [key for key in targets_n_values.keys()]
    trgt_idxs = [np.argwhere(all_props == prop)[0][0] for prop in target_props]
    non_trgt_idxs = [i for i in range(0,len(all_props)) if i not in trgt_idxs]
    if check_new_comp:
        registry_comp = torch.load('./special/registry_of_comp.pt')
        delta_comp_min = 0
    samples = []

    mus, Es, weights = best_fitted_conditional_distributions(
        trgt_idxs,
        non_trgt_idxs,
        [value for value in targets_n_values.values()],
        means,
        covariances
    )   

    cat = Categorical(torch.tensor(weights))
    print('. sampling...')
    k = 0
    l = 0
    while len(samples)<=N_samples:
        
        index = cat.sample([1])
        mu = mus[index]
        E = Es[index]
        sample = sample_conditional_properties_vec(
            trgt_idxs,
            non_trgt_idxs,
            mu,
            E
        )
        if k>=1:
            registry = torch.tensor(samples)
            reconstructed_CMs, Z = modello.test_generation_from_Y(sample.view(1,-1), sampling=False)
            
            reconstructed_CMs[reconstructed_CMs<=0] = 0.
            n = len(reconstructed_CMs[0,:])
            lun=int((-1+(1+2*4*n)**0.5)/2)
            resized_CMs = torch.zeros(reconstructed_CMs.size(0), lun, lun).to(torch.float32)
            i,j=np.triu_indices(lun)
            resized_CMs[:, i, j] = reconstructed_CMs.to(torch.float32)
            resized_CMs[:, j, i] = reconstructed_CMs.to(torch.float32)
            mask = copy.deepcopy(torch.diagonal(resized_CMs, dim1=1, dim2=2))
            mask[mask<=18.5] = 0
            mask[mask>18.5] = 1
            mask = torch.einsum('ij, ik -> ijk', mask, mask)
            resized_CMs = torch.mul(resized_CMs, mask)
            reconstructed_CMs = resized_CMs[:,i,j]
            new_Z, _ = modello.VAE.encode(reconstructed_CMs)
            delta_Z = torch.norm(new_Z - Z, dim = 1)
            pos, comp = get_cartesian(reconstructed_CMs[0,:].tolist())
            atoms = len(comp)
            atom = Atoms(comp, pos)
            n_comp = get_connected(atom)
            if n_comp == 1:
                if verbose:
                    print('okay, so I found a molecule...')
                if check_new_comp:
                    paddin_l = 9 - len(comp)
                    pad = torch.zeros(paddin_l)
                    tens_comp = torch.tensor(comp)
                    tens_comp = torch.cat((tens_comp, pad), dim = 0)
                    delta_comp_min = (tens_comp.view(1,-1) - registry_comp).abs().sum(dim = 1).min()
                    if delta_comp_min!=0:
                        if verbose:
                            print('... and it has a new composition')
                    else:
                        if verbose:
                            print('. but the composition is boring, and that is the {}-th time I try'.format(k))
                else:
                    delta_comp_min = 1
                    
                if delta_comp_min!=0:
                    if delta_Z.item()<deltaz:
                        if verbose:
                            print('... and I am pretty sure it is okay')
                        deltas = torch.linalg.norm(registry-reconstructed_CMs.view(1,-1), dim = 1)
                        if deltas.min()>cm_diff:
                            if verbose:
                                print('... and it is not very similar to the previous ones I found, so:')
                                print('. {}th sample added'.format(l+2))
                            samples.append(reconstructed_CMs[0].tolist())
                            l+=1
                        else:
                            if verbose:
                                print('. but I already foudn this more or less, and that is the {}-th time I try'.format(k))
                    else:
                        if verbose:
                            print('. but I am not sure enough and that is the {}-th time I try'.format(k))
        else:
            reconstructed_CMs, Z = modello.test_generation_from_Y(sample.view(1,-1), sampling=False)
            reconstructed_CMs[reconstructed_CMs<=0] = 0.
            n = len(reconstructed_CMs[0,:])
            lun=int((-1+(1+2*4*n)**0.5)/2)
            resized_CMs = torch.zeros(reconstructed_CMs.size(0), lun, lun).to(torch.float32)
            i,j=np.triu_indices(lun)
            resized_CMs[:, i, j] = reconstructed_CMs.to(torch.float32)
            resized_CMs[:, j, i] = reconstructed_CMs.to(torch.float32)
            mask = copy.deepcopy(torch.diagonal(resized_CMs, dim1=1, dim2=2))
            mask[mask<=18.5] = 0
            mask[mask>18.5] = 1
            mask = torch.einsum('ij, ik -> ijk', mask, mask)
            resized_CMs = torch.mul(resized_CMs, mask)
            reconstructed_CMs = resized_CMs[:,i,j]
            samples.append(reconstructed_CMs[0].tolist())
        k+=1
        
        if k >=tolerance:
            break
        
    samples = torch.tensor(samples)
    return samples[1::]
    