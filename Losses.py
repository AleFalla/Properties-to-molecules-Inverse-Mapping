import torch 
from torch import nn
import numpy as np
import copy

def reconstruction(
    x,
    mu,
    logvar,
    weight
    ):
    
    logscale=nn.Parameter(logvar)
    scale=logscale.mul(0.5).exp()
    dist=torch.distributions.Normal(mu,scale)
    log_pxz=dist.log_prob(x)
    if len(weight)!=1:
        log_pxz = log_pxz*weight
    reco=log_pxz.sum(-1)
    return reco.mean()

def kl_divergence_extra(
    mu, 
    logvar,mu_2,
    logvar_2
    ):
    kl=torch.sum(logvar_2.mul(0.5)-logvar.mul(0.5)+(logvar.exp()+(mu-mu_2).pow(2))/(2*logvar_2.exp())-0.5*torch.ones_like(mu),dim=1)
    return kl.mean()

def ELBO_custom_loss(
    target_mol,
    target_latent,
    pred_mol_mean,
    pred_mol_logvar,
    pred_latent_mean,
    pred_latent_logvar,
    prior_mu,
    prior_logvar,
    latent_mean,
    latent_logvar,
    weight,
    weight_2 = [None]
    ):
    
    pred_latent_logvar=torch.clamp(pred_latent_logvar, min = -1e4)
    pred_mol_logvar = torch.clamp(pred_mol_logvar, min = -1e4)
    latent_logvar = torch.clamp(latent_logvar, min  = -1e4)
    kld = kl_divergence_extra(mu=latent_mean,logvar=latent_logvar,mu_2=prior_mu,logvar_2=prior_logvar)
    reco_mol = reconstruction(target_mol,pred_mol_mean,pred_mol_logvar, weight = weight)
    reco_lat = reconstruction(target_latent,pred_latent_mean,pred_latent_logvar, weight = weight_2)
    return kld, reco_mol, reco_lat

def make_mask(batch, p = 0.5, atoms = 12):
    c = torch.empty(batch.size(0), atoms).bernoulli_(p).to(batch.device)
    mat_c = torch.einsum('ij, ik->ijk', c, c)
    t,s = np.triu_indices(atoms)
    mat_c = mat_c[:,t,s]
    return mat_c
