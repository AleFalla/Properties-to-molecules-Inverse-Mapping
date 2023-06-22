import torch
from torch import nn
import pytorch_lightning as pl
import Losses as l
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from models_old import base_VAE, prop_ls_NN
import numpy as np

def beta_schedule(e, beta_0 = 1, beta_1 = 3, freq = .01):
    beta = beta_0 + (beta_1-beta_0)*0.5*(1+np.cos(e*freq))
    return beta

class Multi_VAE(pl.LightningModule):
    def __init__(
        self,
        structures_dim,
        properties_dim,
        latent_size,
        extra_dim,
        initial_lr,
        properties_means,
        properties_stds,
        alpha = 2,
        beta_init = 3,
        beta_0 = .5,
        beta_1 = 1.5,
        decay = .995,
        freq = 0.01,
        ):
        
        super().__init__()
        self.initial_lr = initial_lr
        self.VAE = base_VAE(structures_dim, structures_dim, latent_size)
        self.property_encoder = prop_ls_NN(latent_size, properties_dim, extra_dim)
        self.properties_means = properties_means
        self.properties_stds = properties_stds
        self.alpha = alpha
        self.beta = beta_init
        self.decay = decay
        self.clamp = 1
        self.epochs = 0
        self.iteration = 0
        self.beta_0 = beta_0
        self.upper_beta = beta_1
        self.freq = freq
        self.beta_sch = 1.

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        X, Y = batch
        
        mask = l.make_mask(X)
        mask[X==0] = mask[X==0]*0.1
        mask[mask==1] = mask[mask==1]*(2)
        
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        Y = (Y - self.properties_means.to(X.device))/self.properties_stds.to(X.device)
        mu_zx, logvar_zx = self.VAE.encode(X)
        Z = self.VAE.reparameterize(mu_zx, logvar_zx)
        mu_zy, logvar_zy = self.property_encoder(Y)
        mu_xz, logvar_xz = self.VAE.decode(Z)
        
        KLD, reco_mol, reco_lat = l.ELBO_custom_loss(
            X,
            Z,
            mu_xz,
            logvar_xz,
            mu_zy,
            logvar_zy,
            torch.zeros_like(Z),
            torch.zeros_like(Z),
            mu_zx,
            logvar_zx,
            weight = mask
        )
        
        loss = self.beta_sch * KLD - reco_mol - self.alpha * reco_lat
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        curr_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", curr_lr, prog_bar=True, on_step=True, sync_dist = True)
        self.iteration += 1
        self.beta_sch = beta_schedule(self.iteration, beta_0=self.beta_0, beta_1=self.beta, freq=self.freq)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        X, Y = batch
        mask = torch.ones_like(X).to(X.device)
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        Y = (Y - self.properties_means.to(X.device))/self.properties_stds.to(X.device)
        mu_zx, logvar_zx = self.VAE.encode(X)
        Z = self.VAE.reparameterize(mu_zx, logvar_zx)
        mu_zy, logvar_zy = self.property_encoder(Y)
        mu_xz, logvar_xz = self.VAE.decode(Z)
        KLD, reco_mol, reco_lat = l.ELBO_custom_loss(
            X,
            Z,
            mu_xz,
            logvar_xz,
            mu_zy,
            logvar_zy,
            torch.zeros_like(Z),
            torch.zeros_like(Z),
            mu_zx,
            logvar_zx,
            weight = mask,
            zero_weight = self.zero_weight
        )
        
        loss = self.beta_sch * KLD - reco_mol - self.alpha * reco_lat
        mol_from_prop, _ = self.VAE.decode(mu_zy)
        mol_from_prop = mol_from_prop.abs()
        print('KLD: ', KLD.item())
        print('reco_mol: ', -reco_mol.item())
        print('abs_mol ', (mu_xz[X != 0] - X[X != 0]).abs().mean().item())
        print('reco_lat: ', -reco_lat.item())
        print('abs_lat: ', (mu_zy - Z).abs().mean().item())
        print('abs_mol_from_prop: ', (mol_from_prop[X != 0] - X[X != 0]).abs().mean().item())
        self.log('proptomol', (mol_from_prop[X != 0] - X[X != 0]).abs().mean().item(), on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        print('total_val_loss: ', loss.item())
        print('Beta:', self.beta_sch)
        self.log("val_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        
        if self.beta>(2-self.decay)*self.upper_beta:
            self.beta = self.beta * self.decay
            

    def test_step(self, batch, batch_idx):
        """
        Test step
        """
        X, Y = batch
        Y = (Y - self.properties_means)/self.properties_stds
        mu_zx, logvar_zx = self.VAE.encode(X)
        Z = self.VAE.reparameterize(mu_zx, logvar_zx)
        mu_zy, logvar_zy = self.property_encoder(Y)
        mu_xz, logvar_xz = self.VAE.decode(Z)
        KLD, reco_mol, reco_lat = l.ELBO_custom_loss(
            X,
            Z,
            mu_xz,
            logvar_xz,
            mu_zy,
            logvar_zy,
            torch.zeros_like(Z),
            torch.zeros_like(Z),
            mu_zx,
            logvar_zx,
            weight = self.w,
            zero_weight = self.zero_weight
        )
        loss = self.beta * KLD - reco_mol - self.alpha * reco_lat
        mol_from_prop, _ = self.decode(mu_zy)
        print('KLD: ', KLD.item())
        print('reco_mol: ', -reco_mol.item())
        print('abs_mol ', (mu_xz[X != 0] - X[X != 0]).abs().mean().item())
        print('reco_lat: ', -reco_lat.item())
        print('abs_lat: ', (mu_zy - Z).abs().mean().item())
        print('abs_mol_from_prop: ', (mol_from_prop[X != 0] - X[X != 0]).abs().mean().item())
        print('total_val_loss: ', loss.item())
        self.log("test_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)

    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr = self.initial_lr)
        scheduler = {"scheduler": ReduceLROnPlateau(optimizer, factor = 0.9, patience = 25), "monitor": "val_loss"}
        return [optimizer], scheduler

    def test_generation_from_Y(self, Y, sampling = False, normalize_latent = False):
        mu_zy, logvar_zy = self.property_encoder(Y)
        if sampling:
            Z = self.VAE.reparameterize(mu_zy, logvar_zy)
        else:
            Z = mu_zy
        if normalize_latent:
            Z = Z/torch.norm(Z, dim = 1)
        mu_xz, logvar_xz = self.VAE.decode(Z)
        return mu_xz, Z
    
    def latent_interpolate_two_Y(self, Y_0, Y_1, steps):
        mu_zy_0, logvar_zy_0 = self.property_encoder(Y_0)
        mu_zy_1, logvar_zy_1 = self.property_encoder(Y_1)
        delta = (mu_zy_1 - mu_zy_0 )/steps
        temp = mu_zy_0.view(1,-1)
       
        for i in range(0, steps):
            temp = temp + delta.view(1,-1)
            latent_path += (temp,)
      
        latent_path = torch.cat(latent_path, dim = 0)
        mu_xz, logvar_xz = self.VAE.decode(latent_path)
        
        return mu_xz