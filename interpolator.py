import torch
from torch import nn
from torchani.utils import _get_derivatives_not_none as derivative
import copy
from torch.autograd import Variable
import torch
from Model import Multi_VAE
import numpy as np
from Data_Handler import Data_Handler
from matplotlib import pyplot as plt
from ase.io import read, write
from invert_CM import *
from ase import Atoms
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = 'cuda'
devicet = torch.device(device)

class Interpolator(nn.Module):
    def __init__(self,
                 initial_props,
                 final_props,
                 steps = 10,
                 device = 'cuda'
                 ) -> None:
        super().__init__()
        devicet = torch.device(device)
        
        self.initial_props = initial_props.to(devicet)
        
        self.final_props = final_props.to(devicet)
        
        self.delta = (final_props - initial_props)/steps
        
        property_path = (initial_props.view(1,-1),)
       
        temp = initial_props.view(1,-1)
       
        for i in range(0, steps):
            temp = temp + self.delta.view(1,-1)
            property_path += (temp,)
      
        property_path = torch.cat(property_path, dim = 0)

        property_path = property_path[1:-1,:]
        
        self.property_path = torch.nn.Parameter(property_path, requires_grad = True)
        
        self.steps = steps
    
    def forward(self, modello):
        modello.freeze()
        modello.eval()
        path = torch.cat((self.initial_props, self.property_path, self.final_props), dim = 0)
        CMs, Zs = modello.test_generation_from_Y(path)
        Z_1 = torch.cat((torch.zeros_like(Zs[0,:]).view(1,-1), Zs),  dim = 0)
        Z_0 = torch.cat((Zs, torch.zeros_like(Zs[0,:]).view(1,-1)),  dim = 0)
        delta_Z = torch.norm(Z_1 - Z_0, dim = 1)[1:-1]**2
        p_1 = torch.cat((torch.zeros_like(path[0,:]).view(1,-1), path),  dim = 0)
        p_0 = torch.cat((path, torch.zeros_like(path[0,:]).view(1,-1)),  dim = 0)
        delta_p = torch.norm(p_1 - p_0, dim = 1)[1:-1]**2
        potential_z = (0.5*(self.steps))*(delta_Z.sum()) 
        potential_p = 1e-3*delta_p.sum()/self.steps
        
        return potential_z, potential_p, CMs, Zs

paper_path = 'special/'

p_means = torch.load('./{}data/properties_means.pt'.format(paper_path)).to(devicet)
p_stds = torch.load('./{}data/properties_stds.pt'.format(paper_path)).to(devicet)

modello = Multi_VAE(
    structures_dim = len(torch.load('./{}data/data_val/CMs.pt'.format(paper_path))[0,:]),
    properties_dim = len(torch.load('./{}data/data_val/properties.pt'.format(paper_path))[0,:]),
    latent_size = 21,
    extra_dim = 32 - len(torch.load('./{}data/data_val/properties.pt'.format(paper_path))[0,:]),
    initial_lr = 1e-3,
    properties_means = p_means,
    properties_stds = p_stds,
    beta_init = 3.,
    beta_0=1,
    beta_1=1.1,
    alpha = 2,
    decay = .995,
    freq=0,
)

PATH='./special/VAE_reduced_21'
modello.VAE.load_state_dict(torch.load(PATH,map_location=torch.device(device)))
PATH='./special/prop_ecoder_reduced_21'
modello.property_encoder.load_state_dict(torch.load(PATH,map_location=torch.device(device)))

modello.to(devicet)
modello.freeze()
modello.eval()

ni = 39582#40006#40850
nf = 39583#40004#40849

initial_props = torch.load(f'./{ni}.pt').to(devicet)
final_props = torch.load(f'./{nf}.pt').to(devicet)
N = 6

interp = Interpolator(
initial_props.view(1,-1),
final_props.view(1,-1),
steps = N,
device = device
)

grad_norm = 1e3
optimizer = torch.optim.AdamW(interp.parameters(), lr=1e-3)
sch = ReduceLROnPlateau(optimizer, factor = 0.9, patience = 100)
i = 0

while grad_norm >=1e-3 and i < 1e6:
    i+=1
    optimizer.zero_grad()
    loss_z, loss_p, CMs, Zs = interp(modello)
    (loss_z + loss_p).backward()
    optimizer.step()
    for param in interp.parameters():   
        grad_norm = torch.linalg.norm(param.grad)
        param = param + 1e-4*torch.randn_like(param)     
        
    if i%10 == 0:
        sch.step(loss_z + loss_p)
        print(i, (loss_z).item(),  (loss_p).item(), (loss_z + loss_p).item(), grad_norm.item())

proppath = torch.cat((interp.initial_props, interp.property_path.detach(), interp.final_props), dim = 0)
    
for k in range(0, N+1):
    if k >= 1:
        pos_old = pos
    out_cm, _ = modello.test_generation_from_Y(proppath[k,:].view(1, -1).to(torch.float32), sampling=False, normalize_latent = False)
    out_cm[out_cm<=0] = 0.
    n = len(out_cm[0,:])
    lun=int((-1+(1+2*4*n)**0.5)/2)
    resized_CMs = torch.zeros(out_cm.size(0), lun, lun).to(torch.float32)
    resized_CMs = resized_CMs.to(devicet)
    i,j=np.triu_indices(lun)
    resized_CMs[:, i, j] = out_cm.to(torch.float32)
    resized_CMs[:, j, i] = out_cm.to(torch.float32)
    mask = copy.deepcopy(torch.diagonal(resized_CMs, dim1=1, dim2=2))
    mask[mask<=18.5] = 0
    mask[mask>18.5] = 1
    mask = torch.einsum('ij, ik -> ijk', mask, mask)
    resized_CMs = torch.mul(resized_CMs, mask)
    out_cm = resized_CMs[:,i,j]
    pos, comp = get_cartesian(out_cm[0].cpu())#reconstructed_CMs[n,:].tolist())
    if k >= 1:
        pass
        #pos = pos-rmsd.centroid(pos_old)
    atom = Atoms(comp, pos)
    try:
        write('./interpolation/interp_{}.png'.format(k), atom)
        write('./interpolation/interp_{}.xyz'.format(k), atom)
    except:
        print('some weird error')
        
torch.save(Zs, './interpolation/interpolated_Zs.pt')
torch.save(CMs, './interpolation/interpolated_CMs.pt')
torch.save(proppath, './interpolation/interpolated_props.pt')

