import torch 
from torch import nn

class base_VAE(nn.Module):
    
    def __init__(self,input_dim, latent_size):
        super().__init__()
        
        self.input_dim=input_dim
        self.latent_size=latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.LayerNorm(512),
            nn.SiLU(),  
            nn.Dropout(p=0.1),
            

            nn.Linear(512,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128,latent_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.LayerNorm(128),
            nn.Tanh(),  
            nn.Dropout(p=0.1),
            nn.Linear(128,latent_size),
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128,512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            
            nn.Linear(512, input_dim)
        )

        self.decoder_var = nn.Sequential(
            nn.Linear(latent_size,256),
            nn.Tanh(),  
            nn.Linear(256, input_dim),
        )

    def reparameterize(self,mu,logvar):
        
        mu.to(torch.float32)
        logvar.to(torch.float32)
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        
        x = x.to(torch.float32)
        mu = self.encoder(x.view(-1,self.input_dim))
        logvar = self.encoder_var(x.view(-1,self.input_dim))
        return mu, logvar
    
    def decode(self,z):
        
        mu = self.decoder(z)#.view(-1,1,self.input_dim)#.view(-1,2,self.input_dim)
        logvar = self.decoder_var(z)#.view(-1,1,self.input_dim)
        #mu=mu_logvar[:,0,:]
        #logvar=mu_logvar[:,1,:]
        return mu, logvar
    

class prop_ls_NN(nn.Module):
    
    def __init__(self, latent_size, prop_size, extra_size):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
  
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128,extra_size),
            nn.LayerNorm(extra_size)
        )

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            nn.Linear(prop_size+extra_size,512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128,128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128,latent_size)        
        )

        self.var=nn.Sequential(
            nn.Linear(prop_size+extra_size,128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(128,latent_size)        
        )
        
    def forward(self,x):
        x.to(torch.float32)
        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        #compute output
        mu_p=self.model(y)
        logvar_p = self.var(y)
        return mu_p, logvar_p