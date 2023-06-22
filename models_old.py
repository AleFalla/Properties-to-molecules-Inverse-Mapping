import torch 
from torch import nn
import torch.nn.functional as F
import math
#base VAE model

input_dim=528
output_dim=528
latent_size=30
prop_size=3

class base_VAE(nn.Module):
    def __init__(self,input_dim=input_dim, output_dim = input_dim, latent_size=latent_size):
        super().__init__()
        
        self.input_dim=input_dim
        self.output_dim = output_dim
        self.latent_size=latent_size
        self.prior_mu = nn.Parameter(torch.zeros(latent_size), requires_grad = True)
        self.prior_logvar = nn.Parameter(torch.zeros(latent_size), requires_grad = True)

        self.encoder=nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.SiLU(),  
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),

            nn.Linear(2048,512),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),

            # nn.Linear(512,1024),
            # nn.Tanh(),
            # nn.Dropout(p=0.1),
            # nn.BatchNorm1d(1024),

            nn.Linear(512,latent_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.Tanh(),  
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256,latent_size),
        )
        
        
        self.decoder=nn.Sequential(
            #nn.Tanh(),
            #nn.Dropout(p=0.1),
            nn.Linear(latent_size,512),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),

            # nn.Linear(1024,512),
            # nn.Tanh(),
            # nn.Dropout(p=0.1),
            # nn.BatchNorm1d(512),

            nn.Linear(512,2048),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048,output_dim)
        )

        self.decoder_var = nn.Sequential(
            nn.Linear(latent_size,256),
            nn.Tanh(),  
            #nn.Dropout(p=0.1),
            #nn.BatchNorm1d(256),
            nn.Linear(256,output_dim),
        )

        # self.decoder_dist=nn.Sequential(
        #     #nn.Tanh(),
        #     #nn.Dropout(p=0.1),
        #     nn.Linear(latent_size,512),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.1),
        #     nn.BatchNorm1d(512),

        #     # nn.Linear(1024,512),
        #     # nn.Tanh(),
        #     # nn.Dropout(p=0.1),
        #     # nn.BatchNorm1d(512),

        #     nn.Linear(512,2048),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.1),
        #     nn.BatchNorm1d(2048),
            
        #     nn.Linear(2048,output_dim)
        # )

        # self.decoder_dist_var = nn.Sequential(
        #     nn.Linear(latent_size,256),
        #     nn.Tanh(),  
        #     #nn.Dropout(p=0.1),
        #     #nn.BatchNorm1d(256),
        #     nn.Linear(256,output_dim),
        # )
        
    def reparameterize(self,mu,logvar):
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        
        mu=self.encoder(x.view(-1,self.input_dim))#.view(-1,1,self.latent_size)#.view(-1,2,self.latent_size)
        logvar = self.encoder_var(x.view(-1,self.input_dim))#.view(-1,1,self.latent_size)
        #mu=mu_logvar[:,0,:]
        #logvar=mu_logvar[:,1,:]
        return mu, logvar
    
  
    def decode(self,z):
        
        mu = self.decoder(z)#.view(-1,1,self.input_dim)#.view(-1,2,self.input_dim)
        logvar = self.decoder_var(z)#.view(-1,1,self.input_dim)
        #mu=mu_logvar[:,0,:]
        #logvar=mu_logvar[:,1,:]
        return mu, logvar
    
    # def decode_dist(self,z):
        
    #     mu = self.decoder_dist(z)#.view(-1,1,self.input_dim)#.view(-1,2,self.input_dim)
    #     logvar = self.decoder_dist_var(z)#.view(-1,1,self.input_dim)
    #     #mu=mu_logvar[:,0,:]
    #     #logvar=mu_logvar[:,1,:]
    #     return mu, logvar
            
    def forward(self,x):
        
        mu,logvar=self.encode(x)
        z=self.reparameterize(mu,logvar)
        mu_x,logvar_x=self.decode(z)
        return mu_x,logvar_x,mu,logvar,z

    def sample(self, n_samples):
        z=torch.randn((n_samples,self.latent_size))
        mu,logvar=self.decode(z)
        return mu

extra_size = 32-prop_size

class prop_ls_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
  
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(prop_size, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,extra_size),
            nn.BatchNorm1d(extra_size)
        )

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            nn.Linear(prop_size+extra_size,2048),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,latent_size)        
        )

       
        

        self.var=nn.Sequential(
            #nn.Dropout(p=0.1),
            nn.Linear(prop_size+extra_size,128),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128,latent_size)        
        )
        
        # self.decoder_p=nn.Sequential(
        #     #nn.Tanh(),
        #     #nn.Dropout(p=0.1),
        #     nn.Linear(latent_size,512),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.1),
        #     nn.BatchNorm1d(512),

        #     # nn.Linear(1024,512),
        #     # nn.Tanh(),
        #     # nn.Dropout(p=0.1),
        #     # nn.BatchNorm1d(512),

        #     nn.Linear(512,2048),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.1),
        #     nn.BatchNorm1d(2048),
            
        #     nn.Linear(2048,prop_size)
        # )

        # self.decoder_p_var = nn.Sequential(
        #     nn.Linear(latent_size,256),
        #     nn.Tanh(),  
        #     #nn.Dropout(p=0.1),
        #     #nn.BatchNorm1d(256),
        #     nn.Linear(256,prop_size),
        # )
    
    # def decode_p(self,z):
        
    #     mu = self.decoder_p(z)#.view(-1,1,self.input_dim)#.view(-1,2,self.input_dim)
    #     logvar = self.decoder_p_var(z)#.view(-1,1,self.input_dim)
    #     #mu=mu_logvar[:,0,:]
    #     #logvar=mu_logvar[:,1,:]
    #     return mu, logvar

    def forward(self,x):

        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        #compute output
        mu_p=self.model(y)#.view(-1,1,self.latent_size)#.view(-1,2,self.latent_size)
        logvar_p = self.var(y)#.view(-1,1,self.latent_size)
        #mu_p=mu_logvar[:,0,:]
        #logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p

