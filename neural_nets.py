import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from hyperparams import *
from torch import nn, optim
import torch.nn.functional as F
import GP_fourier as gpf

torch.manual_seed(my_seed)
np.random.seed(my_seed)



class GPVAE(nn.Module):
    def __init__(self, featureDim=256, zimg_Dim=N_shared, n_neurons = N_NEURONS,condthresh = 1e8,  minlens= None, vy=1000, Fourier = False):#featureDim = 64*16*16 
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)
     
        super(GPVAE, self).__init__()
        self.N = TIME_POINTS # time
        self.K = PIXEL  #pixels
        self.zimg_Dim = zimg_Dim
        self.len_sc = nn.Parameter(torch.Tensor([30]), requires_grad=True)
        

        self.vy = nn.Parameter(torch.Tensor([vy]), requires_grad=True)
      

        
        
        self.enc1 = nn.Linear(PIXEL, 784)
        self.enc2 = nn.Linear(784, 512)
        self.exper1 = nn.Linear(512, featureDim)
        self.exper2 = nn.Linear(featureDim, featureDim)
        

        if Fourier:
            nxc_ext = .2
            if minlens == None:
                print('specify a minlens')
            ### same for condthresh
            [By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y, self.N, minlens,n_neurons,nxcirc = np.array([self.N+nxc_ext*self.N]),condthresh = condthresh)
            self.Bf = Bffts[0]
            self.N_four = Bffts[0].shape[0]
            self.nxcirc = nxcirc
            self.wwnrm = torch.Tensor(wwnrm)
            
            self.encMeanFour = nn.Linear(self.N, self.N_four)
            self.encVarFour = nn.Linear(self.N, self.N_four, bias = False)

        self.encFC1 = nn.Linear(featureDim, zimg_Dim)
        self.encFC2 = nn.Linear(featureDim, zimg_Dim)
        
        self.decFC2 = nn.Linear(zimg_Dim, featureDim)
        self.decFC3 = nn.Linear(featureDim, featureDim)
        self.dexper1 = nn.Linear(featureDim, featureDim)
        self.dexper2 = nn.Linear(featureDim, 512)
        self.dec2 = nn.Linear(512, 784)
        self.dec1 = nn.Linear(784, PIXEL)


    def encode(self, x, Fourier = False):
      
        x = F.elu(self.enc1(x))
        x = F.elu(self.enc2(x))
        x = F.elu(self.exper1(x))
        x = F.elu(self.exper2(x))
        zm = self.encFC1(x)
        zs = self.encFC2(x)
        return zm, zs


    def sample(self, zm, zs, eps):
        eps = torch.randn_like(zs)        
        z = zm + eps * torch.exp(zs) 
  
        return z

    def decoder(self, z):
        x = F.elu(self.decFC2(z))
        x = F.elu(self.decFC3(x))
        x = F.elu(self.dexper1(x))
        x = F.elu(self.dexper2(x))
        x = F.elu(self.dec2(x))
        x = F.elu(self.dec1(x))
        return x


    def nll(self, x, x_hat):
        mse = ((x_hat - x) ** 2).view(x.shape[0], self.N, PIXEL).sum((2), True).sum(1)
        
        nll = mse / (2 * self.vy)
        nll += 0.5 *self.N* torch.log(self.vy) * PIXEL
        return nll, mse

    def gp_make_cov(self, nxcirc= None, wwnrm= None, rho = 10, l_scale = torch.Tensor([15]), eps = 1e-4, Fourier = False):
        
        if Fourier:
            K_cov = gpf.mkcovs.mkcovdiag_ASD_wellcond(self.len_sc, 1, nxcirc, wwnrm = wwnrm,addition = eps)

        else:
            M1 = torch.arange(self.N).unsqueeze(0)-torch.arange(self.N).unsqueeze(1)
            K_cov = torch.exp(-torch.square(M1)/(2*torch.square(self.len_sc))) + eps*torch.eye(self.N)

        return K_cov* rho


    def gp_make_cov_periodic(self, nxcirc= None, wwnrm= None, rho = 1000, l_scale = torch.Tensor([15]), eps = 1e-7, Fourier = False):
        
        if Fourier:
            K_cov = gpf.mkcovs.mkcovdiag_ASD_wellcond(self.len_sc.cpu(), 1, nxcirc, wwnrm = wwnrm,addition = eps)

        else:
            M1 = torch.arange(self.N).unsqueeze(0)-torch.arange(self.N).unsqueeze(1)
    
            up = (torch.sin(torch.pi * M1 / torch.pi))**2
            num = rho * (torch.exp(-up / self.len_sc))
            K_cov = num + self.vy *eps

        return K_cov
    

    def gp_samp(self, K_cov, Fourier = False):
        if Fourier:
            print('ERROR')
        else:
            return np.array(np.random.multivariate_normal(np.zeros(self.N), K_cov, 1)).T 
            
    def forward(self, x, eps,  Fourier = False):

        zm, zs = self.encode(x)
        
        return zm, zs

###############################################################################
## Neural Encoder
###############################################################################

class Spike_Encode(nn.Module):
    
    def __init__(self, zDim=N_lats_spikes, n_neurons= N_NEURONS, condthresh = 1e8, minlens= None, Fourier = False):
        super(Spike_Encode, self).__init__()
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

        self.N = TIME_POINTS
        self.zDim = zDim

        self.en1 = nn.Linear(n_neurons, 80)
        self.en2 = nn.Linear(80, 40)
        self.en3 = nn.Linear(40, 20)
        

        if Fourier:
            nxc_ext = .2
            if minlens == None:
                print('specify a minlens')
    
            [By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y, self.N, minlens,n_neurons,nxcirc = np.array([self.N+nxc_ext*self.N]),condthresh = condthresh)
            self.Bf = Bffts[0]
            self.N_four = Bffts[0].shape[0]
            self.nxcirc = nxcirc
            self.wwnrm = torch.Tensor(wwnrm)
            
            self.spikeMeanFour = nn.Linear(self.N, self.N_four)
            self.spikeVarFour = nn.Linear(self.N, self.N_four, bias = False)

        self.en4 = nn.Linear(20, zDim)
        self.en5 = nn.Linear(20, zDim)
        

    def encode(self, spike):
        spike = F.elu(self.en1(spike))
        spike = F.elu(self.en2(spike))
        spike = F.elu(self.en3(spike))

        zm = self.en4(spike)
        zs = self.en5(spike) 
        return zm, zs

    def get_z(self, zm, zs, eps, Fourier=False):
        eps = torch.randn_like(zs) 
        z_neurons = zm + eps * torch.exp(zs) 
        return z_neurons

    def forward(self, spikes, eps, Fourier):
        zm, zs = self.encode(spikes)
        return zm, zs

###############################################################################
## Neural Decoder
###############################################################################
class Spike_Decode(nn.Module):
    def __init__(self, meanrates, featureDim=256, zDim=1, n_neurons = N_NEURONS, condthresh = 1e8, minlens= None, Fourier = False):
        super(Spike_Decode, self).__init__()
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

        self.N = TIME_POINTS
        self.K = PIXEL
        self.zDim = zDim
       
        
        if Fourier:
            nxc_ext = .2
            if minlens == None:
                print('specify a minlens')
         
            [By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y, self.N, minlens,n_neurons,nxcirc = np.array([self.N+nxc_ext*self.N]),condthresh = condthresh)
            self.Bf = Bffts[0]
            self.N_four = Bffts[0].shape[0]
            self.nxcirc = nxcirc
            self.wwnrm = torch.Tensor(wwnrm)

        self.dec3 = nn.Linear(zDim, n_neurons)
        self.dec3.bias = nn.Parameter(torch.log(torch.Tensor(meanrates))) 
       

    def decoder_lambda(self, z, Fourier = True):
        lambda_hat = self.dec3(z)
        return lambda_hat
    
    def pois_loglike(self, lambda_hat, spikes): 
        p_nll = - ( -(torch.exp(lambda_hat)) -torch.lgamma(spikes + 1) +  lambda_hat * spikes).sum([1,2])
        return p_nll
    

    def forward(self, z, spikes,  Fourier = False):
        lambda_hat= self.decoder_lambda(z)
        neural_loss = self.pois_loglike(lambda_hat, spikes)       
        return neural_loss,lambda_hat
        
  
###############################################################################
## Behavior Encoder + Decoder
###############################################################################
class Behavior_Encoder_Decoder(nn.Module):
    def __init__(self,featureDim=256, zimg_Dim=N_shared, n_neurons = N_NEURONS,condthresh = 1e8, minlens= None, vy=100, Fourier = False):
        super(Behavior_Encoder_Decoder, self).__init__()
        self.N = TIME_POINTS # time
        self.K = PIXEL  #pixels
        self.zimg_Dim = zimg_Dim
        self.len_sc = nn.Parameter(torch.Tensor([30]), requires_grad=True)
    
        self.vy = nn.Parameter(torch.Tensor([vy]), requires_grad=True)
      
        self.enc1 = nn.Linear(PIXEL, 784)
        self.enc2 = nn.Linear(784, 512)
        self.exper1 = nn.Linear(512, featureDim)
        self.exper2 = nn.Linear(featureDim, featureDim)
        
        if Fourier:
            nxc_ext = .2
            if minlens == None:
                print('specify a minlens')
            [By, wwnrm, Bffts, nxcirc] = gpf.comp_fourier.conv_fourier_mult_neuron(y, self.N, minlens,n_neurons,nxcirc = np.array([self.N+nxc_ext*self.N]),condthresh = condthresh)
            self.Bf = Bffts[0]
            self.N_four = Bffts[0].shape[0]
            self.nxcirc = nxcirc
            self.wwnrm = torch.Tensor(wwnrm)
            
            self.encMeanFour = nn.Linear(self.N, self.N_four)
            self.encVarFour = nn.Linear(self.N, self.N_four, bias = False)

        self.encFC1 = nn.Linear(featureDim, zimg_Dim)
        self.encFC2 = nn.Linear(featureDim, zimg_Dim)

        self.decFC2 = nn.Linear(zimg_Dim, featureDim)
        self.dexper1 = nn.Linear(featureDim, featureDim)
        self.dexper2 = nn.Linear(featureDim, 512)
        self.dec2 = nn.Linear(512, 784)
        self.dec1 = nn.Linear(784, PIXEL)


    def encode(self, x, Fourier = False):      
        x = F.elu(self.enc1(x))
        x = F.elu(self.enc2(x))
        x = F.elu(self.exper1(x))
        x = F.elu(self.exper2(x))
        zm = self.encFC1(x)
        zs = self.encFC2(x)
        return zm, zs


    def sample(self, zm, zs, eps):
        eps = torch.randn_like(zs)        
        z = zm + eps * torch.exp(zs) 
        return z

    def decoder(self, z):
        x = F.elu(self.decFC2(z))
        x = F.elu(self.dexper1(x))
        x = F.elu(self.dexper2(x))
        x = F.elu(self.dec2(x))
        x = F.elu(self.dec1(x))
        return x


    def nll(self, x, x_hat):
        mse = ((x_hat - x) ** 2).view(x.shape[0], self.N, PIXEL).sum((2), True).sum(1)     
        nll = mse / (2 * self.vy)
        nll += 0.5 *self.N* torch.log(self.vy) * PIXEL
        return nll, mse

    def gp_make_cov(self, nxcirc= None, wwnrm= None, rho = 1, l_scale = torch.Tensor([15]), eps = 1e-4, Fourier = False):
        
        if Fourier:
            K_cov = gpf.mkcovs.mkcovdiag_ASD_wellcond(self.len_sc, 1, nxcirc, wwnrm = wwnrm,addition = eps)

        else:
            M1 = torch.arange(self.N).unsqueeze(0)-torch.arange(self.N).unsqueeze(1)
            K_cov = torch.exp(-torch.square(M1)/(2*torch.square(self.len_sc))) + eps*torch.eye(self.N)

        return K_cov* rho

    def gp_samp(self, K_cov, Fourier = False):
        if Fourier:
            print('ERROR')
        else:
            return np.array(np.random.multivariate_normal(np.zeros(self.N), K_cov, 1)).T 
            
    def forward(self, x, eps, spikes, Fourier = False):
        zm, zs = self.encode(x)      
        return zm, zs





