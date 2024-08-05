import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
import GP_fourier as gpf
from train import *
from misc import *
from neural_nets import *

torch.manual_seed(my_seed)
np.random.seed(my_seed)


### Data paths
DATA_PATH = './mmgpvae_data_seed32.npy'
FOURIER = True

###################################################################
full_dataset = TimePointCustomDataset(DATA_PATH)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
generator=torch.Generator().manual_seed(DATA_SPLIT_SEED))
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH)
loader_test= torch.utils.data.DataLoader(test_dataset, batch_size=BATCH)

## Unbatched data
torch.manual_seed(my_seed)
full_data = torch.utils.data.DataLoader(full_dataset, batch_size=BATCH)
train_data_unbatched = torch.utils.data.DataLoader(train_dataset, 
batch_size=train_dataset.__len__())
test_data_unbatched = torch.utils.data.DataLoader(test_dataset, 
batch_size=test_dataset.__len__())

for data, labels, spikes, neural_rates, gp1, gp2, zoom in train_data_unbatched:
    meanrates = np.mean(neural_rates.detach().numpy(), axis = (0,1))
    break
###################################################################


# net= Behavior_Encoder_Decoder(zimg_Dim=N_lats_img, 
#                               Fourier = FOURIER,  minlens =10)

net_encode= Behavior_Encoder(zimg_Dim=N_lats_img, 
                              Fourier = FOURIER,  minlens =10)

net_decode= Behavior_Decoder(zimg_Dim=N_lats_img, 
                              Fourier = FOURIER,  minlens =10)

n_encode = Spike_Encode(zDim=N_lats_spikes, Fourier = FOURIER,
                        minlens =10)

n_decode = Spike_Decode(meanrates=meanrates,
                        zDim=N_lats_spikes, 
                        Fourier = FOURIER, minlens =10)


trained = train_mmgpvae_main_alternative(net_encode, net_decode, n_encode, n_decode, loader_train,
                EPOCH = TOTAL_EPOCH, lr1=0.00016, lr2=0.000124, lr3=0.000772, lr4=0.0088, 
                Fourier = FOURIER, visualize_ELBO = True)
#lr1=0.00016, lr2=0.00012,
###################################################################

with torch.no_grad():
    regressions_main_alternative(net_encode = net_encode, net_decode=net_decode, n_encode=n_encode, n_decode= n_decode,
                data_load=loader_test, BATCH=BATCH, cols = 5, trial_number=7, 
                batch_number=0, Fourier=FOURIER)