import numpy as np

##################################
## Uncomment this block if want to run the GPVAE.ipynb 
##################################
## gpvae
##################################
my_seed = 20

DATA_SPLIT_SEED = 10

BATCH = 5

TIME_POINTS = 40 

N_NEURONS = 100

PIXEL = 36*36 

TOTAL_EPOCH = 100

trial_number = 1

image_lr = 0.0005

N_lats_img = 1
N_lats_spikes = 0
N_shared = 0

latent_dim = 1
y = np.ones([1,N_NEURONS,TIME_POINTS])

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


##################################
## Uncomment the block below if you want to run MMGPVAE.ipynb
##################################
## mmgpvae
##################################

# BATCH = 20

# TIME_POINTS = 60 

# N_NEURONS = 100

# PIXEL = 36*36 

# TOTAL_EPOCH = 700

# trial_number = 1

# my_seed = 77

# DATA_SPLIT_SEED = 10

# image_lr =0.00016

# N_lats_img = 2
# N_lats_spikes = 2
# N_shared = 1

# latent_dim = 1
# y = np.ones([1,N_NEURONS,TIME_POINTS])