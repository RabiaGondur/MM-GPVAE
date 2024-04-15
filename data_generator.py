from cgi import test
from numpy.random.mtrand import sample
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import zoom

## Directory info and hyperparameters
########################################################################
seed_no = 32
np.random.seed(seed_no)

PATH = '../trainingSet/'
NO_ZOOM = False
if NO_ZOOM == False:
    SAVED_PATH = f'./mmgpvae_data_seed{seed_no}'
    N = 60
if NO_ZOOM == True:
    SAVED_PATH = f'./gpvae_data_seed{seed_no}'
    N = 40 
########################################################################

num_trials = 300  
tps = np.arange(0, N)  
test_array = np.empty((0, 1))
rotated_MNIST_alltrials = np.empty((num_trials, N, 1296))

len_sc_true = 10
rh1 = .3
rh2 = .5
rh3 = .8

M1 = np.array([range(N)]) - np.transpose(np.array([range(N)]))
K = rh1*np.exp(-(np.square(M1)/(2*np.square(len_sc_true))))
samp_GP = np.array(np.random.multivariate_normal(np.zeros(N), K, (num_trials)))

K = rh2*np.exp(-(np.square(M1)/(2*np.square(len_sc_true))))
samp_GP2 = np.array(np.random.multivariate_normal(
    np.zeros(N), K, (num_trials)))

K = rh3*np.exp(-(np.square(M1)/(2*np.square(len_sc_true))))
samp_GP3 = np.array(np.random.multivariate_normal(
    np.zeros(N), K, (num_trials))) + 1


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple)


    elif zoom_factor > 1:
        zh = int(np.ceil(h / zoom_factor))
        zw = int(np.ceil(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple)
        prev_out = out
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
        if out.shape[0] == 1:
            print("broken")
    else:
        out = img

    return out


GP_angs = samp_GP*2*np.pi*(180/np.pi)/3
GP_angs = np.expand_dims(GP_angs, 2)
GP_zooms = np.expand_dims(samp_GP3, 2)


GPs = np.stack([samp_GP, samp_GP2], axis=1)
n_neurons = 100
W = 1.5 * np.random.uniform(low=-1, high=1.0, size=(n_neurons,2))
offsets = 0.5 * np.random.uniform(low=-1, high=1.0, size=n_neurons)
neural_rates = np.exp(np.array([W@GPs[ii, :] + offsets[:, None]
                      for ii in range(num_trials)]))  
neural_rates = np.transpose(neural_rates, [0, 2, 1])

spikes = np.random.poisson(neural_rates)
num_3 = N * num_trials
digit_mod = {'3': num_3}


pth = PATH

for digit in digit_mod.keys():
    start_range = 0
    end_range = N
    count = 0
    for n in np.arange(num_trials):
        print("Creating instances of digit {}".format(digit))
        print("for trial {}".format(n))
        rotated_MNIST = np.empty((0, 1296))
        angle_img_idx = 0
        data_path = os.path.join(pth, digit)
        files = glob.glob('{}/*.jpg'.format(data_path))

        while angle_img_idx < len(GP_angs[n]):
            if count < len(files):
                original_image = plt.imread(files[count])
                count += 1
            else:
                count = 0
                original_image = plt.imread(files[count])
                count += 1
            original_image_pad = np.pad(
                original_image, ((4, 4), (4, 4)), 'constant')
            img = ndimage.rotate(original_image_pad,
                                 angle=np.squeeze(GP_angs[n, angle_img_idx]), reshape=False)

            if NO_ZOOM == False:
                img = clipped_zoom(
                    img, abs(np.squeeze(GP_zooms[n, angle_img_idx])))

            # if np.all(img == 0):
            #     print("all zeros")
            rotated_MNIST = np.append(
                rotated_MNIST, np.reshape(img, (1, 1296)), axis=0)


            angle_img_idx += 1

        start_range = start_range + N
        end_range = end_range + N

        rotated_MNIST_alltrials[n, :, :] = np.reshape(
            rotated_MNIST, (1, N, 1296))

np.save(SAVED_PATH,
        {'GPs': GPs, 'GP_angs': GP_angs, 'GP_zooms': GP_zooms, 
        'rates': neural_rates, 'spikes': spikes, 
        'imgs': rotated_MNIST_alltrials})


