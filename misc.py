import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset
from hyperparams import *

torch.manual_seed(my_seed)
np.random.seed(my_seed)


def return_partitioned_latents(neural_embeds_m, image_embeds_m,
                               neural_embeds_s, image_embeds_s, N_lats_spikes, N_lats_img,
                               N_shared):
    z_m_neurons = neural_embeds_m[:,:,:N_lats_spikes-N_shared] 
    z_m_img = image_embeds_m[:,:,:N_lats_img-N_shared] 

    z_m_shared = (neural_embeds_m[:,:,N_lats_spikes-N_shared:] + 
                    image_embeds_m[:,:,N_lats_img-N_shared:])/2
    

    z_s_neurons = neural_embeds_s[:,:,:N_lats_spikes-N_shared] 
    z_s_img = image_embeds_s[:,:,:N_lats_img-N_shared] 


    z_s_shared = (neural_embeds_s[:,:,N_lats_spikes-N_shared:] + 
                    image_embeds_s[:,:,N_lats_img-N_shared:])/2

    z_m_tot = torch.cat((z_m_neurons,z_m_shared, z_m_img), dim=2) 
    z_s_tot = torch.cat((z_s_neurons,z_s_shared, z_s_img), dim=2) 

    return z_m_tot, z_s_tot

def regression_plot_helper(latent, true):
    z_latent = np.concatenate(latent.detach().numpy(), axis=0)
    true_trace = np.concatenate(true.detach().numpy(), axis=0)

    z_zoom = true_trace
    regr_list1 = []
    regr_list2 = []

    recon_z_const_zoom =np.stack((z_latent.squeeze(), np.ones(np.shape(true_trace))))


    W1 = np.linalg.lstsq(recon_z_const_zoom.T, z_zoom.T)[0]
    new_zoom = (W1.T@(recon_z_const_zoom)).T
    new_zoom = new_zoom.reshape(true.shape[0],TIME_POINTS,1)

    regr_list2.append(new_zoom)

    regr_list1.append(np.array(regr_list2).squeeze())

    regr_list1  = np.array(regr_list1)

    return regr_list1












def Fourer_lat_conv(z, enc):
        zm_transposed = torch.transpose(z, 1, 2)
        zm_four = enc(zm_transposed)
        zm_four = torch.transpose(zm_four, 1, 2)
        return zm_four

def Time_lat_conv(z, Bf):
        z_t = torch.transpose(z, 1, 2)
        z_time = torch.matmul(z_t, torch.Tensor(Bf))
        z_time =torch.transpose(z_time, 1, 2)
        return z_time




def gp_nll(K_cov, z, Fourier = False):
    
    zDim = z.shape[-1]
    if Fourier:
        log_prior = 0
        for ii in range(zDim):
            zonedim = z[:,:,ii]
            log_prior += (1/2)*(torch.sum(torch.square(zonedim)/K_cov+ torch.log(2*torch.pi*K_cov), axis = 1))
        log_prior = torch.squeeze(log_prior)
    else:
        K_inverse = torch.linalg.inv(K_cov) 
        log_prior = 0
        for ii in range(zDim):
            zonedim = z[:,:,ii].unsqueeze(2)
            zonedim_transposed = torch.transpose(zonedim, 1, 2)
            log_prior +=  0.5 * K_cov.shape[0]*torch.log(torch.Tensor([2*np.pi])) + 0.5 *torch.matmul(zonedim_transposed, torch.matmul(K_inverse, zonedim)) + torch.linalg.slogdet(K_cov)[1]

        log_prior = log_prior
    return log_prior




def calc_int_var(z, Bf):
        z_t = torch.transpose(z, 1, 2) 
        
        z_full = torch.diag_embed(z_t) 
        z_bf = torch.matmul(z_full, torch.Tensor(Bf))

        diag_covs = torch.sum(torch.Tensor(Bf)*z_bf, axis = 2)
        
        return diag_covs




def save_model(net, data, Fourier, model_saved):
    torch.manual_seed(my_seed)

    lambda_hats = []
    true_rates = []
    angles = []
    image_z = []
    images_end = []
    x_hat_end = []
    gp1list = []
    gp2list = []
    z_neuronLIST = []


    net.eval()
    with torch.no_grad():
        for imgs, angle in data:

            if Fourier == True:
            
                eps = Variable(torch.randn(
                    imgs.shape[0], latent_dim), requires_grad=False)
        
                embs_m_img, embs_s_img = net.encode(imgs.float())

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)

            
                z_m_img = embs_m_img
                z_s_img = embs_s_img
                z_m_tot = z_m_img
                z_s_tot = z_s_img
                z_tot = net.sample(z_m_tot, z_s_tot, eps)

                z_tot_time = Time_lat_conv(z_tot, net.Bf)

                z_m_tot_time = Time_lat_conv(z_m_tot, net.Bf)

                x_hat = net.decoder(z_tot_time[:, :, -(N_lats_img):])
               
                x_hat = x_hat.reshape(imgs.shape[0], TIME_POINTS, 36, 36)
 
                angles.append(angle.detach().numpy())
                image_z.append(z_m_tot_time[:, :, -(N_lats_img):].detach().numpy())
                x_hat_end.append(x_hat.detach().numpy())
                images_end.append(imgs.detach().numpy())

            if Fourier == False:
                eps = Variable(torch.randn(
                imgs.shape[0], latent_dim), requires_grad=False)
    
                embs_m_img, embs_s_img = net.encode(imgs.float())

                z_m_img = embs_m_img
                z_s_img = embs_s_img
                z_m_tot = z_m_img
                z_s_tot = z_s_img
                z_tot = net.sample(z_m_tot, z_s_tot, eps)

                x_hat = net.decoder(z_tot[:, :, -(N_lats_img):])
            
                x_hat = x_hat.reshape(imgs.shape[0], TIME_POINTS, 36, 36)

                angles.append(angle.detach().numpy())
                image_z.append(z_m_tot[:, :, -(N_lats_img):].detach().numpy())
                x_hat_end.append(x_hat.detach().numpy())
                images_end.append(imgs.detach().numpy())


    return np.save(f'{model_saved}.npy', { 'image_z': image_z, 'angle': angles,
                                                                'xhat': x_hat_end, 'images': images_end,})




def save_multimodal(net, n_encode, n_decode, data_load, mmgpvae ,gpvae, gpfa, FOURIER = True,
                    name = 'fourier_mmgpvae03closedform', zero_image = 1,
                    zero_neuron = 1):

    torch.manual_seed(my_seed)

    lambda_hats = []
    true_rates = []
    angles = []
    image_z = []
    images_end = []
    x_hat_end = []
    gp1list = []
    gp2list = []
    z_neuronLIST = []
    zooms = []
    true_zoom = []
    n_encode.eval()
    n_decode.eval()
    net.eval()
    with torch.no_grad():
        for imgs, angle, spikes_n, neural, gp1, gp2, zoom in data_load:

            if FOURIER == True:
          
                eps = Variable(torch.randn(
                    imgs.shape[0], latent_dim), requires_grad=False)
                embs_m_n, embs_s_n = n_encode.encode(spikes_n.float())
                embs_m_img, embs_s_img = net.encode(imgs.float())

                embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour)
                embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour)

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)

                z_m_neurons = embs_m_n[:, :, :N_lats_spikes-N_shared] * zero_neuron
                z_m_img = embs_m_img[:, :, :N_lats_img-N_shared] * zero_image

               
                if gpvae:
                    z_m_shared = embs_m_img[:,:,N_lats_img-N_shared:]
                if gpfa:
                    z_m_shared = embs_m_n[:,:,N_lats_spikes-N_shared:]
                if mmgpvae:
                    z_m_shared = (embs_m_n[:,:,N_lats_spikes-N_shared:] + 
                                embs_m_img[:,:,N_lats_img-N_shared:])/2

                z_s_neurons = embs_s_n[:, :, :N_lats_spikes-N_shared] * zero_neuron
                z_s_img = embs_s_img[:, :, :N_lats_img-N_shared] * zero_image
                if gpvae:
                    z_s_shared = embs_s_img[:,:,N_lats_img-N_shared:]
                if gpfa:
                    z_s_shared = embs_s_n[:,:,N_lats_spikes-N_shared:]
                if mmgpvae:
                    z_s_shared = (embs_s_n[:,:,N_lats_spikes-N_shared:] + 
                                embs_s_img[:,:,N_lats_img-N_shared:])/2

                # pay attention to order
                z_m_tot = torch.cat((z_m_neurons, z_m_shared, z_m_img), dim=2)
                z_s_tot = torch.cat((z_s_neurons, z_s_shared, z_s_img), dim=2)
                z_tot = net.sample(z_m_tot, z_s_tot, eps)

                z_tot_time = Time_lat_conv(z_tot, net.Bf)

                z_m_tot_time = Time_lat_conv(z_m_tot, net.Bf)

                x_hat = net.decoder(z_tot_time[:, :, -(N_lats_img):])
      
                x_hat = x_hat.reshape(imgs.shape[0], TIME_POINTS, 36, 36)

                combine = n_decode.decoder_lambda(z_tot_time[:, :, :N_lats_spikes])
                print(z_m_tot_time[:,:,N_lats_img].shape)
                print(z_m_tot_time[:,:,N_lats_img-N_shared].shape)

                lambda_hats.append(combine.detach().numpy())
                true_rates.append(neural.detach().numpy())
                angles.append(angle.detach().numpy())
                image_z.append(z_m_tot_time[:,:,N_lats_img-N_shared].detach().numpy()) 
                x_hat_end.append(x_hat.detach().numpy())
                images_end.append(imgs.detach().numpy())
                zooms.append(z_m_tot_time[:,:,N_lats_img].detach().numpy())

                gp1list.append(gp1.detach().numpy())
                z_neuronLIST.append(
                    z_m_tot_time[:, :, :N_lats_spikes-N_shared].detach().numpy())
                gp2list.append(gp2.detach().numpy())
                true_zoom.append(zoom.detach().numpy())

            if FOURIER == False:
                eps = Variable(torch.randn(
                    imgs.shape[0], latent_dim), requires_grad=False)
                embs_m_n, embs_s_n = n_encode.encode(spikes_n.float())
                embs_m_img, embs_s_img = net.encode(imgs.float())

                z_m_neurons = embs_m_n[:, :, :N_lats_spikes-N_shared] * zero_neuron
                z_m_img = embs_m_img[:, :, :N_lats_img-N_shared] * zero_image
                z_m_shared = (embs_m_n[:, :, N_lats_spikes-N_shared:] +
                            embs_m_img[:, :, N_lats_img-N_shared:])/2

                z_s_neurons = embs_s_n[:, :, :N_lats_spikes-N_shared] * zero_neuron
                z_s_img = embs_s_img[:, :, :N_lats_img-N_shared] * zero_image
                z_s_shared = (embs_s_n[:, :, N_lats_spikes-N_shared:] +
                            embs_s_img[:, :, N_lats_img-N_shared:])/2

                
                z_m_tot = torch.cat((z_m_neurons, z_m_shared, z_m_img), dim=2)
                z_s_tot = torch.cat((z_s_neurons, z_s_shared, z_s_img), dim=2)
                z_tot = net.sample(z_m_tot, z_s_tot, eps)

                x_hat_noF = net.decoder(z_tot[:, :, -(N_lats_img):])

                combine_noF = n_decode.decoder_lambda(z_tot[:, :, :N_lats_spikes])

                lambda_hats.append(combine_noF.detach().numpy())
                true_rates.append(neural.detach().numpy())
                angles.append(angle.detach().numpy())
                image_z.append(z_m_tot[:,:,N_lats_img-N_shared].detach().numpy())
                x_hat_end.append(x_hat_noF.detach().numpy())
                images_end.append(imgs.detach().numpy())
                zooms.append(z_m_tot[:,:,N_lats_img].detach().numpy())

                gp1list.append(gp1.detach().numpy())
                z_neuronLIST.append(
                    z_m_tot[:, :, :N_lats_spikes-N_shared].detach().numpy())
                gp2list.append(gp2.detach().numpy())
                true_zoom.append(zoom.detach().numpy())

            

    if FOURIER == True:
        np.save(f'./{name}.npy', {'true_rates': true_rates, 'lambda_hat': lambda_hats, 'image_z': image_z, 'angle': angles,
                                                                'xhat': x_hat_end, 'images': images_end, 'gp1': gp1list, 'gp2': gp2list, 'zneuron': z_neuronLIST,
                                                                'zooms': zooms,
                                                                'true_z': true_zoom})
    if FOURIER == False:
        np.save(f'./{name}.npy', {'true_rates': true_rates, 'lambda_hat': lambda_hats, 'image_z': image_z, 'angle': angles,
                                                                'xhat': x_hat_end, 'images': images_end, 'gp1': gp1list, 'gp2': gp2list, 
                                                                'zneuron': z_neuronLIST,'zooms': zooms,
                                                                'true_z': true_zoom})








def plot_figures(net, n_encode, n_decode, data_load, BATCH, 
                image = False, current_batch = 1, 
                trial_number = 1, cols = 5, batch_number = 10,
                Fourier= True, save= False, save_name='output'):

    
    plt.figure(figsize=(10,2), dpi=150)
    np.random.seed(my_seed)
    images = []
    angles = []
    spikes = []
    neuralRates = []
    sharedGP = []
    neuronGP = []
    zooms = []
    for imgs, angle, spikes_n, neural, gp1, gp2, zoom in data_load:
        images.append(imgs)
        angles.append(angle)
        spikes.append(spikes_n)
        neuralRates.append(neural)
        sharedGP.append(gp1)
        neuronGP.append(gp2)
        zooms.append(zoom)

    current_batch =batch_number

    batch_imgs = images[current_batch]
    batch_angles = angles[current_batch]
    batch_spikes = spikes[current_batch]
    batch_neuralRates = neuralRates[current_batch]
    batch_sharedGP = sharedGP[current_batch]
    batch_neuronGP = neuronGP[current_batch]
    batch_zoom = zooms[current_batch]

    eps = Variable(torch.randn(imgs.shape[0],latent_dim ), requires_grad=False)
    embs_m_n, embs_s_n = n_encode.encode(batch_spikes.float())
    embs_m_img, embs_s_img= net.encode(batch_imgs.float())


    if Fourier == True:
        embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour)
        embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) 
                    
                
        embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
        embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour) 


    z_m_neurons = embs_m_n[:,:,:N_lats_spikes-N_shared]
    z_m_img = embs_m_img[:,:,:N_lats_img-N_shared]
    z_m_shared = (embs_m_n[:,:,N_lats_spikes-N_shared:] + embs_m_img[:,:,N_lats_img-N_shared:])/2

    z_s_neurons = embs_s_n[:,:,:N_lats_spikes-N_shared]
    z_s_img = embs_s_img[:,:,:N_lats_img-N_shared]
    z_s_shared = (embs_s_n[:,:,N_lats_spikes-N_shared:] + embs_s_img[:,:,N_lats_img-N_shared:])/2

    z_m_tot = torch.cat((z_m_neurons,z_m_shared, z_m_img), dim=2) 
    z_s_tot = torch.cat((z_s_neurons,z_s_shared, z_s_img), dim=2)
    z_tot= net.sample(z_m_tot, z_s_tot, eps) 

    if Fourier:
        z_tot_time = Time_lat_conv(z_tot, net.Bf)
        z_m_tot_time = Time_lat_conv(z_m_tot, net.Bf)
        x_hat = net.decoder(z_m_tot_time[:,:,-(N_lats_img):])
        combine = n_decode.decoder_lambda(z_m_tot_time[:,:,:N_lats_spikes])
        
    x_hat_noF = net.decoder(z_m_tot[:,:,-(N_lats_img):])
    

    combine_noF = n_decode.decoder_lambda(z_m_tot[:,:,:N_lats_spikes])


    
    rows = 2
    n = cols
    fig, axs = plt.subplots(rows, n)
    fig.set_figheight(8)
    fig.set_figwidth(16)

    for i in range(n):
        if image == True: 
            
            batch_imgs = batch_imgs.reshape(BATCH, TIME_POINTS, 36,36)
            if Fourier:
                x_hat = x_hat.reshape(BATCH, TIME_POINTS, 36,36)
            else:
                x_hat_noF = x_hat_noF.reshape(BATCH, TIME_POINTS, 36,36)

            axs[0,i].title.set_text('X')
            axs[0,i].imshow(batch_imgs[trial_number][i].detach().numpy(), cmap='gray')

            if Fourier:
                axs[1,i].title.set_text('Recon X')
                axs[1,i].imshow(x_hat[trial_number][i].detach().numpy(), cmap='gray')
            else:
                axs[1,i].title.set_text('Reconstructed X')
                axs[1,i].imshow(x_hat_noF[trial_number][i].detach().numpy(), cmap='gray')
            axs[0,i].axis('off')
            axs[1,i].axis('off')
            
        else:
            if Fourier:
                axs[0,i].title.set_text('Est neural rates')
                axs[0,i].plot(((torch.exp(combine)[trial_number+i, :, 0:3]).detach().numpy()))
            else:
                axs[0,i].title.set_text('Est neural rates')
                axs[0,i].plot(((torch.exp(combine_noF)[trial_number+i, :, 0:3]).detach().numpy()))

            axs[1,i].title.set_text('True neural rates')
            axs[1,i].plot((batch_neuralRates[trial_number+i, :, 0:3]).detach().numpy())

            axs[0,i].spines['right'].set_visible(False)
            axs[0,i].spines['top'].set_visible(False)
            axs[1,i].spines['right'].set_visible(False)
            axs[1,i].spines['top'].set_visible(False)
        if save == True:
            fig.savefig(save_name,  dpi=800)





def regressions(net, n_encode, n_decode, data_load, BATCH, 
                current_batch = 1, 
                trial_number = 1, cols = 5, batch_number = 10,
                save= False, save_name='output', Fourier = True):
    np.random.seed(my_seed)
    images = []
    angles = []
    spikes = []
    neuralRates = []
    sharedGP = []
    neuronGP = []
    regr_four = []
    regr_four_zoom = []
    regr_n_only = []
    zooms = []


    for imgs, angle, spikes_n, neural, gp1, gp2, zoom in data_load:
        images.append(imgs)
        angles.append(angle)
        spikes.append(spikes_n)
        neuralRates.append(neural)
        sharedGP.append(gp1)
        neuronGP.append(gp2)
        zooms.append(zoom)

    current_batch = 0

    batch_imgs = images[current_batch]
    batch_angles = angles[current_batch]
    batch_spikes = spikes[current_batch]
    batch_neuronGP = neuronGP[current_batch]
    batch_zoom = zooms[current_batch]

    eps = Variable(torch.randn(imgs.shape[0],latent_dim ), requires_grad=False)
    embs_m_n, embs_s_n = n_encode.encode(batch_spikes.float())
    embs_m_img, embs_s_img= net.encode(batch_imgs.float())


    if Fourier == True:
        embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour)
        embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) 
                    
                
        embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
        embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour) 

    z_m_tot, z_s_tot = return_partitioned_latents(neural_embeds_m=embs_m_n, 
                                                          image_embeds_m=embs_m_img,
                               neural_embeds_s = embs_s_n, image_embeds_s =embs_s_img, 
                               N_lats_spikes=N_lats_spikes, N_lats_img=N_lats_img,
                               N_shared=N_shared)
    z_tot= net.sample(z_m_tot, z_s_tot, eps) 

    if Fourier:
        z_tot_time = Time_lat_conv(z_tot, net.Bf)
        z_m_tot_time = Time_lat_conv(z_m_tot, net.Bf)
        x_hat = net.decoder(z_m_tot_time[:,:,-(N_lats_img):])
        x_hat = x_hat.reshape(imgs.shape[0], TIME_POINTS, 36,36)
     
        regr_four  = regression_plot_helper(latent = z_m_tot_time[:,:,N_lats_img-N_shared], true= batch_angles)

        new_regr_four = np.transpose(regr_four, (1,2,0))

        #######################################################################
        # for zoom
      
        regr_four_zoom = regression_plot_helper(latent = z_m_tot_time[:,:,N_lats_img], true= batch_zoom)

        new_regr_four_zoom = np.transpose(regr_four_zoom, (1,2,0))
   
    #########################################
      
        regr_n_only = regression_plot_helper(latent = z_m_tot_time[:, :, :N_lats_spikes-N_shared], true= batch_neuronGP)

        regr_n_only = regr_n_only.reshape(imgs.shape[0],TIME_POINTS,1)

    trial_number = trial_number
    n = cols
    fig, axs = plt.subplots(3, n)
    fig.set_figheight(12)
    fig.set_figwidth(20)


    for i in range(n):

            axs[0,i].title.set_text('Shared')
            axs[0,i].plot((new_regr_four[trial_number+i]), color='red')
            axs[0,i].plot((batch_angles[trial_number+i][:]), color='blue')

            axs[1,i].title.set_text('Behavior Independent')
            axs[1,i].plot((new_regr_four_zoom[trial_number+i]),color='red')
            axs[1,i].plot((batch_zoom[trial_number+i][:]), color='blue')

            axs[2,i].title.set_text('Neural Independent')
            axs[2,i].plot((regr_n_only[trial_number+i]),color='red')
            axs[2,i].plot((batch_neuronGP[trial_number+i][:]), color='blue')


            axs[0,i].spines['right'].set_visible(False)
            axs[0,i].spines['top'].set_visible(False)
            axs[1,i].spines['right'].set_visible(False)
            axs[1,i].spines['top'].set_visible(False)
            axs[2,i].spines['right'].set_visible(False)
            axs[2,i].spines['top'].set_visible(False)


    plt.show()


def regressions_main_alternative(net_encode, net_decode,n_encode, n_decode, data_load, BATCH, 
                current_batch = 1, 
                trial_number = 1, cols = 5, batch_number = 10,
                save= False, save_name='output', Fourier = True):
    np.random.seed(my_seed)
    images = []
    angles = []
    spikes = []
    neuralRates = []
    sharedGP = []
    neuronGP = []

    regr_four = []
    regr_four_zoom = []
    regr_n_only = []
    zooms = []


    for imgs, angle, spikes_n, neural, gp1, gp2, zoom in data_load:
        images.append(imgs)
        angles.append(angle)
        spikes.append(spikes_n)
        neuralRates.append(neural)
        sharedGP.append(gp1)
        neuronGP.append(gp2)
        zooms.append(zoom)

    current_batch = 0

    batch_imgs = images[current_batch]
    batch_angles = angles[current_batch]
    batch_spikes = spikes[current_batch]
    batch_neuronGP = neuronGP[current_batch]
    batch_zoom = zooms[current_batch]

    eps = Variable(torch.randn(imgs.shape[0],latent_dim), requires_grad=False)
    eps_n = Variable(torch.randn(imgs.shape[0],latent_dim), requires_grad=False)
  
    
    embs_m_n, embs_s_n = n_encode.forward(batch_spikes.float(), eps_n, Fourier = Fourier)
    embs_m_img, embs_s_img= net_encode.forward(batch_imgs.float(), Fourier = Fourier) #only encoding the z mean and var in time domain


    if Fourier:

        embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour) 
        embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) 
    
        embs_m_img = Fourer_lat_conv(embs_m_img, net_encode.encMeanFour) 
        embs_s_img = Fourer_lat_conv(embs_s_img, net_encode.encVarFour)  
        

    
    z_m_tot, z_s_tot = return_partitioned_latents(neural_embeds_m=embs_m_n, 
                                                          image_embeds_m=embs_m_img,
                               neural_embeds_s = embs_s_n, image_embeds_s =embs_s_img, 
                               N_lats_spikes=N_lats_spikes, N_lats_img=N_lats_img,
                               N_shared=N_shared)

    z_tot= net_encode.sample(z_m_tot, z_s_tot, eps) 

    if Fourier:
        z_tot_time = Time_lat_conv(z_tot, net_encode.Bf)
        z_m_tot_time = Time_lat_conv(z_m_tot, net_encode.Bf)
        x_hat, _ = net_decode.forward(z_m_tot_time[:,:,-(N_lats_img):], batch_imgs)
        x_hat = x_hat.reshape(imgs.shape[0], TIME_POINTS, 36,36)

        regr_four  = regression_plot_helper(latent = z_m_tot_time[:,:,N_lats_img-N_shared], true= batch_angles)

        new_regr_four = np.transpose(regr_four, (1,2,0))


        #######################################################################
        # for zoom
        regr_four_zoom = regression_plot_helper(latent = z_m_tot_time[:,:,N_lats_img], true= batch_zoom)


        new_regr_four_zoom = np.transpose(regr_four_zoom, (1,2,0))
   
    #########################################
        regr_n_only = regression_plot_helper(latent = z_m_tot_time[:, :, :N_lats_spikes-N_shared], true= batch_neuronGP)

        regr_n_only = regr_n_only.reshape(imgs.shape[0],TIME_POINTS,1)

    trial_number = trial_number
    n = cols
    fig, axs = plt.subplots(3, n)
    fig.set_figheight(12)
    fig.set_figwidth(20)


    for i in range(n):

            axs[0,i].title.set_text('Shared')
            axs[0,i].plot((new_regr_four[trial_number+i]), color='red')
            axs[0,i].plot((batch_angles[trial_number+i][:]), color='blue')

            axs[1,i].title.set_text('Behavior Independent')
            axs[1,i].plot((new_regr_four_zoom[trial_number+i]),color='red')
            axs[1,i].plot((batch_zoom[trial_number+i][:]), color='blue')

            axs[2,i].title.set_text('Neural Independent')
            axs[2,i].plot((regr_n_only[trial_number+i]),color='red')
            axs[2,i].plot((batch_neuronGP[trial_number+i][:]), color='blue')


            axs[0,i].spines['right'].set_visible(False)
            axs[0,i].spines['top'].set_visible(False)
            axs[1,i].spines['right'].set_visible(False)
            axs[1,i].spines['top'].set_visible(False)
            axs[2,i].spines['right'].set_visible(False)
            axs[2,i].spines['top'].set_visible(False)


    plt.show()


class TimePointCustomDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        all_data = np.load(data_path, allow_pickle=True)
        
        self.angles = torch.from_numpy(np.squeeze(all_data.item()['GP_angs']))
        self.neural_rates = torch.from_numpy(all_data.item()['rates'])
        self.spikes = torch.from_numpy(all_data.item()['spikes'])
        self.trials = torch.from_numpy(all_data.item()['imgs'])
        # print(self.trials.shape)
        self.gp1 = torch.from_numpy(all_data.item()['GPs'][:,0])
        self.gp2 = torch.from_numpy(all_data.item()['GPs'][:,1])
        self.zoom = torch.from_numpy(np.squeeze(all_data.item()['GP_zooms']))
        # print(self.gp2.shape)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx): 
        trials = self.trials[idx]
        angles = self.angles[idx,:]
        spike_rates = self.spikes[idx,:]
        neural_rates = self.neural_rates[idx,:]
        gp1 = self.gp1[idx,:]
        gp2 = self.gp2[idx,:]
        zoom = self.zoom[idx,:]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transforma(label)

        return trials, angles, spike_rates, neural_rates, gp1, gp2, zoom 
    

class GPVAE_dataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        all_data = np.load(data_path, allow_pickle=True)
        self.angles = torch.from_numpy(np.squeeze(all_data.item()['GP_angs']))
        self.neural_rates = torch.from_numpy(all_data.item()['rates'])
        self.spikes = torch.from_numpy(all_data.item()['spikes'])
        self.trials = torch.from_numpy(all_data.item()['imgs'])
        self.gp1 = torch.from_numpy(all_data.item()['GPs'][:, 0])
        self.gp2 = torch.from_numpy(all_data.item()['GPs'][:, 1])
    
        self.transform = transform
        self.target_transform = target_transform
        self.imagemean = (self.trials).mean()

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):  
        trials = self.trials[idx] - self.imagemean
        angles = self.angles[idx, :]
        spike_rates = self.spikes[idx, :]
        neural_rates = self.neural_rates[idx, :]
        gp1 = self.gp1[idx, :]
        gp2 = self.gp2[idx, :]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transforma(label)
        return trials, angles
