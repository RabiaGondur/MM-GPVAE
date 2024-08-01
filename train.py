import matplotlib.pyplot as plt
import torch
import numpy as np
from hyperparams import *
from torch.autograd import Variable
from misc import *
from torch import nn, optim

torch.manual_seed(my_seed)
np.random.seed(my_seed)

#############################################################
## gpvae
#############################################################

def train_model_fourier_gpvae(net, loader_train, EPOCH = TOTAL_EPOCH,
 lr1=0.00012, 
 Fourier=True, visualize_ELBO=True, save_dict = False, model_dict='test', eps_v= 1e-2):
 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)

    N_tot = N_lats_img + N_lats_spikes - N_shared



    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)


    losses = []
    recons = []
    k_covList = []
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        recon_term_epochs = 0

        for data, label in loader_train:
        
            optimizer_behave.zero_grad()
           

            net.train()
           

            labels = label
       
            new_data = data

        
            eps = Variable(torch.randn(
                new_data.shape[0], N_lats_img), requires_grad=False)
         
         
            embs_m_img, embs_s_img = net.forward(
                new_data.float(), eps, Fourier=Fourier)
          
            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)

           
           
            z_m_img = embs_m_img
       
        
            z_s_img = embs_s_img
         

            z_tot = net.sample(z_m_img, z_s_img, eps)
          
            pen_term = -z_s_img.sum([1, 2])  

            if Fourier == True:
           
                K_cov = net.gp_make_cov(
                    eps=eps_v, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)

            gpNLL = gp_nll(K_cov, z_tot, Fourier=Fourier)  

     
            
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
            elif Fourier == False:
                z_tot_time = z_tot

          

            x_hat = net.decoder(z_tot_time[:, :, -(N_lats_img):])
     
            recon_term, mse = net.nll(new_data, x_hat)
            recon_term = recon_term
            
        
            ELBO = (pen_term +  recon_term+ gpNLL).sum()
            loss = ELBO / BATCH

            trainloss += loss.item()
       

            loss.backward()

            optimizer_behave.step()
            

        if nan:
            break

        trainloss = trainloss/len(loader_train)
        losses.append(trainloss)
      
        k_covList.append(K_cov)

        if epoch % 50 == 0:
            print(
                f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5, 3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
    if save_dict == True:
        if Fourier == True:
            torch.save(net.state_dict(), f'./{model_dict}.pt')
          
        if Fourier == False:
            torch.save(net.state_dict(), f'./{model_dict}.pt')
           
    return losses, k_covList









#############################################################
## vae
#############################################################

def train_model_vae(net, loader_train, EPOCH = TOTAL_EPOCH,
 lr1=0.00012, 
 Fourier=True, visualize_ELBO=True, save_dict = False, model_dict='test'):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)

    N_tot = N_lats_img + N_lats_spikes - N_shared


    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)


    losses = []
    recons = []
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        recon_term_epochs = 0

        for data, label in loader_train:
        
            optimizer_behave.zero_grad()
           

            net.train()
           

            labels = label
          
            new_data = data

            eps = Variable(torch.randn(
                new_data.shape[0], N_lats_img), requires_grad=False)
         
         
            embs_m_img, embs_s_img = net.forward(
                new_data.float(), eps,  Fourier=Fourier)
      
            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)

           
            z_m_img = embs_m_img
       
        
            z_s_img = embs_s_img
         

            z_tot = net.sample(z_m_img, z_s_img, eps)
       
            pen_term = -z_s_img.sum([1, 2]) 

            if Fourier == True:
      
                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)

            gpNLL = gp_nll(K_cov, z_tot, Fourier=Fourier)  

         
            
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
            elif Fourier == False:
                z_tot_time = z_tot

        

            x_hat = net.decoder(z_tot_time[:, :, -(N_lats_img):])
       
            recon_term, mse = net.nll(new_data, x_hat)
            recon_term = recon_term
           
            criterion = nn.GaussianNLLLoss(full=True, reduction = 'sum')
            target = torch.zeros_like(z_tot_time)
            var = torch.ones_like(z_tot_time) 
            normalnll = criterion(z_tot_time, target, var)
         
            ELBO = (pen_term +  recon_term + normalnll).sum()
            
            loss = ELBO / BATCH

            trainloss += loss.item()
          

            loss.backward()

            optimizer_behave.step()
            

        if nan:
            break

        trainloss = trainloss/len(loader_train)
        losses.append(trainloss)
    

        if epoch % 50 == 0:
            print(
                f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5, 3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
    if save_dict == True:
        if Fourier == True:
            torch.save(net.state_dict(), f'./{model_dict}.pt')
          
        if Fourier == False:
            torch.save(net.state_dict(), f'./{model_dict}.pt')
           
    return losses  









#####################################################################################
## mmgpvae closed form (use it for gpfa and gpvae if you want to encode both modality)
#####################################################################################

def train_model_mmgpvae_closed(net, n_encode, n_decode, data_load,
                EPOCH = 650, lr1=0.00016, lr2=0.000772, lr3=0.0088, 
                Fourier = True, visualize_ELBO = True, zero_neuro = 1,
                zero_image = 1, gpvae_mode = False,
                gpfa_mode = False, mmgpvae = False):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)
    
    N_tot = N_lats_img + N_lats_spikes - N_shared

    y = np.ones([1,N_NEURONS,TIME_POINTS])

    if gpvae_mode:
        for param in n_encode.parameters():
            param.requires_grad = False
        for param in n_decode.parameters():
            param.requires_grad = False
        for param in net.parameters():
            param.requires_grad = True
    if gpfa_mode:
        for param in n_encode.parameters():
            param.requires_grad = True
        for param in n_decode.parameters():
            param.requires_grad = True
        for param in net.parameters():
            param.requires_grad = False
    if mmgpvae:
        for param in n_encode.parameters():
            param.requires_grad = True
        for param in n_decode.parameters():
            param.requires_grad = True
        for param in net.parameters():
            param.requires_grad = True

    
    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)
    optimizer_n_encode = torch.optim.Adam(n_encode.parameters(), lr=lr2)
    optimizer_n_decode = torch.optim.Adam(n_decode.parameters(), lr=lr3) #.0088


    
    losses = []
    max_grad_norm = 1
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        


        for data, label, spikes, neural_rates, gp1, gp2, zoom in data_load:
            

            if gpfa_mode:
             
                optimizer_n_encode.zero_grad()
                optimizer_n_decode.zero_grad()

            if gpvae_mode:
                optimizer_behave.zero_grad()
            
            if mmgpvae:
                optimizer_behave.zero_grad()
                optimizer_n_encode.zero_grad()
                optimizer_n_decode.zero_grad()
            
        
            if gpfa_mode:
             
                n_decode.train()
                n_encode.train()

            if gpvae_mode:
                net.train()
            
            if mmgpvae:
                net.train()
                n_decode.train()
                n_encode.train()


            labels = label
            spikes = spikes 
            new_data = data
        
          
            eps = Variable(torch.randn(new_data.shape[0],N_lats_img ), requires_grad=False)
            eps_n = Variable(torch.randn(new_data.shape[0], N_lats_spikes ), requires_grad=False)
            new_data, eps = new_data.to(device), eps.to(device)
            
            embs_m_n, embs_s_n = n_encode.forward(spikes.float(), eps_n, Fourier = Fourier)
            embs_m_img, embs_s_img= net.forward(new_data.float(), eps, spikes, Fourier = Fourier) #only encoding the z mean and var in time domain

        
            if Fourier:

                embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour) * zero_neuro
                embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) * zero_neuro
                
            

            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour) * zero_image
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)  * zero_image
                

         
            z_m_neurons = embs_m_n[:,:,:N_lats_spikes-N_shared] * zero_neuro
            z_m_img = embs_m_img[:,:,:N_lats_img-N_shared] * zero_image
            if gpvae_mode:
                z_m_shared = embs_m_img[:,:,N_lats_img-N_shared:]
            if gpfa_mode:
                z_m_shared = embs_m_n[:,:,N_lats_spikes-N_shared:]
            if mmgpvae:
                z_m_shared = (embs_m_n[:,:,N_lats_spikes-N_shared:] + 
                              embs_m_img[:,:,N_lats_img-N_shared:])/2
                

            z_s_neurons = embs_s_n[:,:,:N_lats_spikes-N_shared] * zero_neuro
            z_s_img = embs_s_img[:,:,:N_lats_img-N_shared] * zero_image

            if gpvae_mode:
                z_s_shared = embs_s_img[:,:,N_lats_img-N_shared:]
            if gpfa_mode:
                z_s_shared = embs_s_n[:,:,N_lats_spikes-N_shared:]
            if mmgpvae:
                z_s_shared = (embs_s_n[:,:,N_lats_spikes-N_shared:] + 
                              embs_s_img[:,:,N_lats_img-N_shared:])/2
      
            z_m_tot = torch.cat((z_m_neurons,z_m_shared, z_m_img), dim=2) 
            z_s_tot = torch.cat((z_s_neurons,z_s_shared, z_s_img), dim=2) 
    
            z_tot= net.sample(z_m_tot, z_s_tot, eps) 
          
            pen_term = -z_s_tot.sum([1,2])  


            if Fourier == True:
               
                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)



            gpNLL = gp_nll(K_cov, (z_m_tot+torch.exp(z_s_tot)), Fourier=Fourier)  
            
            
            
            
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
                
                z_m_tot_time = Time_lat_conv(z_m_tot[:,:,:N_lats_spikes], net.Bf) 
                lat_var_diag = calc_int_var(torch.exp(z_s_tot[:,:,:N_lats_spikes]), net.Bf)
          
                int_var = torch.matmul(torch.square(n_decode.dec3.weight), lat_var_diag)
            elif Fourier == False:
                "NOT IMPLEMENTED"
            lambda_hat = n_decode.dec3(z_m_tot_time)
            neural_loss = - ( -(torch.exp(lambda_hat + 0.5*torch.transpose(int_var,1,2))) -torch.lgamma(spikes + 1) +  lambda_hat * spikes).sum([1,2])


            neural_loss = neural_loss 
            
         
        
            x_hat = net.decoder(z_tot_time[:,:,-(N_lats_img):])

            recon_term, mse = net.nll(new_data, x_hat) 
            recon_term = recon_term 

         
            if gpvae_mode:
                ELBO = (pen_term + recon_term+ gpNLL).sum() 
            if gpfa_mode:
                ELBO = (pen_term + neural_loss + gpNLL).sum()
            if mmgpvae: 
                ELBO = (pen_term + neural_loss + recon_term+ gpNLL).sum() 
         
            loss = ELBO / BATCH

    

            trainloss += loss.item()

            
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n_decode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(n_encode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            if mmgpvae:
                optimizer_behave.step()
                optimizer_n_encode.step()
                optimizer_n_decode.step()
            if gpvae_mode:
                optimizer_behave.step()
            if gpfa_mode:
                optimizer_n_encode.step()
                optimizer_n_decode.step()

    
        if nan:
            break

        trainloss = trainloss/len(data_load)
        losses.append(trainloss)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5,3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        
    return losses






#####################################################################################
## Run GPVAE (for only encoding one modality)
#####################################################################################

def train_model_gpvae(net, n_encode, n_decode, data_load,
                EPOCH = 650, lr1=0.00016, lr2=0.000772, lr3=0.0088, 
                Fourier = True, visualize_ELBO = True):
   
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)
    
    N_tot = N_lats_img + N_lats_spikes - N_shared

    y = np.ones([1,N_NEURONS,TIME_POINTS])

    
    for param in n_encode.parameters():
        param.requires_grad = False
    for param in n_decode.parameters():
        param.requires_grad = False
    for param in net.parameters():
        param.requires_grad = True
    
    


    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)
    
    
    losses = []
    max_grad_norm = 1
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        


        for data, label, spikes, neural_rates, gp1, gp2, zoom in data_load:
            

            
            optimizer_behave.zero_grad()
            
            net.train()
            
            labels = label
            spikes = spikes 
            new_data = data
        
        
            eps = Variable(torch.randn(new_data.shape[0],N_lats_img ), requires_grad=False)
            
            new_data, eps = new_data.to(device), eps.to(device)
            
          
            embs_m_img, embs_s_img= net.forward(new_data.float(), eps, spikes, Fourier = Fourier) #only encoding the z mean and var in time domain

        

            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour) 
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)  
                

         
            
            z_m_img = embs_m_img[:,:,:N_lats_img-N_shared] 
            
            z_m_shared = embs_m_img[:,:,N_lats_img-N_shared:]
           
                

          
            z_s_img = embs_s_img[:,:,:N_lats_img-N_shared] 

         
            z_s_shared = embs_s_img[:,:,N_lats_img-N_shared:]
           

            z_m_tot = torch.cat((z_m_shared, z_m_img), dim=2) 
            z_s_tot = torch.cat((z_s_shared, z_s_img), dim=2) 
    
            z_tot= net.sample(z_m_tot, z_s_tot, eps) 
       
            pen_term = -z_s_tot.sum([1,2])  

  
            if Fourier == True:

                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)


            gpNLL = gp_nll(K_cov, (z_m_tot+torch.exp(z_s_tot)), Fourier=Fourier)  
            
         
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
        
            x_hat = net.decoder(z_tot_time[:,:,-(N_lats_img):])

            recon_term, mse = net.nll(new_data, x_hat) 
            recon_term = recon_term 

        
           
            ELBO = (pen_term + recon_term+ gpNLL).sum() 
            
            loss = ELBO / BATCH

    

            trainloss += loss.item()

            
        
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
        
            optimizer_behave.step()
       

    
        if nan:
            break

        trainloss = trainloss/len(data_load)
        losses.append(trainloss)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5,3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        
    return losses





###############################################################################
## Run GPFA (for only encoding one modality)
###############################################################################


def train_model_gpfa(net, n_encode, n_decode, data_load,
                EPOCH = 650, lr1=0.00016, lr2=0.000772, lr3=0.0088, 
                Fourier = True, visualize_ELBO = True):
  
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)
    
    N_tot = N_lats_img + N_lats_spikes - N_shared

    y = np.ones([1,N_NEURONS,TIME_POINTS])

    
    for param in n_encode.parameters():
        param.requires_grad = True
    for param in n_decode.parameters():
        param.requires_grad = True
    for param in net.parameters():
        param.requires_grad = False
   

    optimizer_n_encode = torch.optim.Adam(n_encode.parameters(), lr=lr2)
    optimizer_n_decode = torch.optim.Adam(n_decode.parameters(), lr=lr3) #.0088


    
    losses = []
    max_grad_norm = 1
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        


        for data, label, spikes, neural_rates, gp1, gp2, zoom in data_load:
            

            
            optimizer_n_encode.zero_grad()
            optimizer_n_decode.zero_grad()

           

            n_decode.train()
            n_encode.train()

            

            labels = label
            spikes = spikes 
            new_data = data
        
           
            
            eps_n = Variable(torch.randn(new_data.shape[0], N_lats_spikes ), requires_grad=False)
            
            
            embs_m_n, embs_s_n = n_encode.forward(spikes.float(), eps_n, Fourier = Fourier)
           

     
            if Fourier:

                embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour) 
                embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour)
                
        
            z_m_neurons = embs_m_n[:,:,:N_lats_spikes-N_shared] 
           
            z_m_shared = embs_m_n[:,:,N_lats_spikes-N_shared:]
          
                

            z_s_neurons = embs_s_n[:,:,:N_lats_spikes-N_shared] 
            
            
            z_s_shared = embs_s_n[:,:,N_lats_spikes-N_shared:]


            z_m_tot = torch.cat((z_m_neurons,z_m_shared), dim=2)
            z_s_tot = torch.cat((z_s_neurons,z_s_shared), dim=2) 
    
            z_tot= net.sample(z_m_tot, z_s_tot, eps_n) 
        
            pen_term = -z_s_tot.sum([1,2])  

            if Fourier == True:
                
                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)


            gpNLL = gp_nll(K_cov, (z_m_tot+torch.exp(z_s_tot)), Fourier=Fourier)  
            
            
            
            
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
                
                z_m_tot_time = Time_lat_conv(z_m_tot[:,:,:N_lats_spikes], net.Bf) 
                lat_var_diag = calc_int_var(torch.exp(z_s_tot[:,:,:N_lats_spikes]), net.Bf)
               

                int_var = torch.matmul(torch.square(n_decode.dec3.weight), lat_var_diag)
            elif Fourier == False:
                "NOT IMPLEMENTED"
            lambda_hat = n_decode.dec3(z_m_tot_time)
            neural_loss = - ( -(torch.exp(lambda_hat + 0.5*torch.transpose(int_var,1,2))) -torch.lgamma(spikes + 1) +  lambda_hat * spikes).sum([1,2])


            neural_loss = neural_loss 
            
            
           
            ELBO = (pen_term + neural_loss + gpNLL).sum()
           
            loss = ELBO / BATCH

    

            trainloss += loss.item()

            
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n_decode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(n_encode.parameters(), max_grad_norm)
          
           
           
            optimizer_n_encode.step()
            optimizer_n_decode.step()

    
        if nan:
            break

        trainloss = trainloss/len(data_load)
        losses.append(trainloss)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5,3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        
    return losses






###############################################################################
## Run MM-GPVAE (not closed form)
###############################################################################

def train_model_mmgpvae(net, n_encode, n_decode, data_loader,
                EPOCH = 650, lr1=0.00016, lr2=0.000772, lr3=0.0088, 
                Fourier = True, visualize_ELBO = True, zero_neuro = 1,
                zero_image = 1, gpvae_mode = False,
                gpfa_mode = False, mmgpvae = False):
 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)
    
    N_tot = N_lats_img + N_lats_spikes - N_shared

    y = np.ones([1,N_NEURONS,TIME_POINTS])
    if gpvae_mode:
        for param in n_encode.parameters():
            param.requires_grad = False
        for param in n_decode.parameters():
            param.requires_grad = False
        for param in net.parameters():
            param.requires_grad = True
    if gpfa_mode:
        for param in n_encode.parameters():
            param.requires_grad = True
        for param in n_decode.parameters():
            param.requires_grad = True
        for param in net.parameters():
            param.requires_grad = False
    if mmgpvae:
        for param in n_encode.parameters():
            param.requires_grad = True
        for param in n_decode.parameters():
            param.requires_grad = True
        for param in net.parameters():
            param.requires_grad = True
    


    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)
    optimizer_n_encode = torch.optim.Adam(n_encode.parameters(), lr=lr2)
    optimizer_n_decode = torch.optim.Adam(n_decode.parameters(), lr=lr3) #.0088


    
    losses = []
    max_grad_norm = 1
    loss_list = []
    nan = False
    first_epoch = True
    for epoch in range(EPOCH):
        train_loss = 0
        trainloss = 0
        


        for data, label, spikes, neural_rates, gp1, gp2, zoom in data_loader:
       

            if gpfa_mode:
             
                optimizer_n_encode.zero_grad()
                optimizer_n_decode.zero_grad()

            if gpvae_mode:
                optimizer_behave.zero_grad()
            
            if mmgpvae:
                optimizer_behave.zero_grad()
                optimizer_n_encode.zero_grad()
                optimizer_n_decode.zero_grad()
            
     
            if gpfa_mode:
             
                n_decode.train()
                n_encode.train()

            if gpvae_mode:
                net.train()
            
            if mmgpvae:
                net.train()
                n_decode.train()
                n_encode.train()


            labels = label
            spikes = spikes 
            new_data = data 
        
         
            eps = Variable(torch.randn(new_data.shape[0],N_lats_img ), requires_grad=False)
            eps_n = Variable(torch.randn(new_data.shape[0], N_lats_spikes ), requires_grad=False)
            new_data, eps = new_data.to(device), eps.to(device)
            
            embs_m_n, embs_s_n = n_encode.forward(spikes.float(), eps_n, Fourier = Fourier)
            embs_m_img, embs_s_img= net.forward(new_data.float(), eps, spikes, Fourier = Fourier) #only encoding the z mean and var in time domain

        
            if Fourier:

                embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour)
                embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) 
                
            

            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour)
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour) 
                

            z_m_neurons = embs_m_n[:,:,:N_lats_spikes-N_shared] * zero_neuro
            z_m_img = embs_m_img[:,:,:N_lats_img-N_shared] * zero_image
            if gpvae_mode:
                z_m_shared = embs_m_img[:,:,N_lats_img-N_shared:]
            if gpfa_mode:
                z_m_shared = embs_m_n[:,:,N_lats_spikes-N_shared:]
            if mmgpvae:
                z_m_shared = (embs_m_n[:,:,N_lats_spikes-N_shared:] + 
                              embs_m_img[:,:,N_lats_img-N_shared:])/2
                

            z_s_neurons = embs_s_n[:,:,:N_lats_spikes-N_shared] * zero_neuro
            z_s_img = embs_s_img[:,:,:N_lats_img-N_shared] * zero_image

            if gpvae_mode:
                z_s_shared = embs_s_img[:,:,N_lats_img-N_shared:]
            if gpfa_mode:
                z_s_shared = embs_s_n[:,:,N_lats_spikes-N_shared:]
            if mmgpvae:
                z_s_shared = (embs_s_n[:,:,N_lats_spikes-N_shared:] + 
                              embs_s_img[:,:,N_lats_img-N_shared:])/2
           
            z_m_tot = torch.cat((z_m_neurons,z_m_shared, z_m_img), dim=2) 
            z_s_tot = torch.cat((z_s_neurons,z_s_shared, z_s_img), dim=2) 
    
            z_tot= net.sample(z_m_tot, z_s_tot, eps) 
            
            pen_term = -z_s_tot.sum([1,2])  

            if Fourier == True:
               
                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)

            gpNLL = gp_nll(K_cov, z_tot, Fourier=Fourier)  

           
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
                zs_tot_time = Time_lat_conv(z_s_tot, net.Bf)
            elif Fourier == False:
                z_tot_time = z_tot
            
        

            neural_loss, lambda_hat = n_decode.forward(z_tot_time[:,:,:N_lats_spikes], spikes, Fourier = Fourier)
            
            neural_loss = neural_loss 
            
        
            x_hat = net.decoder(z_tot_time[:,:,-(N_lats_img):])

            recon_term, mse = net.nll(new_data, x_hat) 
            
           
            if gpvae_mode:
                ELBO = (pen_term + recon_term+ gpNLL).sum() 
            if gpfa_mode:
                ELBO = (pen_term + neural_loss + gpNLL).sum()
            if mmgpvae: 
                ELBO = (pen_term + neural_loss + recon_term+ gpNLL).sum() 
            loss = ELBO / BATCH

    

            trainloss += loss.item()

            
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n_decode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(n_encode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

            if mmgpvae:
                optimizer_behave.step()
                optimizer_n_encode.step()
                optimizer_n_decode.step()
            if gpvae_mode:
                optimizer_behave.step()
            if gpfa_mode:
                optimizer_n_encode.step()
                optimizer_n_decode.step()


    
        if nan:
            break

        trainloss = trainloss/len(data_loader)
        losses.append(trainloss)
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5,3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        
    return losses


def train_model_mmgpvae_closed_no_flags(net, n_encode, n_decode, data_load,
                EPOCH = 650, lr1=0.00016, lr2=0.000772, lr3=0.0088, 
                Fourier = True, visualize_ELBO = True):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(my_seed)
    
    N_tot = N_lats_img + N_lats_spikes - N_shared

    
    for param in n_encode.parameters():
        param.requires_grad = True
    for param in n_decode.parameters():
        param.requires_grad = True
    for param in net.parameters():
        param.requires_grad = True

    
    optimizer_behave = torch.optim.Adam(net.parameters(), lr=lr1)
    optimizer_n_encode = torch.optim.Adam(n_encode.parameters(), lr=lr2)
    optimizer_n_decode = torch.optim.Adam(n_decode.parameters(), lr=lr3) #.0088


    
    losses = []
    max_grad_norm = 1
    for epoch in range(EPOCH):
    
        trainloss = 0
        
        for data, label, spikes, neural_rates, gp1, gp2, zoom in data_load:
            
            optimizer_behave.zero_grad()
            optimizer_n_encode.zero_grad()
            optimizer_n_decode.zero_grad()
        
        
            net.train()
            n_decode.train()
            n_encode.train()


            labels = label
            spikes = spikes 
            new_data = data
        
          
            eps = Variable(torch.randn(new_data.shape[0],N_lats_img ), requires_grad=False)
            eps_n = Variable(torch.randn(new_data.shape[0], N_lats_spikes ), requires_grad=False)
            new_data, eps = new_data.to(device), eps.to(device)
            
            embs_m_n, embs_s_n = n_encode.forward(spikes.float(), eps_n, Fourier = Fourier)
            embs_m_img, embs_s_img= net.forward(new_data.float(), eps, spikes, Fourier = Fourier) #only encoding the z mean and var in time domain

        
            if Fourier:

                embs_m_n = Fourer_lat_conv(embs_m_n, n_encode.spikeMeanFour) 
                embs_s_n = Fourer_lat_conv(embs_s_n, n_encode.spikeVarFour) 
                
            

            if Fourier:

                embs_m_img = Fourer_lat_conv(embs_m_img, net.encMeanFour) 
                embs_s_img = Fourer_lat_conv(embs_s_img, net.encVarFour)  
                

         
            z_m_neurons = embs_m_n[:,:,:N_lats_spikes-N_shared] 
            z_m_img = embs_m_img[:,:,:N_lats_img-N_shared] 
        
            z_m_shared = (embs_m_n[:,:,N_lats_spikes-N_shared:] + 
                            embs_m_img[:,:,N_lats_img-N_shared:])/2
            

            z_s_neurons = embs_s_n[:,:,:N_lats_spikes-N_shared] 
            z_s_img = embs_s_img[:,:,:N_lats_img-N_shared] 

    
            z_s_shared = (embs_s_n[:,:,N_lats_spikes-N_shared:] + 
                            embs_s_img[:,:,N_lats_img-N_shared:])/2
      
            z_m_tot = torch.cat((z_m_neurons,z_m_shared, z_m_img), dim=2) 
            z_s_tot = torch.cat((z_s_neurons,z_s_shared, z_s_img), dim=2) 
    
            z_tot= net.sample(z_m_tot, z_s_tot, eps) 
          
            pen_term = -z_s_tot.sum([1,2])  


            if Fourier == True:
               
                K_cov = net.gp_make_cov(
                    eps=1e-2, wwnrm=net.wwnrm, nxcirc=net.nxcirc, Fourier=Fourier)
            if Fourier == False:
                K_cov = net.gp_make_cov(eps=1e-2, Fourier=Fourier)



            gpNLL = gp_nll(K_cov, (z_m_tot+torch.exp(z_s_tot)), Fourier=Fourier)  
            
            
            if Fourier:
                z_tot_time = Time_lat_conv(z_tot, net.Bf)
                
                z_m_tot_time = Time_lat_conv(z_m_tot[:,:,:N_lats_spikes], net.Bf) 
                lat_var_diag = calc_int_var(torch.exp(z_s_tot[:,:,:N_lats_spikes]), net.Bf)
          
                int_var = torch.matmul(torch.square(n_decode.dec3.weight), lat_var_diag)
            elif Fourier == False:
                "NOT IMPLEMENTED"

            lambda_hat = n_decode.dec3(z_m_tot_time)
            neural_loss = - ( -(torch.exp(lambda_hat + 0.5*torch.transpose(int_var,1,2))) -torch.lgamma(spikes + 1) +  lambda_hat * spikes).sum([1,2])

            x_hat = net.decoder(z_tot_time[:,:,-(N_lats_img):])

            recon_term, mse = net.nll(new_data, x_hat) 
          

            ELBO = (pen_term + neural_loss + recon_term+ gpNLL).sum() 
         
            loss = ELBO / BATCH
            trainloss += loss.item()

            
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(n_decode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(n_encode.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            optimizer_behave.step()
            optimizer_n_encode.step()
            optimizer_n_decode.step()
    

        trainloss = trainloss/len(data_load)
        losses.append(trainloss)
        if epoch % 50 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.2f}')
    if visualize_ELBO == True:
        plt.figure(figsize=(5,3), dpi=150)
        plt.title('ELBO')
        plt.plot(losses[:])
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.show()
        
    return losses


