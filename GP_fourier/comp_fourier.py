import numpy as np
import warnings

from . import mkcovs, kron_ops
from . import fft_ops as rffb


def conv_fourier(x,dims,minlens=0,nxcirc = None,condthresh = 1e8):
	# Version of this NOT complete for higher dimensions 9/15/17!
	#
	# INPUT:
	# -----
	#           x [D x n x p] - stimulus, where each row vector is the spatial stim at a single time, D is number of batches 
	#        dims [m x 1] - number of coefficients along each stimulus dimension
	#     minlens [m x 1] - minimum length scale for each dimension (can be scalar)
	#      nxcirc [m x 1] - circular boundary in each stimulus dimension (minimum is dims) OPTIONAL
	#  condthresh [1 x 1] - condition number for thresholding for small eigenvalues OPTIONAL
	#
	# OUTPUT:
	# ------
	#     Bx  - output data, x, in fourier domain
	#  wwnrm [nf x 1] - squared "effective frequencies" in vector form for each dim (normalized)
	#   Bfft  {1 x p} - cell array with DFT bases for each dimension (list of numpy arrays for each dimension)
	# 	1e8 is default value (condition number on prior covariance)


	dims = np.array(np.reshape(dims,(1,-1)))
	minlens = np.array(np.reshape(minlens,(1,-1)))

	# Set circular bounardy (for n-point fft) to avoid edge effects, if needed
	if nxcirc == None:
	    #nxcirc = np.ceil(max([dims(:)'+minlens(:)'*4; dims(:)'*1.25]))'
	    nxcirc = np.ceil(np.max(np.concatenate((dims+minlens*4 ,dims), axis = 0), axis = 0))


	nd = np.size(dims) # number of filter dimensions
	if np.size(minlens) is 1 and nd is not 1: #% make vector out of minlens, if necessary
	    minlens = np.repmat(minlens,nd,1)

	# generate here a list of your
	#None of these quantities depend on the data directly

	wvecs = [rffb.comp_wvec(nxcirc[jj],minlens[0][jj], condthresh) for jj in np.arange(nd)]
	Bffts = [rffb.realfftbasis(dims[jj],nxcirc[jj],wvecs[jj])[0] for jj in np.arange(nd)]


	def f(switcher):  
	    # switch based on stimulus dimension
	    if switcher == 2:
	    	pass
	    if switcher == 3:
	    	pass
	    return{
	    1: #% 1 dimensional stimulus
	         [np.square(2*np.pi/nxcirc[0]) * np.square(wvecs[0]), #normalized wvec
	         np.ones([np.size(wvecs[0]),1])==1] #indices to keep 

	        
	    # 2: % 2 dimensional stimulus
	        
	    #     % Form full frequency vector and see which to cut
	    #     Cdiag = kron(cdiagvecs{2},cdiagvecs{1});
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep 
	                    
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2] = ndgrid(wvecs{1},wvecs{2});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).^2 ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2];
	        
	    # 3: % 3 dimensional stimulus

	    #     Cdiag = kron(cdiagvecs{3},(kron(cdiagvecs{2},cdiagvecs{1})));
	    #     ii = (Cdiag/max(Cdiag))>1/condthresh; % indices to keep
	        
	    #     % compute vector of normalized frequencies squared
	    #     [ww1,ww2,ww3] = ndgrid(wvecs{1},wvecs{2},wvecs{3});
	    #     wwnrm = [(ww1(ii)*(2*pi/nxcirc(1))).mv ^2, ...
	    #         (ww2(ii)*(2*pi/nxcirc(2))).^2, ....,
	    #         (ww3(ii)*(2*pi/nxcirc(3))).^2];
	        
	    # otherwise
	    #     error('compLSsuffstats_fourier.m : doesn''t yet handle %d dimensional filters\n',nd);
		}[switcher] 
	try:
	    [wwnrm, ii] = f(nd)
	except KeyError:
	    print('\n\n Does not handle values of dimension', nd, 'yet')    
	

	# Calculate stimulus sufficient stats in Fourier domain

	
	# if x.shape[0] == 1: 

	# 	#originally this used the transpose operation (kronmulttrp) ! !!!might be a transpositional issue.
	# 	Bx = kron_ops.kronmult(Bffts,np.transpose(x)) # convert to Fourier domain
	# 	Bx = Bx[ii] # prune unneeded freqs

	# elif x.shape[0]>1: #Batched data. when the shape of x is 3 and dims is 2, for example. 

	Bx = [kron_ops.kronmult(Bffts,np.transpose(batch)) for batch in x] 

	Bx = [prune[ii] for prune in Bx]

	return Bx, wwnrm, Bffts, nxcirc




def conv_fourier_mult_neuron(x,dims,minlens,num_neurons,nxcirc = None,condthresh = 1e8):

	#print(x[:,0,:].shape)
	Bys, wwnrm, Bffts, nxcirc = conv_fourier(x[:,0,:],dims,minlens,nxcirc = nxcirc,condthresh = condthresh)
	N_four = np.array(Bys).shape[1]
	if num_neurons >1:
		for i in np.arange(1,num_neurons):
			Bys = np.vstack((Bys,conv_fourier(x[:,i,:],dims,minlens,nxcirc = nxcirc,condthresh = condthresh)[0]))
	Bys = np.reshape(Bys, [x.shape[0],num_neurons,N_four])

	return Bys, wwnrm, Bffts, nxcirc




# def conv_fourier_batch(x,dims,minlens,nxcirc = None,condthresh = 1e8):

# 	if len(x.shape) <= len(dims):
# 		warnings.warn('\n\n shape of input vector is not longer than dims vector. Try using conv_fourier, not conv_fourier_batch \n\n')

# 	#return a list of arrays for the Bx all the batched data, 	
# 	return [conv_fourier(batch,dims,minlens,nxcirc,condthresh)[0] 
# 	for batch in arange(x)] + [conv_fourier(x[0],dims,minlens,nxcirc,condthresh)[1,:]]




def compLSsuffstats_fourier(x,y,dims,num_neurons, minlens=0,nxcirc = None,condthresh = 1e8):
	# Compute least-squares regression sufficient statistics in DFT basis
	if nxcirc == None:
		nxcirc = dims

	print(len(y.shape))
	if len(y.shape) is 3:
		#if we have many y observations (multi-neuron)
		[Bys, wwnrm, Bffts, nxcirc]  =conv_fourier_mult_neuron(y,dims,minlens,num_neurons,condthresh = condthresh,nxcirc = nxcirc)


	else:
		[Bys, wwnrm, Bffts, nxcirc] = conv_fourier(y,dims,minlens,condthresh = condthresh,nxcirc = nxcirc)

	y = np.reshape(y,[num_neurons,-1])
	dd = {}
	dd['x'] = Bffts[0]@x.T
	dd['xx'] = dd['x']@dd['x'].T
	dd['xy'] = dd['x'] @ y.T
	# Fill in other statistics
	dd['yy'] = y@y.T# marginal response variance

	return dd, Bys, wwnrm, Bffts, nxcirc








