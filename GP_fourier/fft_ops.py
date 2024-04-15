
import numpy as np
import warnings

def comp_wvec(nn, len_sc = 0, condthresh = None):
	#return frequency vector, wvec

	#compute wvec if it does not depend on a rho scale or a condthresh. 
	wvec = None
	if len_sc != 0:
		maxfreq = np.floor(nn/(np.pi*len_sc)*np.sqrt(.5*np.log(condthresh))) # max
		if maxfreq < nn/2:
			wvec = np.concatenate(([np.arange(int(maxfreq))],[np.arange(-int(maxfreq),0)]),axis = 1)[0]


	if wvec is None:
		ncos = np.ceil((nn+1)/2) # number of cosine terms (positive freqs)
		nsin = np.floor((nn-1)/2) # number of sine terms (negative freqs)
		wcos = np.arange(ncos)
		wsin = np.arange(-nsin, 0)
		wvec = np.concatenate((wcos, wsin),axis = 0) # vector of frequencies
	return wvec





def realfftbasis(nx,nn=None,wvec=None):
#  Basis of sines+cosines for nn-point discrete fourier transform (DFT)
# 
#  [B,wvec] = realfftbasis(nx,nn,w)
# 
#  For real-valued vector x, realfftbasis(nx,nn)*x is equivalent to realfft(x,nn) 
# 
#  INPUTS:
#   nx - number of coefficients in original signal
#   nn - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
#   wvec (optional) - frequencies: positive = cosine
# 
#  OUTPUTS:
#    B [nn x nx]  - DFT basis 
#    wvec - frequencies associated with rows of B
# 
	

	if nn is None:
	    nn = nx

	# if nn<nx:
	#     warnings.warn('realfftbasis: nxcirc < nx. SOMETHING IS WRONG')

	if wvec is None: 
	    # Make frequency vector
		wvec = comp_wvec(nn)

	# Divide into pos (for cosine) and neg (for sine) frequencies
	wcos = wvec[wvec>=0] 
	wsin = wvec[wvec<0]  

	x = np.arange(nx) # spatial np.pixel indices
	if wsin.any():
	    B = np.concatenate((np.cos(np.outer(wcos*2*np.pi/nn,x)), np.sin(np.outer(wsin*2*np.pi/nn,x))),axis = 0)/np.sqrt(nn/2)
	else:
	    B = np.cos(np.outer(wcos*2*np.pi/nn,x))/np.sqrt(nn/2)


	# make DC term into a unit vector
	izero = [wvec==0][0] # index for DC term... careful of indexing here!
	inds = [i for i, x in enumerate(izero) if x]
	newthing = B[inds]/np.sqrt(2)
	B[inds] = newthing
	# if nn is even, make Nyquist term (highest cosine term) a unit vector
	if (nn/2 == np.max(wvec)):
	    ncos = np.int(np.ceil((nn)/2)) # this is the index of the nyquist freq
	    B[ncos] = B[ncos]/np.sqrt(2)

	return B, wvec
