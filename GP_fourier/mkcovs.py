import torch
import warnings
import numpy as np

from . import fft_ops as rffb

def mkcovdiag_ASD(len_sc,rho,nxcirc,wvec= None,wwnrm= None):
#  Eigenvalues of ASD covariance (as diagonalized in Fourier domain)
# 
#  [cdiag,dcdiag,ddcdiag] = mkcovdiag_ASD(rho,l,nxcirc,wvecsq)
# 
#  Compute discrete ASD (RBF kernel) eigenspectrum using frequencies in [0, nxcirc].
#  See mkCov_ASD_factored for more info
# 
#  INPUT (all python 1d lists!):
#          len - length scale of ASD kernel (determines smoothness)
#          rho - maximal prior variance ("overall scale")
#       nxcirc - number of coefficients to consider for circular boundary 
#         wvec - vector of freq for DFT 
#		wwnrm - vector of freq for DFT (normalized)
#         
#  OUTPUT:
#      cdiag [nxcirc x 1] - vector of eigenvalues of C for frequencies in w
# 
# Note: nxcirc = nx corresponds to having a circular boundary 


# Compute diagonal of ASD covariance matrix
	if wvec is not None:
		wvecsq = torch.square(wvec)
		const = torch.square(torch.Tensor(2*torch.pi/nxcirc)) # constant 
		ww = wvecsq*const  # effective frequency vector
	elif wwnrm is not None:
		ww = wwnrm
	else:
		print("please provide either wvec or a normalized wvec into this function")

	cdiag = torch.squeeze(torch.sqrt(torch.Tensor([2*torch.pi]))*rho*len_sc*torch.exp(-.5*torch.outer(ww,torch.square(len_sc))))
	return cdiag




def mkcov_ASDfactored(prs,nx,nxcirc=None,condthresh = 1e8,compfftbasis = None):
# % Factored representation of ASD covariance matrix in Fourier domain
# %
# % [Cdiag,U,wvec] = mkcov_ASDfactored(prs,nx,opts)
# %
# % Covariance represented as C = U*sdiag*U'
# % where U is unitary (in some larger basis) and sdiag is diagonal
# %
# %  C_ij = rho*exp(((i-j)^2/(2*l^2))
# %
# % INPUT:
# % ------
# %   prs [2 x 1]  - ASD parameters [len_sc = length scale; rho - maximal variance; ]:
# %    nx [1 x 1]  - number of regression coeffs
# % 
# % Note: nxcirc = nx gives circular boundary
# %
# % OUTPUT:
# % -------
# %   cdiag [ni x 1] - vector with thresholded eigenvalues of C
# %       U [ni x nxcirc] - column vectors define orthogonal basis for C (on Reals)
# %    wvec [nxcirc x 1] - vector of Fourier frequencies

	len_sc = prs[0]
	rho = prs[1]

	# % Parse inputs
	if nxcirc is None:
		nxcirc = nx + torch.ceil(4*len_sc) # extends support by 4 stdevs of ASD kernel width



	# % Check that nxcirc isn't bigger than nx
	if nxcirc < nx:
	    warnings.warn('mkcov_ASDfactored: nxcirc < nx. Some columns of x will be ignored')


	# % compute vector of Fourier frequencies
	# maxfreq = np.floor(nxcirc/(np.pi*len_sc)*np.sqrt(.5*np.log(condthresh))) # max
	# if maxfreq < nxcirc/2:
	#     wvec = np.concatenate(([np.arange(int(maxfreq))],[np.arange(-int(maxfreq),0)]),axis = 1)
	#else:
	    # % in case cutoff is above max number of frequencies
	wvec = rffb.comp_wvec(nxcirc, len_sc, condthresh)



	# % Compute diagonal in Fourier domain
	cdiag = mkcovdiag_ASD(len_sc,rho,nxcirc,wvec = wvec) # compute diagonal and Fourier freqs

	# % Compute real-valued discrete Fourier basis U
	if compfftbasis is not None:
	    U = rffb.realfftbasis(nx,nxcirc,wvec)[0]
	    return cdiag, U, wvec
	else:
		return cdiag




def mkcovdiag_ASD_wellcond(len_sc,rho,nxcirc,addition = 1e-8, wvec= None,wwnrm= None):

	cdiag = mkcovdiag_ASD(len_sc,rho,nxcirc,wvec,wwnrm)

	return cdiag + addition




	
