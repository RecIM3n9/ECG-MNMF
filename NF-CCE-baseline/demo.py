# add current directory

import numpy as np
import pylab as pl
import time
#import os, sys
#sys.path.append("/Users/vgligorijevic/Projects/NF-CCE/data/mostafavi/")

from snmf import SNMF
from csnmf import CSNMF
from preprocessing import nets_from_mat, mltplx_from_mat, net_normalize
from cluster import nmf_clust, spect_clust, clust_eval

t_start = time.perf_counter()

## create symmetric matrix (test)
#X = np.asmatrix(np.random.rand(10, 10))
#X = 0.5*(X + X.T)
#X = X - np.diag(np.diag(X))

## load Mostafavi data
#genes, _, Nets, _ = nets_from_mat("../data/mostafavi/human_and_go.mat")
#Nets = Nets[:3]

## load Dong data
Nets, ground_idx = mltplx_from_mat("data/nets/CiteSeer.mat", 'citeseer')
Nets = net_normalize(Nets)

#objSNMF = SNMF(Nets[1], 50, init='rnda', displ='true')
#res_snmf = objSNMF.fit()

objCSNMF = CSNMF(Nets, max(ground_idx), alpha = 0.5, init = 'rnda', displ = True)
res_csnmf = objCSNMF.fit()


H = res_csnmf.matrices[0]
J = res_csnmf.objfun_vals

# objSNMF = SNMF(Nets[1], max(ground_idx), init='rnda', displ='true')
# res_snmf = objSNMF.fit()

# H = res_snmf.matrices[0]
# J = res_snmf.objfun_vals

## Clustering performance
spect_idx = spect_clust(H, max(ground_idx))
nmf_idx = nmf_clust(H)

print(ground_idx.shape)
print(spect_idx.shape)
print(nmf_idx.shape)

print("### NMF clustering: ")
print(clust_eval(ground_idx, nmf_idx))
print("### Spectral clustering: ")
print(clust_eval(ground_idx, spect_idx))


t_end = time.perf_counter()
print("=" * 50)
print(f"[Time] Total runtime: {t_end - t_start:.4f} seconds")
print("=" * 50)

# show results
pl.figure(1)
pl.title("Cluster indicator matrix")
pl.imshow(H, interpolation='nearest')
pl.imshow(H @ H.T, interpolation='nearest')
pl.colorbar()

pl.figure(2)
pl.title("Obj function")
pl.plot(J, 'o-')

pl.show()
