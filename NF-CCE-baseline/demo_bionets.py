# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
import scipy.sparse
from scipy.sparse import issparse, diags

from snmf import SNMF
from csnmf import CSNMF
from preprocessing import nets_from_mat
import time


def sparse_net_normalize(W):
    """
    D^-0.5 * W * D^-0.5
    """
    if not issparse(W):
        if isinstance(W, np.ndarray) and W.ndim == 0:
            W = W.item()
        W = scipy.sparse.csr_matrix(W)
        
    # 2. D 
    d = np.array(W.sum(axis=1)).flatten()
    
    # 3. D^-0.5
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # 4. dig
    D_inv_sqrt = diags(d_inv_sqrt)
    
    # 5. D^-0.5 * W * D^-0.5
    return D_inv_sqrt @ W @ D_inv_sqrt

def calculate_average_redundancy(H_matrix, GO_matrix):
    """ 
    Average Redundancy 
    H_matrix: (n_nodes x k)
    GO_matrix: (n_nodes x n_go_terms)
    """
    # H (dense) -> Labels (Hard Clustering)
    cluster_labels = np.array(H_matrix.argmax(axis=1)).flatten()
    n_clusters = H_matrix.shape[1]
    
    if isinstance(GO_matrix, np.ndarray) and GO_matrix.ndim == 0:
        GO_matrix = GO_matrix.item()

    if not scipy.sparse.issparse(GO_matrix):
        GO_matrix = scipy.sparse.csr_matrix(GO_matrix)
        
    n_go_terms = GO_matrix.shape[1]
    redundancy_sum = 0
    valid_clusters = 0

    for k in range(n_clusters):
        nodes_in_cluster = np.where(cluster_labels == k)[0]
        if len(nodes_in_cluster) == 0: continue

        cluster_go_vectors = GO_matrix[nodes_in_cluster]

        go_counts = np.array(cluster_go_vectors.sum(axis=0)).flatten()
        
        total_counts = go_counts.sum()
        if total_counts == 0: continue
            
        probs = go_counts / total_counts
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        
        # Redundancy
        if n_go_terms > 1:
            redundancy = 1 - (entropy / np.log2(n_go_terms))
        else:
            redundancy = 0
        
        redundancy_sum += redundancy
        valid_clusters += 1

    # Average Redundancy
    return redundancy_sum / valid_clusters if valid_clusters > 0 else 0

# ==================== Main Loop ====================

t_start = time.perf_counter()

print("### SNMF MODULE LOADED ###")
print("### Loading *.mat file...")

# 1. dataload
filename = "data/bionets/mouse_and_go.mat"
genes, _, Nets, GO_data = nets_from_mat(filename)

Nets = Nets[:3]

print("### Preprocessing: Unwrapping and Normalizing Networks...")
cleaned_Nets = []
for i, n in enumerate(Nets):
    # Peeling the onion
    if isinstance(n, np.ndarray) and n.ndim == 0:
        n = n.item()
    
    # Sparse Normalization
    norm_n = sparse_net_normalize(n)
    cleaned_Nets.append(norm_n)
    print(f"   - Layer {i+1} shape: {norm_n.shape}, Non-zeros: {norm_n.nnz}")

# update Nets 
Nets = cleaned_Nets

# 3. training
k = 100 
print(f"### Starting CSNMF (k={k})...")

objCSNMF = CSNMF(Nets, k, alpha=0.5, init='rnda', displ=True)
res_csnmf = objCSNMF.fit()

# results
H = res_csnmf.matrices[0]
J = res_csnmf.objfun_vals

# objSNMF = SNMF(Nets[1], 100, init='rnda', displ='true')
# res_snmf = objSNMF.fit()

# H = res_snmf.matrices[0]
# J = res_snmf.objfun_vals


# 4. Redundancy

t_end = time.perf_counter()
print("=" * 50)
print(f"[Time] Total runtime: {t_end - t_start:.4f} seconds")
print("=" * 50)


print("\n### Calculating Average Redundancy...")
try:
    avg_red = calculate_average_redundancy(H, GO_data)
    print("=" * 40)
    print(f"Dataset: {filename}")
    print(f"Number of Clusters (k): {k}")
    print(f"Average Redundancy: {avg_red:.4f}")
    print("=" * 40)
except Exception as e:
    print(f"Error calculating redundancy: {e}")
    print("Check if GO_data is loaded correctly.")

# 5. 可视化
pl.figure(1)
pl.title("Cluster indicator matrix H")
pl.imshow(H, aspect='auto', interpolation='nearest')
pl.colorbar()

pl.figure(2)
pl.title("Objective Function Convergence")
pl.xlabel("Iterations (x10)")
pl.ylabel("Loss")
pl.plot(J, 'o-')

pl.show()


