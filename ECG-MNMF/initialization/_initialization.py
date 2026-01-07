# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
from scipy import sparse
import scipy.sparse.linalg
from math import ceil, sqrt
from operator import itemgetter
from numpy import linalg as la

def _to_dense_if_small(X, threshold_bytes=1e8):
    """
    Helper: Only convert to dense if matrix is small enough.
    Otherwise, return as is (sparse).
    """
    if sparse.issparse(X):
        size = X.shape[0] * X.shape[1] * 8
        if size < threshold_bytes:
            return X.toarray()
    return X # Keep sparse if too big or already dense

def rnda_init(X, k, p=None):
    """
    RandomAcol initialization (Sparse Compatible).
    Initialization using the average of p random columns of X.
    """
    n, m = X.shape

    if p is None:
        p = int(ceil(0.2 * m)) # or 1/5 as used before

    prng = np.random.RandomState()
    
    # Init H as ndarray
    H = np.zeros((n, k))

    for i in range(k):
        # Randomly select p columns
        cols = prng.randint(low=0, high=m, size=p)
        
        avg_vec = X[:, cols].mean(axis=1)
        
        H[:, i] = np.array(avg_vec).ravel()

    return H

def sparse_svd_init(X, k):
    """
    [新增] Sparse SVD-based initialization.
    Uses scipy.sparse.linalg.svds for efficiency on large graphs.
    """
    print("Initializing H with Sparse SVD...")

    if not scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    n, m = X.shape
    k = min(k, min(n, m) - 1)

    
    # Ensure float type for SVD
    if X.dtype != float:
        X = X.astype(float)
        
    n = X.shape[0]
    
    # 1. Randomized Truncated SVD (Fast for sparse)
    # k: number of components
    # which='LM': Largest Magnitude
    u, s, vt = scipy.sparse.linalg.svds(X, k=k, which='LM')
    
    # 2. Sort components (svds output is often unsorted)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    
    # 3. Construct H ~ U * S^0.5
    # (Simplified NNDSVD strategy)
    H = u @ np.diag(np.sqrt(s))
    
    # 4. Enforce Non-negativity (Abs)
    H = np.abs(H)
    
    # 5. Handle near-zero values
    H[H < 1e-10] = 0.0
    
    # 6. Normalize rows (Optional but recommended for NMF stability)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    H = H / norms
    
    return H

def rnd_init(X, k):
    """ Random initialization. """
    n = X.shape[0]
    H = np.random.rand(n, k)
    return H

def rndc_init(X, k, p=None, l=None):
    """
    RandomC initialization.
    (Kept for compatibility, mostly suited for dense/smaller matrices)
    """
    X_dense = _to_dense_if_small(X) 
    
    n = X_dense.shape[0]
    if p is None: p = int(ceil(0.2 * n))
    if l is None: l = int(ceil(0.5 * n))

    prng = np.random.default_rng()

    # Calculation of norms can be expensive on huge sparse matrices
    # Here we assume X is manageable or dense
    if sparse.issparse(X_dense):
         # Sparse norm calculation
         norms = [(i, scipy.sparse.linalg.norm(X_dense[i, :])) for i in range(n)]
    else:
         norms = [(i, la.norm(X_dense[i, :], 2)) for i in range(n)]
         
    top = sorted(norms, key=itemgetter(1), reverse=True)[:l]
    top_idx = np.array([i for i, _ in top])

    H = np.zeros((n, k))
    for i in range(k):
        cols = prng.choice(top_idx, size=p, replace=True)
        # Sparse safe slicing
        val = X_dense[:, cols].mean(axis=1)
        H[:, i] = np.array(val).ravel()

    return H

def svd_init(X, k, flag=0):
    """
    Legacy dense SVD initialization.
    (Kept for compatibility with original code structure)
    """
    # Warning: This calls dense SVD
    X = _to_dense_if_small(X)
    if sparse.issparse(X):
        return sparse_svd_init(X, k) # Fallback to sparse if still sparse

    n = X.shape[0]
    H = np.zeros((n, k))
    U, S, _ = la.svd(X, full_matrices=False)

    H[:, 0] = sqrt(S[0]) * np.abs(U[:, 0])
    for i in range(1, k):
        uu = U[:, i]
        uup = _pos(uu)
        n_uup = la.norm(uup, 2)
        if n_uup > 0:
            H[:, i] = sqrt(S[i]) * uup / n_uup

    H[H < 1e-10] = 0.0
    
    if flag in (1, 2):
        avg = X.mean()
        mask = (H == 0)
        if flag == 1:
            H[mask] = avg
        else:
            H[mask] = avg * np.random.random(mask.sum()) / 100.0
            
    return H

def _pos(x):
    return np.maximum(x, 0)

def _neg(x):
    return np.maximum(-x, 0)