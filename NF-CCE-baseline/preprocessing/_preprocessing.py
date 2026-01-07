# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import issparse, diags
from sklearn.neighbors import kneighbors_graph

# ==========================================================
# 1) BioNet loader 
# ==========================================================

def nets_from_mat(filename):
    """
    Load Mostafavi BioNet dataset (*.mat) with GO annotations.
    Return:
        genes, goterms, Nets(list), GO_matrix
    """
    print("### Loading *.mat file...")
    D = sio.loadmat(filename, squeeze_me=True)
    GO_data = D['GO']
    Net_data = D['networks']

    Nets = []
    for i in range(Net_data.shape[0]):
        A = Net_data[i]['data']
        # 保留稀疏格式更省内存
        if hasattr(A, "tocsr"):
            A = A.tocsr()
        Nets.append(A)

    goterms = GO_data['collabels'].tolist().tolist()
    goterms = [item.encode('utf-8') for item in goterms]

    genes = GO_data['rowlabels'].tolist().tolist()
    genes = [item.encode('utf-8') for item in genes]

    GO = GO_data['data']
    if hasattr(GO, "tocsr"):
        GO = GO.tocsr()

    return genes, goterms, Nets, GO


# ==========================================================
# 2) Multiplex benchmark loader (Cora / MIT / CiteSeer)
# ==========================================================

def _unwrap_mat_obj(x):
    """Handle matlab cell / 0-d objects."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x

def _to_sparse(A):
    """Convert dense to csr sparse if needed."""
    if sp.issparse(A):
        return A.tocsr()
    return sp.csr_matrix(np.asarray(A))

def mltplx_from_mat(filename, net_name):
    """
    Load multiplex benchmark datasets.
    Returns:
        Nets: list of adjacency matrices (recommended: csr_matrix)
        ground_idx: label vector (n,)
    """
    D = sio.loadmat(filename, squeeze_me=True)
    Nets = []
    ground_idx = None

    if net_name == 'cora':
        print("### Loading CoRA file...")
        # Author-custom: A is tensor (n x n x M)
        A = D['A']
        for i in range(A.shape[2]):
            Nets.append(_to_sparse(A[:, :, i]))
        ground_idx = np.asarray(D['C']).reshape(-1).astype(int)
        if ground_idx.min() == 0:
            ground_idx += 1

    elif net_name == 'mit':
        print("### Loading MIT file...")
        Nets.append(_to_sparse(D['celltower_graph']))
        Nets.append(_to_sparse(D['phone_graph']))
        Nets.append(_to_sparse(D['bt_graph']))

        # MIT labels stored as cell array C: node indices per class
        ground_idx = np.zeros((Nets[0].shape[0], 1), dtype=int)
        for k in range(D['C'].shape[0]):
            ground_idx[D['C'][k] - 1] = k + 1
        ground_idx = np.asarray(ground_idx).reshape(-1)

    
    elif net_name == 'citeseer':
        print("### Loading CiteSeer file...")

        X = _unwrap_mat_obj(D['X'])
        y = _unwrap_mat_obj(D['y'])

        # X can be list-like or object ndarray
        if isinstance(X, (list, tuple)):
            layers = X
        else:
            layers = list(X)

        Nets = []
        n = None

        for idx, A in enumerate(layers):
            A = _unwrap_mat_obj(A)

            # Convert to sparse if possible
            if sp.issparse(A):
                A_sp = A.tocsr()
                shape = A_sp.shape
            else:
                A_arr = np.asarray(A)
                shape = A_arr.shape

            print(f"  - Raw layer {idx}: shape={shape}")

            # Determine node count from labels
            if n is None:
                n = np.asarray(y).reshape(-1).shape[0]

            # ===============================
            # Case 1: adjacency matrix (n x n)
            # ===============================
            if shape[0] == shape[1] == n:
                A_sp = _to_sparse(A)

                # symmetrize + remove diagonal
                A_sp = 0.5 * (A_sp + A_sp.T)
                A_sp.setdiag(0)
                A_sp.eliminate_zeros()

                Nets.append(A_sp)
                print(f"    -> treated as adjacency, nnz={A_sp.nnz}")

            # ===============================
            # Case 2: feature matrix (n x d)
            # ===============================
            elif shape[0] == n and shape[1] != n:
                print("    -> treated as features, building KNN attribute graph...")

                # Ensure sparse CSR for sklearn KNN graph (fast)
                if not sp.issparse(A):
                    A_feat = sp.csr_matrix(np.asarray(A))
                else:
                    A_feat = A_sp

                # Build KNN graph (cosine)
                S_attr = kneighbors_graph(
                    A_feat,
                    n_neighbors=15,
                    mode="connectivity",
                    metric="cosine",
                    include_self=False
                )

                S_attr = 0.5 * (S_attr + S_attr.T)
                S_attr.setdiag(0)
                S_attr.eliminate_zeros()

                Nets.append(S_attr.tocsr())
                print(f"    -> attribute graph built, nnz={S_attr.nnz}")

            else:
                raise ValueError(
                    f"Layer {idx} has incompatible shape {shape}, expected ({n},{n}) or ({n},d)."
                )

        ground_idx = np.asarray(y).reshape(-1).astype(int)
        if ground_idx.min() == 0:
            ground_idx += 1

        print(f"### CiteSeer loaded: layers={len(Nets)}, n={n}, classes={len(np.unique(ground_idx))}")

    return Nets, ground_idx


# ==========================================================
# 3) Network normalization (Dense + Sparse supported)
# ==========================================================

def _net_normalize_sparse(W):
    """
    Sparse symmetric normalization: D^-0.5 * W * D^-0.5
    """
    if not issparse(W):
        W = sp.csr_matrix(np.asarray(W))

    # Nonnegativity enforcement (sparse)
    if W.data.size > 0 and W.data.min() < 0:
        W.data[W.data < 0] = 0
        W.eliminate_zeros()

    # Symmetrize
    if (W != W.T).nnz != 0:
        W = 0.5 * (W + W.T)
        W.setdiag(0)
        W.eliminate_zeros()

    d = np.array(W.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    D_inv_sqrt = diags(d_inv_sqrt)
    return D_inv_sqrt @ W @ D_inv_sqrt


def _net_normalize_dense(X):
    """
    Original dense normalization (kept for compatibility).
    """
    X = np.asarray(X)

    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print("### Matrix converted to nonnegative matrix.")
        print()

    # Symmetrize
    if not np.allclose(X.T, X):
        print("### Matrix not symmetric.")
        X = X + X.T
        np.fill_diagonal(X, 0)
        print("### Matrix converted to symmetric.")

    deg = X.sum(axis=1).flatten()
    with np.errstate(divide='ignore'):
        deg = 1.0 / np.sqrt(deg)
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))
    return X


def net_normalize(Net):
    """
    Normalize a single network or a list of networks.
    Automatically supports sparse matrices.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            if issparse(Net[i]):
                Net[i] = _net_normalize_sparse(Net[i])
            else:
                Net[i] = _net_normalize_dense(Net[i])
        return Net
    else:
        if issparse(Net):
            return _net_normalize_sparse(Net)
        else:
            return _net_normalize_dense(Net)

