# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import issparse, diags
from sklearn.neighbors import kneighbors_graph


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


def nets_from_mat(filename):
    print("### Loading *.mat file...")
    D = sio.loadmat(filename, squeeze_me=True)
    GO_data = D['GO']
    Net_data = D['networks']

    Nets = []
    for i in range(Net_data.shape[0]):
        if hasattr(Net_data[i]['data'], 'todense'):
            Nets.append(Net_data[i]['data'].todense())
        else:
            Nets.append(Net_data[i]['data'])

    goterms = GO_data['collabels'].tolist().tolist()
    goterms = [item.encode('utf-8') for item in goterms]

    genes = GO_data['rowlabels'].tolist().tolist()
    genes = [item.encode('utf-8') for item in genes]

    GO = GO_data['data'].tolist()
    GO = GO.todense()

    return genes, goterms, Nets, GO

def mltplx_from_mat(filename, net_name):
    D = sio.loadmat(filename, squeeze_me=True)
    Nets = []
    ground_idx = []
    if net_name == 'cora':
        print("### Loading CoRA file...")
        Nets = [D['A'][:,:,i] for i in range(D['A'].shape[2])]
        ground_idx = D['C']
        ground_idx = np.reshape(ground_idx, Nets[0].shape[0])
    elif net_name == 'mit':
        print("### Loading MIT file...")
        Nets.append(D['celltower_graph'])
        Nets.append(D['phone_graph'])
        Nets.append(D['bt_graph'])
        ground_idx = np.zeros((Nets[0].shape[0], 1), dtype=int)
        for k in range(D['C'].shape[0]):
            ground_idx[D['C'][k]-1] = k+1
        ground_idx = np.reshape(ground_idx, Nets[0].shape[0])

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

def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print("### Negative entries in the matrix are not allowed!")
        X[X<0] = 0
        print("### Matrix converted to nonnegative matrix.")
        print()
    if (X.T == X).all():
        pass
    else:
        print("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print("### Matrix converted to symmetric.")

    ### normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X

def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
    else:
        Net = _net_normalize(Net)

    return Net
