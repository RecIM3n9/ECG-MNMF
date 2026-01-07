# -*- coding: utf-8 -*-
"""
Improved demo for BioNet dataset (NF-CCE) with:
1) Sparse SVD initialization
2) Graph Regularization (Laplacian smoothing)
3) CDNMF-style Contrastive Learning (InfoNCE) with:
   - Topology Views (Step1 SNMF per layer)
   - Attribute View (GO-based KNN graph)

Evaluation:
- Average Redundancy (same as original NF-CCE pipeline)
"""

import numpy as np
import pylab as pl
import scipy.sparse as sp
from scipy.sparse import issparse, diags

# project modules
from preprocessing import nets_from_mat
from snmf import SNMF
from csnmf import CSNMF
import time 

from sklearn.neighbors import kneighbors_graph


# ===========================
# Utils
# ===========================

def unwrap_mat_obj(x):
    """Peel the onion: unwrap 0-d arrays / object arrays from .mat load."""
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.dtype == object and x.size == 1:
            try:
                return x.item()
            except Exception:
                return x
    return x


def ensure_csr(W):
    """Ensure W is csr sparse."""
    W = unwrap_mat_obj(W)
    if issparse(W):
        return W.tocsr()
    return sp.csr_matrix(np.asarray(W))


def sparse_net_normalize(W):
    """
    Symmetric normalization for sparse adjacency:
        D^{-1/2} W D^{-1/2}
    """
    W = ensure_csr(W)
    d = np.array(W.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = diags(d_inv_sqrt)
    return D_inv_sqrt @ W @ D_inv_sqrt


def build_reg_graph(Nets):
    """
    Build graph regularization matrices:
        W_reg = sum(topology layers)
        D_reg = diag(sum(W_reg))
    """
    W_reg = sum(Nets)
    d = np.array(W_reg.sum(axis=1)).flatten()
    D_reg = sp.diags(d)
    return W_reg.tocsr(), D_reg.tocsr()


def build_feature_graph(feature_matrix, knn_k=15, metric="cosine"):
    """
    Build attribute-view adjacency using KNN on GO annotation matrix.
    GO is sparse binary; cosine is a good choice.
    Returns symmetric sparse adjacency.
    """
    feature_matrix = unwrap_mat_obj(feature_matrix)
    if not sp.issparse(feature_matrix):
        feature_matrix = sp.csr_matrix(feature_matrix)

    adj = kneighbors_graph(
        feature_matrix,
        knn_k,
        mode="connectivity",
        metric=metric,
        include_self=False
    )
    adj = 0.5 * (adj + adj.T)
    return adj.tocsr()


def calculate_average_redundancy(H_matrix, GO_matrix):
    """Average Redundancy metric used in NF-CCE."""
    H_matrix = np.asarray(H_matrix)
    cluster_labels = H_matrix.argmax(axis=1).flatten()
    n_clusters = H_matrix.shape[1]

    GO_matrix = unwrap_mat_obj(GO_matrix)
    if not sp.issparse(GO_matrix):
        GO_matrix = sp.csr_matrix(GO_matrix)

    n_go_terms = GO_matrix.shape[1]
    redundancy_sum = 0.0
    valid_clusters = 0

    for k in range(n_clusters):
        nodes_in_cluster = np.where(cluster_labels == k)[0]
        if len(nodes_in_cluster) == 0:
            continue

        cluster_go_vectors = GO_matrix[nodes_in_cluster]
        go_counts = np.array(cluster_go_vectors.sum(axis=0)).flatten()
        total_counts = go_counts.sum()
        if total_counts == 0:
            continue

        probs = go_counts / total_counts
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        if n_go_terms > 1:
            redundancy = 1 - (entropy / np.log2(n_go_terms))
        else:
            redundancy = 0.0

        redundancy_sum += redundancy
        valid_clusters += 1

    return redundancy_sum / valid_clusters if valid_clusters > 0 else 0.0


# ===========================
# Main
# ===========================

if __name__ == "__main__":

    print("### Improved BioNet Demo (SVD + GraphReg + Contrastive) ###")

    # -----------------------
    # 1) Load data
    # -----------------------
    filename = "data/bionets/yeast_and_go.mat"
    print(f"\nLoading data: {filename}")
    genes, _, Nets, GO_data = nets_from_mat(filename)

    Nets = unwrap_mat_obj(Nets)
    GO_data = unwrap_mat_obj(GO_data)

    # choose first few layers for faster debugging
    num_layers = 3
    Nets = Nets[:num_layers]

    t_start = time.perf_counter()
    # -----------------------
    # 2) Preprocess topology nets
    # -----------------------
    print("\nPreprocessing: unwrap + normalize topology layers...")
    cleaned_Nets = []
    for i, A in enumerate(Nets):
        A = ensure_csr(A)
        A = sparse_net_normalize(A)
        cleaned_Nets.append(A)
        print(f"  - Layer {i+1}: shape={A.shape}, nnz={A.nnz}")

    Nets = cleaned_Nets
    n = Nets[0].shape[0]
    print(f"Total nodes: {n}")

    # -----------------------
    # 3) Step1: Topological Views
    # -----------------------
    print("\nStep1: generating topological views with SNMF...")
    k = 100
    step1_iter = 25  # fast views (no need full convergence)
    H_views = []

    for i, A in enumerate(Nets):
        snmf_model = SNMF(A, rank=k, init="svd", displ=False)
        snmf_model.maxiter = step1_iter
        res = snmf_model.fit()
        H_v = np.asarray(res.matrices[0])
        H_views.append(H_v)
        print(f"  - View {i+1}: H_view shape={H_v.shape}")

    # -----------------------
    # 4) Attribute View from GO_data
    # -----------------------
    print("\nBuilding attribute view from GO annotations (KNN graph)...")
    # Build attribute adjacency S_attr
    S_attr = build_feature_graph(GO_data, knn_k=15, metric="cosine")
    S_attr = sparse_net_normalize(S_attr)
    print(f"  - S_attr: shape={S_attr.shape}, nnz={S_attr.nnz}")

    # Factorize attribute graph -> H_attr (also via SNMF)
    attr_model = SNMF(S_attr, rank=k, init="svd", displ=False)
    attr_model.maxiter = step1_iter
    res_attr = attr_model.fit()
    H_attr = np.asarray(res_attr.matrices[0])
    print(f"  - H_attr shape={H_attr.shape}")

    # -----------------------
    # 5) Graph Regularization matrices
    # -----------------------
    print("\nBuilding graph regularization matrices W_reg, D_reg...")
    W_reg, D_reg = build_reg_graph(Nets)
    print(f"  - W_reg nnz={W_reg.nnz}")

    # -----------------------
    # 6) Step2: Improved CSNMF (MU + GraphReg + Contrastive)
    # -----------------------
    print("\nStep2: running improved CSNMF...")

    # Hyperparameters
    max_iter = 150
    warmup = 10
    beta_reg = 0.5 # 0.1 - 0.5
    use_cl = True
    lr_cl = 0.005
    tau = 0.5

    model = CSNMF(Nets, k=k, alpha=0.5, init="svd", displ=True)

    res = model.fit(
        H_views=H_views,
        H_attr=H_attr,
        W_reg=W_reg,
        D_reg=D_reg,
        beta_reg=beta_reg,

        use_cl=use_cl,
        lr_cl=lr_cl,
        tau=tau,
        max_iter=max_iter,

        warmup=warmup,
        update_label_freq=5,
        neg_sample_size=256,

        normalize_after=True,
        clip_cl_grad=1.0,            # set None to disable
        normalize_cl_grad=False      # True if unstable
    )

    H_final = np.asarray(res.matrices[0])

    # logs (optional)
    J_mu = getattr(res, "obj_mu", None)
    J_cl = getattr(res, "obj_cl", None)

    # -----------------------
    # 7) Redundancy Evaluation
    # -----------------------
    print("\n### Calculating Average Redundancy...")
    avg_red = calculate_average_redundancy(H_final, GO_data)

    print("=" * 60)
    print(f"Dataset: {filename}")
    print(f"Layers used: {num_layers}")
    print(f"Clusters (k): {k}")
    print(f"GraphReg beta: {beta_reg}")
    print(f"Contrastive: {use_cl} | lr_cl={lr_cl} | tau={tau}")
    print(f"Average Redundancy: {avg_red:.4f}")
    print("=" * 60)

    # time
    t_end = time.perf_counter()
    print("=" * 50)
    print(f"[Time] Total runtime: {t_end - t_start:.4f} seconds")
    print("=" * 50)

    # -----------------------
    # 8) Visualization
    # -----------------------
    pl.figure(1)
    pl.title("Consensus Cluster Indicator H (Improved)")
    pl.imshow(H_final, aspect="auto", interpolation="nearest")
    pl.colorbar()

    pl.figure(2)
    pl.title("Consensus Similarity H H^T")
    pl.imshow(H_final @ H_final.T, interpolation="nearest")
    pl.colorbar()

    if J_mu is not None and len(J_mu) > 0:
        pl.figure(3)
        pl.title("MU Objective (every 10 iters)")
        pl.plot(J_mu, "o-")
        pl.xlabel("log step")
        pl.ylabel("loss")

    if J_cl is not None and len(J_cl) > 0:
        pl.figure(4)
        pl.title("Contrastive Loss (InfoNCE)")
        pl.plot(J_cl, "o-")
        pl.xlabel("log step")
        pl.ylabel("loss")

    pl.show()
