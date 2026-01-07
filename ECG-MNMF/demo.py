# -*- coding: utf-8 -*-
"""
Improved demo for non-bionet multiplex datasets (MIT / CORA / MDP).
Pipeline:
  Step1: Per-layer SNMF -> H_views
  Step2: Improved CSNMF (MU + GraphReg + Contrastive + SVD init) -> H_final
  Eval : NMF clustering + Spectral clustering

This demo assumes you already modified:
  - snmf.py  (correct SNMF denominator (HH^T)H and sparse_svd_init)
  - csnmf.py (correct denominator (HH^T)H, graph reg + contrastive PGD)
  - contrastive.py (InfoNCE + debiased negatives)
  - initialization.py (sparse_svd_init)
"""

import numpy as np
import pylab as pl
import scipy.sparse as sp

from snmf import SNMF
from csnmf import CSNMF
from preprocessing import mltplx_from_mat, net_normalize
from cluster import nmf_clust, spect_clust, clust_eval
import time


# -------------------------
# Utilities
# -------------------------
def unwrap_mat_obj(x):
    """Unwrap possible 0-d arrays / object arrays from .mat loading."""
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.dtype == object and x.size == 1:
            try:
                return x.item()
            except Exception:
                return x
    return x


def ensure_csr(A):
    """Ensure adjacency matrix is CSR sparse."""
    A = unwrap_mat_obj(A)
    if sp.issparse(A):
        return A.tocsr()
    return sp.csr_matrix(np.asarray(A))


def sparse_net_normalize(W):
    """
    Symmetric normalization: D^{-1/2} W D^{-1/2}
    (more stable than plain normalization for SNMF/CSNMF).
    """
    W = ensure_csr(W)
    d = np.array(W.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ W @ D_inv_sqrt


def build_reg_graph(Nets):
    """
    Graph regularization matrices:
      W_reg = sum(Nets)
      D_reg = diag(sum(W_reg))
    """
    W_reg = sum(Nets)
    d = np.array(W_reg.sum(axis=1)).flatten()
    D_reg = sp.diags(d)
    return W_reg.tocsr(), D_reg.tocsr()


def compute_k_from_labels(labels):
    """
    Determine number of clusters from ground truth labels.
    Handles labels starting at 0 or 1.
    """
    labels = np.asarray(labels).flatten()
    uniq = np.unique(labels)
    # sometimes labels include 0 as "unlabeled", but in MIT dataset usually all labeled
    # if you have 0 unlabeled nodes, you might want to ignore them in eval.
    return int(len(uniq))


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # ==============================
    # 1) Load dataset (MIT / CORA / MDP)
    # ==============================
    dataset_path = "data/nets/cora.mat"
    dataset_name = "cora"  # could be "cora" or "mdp" if your loader supports

    t_start = time.perf_counter()

    Nets, ground_idx = mltplx_from_mat(dataset_path, dataset_name)
    Nets = unwrap_mat_obj(Nets)
    ground_idx = unwrap_mat_obj(ground_idx)

    # Ensure list
    if not isinstance(Nets, list):
        Nets = list(Nets)

    # Convert to CSR + normalize
    Nets = [ensure_csr(A) for A in Nets]

    # You can keep original net_normalize if it works well,
    # but symmetric normalization is usually more stable.
    # Option A: use existing net_normalize
    # Nets = net_normalize(Nets)

    # Option B: use our symmetric normalization (recommended)
    Nets = [sparse_net_normalize(A) for A in Nets]

    # Basic checks
    n = Nets[0].shape[0]
    for A in Nets:
        assert A.shape[0] == n and A.shape[1] == n, "All layers must be n x n."

    ground_idx = np.asarray(ground_idx).flatten()
    k = compute_k_from_labels(ground_idx)

    print("=" * 60)
    print(f"Loaded dataset: {dataset_name}")
    print(f"#layers = {len(Nets)}, #nodes = {n}, #clusters(k) = {k}")
    print("=" * 60)

    # ==============================
    # 2) Step1: Generate Topological Views with SNMF
    # ==============================
    print("\nStep1: Generating per-layer views with SNMF...")
    H_views = []

    # For non-bionet tasks, Step1 doesn't need full convergence.
    step1_iter = 20

    for i, A in enumerate(Nets):
        snmf_model = SNMF(A, rank=k, init="svd", displ=False)  ####
        snmf_model.maxiter = step1_iter
        res = snmf_model.fit()
        H_v = res.matrices[0]  # (n, k)
        H_views.append(np.asarray(H_v))
        print(f"  - Layer {i+1}/{len(Nets)} done. H_view shape = {H_v.shape}")

    # ==============================
    # 3) Graph regularization matrices
    # ==============================
    W_reg, D_reg = build_reg_graph(Nets)

    # ==============================
    # 4) Step2: Improved CSNMF (MU + GraphReg + Contrastive)
    # ==============================
    print("\nStep2: Running Improved CSNMF (MU + GraphReg + Contrastive)...")

    # Recommended hyperparameters
    max_iter = 100
    warmup = 10
    beta_reg = 0.5   # graph reg strength
    use_cl = True
    lr_cl = 0.005
    tau = 0.5

    objCSNMF = CSNMF(Nets, k=k, alpha=0.5, init="svd", displ=True) ###

    res_csnmf = objCSNMF.fit(
        H_views=H_views,
        H_attr=None,                 # no attribute view for MIT/CORA/MDP by default
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
        clip_cl_grad=1.0,            # you can set None to disable
        normalize_cl_grad=False      # set True if CL unstable
    )

    H = res_csnmf.matrices[0]  # (n, k)
    H = np.asarray(H)

    # MU/CL logs (if you want to plot)
    J_mu = getattr(res_csnmf, "obj_mu", None)
    J_cl = getattr(res_csnmf, "obj_cl", None)

    # ==============================
    # 5) Clustering performance
    # ==============================
    print("\nClustering performance:")
    spect_idx = spect_clust(H, k)
    nmf_idx = nmf_clust(H)

    print("ground_idx shape:", ground_idx.shape)
    print("spect_idx shape :", spect_idx.shape)
    print("nmf_idx shape   :", nmf_idx.shape)

    print("\n### NMF clustering:")
    print(clust_eval(ground_idx, nmf_idx))

    print("\n### Spectral clustering:")
    print(clust_eval(ground_idx, spect_idx))

    # time 
    t_end = time.perf_counter()
    print("=" * 50)
    print(f"[Time] Total runtime: {t_end - t_start:.4f} seconds")
    print("=" * 50)

    # ==============================
    # 6) Visualization
    # ==============================
    pl.figure(1)
    pl.title("Cluster indicator matrix H (Consensus)")
    pl.imshow(H, interpolation='nearest', aspect='auto')
    pl.colorbar()

    pl.figure(2)
    pl.title("H H^T (Consensus similarity)")
    pl.imshow(H @ H.T, interpolation='nearest')
    pl.colorbar()

    if J_mu is not None and len(J_mu) > 0:
        pl.figure(3)
        pl.title("MU Objective (every 10 iters)")
        pl.plot(J_mu, 'o-')
        pl.xlabel("log step")
        pl.ylabel("loss")

    if J_cl is not None and len(J_cl) > 0:
        pl.figure(4)
        pl.title("Contrastive Loss (logged when enabled)")
        pl.plot(J_cl, 'o-')
        pl.xlabel("log step")
        pl.ylabel("InfoNCE loss")

    pl.show()
