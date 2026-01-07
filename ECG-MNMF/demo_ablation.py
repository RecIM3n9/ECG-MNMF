# -*- coding: utf-8 -*-
"""
Ablation Demo for non-bionet multiplex datasets (MIT / CORA / MDP / CiteSeer).

Pipeline:
  Step1 (once): Per-layer SNMF -> H_views
  Step2 (loop): Improved CSNMF variants -> H_final
  Eval         : NMI/ARI via clust_eval + runtime

This script automatically runs ablation settings:
  - Baseline NF-CCE (Random init, no reg, no CL)
  - + SVD init
  - + GraphReg
  - + Contrastive
  - Full (SVD + GraphReg + Contrastive)

Requirements:
  - snmf.py  (correct SNMF denominator (HH^T)H and sparse_svd_init)
  - csnmf.py (MU + graph reg + contrastive PGD, supports alpha)
  - contrastive.py (InfoNCE + debiased negatives)
  - initialization.py (sparse_svd_init)
"""

import numpy as np
import scipy.sparse as sp
import time

from snmf import SNMF
from csnmf import CSNMF
from preprocessing import mltplx_from_mat, net_normalize
from cluster import nmf_clust, spect_clust, clust_eval


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
    W_reg = sum(Nets).tocsr()
    d = np.array(W_reg.sum(axis=1)).flatten()
    D_reg = sp.diags(d).tocsr()
    return W_reg, D_reg


def compute_k_from_labels(labels):
    """
    Determine number of clusters from ground truth labels.
    Handles labels starting at 0 or 1.
    """
    labels = np.asarray(labels).flatten()
    uniq = np.unique(labels)

    # Optional: ignore label 0 if it's "unlabeled"
    # if 0 in uniq:
    #     uniq = uniq[uniq != 0]

    return int(len(uniq))


def safe_parse_eval(eval_out):
    """
    clust_eval return format may differ.
    Try to parse NMI/ARI from common formats:
      - dict: {"NMI":..., "ARI":...}
      - tuple/list: (NMI, ARI, ...)
      - string: you can just print it
    """
    if isinstance(eval_out, dict):
        nmi = eval_out.get("NMI", None)
        ari = eval_out.get("ARI", None)
        return nmi, ari
    if isinstance(eval_out, (tuple, list)) and len(eval_out) >= 2:
        return eval_out[0], eval_out[1]
    return None, None


def format_float(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # ==============================
    # 1) Load dataset (MIT / CORA / MDP / CiteSeer)
    # ==============================
    # dataset_path = "data/nets/mit.mat"
    # dataset_name = "mit"

    # If you want CORA / CiteSeer / MDP, change these:
    # dataset_path = "data/nets/cora.mat"
    # dataset_name = "cora"
    #
    dataset_path = "data/nets/CiteSeer.mat"
    dataset_name = "citeseer"

    print("=" * 70)
    print(f"Dataset: {dataset_name} | file: {dataset_path}")
    print("=" * 70)

    Nets, ground_idx = mltplx_from_mat(dataset_path, dataset_name)
    Nets = unwrap_mat_obj(Nets)
    ground_idx = unwrap_mat_obj(ground_idx)

    if not isinstance(Nets, list):
        Nets = list(Nets)

    # Convert to CSR + normalize
    Nets = [ensure_csr(A) for A in Nets]
    Nets = [sparse_net_normalize(A) for A in Nets]

    n = Nets[0].shape[0]
    ground_idx = np.asarray(ground_idx).flatten()
    k = compute_k_from_labels(ground_idx)

    print(f"#layers = {len(Nets)} | #nodes = {n} | #clusters(k) = {k}")
    print("=" * 70)

    # ==============================
    # 2) Step1: Generate Topological Views once (SNMF)
    # ==============================
    print("\n[Step1] Generating per-layer views with SNMF (run once)...")
    t_step1_start = time.perf_counter()

    H_views = []
    step1_iter = 20

    for i, A in enumerate(Nets):
        snmf_model = SNMF(A, rank=k, init="svd", displ=False)
        snmf_model.maxiter = step1_iter
        res = snmf_model.fit()
        H_v = np.asarray(res.matrices[0], dtype=float)
        H_views.append(H_v)
        print(f"  - Layer {i+1}/{len(Nets)} | H_view shape = {H_v.shape}")

    t_step1_end = time.perf_counter()
    step1_time = t_step1_end - t_step1_start
    print(f"[Step1] Done. Time = {step1_time:.4f} sec")

    # ==============================
    # 3) Build Graph Regularization matrices once
    # ==============================
    W_reg, D_reg = build_reg_graph(Nets)

    # ==============================
    # 4) Define Ablation Variants
    # ==============================
    variants = [
        ("+SVD Init",
         dict(init="svd", beta_reg=0.0, use_cl=False)),
        ("+SVD + GraphReg",
         dict(init="svd", beta_reg=0.5, use_cl=False)),
        ("+SVD + Contrastive",
         dict(init="svd", beta_reg=0.0, use_cl=True)),
        ("Full (SVD + GraphReg + CL)",
         dict(init="svd", beta_reg=0.5, use_cl=True)),
    ]

    # Fixed hyperparameters for fair comparison
    alpha = 0.5
    max_iter = 100
    warmup = 10
    lr_cl = 0.005
    tau = 0.5

    update_label_freq = 5
    neg_sample_size = 256
    clip_cl_grad = 1.0
    normalize_cl_grad = False

    # ==============================
    # 5) Run Ablation Loop
    # ==============================
    results = []

    print("\n" + "=" * 70)
    print("[Step2] Running Ablation Variants...")
    print("=" * 70)

    for name, cfg in variants:
        init = cfg["init"]
        beta_reg = cfg["beta_reg"]
        use_cl = cfg["use_cl"]

        print("\n" + "-" * 70)
        print(f"Variant: {name}")
        print(f"  init={init}, beta_reg={beta_reg}, use_cl={use_cl}")
        print("-" * 70)

        # Build CSNMF
        model = CSNMF(Nets, k=k, alpha=alpha, init=init, displ=True)

        # Time Step2 only (fair)
        t0 = time.perf_counter()
        res = model.fit(
            H_views=H_views,
            H_attr=None,
            W_reg=W_reg,
            D_reg=D_reg,
            beta_reg=beta_reg,
            use_cl=use_cl,
            lr_cl=lr_cl,
            tau=tau,
            max_iter=max_iter,
            warmup=warmup,
            update_label_freq=update_label_freq,
            neg_sample_size=neg_sample_size,
            normalize_after=True,
            clip_cl_grad=clip_cl_grad,
            normalize_cl_grad=normalize_cl_grad
        )
        t1 = time.perf_counter()
        step2_time = t1 - t0

        H = np.asarray(res.matrices[0], dtype=float)

        # Evaluate clustering
        nmf_idx = nmf_clust(H)
        spect_idx = spect_clust(H, k)

        eval_nmf = clust_eval(ground_idx, nmf_idx)
        eval_spect = clust_eval(ground_idx, spect_idx)

        nmf_nmi, nmf_ari = safe_parse_eval(eval_nmf)
        sp_nmi, sp_ari = safe_parse_eval(eval_spect)

        # Collect losses (if available)
        J_mu = getattr(res, "obj_mu", None)
        J_cl = getattr(res, "obj_cl", None)

        last_mu = J_mu[-1] if (J_mu is not None and len(J_mu) > 0) else None
        last_cl = J_cl[-1] if (J_cl is not None and len(J_cl) > 0) else None

        total_time = step1_time + step2_time

        print(f"[Time] Step2 runtime = {step2_time:.4f} sec")
        print(f"[TOTAL TIME] TOTAL TIME = {total_time:.4f} sec")
        print("[Eval] NMF clustering:", eval_nmf)
        print("[Eval] Spectral clustering:", eval_spect)

        results.append({
            "Variant": name,
            "init": init,
            "beta_reg": beta_reg,
            "use_cl": use_cl,
            "Step1_time": step1_time,      # constant, optional
            "Step2_time": step2_time,
            "Total_time": step1_time + step2_time,
            "NMF_NMI": nmf_nmi,
            "NMF_ARI": nmf_ari,
            "Spec_NMI": sp_nmi,
            "Spec_ARI": sp_ari,
            "Last_MU": last_mu,
            "Last_CL": last_cl
        })

    # ==============================
    # 6) Print Summary Table
    # ==============================
    print("\n" + "=" * 90)
    print("Ablation Summary (NMF clustering shown; Spectral shown too)")
    print("=" * 90)

    header = (
        f"{'Variant':28s} | {'init':5s} | {'beta':5s} | {'CL':3s} | "
        f"{'NMI':>7s} | {'ARI':>7s} | {'SpecNMI':>7s} | {'SpecARI':>7s} | "
        f"{'Step2(s)':>9s} | {'Total(s)':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['Variant'][:28]:28s} | "
            f"{r['init']:<5s} | "
            f"{r['beta_reg']:<5.2f} | "
            f"{str(r['use_cl'])[0]:<3s} | "
            f"{format_float(r['NMF_NMI']):>7s} | "
            f"{format_float(r['NMF_ARI']):>7s} | "
            f"{format_float(r['Spec_NMI']):>7s} | "
            f"{format_float(r['Spec_ARI']):>7s} | "
            f"{r['Step2_time']:>9.4f} | "
            f"{r['Total_time']:>9.4f}"
        )

    print("=" * 90)
    print("Done.")
    print("=" * 90)
