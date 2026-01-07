# -*- coding: utf-8 -*-
"""
BioNet Ablation Demo (NF-CCE style)

Pipeline (computed once):
  - Load multiplex topology layers Nets
  - Normalize each layer
  - Step1: SNMF per layer -> H_views
  - Attribute View: GO KNN graph -> H_attr (via SNMF)
  - GraphReg: W_reg, D_reg

Ablation (loop):
  - Run CSNMF Step2 with different settings
  - Report Average Redundancy + runtime

Variants:
  1) NF-CCE baseline            : Random init, no GraphReg, no Contrastive
  2) + SVD init                 : SVD init only
  3) + SVD + GraphReg           : GraphReg only
  4) + SVD + Contrastive        : Contrastive only
  5) Full (SVD + GraphReg + CL) : All improvements
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse, diags
import time

from preprocessing import nets_from_mat
from snmf import SNMF
from csnmf import CSNMF
from sklearn.neighbors import kneighbors_graph


# ===========================
# Utils
# ===========================
def unwrap_mat_obj(x):
    """Peel onion: unwrap 0-d arrays / object arrays from .mat load."""
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
    Symmetric normalization:
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
    W_reg = sum(Nets).tocsr()
    d = np.array(W_reg.sum(axis=1)).flatten()
    D_reg = sp.diags(d).tocsr()
    return W_reg, D_reg


def build_feature_graph(feature_matrix, knn_k=15, metric="cosine"):
    """
    Build attribute-view adjacency using KNN on GO annotation matrix.
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
    """
    Average Redundancy metric used in NF-CCE.
    """
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


def format_float(x):
    return f"{x:.4f}"


# ===========================
# Main
# ===========================
if __name__ == "__main__":

    print("### BioNet Ablation Demo (NF-CCE + SVD + GraphReg + Contrastive) ###")

    # -----------------------
    # 1) Choose dataset
    # -----------------------
    filename = "data/bionets/yeast_and_go.mat"
    # filename = "data/bionets/human_and_go.mat"
    # filename = "data/bionets/mouse_and_go.mat"

    # -----------------------
    # 2) Global settings
    # -----------------------
    num_layers = 5        # use first 5 layers like NF-CCE paper
    k = 100               # same as paper for BioNet
    step1_iter = 25       # fast views
    attr_knn_k = 15

    # Step2 hyperparameters (fixed for fairness)
    max_iter = 150
    warmup = 10
    alpha = 0.5

    beta_reg_default = 0.5
    lr_cl_default = 0.005
    tau_default = 0.5

    update_label_freq = 5
    neg_sample_size = 256
    clip_cl_grad = 1.0
    normalize_cl_grad = False

    # -----------------------
    # 3) Load data
    # -----------------------
    print(f"\nLoading data: {filename}")
    genes, _, Nets, GO_data = nets_from_mat(filename)
    Nets = unwrap_mat_obj(Nets)
    GO_data = unwrap_mat_obj(GO_data)

    Nets = Nets[:num_layers]

    # measure total runtime
    t_total_start = time.perf_counter()

    # -----------------------
    # 4) Preprocess topology nets
    # -----------------------
    print("\nPreprocessing: normalize topology layers...")
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
    # 5) Step1: Topological Views (computed once)
    # -----------------------
    print("\n[Step1] Generating topological views with SNMF (once)...")
    t_step1_start = time.perf_counter()

    H_views = []
    for i, A in enumerate(Nets):
        snmf_model = SNMF(A, rank=k, init="svd", displ=False)
        snmf_model.maxiter = step1_iter
        res = snmf_model.fit()
        H_v = np.asarray(res.matrices[0], dtype=float)
        H_views.append(H_v)
        print(f"  - View {i+1}: H_view shape={H_v.shape}")

    t_step1_end = time.perf_counter()
    step1_time = t_step1_end - t_step1_start
    print(f"[Step1] Done. Time = {step1_time:.4f} sec")

    # -----------------------
    # 6) Attribute View (GO KNN graph) (computed once)
    # -----------------------
    print("\n[Attr] Building attribute view (GO KNN graph) once...")
    t_attr_start = time.perf_counter()

    S_attr = build_feature_graph(GO_data, knn_k=attr_knn_k, metric="cosine")
    S_attr = sparse_net_normalize(S_attr)
    print(f"  - S_attr: shape={S_attr.shape}, nnz={S_attr.nnz}")

    attr_model = SNMF(S_attr, rank=k, init="svd", displ=False)
    attr_model.maxiter = step1_iter
    res_attr = attr_model.fit()
    H_attr = np.asarray(res_attr.matrices[0], dtype=float)
    print(f"  - H_attr shape={H_attr.shape}")

    t_attr_end = time.perf_counter()
    attr_time = t_attr_end - t_attr_start
    print(f"[Attr] Done. Time = {attr_time:.4f} sec")

    # -----------------------
    # 7) Graph Regularization (computed once)
    # -----------------------
    print("\n[GraphReg] Building W_reg, D_reg once...")
    W_reg, D_reg = build_reg_graph(Nets)
    print(f"  - W_reg nnz={W_reg.nnz}")

    # -----------------------
    # 8) Ablation Variants
    # -----------------------
    variants = [
        ("+SVD Init",
         dict(init="svd", beta_reg=0.0, use_cl=False, lr_cl=0.0, tau=tau_default)),
        ("+SVD + GraphReg",
         dict(init="svd", beta_reg=beta_reg_default, use_cl=False, lr_cl=0.0, tau=tau_default)),
        ("+SVD + Contrastive",
         dict(init="svd", beta_reg=0.0, use_cl=True, lr_cl=lr_cl_default, tau=tau_default)),
        ("Full (SVD + GraphReg + CL)",
         dict(init="svd", beta_reg=beta_reg_default, use_cl=True, lr_cl=lr_cl_default, tau=tau_default)),
    ]

    results = []

    print("\n" + "=" * 80)
    print("[Step2] Running Ablation Variants...")
    print("=" * 80)

    for name, cfg in variants:
        init = cfg["init"]
        beta_reg = cfg["beta_reg"]
        use_cl = cfg["use_cl"]
        lr_cl = cfg["lr_cl"]
        tau = cfg["tau"]

        print("\n" + "-" * 80)
        print(f"Variant: {name}")
        print(f"  init={init}, beta_reg={beta_reg}, use_cl={use_cl}, lr_cl={lr_cl}, tau={tau}")
        print("-" * 80)

        model = CSNMF(Nets, k=k, alpha=alpha, init=init, displ=True)

        # Step2 time only (fair)
        t_step2_start = time.perf_counter()

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
            update_label_freq=update_label_freq,
            neg_sample_size=neg_sample_size,

            normalize_after=True,
            clip_cl_grad=clip_cl_grad,
            normalize_cl_grad=normalize_cl_grad
        )

        t_step2_end = time.perf_counter()
        step2_time = t_step2_end - t_step2_start

        H_final = np.asarray(res.matrices[0], dtype=float)

        # Evaluate redundancy
        avg_red = calculate_average_redundancy(H_final, GO_data)

        print(f"[Result] Avg Redundancy = {avg_red:.4f}")
        print(f"[Time] Step2 runtime  = {step2_time:.4f} sec")

        results.append({
            "Variant": name,
            "AvgRed": avg_red,
            "Step2_time": step2_time,
            "Total_time": step1_time + attr_time + step2_time
        })

    # total runtime
    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    # -----------------------
    # 9) Print Summary Table
    # -----------------------
    print("\n" + "=" * 90)
    print(f"Ablation Summary (BioNet) | Dataset: {filename}")
    print("=" * 90)
    header = f"{'Variant':30s} | {'AvgRed':>8s} | {'Step2(s)':>10s} | {'Total(s)':>10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['Variant'][:30]:30s} | "
            f"{format_float(r['AvgRed']):>8s} | "
            f"{r['Step2_time']:>10.4f} | "
            f"{r['Total_time']:>10.4f}"
        )

    print("=" * 90)
    print(f"[Total runtime of this script] {total_time:.4f} sec")
    print("=" * 90)
    print("Done.")
