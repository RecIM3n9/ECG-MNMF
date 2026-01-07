# -*- coding: utf-8 -*-
"""
Improved CSNMF (Consensus Symmetric NMF) for NF-CCE Step2 with:
1) Sparse SVD initialization (fast, stable)
2) Graph Regularization (Laplacian smoothing)
3) Contrastive Learning (InfoNCE, CDNMF-style) via Projected Gradient
4) [NEW] Alpha weighting for multiplex layers (NF-CCE alpha)

Core idea:
  - MU step optimizes weighted reconstruction + graph regularization
  - PGD step optimizes contrastive loss (InfoNCE) without breaking MU derivation
  - Always keep H >= 0 via projection, and optionally L1-normalize rows
"""

import numpy as np
import scipy.sparse as sp

from initialization import rnda_init, sparse_svd_init, svd_init
from contrastive import compute_infonce_gradient


# ==========================================================
# Helpers
# ==========================================================

def _fro_norm_sq_sparse(X):
    """Return ||X||_F^2 for sparse/dense matrix without densifying."""
    if sp.issparse(X):
        return float(X.power(2).sum())
    X = np.asarray(X)
    return float(np.dot(X.ravel(), X.ravel()))


def _ensure_csr(M):
    """Ensure matrix is CSR if not None."""
    if M is None:
        return None
    if sp.issparse(M):
        return M.tocsr()
    return sp.csr_matrix(np.asarray(M))


def _sanitize_alpha(alpha, num_layers):
    """
    Convert alpha into a vector alpha_v of length M.
    Supports:
      - scalar: same weight for all layers
      - list/ndarray: per-layer weights
    Returns:
      alpha_vec (M,)
      alpha_sum (float)
    """
    if alpha is None:
        alpha_vec = np.ones(num_layers, dtype=float)
    else:
        if np.isscalar(alpha):
            alpha_vec = np.ones(num_layers, dtype=float) * float(alpha)
        else:
            alpha_vec = np.asarray(alpha, dtype=float).flatten()
            if alpha_vec.size != num_layers:
                raise ValueError(
                    f"alpha length mismatch: got {alpha_vec.size}, expected {num_layers}"
                )

    # avoid negatives / zeros
    alpha_vec = np.maximum(alpha_vec, 0.0)
    alpha_sum = float(alpha_vec.sum())
    if alpha_sum <= 0:
        raise ValueError("alpha must have positive sum.")
    return alpha_vec, alpha_sum


def _weighted_sum_sparse(mats, weights):
    """Compute sum_i weights[i] * mats[i] safely for sparse/dense."""
    acc = None
    for w, A in zip(weights, mats):
        if w == 0:
            continue
        if acc is None:
            acc = A * w
        else:
            acc = acc + A * w
    if acc is None:
        # all weights are 0, shouldn't happen because alpha_sum checked
        acc = mats[0] * 0.0
    return acc


# ==========================================================
# Model
# ==========================================================

class CSNMF:
    """
    Improved CSNMF with:
    - Sparse SVD init
    - Graph Reg
    - Contrastive Learning (Projected GD)
    - [NEW] alpha weighting over multiplex layers
    """

    def __init__(self, Nets, k, alpha=0.5, init='svd', displ=True):
        """
        Nets: list of adjacency matrices A^{(v)} (preferably sparse CSR)
        k: number of clusters / latent dimension
        alpha: scalar or vector of length M (layer weights)
        """
        self.Nets = [_ensure_csr(A) for A in Nets]
        self.k = int(k)
        self.init = init
        self.displ = displ

        # alpha vector for layers
        self.alpha_vec, self.alpha_sum = _sanitize_alpha(alpha, len(self.Nets))

        # Weighted aggregate network used as main reconstruction target:
        # AggNet = sum_v alpha_v * A_v
        self.AggNet = _weighted_sum_sparse(self.Nets, self.alpha_vec)

        # Precompute ||AggNet||_F^2 for fast MU objective logging
        self.norm_A_sq = _fro_norm_sq_sparse(self.AggNet)

    def fit(self,
            H_views=None,                 # list of H^{(v)} from Step1
            H_attr=None,                  # H^{(attr)} from attribute view
            W_reg=None, D_reg=None,       # graph reg matrices
            beta_reg=0.0,                 # strength of graph reg
            use_cl=False,                 # whether to enable contrastive learning
            lr_cl=0.005,                  # CL projected GD step size
            tau=0.5,                      # temperature for InfoNCE
            max_iter=200,
            warmup=5,                     # MU warmup iterations before CL
            update_label_freq=5,          # pseudo-label update frequency
            neg_sample_size=256,          # pass-through for contrastive module
            normalize_after=True,         # L1 normalize after each iteration
            clip_cl_grad=None,            # e.g. 1.0 to clip gradients; None = no clip
            normalize_cl_grad=False       # normalize row-wise gradient direction
            ):
        """
        Returns CSNMFResult with matrices=[H]
        """

        # ---------- sanitize inputs ----------
        max_iter = int(max_iter)
        warmup = int(warmup)
        update_label_freq = int(update_label_freq)

        # ensure sparse types for reg matrices (fast @)
        W_reg = _ensure_csr(W_reg)
        D_reg = _ensure_csr(D_reg)

        # ---------- 1) initialize H ----------
        if self.init == 'svd':
            H = sparse_svd_init(self.AggNet, self.k)
        elif self.init == 'svd_dense':
            H = svd_init(self.AggNet, self.k)
        else:
            H = rnda_init(self.AggNet, self.k)

        H = np.asarray(H, dtype=float)
        H = np.maximum(H, 1e-10)

        # ---------- prepare view_list for contrastive ----------
        view_list = []
        if H_views is not None and len(H_views) > 0:
            view_list.extend([np.asarray(v, dtype=float) for v in H_views])
        if H_attr is not None:
            view_list.append(np.asarray(H_attr, dtype=float))

        # ---------- init pseudo labels ----------
        labels = np.zeros(H.shape[0], dtype=np.int64)

        # ---------- objective logging ----------
        obj_mu = []
        obj_cl = []

        # ---------- 2) main loop ----------
        for it in range(max_iter):

            # ==========================================================
            # Part A: MU update for Weighted Reconstruction + Graph Reg
            #
            # Weighted objective:
            #   sum_v alpha_v ||A_v - HH^T||_F^2
            #
            # MU numerator:
            #   (sum_v alpha_v A_v) H  + beta W_reg H
            # MU denominator:
            #   (sum_v alpha_v) (HH^T H) + beta D_reg H
            #
            # Note: for SNMF, HH^T H = (H H^T) H
            # ==========================================================

            # numerator: AggNet H + beta W H
            grad_neg = self.AggNet @ H
            if beta_reg > 0 and W_reg is not None:
                grad_neg = grad_neg + beta_reg * (W_reg @ H)

            # denominator: alpha_sum * (HH^T H) + beta D H
            HHtH = (H @ H.T) @ H
            grad_pos = self.alpha_sum * HHtH
            if beta_reg > 0 and D_reg is not None:
                grad_pos = grad_pos + beta_reg * (D_reg @ H)

            # damped MU update (stable with CL)
            ratio = grad_neg / (grad_pos + 1e-10)
            H = H * np.power(ratio, 0.5)

            # enforce nonnegativity
            H = np.maximum(H, 1e-10)

            # ==========================================================
            # Part B: Contrastive Learning step (Projected Gradient)
            # ==========================================================
            if use_cl and (it >= warmup) and (len(view_list) > 0):

                # update pseudo-labels at warmup start and periodically
                if it == warmup or (it % update_label_freq == 0):
                    labels = np.argmax(H, axis=1)

                grad_cl, loss_cl = compute_infonce_gradient(
                    H, view_list, labels,
                    tau=tau,
                    neg_sample_size=neg_sample_size
                )

                grad_cl = np.asarray(grad_cl, dtype=float)

                # optional gradient normalization (row-wise)
                if normalize_cl_grad:
                    gn = np.linalg.norm(grad_cl, axis=1, keepdims=True)
                    grad_cl = grad_cl / np.maximum(gn, 1e-8)

                # optional gradient clipping
                if clip_cl_grad is not None:
                    clip_val = float(clip_cl_grad)
                    grad_cl = np.clip(grad_cl, -clip_val, clip_val)

                # projected gradient descent
                H = H - lr_cl * grad_cl
                H = np.maximum(H, 1e-10)

                if self.displ and (it % 10 == 0):
                    print(f"Iter {it:03d} | CL Loss: {loss_cl:.4f}")

                obj_cl.append(float(loss_cl))

            # ==========================================================
            # Part C: L1 row normalization (interpretability)
            # ==========================================================
            if normalize_after:
                row_sum = H.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                H = H / row_sum

            # ==========================================================
            # MU objective logging (optional)
            # using weighted AggNet already includes alpha weighting
            # ==========================================================
            if (it % 10 == 0) and self.displ:
                # ||AggNet - HH^T||_F^2 = ||AggNet||_F^2 - 2 tr(H^T AggNet H) + ||H^T H||_F^2
                AH = self.AggNet @ H
                term_cross = -2.0 * np.sum(AH * H)
                HtH = H.T @ H
                term_HtH = np.sum(HtH * HtH)
                loss_mu = self.norm_A_sq + term_cross + term_HtH

                # add graph reg term
                if beta_reg > 0 and (W_reg is not None) and (D_reg is not None):
                    LH = (D_reg - W_reg) @ H
                    reg_term = np.sum(H * LH)
                    loss_mu = loss_mu + beta_reg * reg_term

                obj_mu.append(float(loss_mu))
                if not use_cl:
                    print(f"Iter {it:03d} | MU Loss: {loss_mu:.4e}")

        return CSNMFResult([H], obj_mu=obj_mu, obj_cl=obj_cl, converged=True)


class CSNMFResult:
    """Simple wrapper compatible with demo scripts."""
    def __init__(self, matrices, obj_mu=None, obj_cl=None, converged=True):
        self.matrices = matrices
        self.obj_mu = obj_mu
        self.obj_cl = obj_cl
        self.converged = converged


