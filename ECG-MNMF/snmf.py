# new
# -*- coding: utf-8 -*-
"""
Implementation of the Symmetric NMF (SNMF).
Modified for High-Performance and Low-Memory usage on Large Networks.

Objective:
    min_{H>=0} ||X - H H^T||_F^2

Update (Damped MU):
    H <- H .* (0.5 + 0.5 * (XH) / ((H H^T) H))
"""

from basenmf import NMF, NMFresult
from initialization import rnd_init, rndc_init, rnda_init, svd_init, sparse_svd_init

import numpy as np
import scipy.sparse as sp


class SNMF(NMF):
    """
    Symmetric Nonnegative Matrix Factorization (SNMF).
    """

    def fit(self):
        """
        Multiplicative update rules for minimizing:
            min ||X - H H^T||_F^2
        """

        X = self.X

        # ============================================================
        # [1] Unwrap: handle 0-d numpy array / object wrapper from .mat
        # ============================================================
        if isinstance(X, np.ndarray):
            if X.ndim == 0:
                X = X.item()
            elif X.dtype == object and X.size == 1:
                try:
                    X = X.item()
                except Exception:
                    pass

        # ============================================================
        # [2] Non-negativity check (safe for sparse/dense)
        # ============================================================
        try:
            if sp.issparse(X):
                if X.data.size > 0 and X.data.min() < 0:
                    raise ValueError("Input matrix X contains negative values.")
            else:
                if np.min(np.asarray(X)) < 0:
                    raise ValueError("Input matrix X contains negative values.")
        except Exception as e:
            print(f"Warning: Could not verify non-negativity of X due to: {e}")
            print("Skipping check and proceeding (assuming X is non-negative)...")

        # Update self.X to avoid later issues
        self.X = X

        # ============================================================
        # [3] Initialize H (ndarray)
        # ============================================================
        if self.init == "rnd":
            H = rnd_init(X, self.rank)
        elif self.init == "rndc":
            H = rndc_init(X, self.rank)
        elif self.init == "rnda":
            H = rnda_init(X, self.rank)
        elif self.init == "svd":
            # Prefer sparse SVD init for sparse/large graphs
            # If X is dense but large, sparse_svd_init will convert to csr internally
            H = sparse_svd_init(X, self.rank)
        elif self.init == "svd_dense":
            # legacy dense SVD for small matrices only
            H = svd_init(X, self.rank, flag=1)
        else:
            H = rnd_init(X, self.rank)

        H = np.asarray(H, dtype=float)  # ensure ndarray

        # ============================================================
        # [4] Precompute ||X||_F^2 (move out of loop)
        # ============================================================
        if sp.issparse(X):
            # Sparse: sum of squares of non-zeros
            norm_X_sq = X.power(2).sum()
        else:
            X_flat = np.asarray(X).ravel()
            norm_X_sq = float(np.dot(X_flat, X_flat))

        self.norm_X_sq = norm_X_sq

        # ============================================================
        # [5] Main loop variables
        # ============================================================
        dist = 0.0
        pdist = 1e10
        converged = False

        # Store objective every 10 iterations
        objfun_vals = np.zeros(max(1, self.maxiter // 10), dtype=float)
        c = 0

        # ============================================================
        # [6] Main optimization loop
        # ============================================================
        for it in range(self.maxiter):

            # --- MU Update ---
            # numerator: XH
            # denominator: (H H^T) H   [IMPORTANT FIX]
            grad_neg = X @ H
            grad_pos = (H @ H.T) @ H

            H = np.multiply(H, 0.5 + 0.5 * np.divide(grad_neg, grad_pos + 1e-10))

            # --- Convergence check every 10 iters (and last iter) ---
            if (it % 10 == 0) or (it == self.maxiter - 1):

                # Reuse grad_neg = X@H (avoid recomputing)
                XtH = grad_neg

                # Using: ||X - HH^T||_F^2 = ||X||_F^2 - 2 tr(H^T X H) + ||H^T H||_F^2
                # term_cross = -2 tr(H^T X H) = -2 sum(H .* (XH))
                term_cross = -2.0 * np.sum(XtH * H)

                HtH = H.T @ H
                term_HtH = np.sum(np.square(HtH))

                dist_sq = float(self.norm_X_sq + term_cross + term_HtH)
                dist_sq = max(dist_sq, 0.0)  # numerical safety
                dist = np.sqrt(dist_sq)

                # record objective
                if c < len(objfun_vals):
                    objfun_vals[c] = dist_sq

                # check convergence
                if (pdist - dist) < self.tol:
                    converged = True
                    break

                if self.displ:
                    print("### Iter = %d | ObjF = %.3e | Rel = %.3e" %
                          (it, dist_sq, pdist - dist))

                pdist = dist
                c += 1

        # ============================================================
        # [7] Post-processing: L1 row normalization for interpretability
        # ============================================================
        norms = H.sum(axis=1)
        norms[norms == 0] = 1.0
        H = H / norms[:, np.newaxis]

        return NMFresult((H,), objfun_vals[:c], dist * dist, converged)



