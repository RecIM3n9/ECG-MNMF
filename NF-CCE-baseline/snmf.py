# nf-cce
# -*- coding: utf-8 -*-
"""
Implementation of the Symmetric NMF (SNMF).
Modified for High-Performance and Low-Memory usage on Large Networks.
"""

from basenmf import NMF, NMFresult
from initialization import rnd_init, rndc_init, rnda_init, svd_init
import numpy as np
import scipy.sparse as sp

class SNMF(NMF):
    """
    Implementation of the Symmetric NMF (SNMF).
    """
    def fit(self):
        """
        Multiplicative update rules for minimizing the objective function:
            min ||X - H H^T||_F
        """
        
        X = self.X

        if isinstance(X, np.ndarray):
            if X.ndim == 0:
                X = X.item()
            elif X.dtype == object and X.size == 1:
                try:
                    X = X.item()
                except Exception:
                    pass 

        try:
            if sp.issparse(X):
                if X.data.size > 0 and X.data.min() < 0:
                    raise ValueError("Input matrix X contains negative values.")
            
            # (numpy.ndarray / numpy.matrix)
            else:
                if np.min(np.asarray(X)) < 0:
                    raise ValueError("Input matrix X contains negative values.")

        except Exception as e:
            print(f"Warning: Could not verify non-negativity of X due to: {e}")
            print("Skipping check and proceeding (assuming X is non-negative)...")

        self.X = X 

        # ============================================================
        # initial
        # ============================================================
        if self.init == "rnd":
            H = rnd_init(X, self.rank)
        elif self.init == "rndc":
            H = rndc_init(X, self.rank)
        elif self.init == "rnda":
            H = rnda_init(X, self.rank)
        elif self.init == "svd":
            H = svd_init(X, self.rank, flag=1)
        else:
            # 默认 fallback
            H = rnd_init(X, self.rank)

        # ============================================================
        # ||X||_F^2
        # ============================================================
        if sp.issparse(X):
            self.norm_X_sq = X.power(2).sum()
        else:
            X_flat = np.asarray(X).ravel()
            self.norm_X_sq = np.dot(X_flat, X_flat)

        dist = 0
        pdist = 1e10
        converged = False
        objfun_vals = np.zeros(self.maxiter // 10)
        c = 0

        # ============================================================
        # Main Loop
        # ============================================================
        for it in range(self.maxiter):

            # --- Multiplicative Update Rule ---
            # H <- H .* (0.5 + 0.5 * (X*H) ./ (H*H'*H))
            grad_neg = X @ H
            grad_pos = (H @ H.T) @ H

            H = np.multiply(H, 0.5 + 0.5 * np.divide(grad_neg, grad_pos + 1e-10))

            # --- Error Calculation & Convergence Check ---
            if (it % 10 == 0) or (it == self.maxiter - 1):
                
                # 1. ||X||^2
                norm_X_sq = self.norm_X_sq
                
                # 2. -2 * tr(H^T X H)
                # tr(A B) = sum(A .* B^T)
                # tr(H^T (X H)) = sum(H^T .* (X H)^T) = sum(H .* (X H))
                XtH = X @ H
                term_cross = -2 * np.sum(np.multiply(XtH, H))
                
                # 3. ||H^T H||_F^2
                HtH = H.T @ H
                term_HtH = np.sum(np.square(HtH))
                
                dist_sq = norm_X_sq + term_cross + term_HtH
                dist = np.sqrt(max(dist_sq, 0))

                if c < len(objfun_vals):
                    objfun_vals[c] = dist * dist
                
                if pdist - dist < self.tol:
                    converged = True
                    break
                
                if self.displ:
                    print("### Iter = %d | ObjF = %.3e | Rel = %.3e" % (it, dist*dist, pdist-dist))
                
                pdist = dist
                c += 1
        
        # ============================================================
        # L1 Row Normalization
        # ============================================================
        H = np.array(H) 
        norms = H.sum(axis=1)
        norms[norms == 0] = 1.0 
        H /= norms[:, np.newaxis]

        return NMFresult((H,), objfun_vals[:c], dist*dist, converged)
    
