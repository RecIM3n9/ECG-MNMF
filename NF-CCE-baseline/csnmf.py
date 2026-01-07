# nf-cce    
# -*- coding: utf-8 -*-

# from snmf import SNMF
# from basenmf import NMF, NMFresult
# import numpy as np

# class CSNMF(NMF):
#     """
#     Implementation of the Collective SNMF (CSNMF).
#     The implementation is based on the paper:
#         V. Gligorijevic, Y. Panagakis, S. Zafeiriou, "Non-Negative Matrix
#         Factorizations for Multiplex Network Analysis", 2017, T-PAMI.
#     """
#     def __init__(self, X, rank, alpha = 0.5, **kwargs):
#         NMF.__init__(self, X, rank, **kwargs)
#         self.alpha = alpha

#     def fit(self):
#         """
#         Multiplicative update rules for minimizing the objective function:
#             sum_i ||X_i - HHT||_F + alpha sum_i ||HHT - H_iH_iT||_F
#         """
#         N = len(self.X)
#         H = []
#         for i in range(N):
#             print("### Factorizing network [%d]..."%(i+1))
#             X = self.X[i]
#             objSNMF = SNMF(X, self.rank, init = self.init, displ = self.displ)
#             res_snmf = objSNMF.fit()
#             H.append(np.mat(res_snmf.matrices[0]))
#             if res_snmf.converged:
#                 print("### Converged.")
#         A_avg = np.mat(np.zeros((self.X[0].shape), dtype=float))
#         for i in range(N):
#             A_avg += self.X[i] + self.alpha*(H[i]*H[i].T)
#         A_avg /= (1.0 + self.alpha)*N
#         objSNMF = SNMF(A_avg, self.rank, init = self.init, displ = self.displ)
#         res_snmf = objSNMF.fit()

#         return res_snmf

# -*- coding: utf-8 -*-

from snmf import SNMF
from basenmf import NMF, NMFresult
import numpy as np
from scipy import sparse

class CSNMF(NMF):
    """
    Implementation of the Collective SNMF (CSNMF).
    """

    def __init__(self, X, rank, alpha=0.5, **kwargs):
        super().__init__(X, rank, **kwargs)
        self.alpha = alpha

    def fit(self):
        """
        Multiplicative update rules for minimizing:
            sum_i ||X_i - H H^T||_F 
          + alpha * sum_i ||H H^T - H_i H_i^T||_F
        """
        N = len(self.X)
        H = []

        # --- Step 1: SNMF on each network ---
        for i in range(N):
            print(f"### Factorizing network [{i+1}]...")
            X = self.X[i]
            
            objSNMF = SNMF(X, self.rank, 
                           init=self.init, 
                           displ=self.displ)
            res_snmf = objSNMF.fit()
            
            # use float32 to save memory
            H.append(res_snmf.matrices[0].astype(np.float32))

            if res_snmf.converged:
                print("### Converged.")

        # --- Step 2: build averaged matrix A_avg (dense) ---
        print("### Constructing Consensus Matrix (Memory Optimized)...")
        
        if isinstance(self.X[0], np.ndarray) and self.X[0].ndim == 0:
            self.X = [x.item() for x in self.X]
            
        n = self.X[0].shape[0]
        
        A_avg = np.zeros((n, n), dtype=np.float32)

        for i in range(N):
            if sparse.issparse(self.X[i]):
                A_avg += self.X[i].astype(np.float32)
            else:
                A_avg += self.X[i].astype(np.float32)

            # 2. alpha * (H @ H.T)
            # Block Processing
            # avoid OOM
            
            Hi = H[i] 
            alpha_val = float(self.alpha)

            chunk_size = 2000
            
            for start_row in range(0, n, chunk_size):
                end_row = min(start_row + chunk_size, n)
                
                # (chunk, k) @ (k, n) -> (chunk, n)
                Hi_chunk = Hi[start_row:end_row, :]
                product_chunk = Hi_chunk @ Hi.T
                
                A_avg[start_row:end_row, :] += alpha_val * product_chunk
                
            print(f"   - Network {i+1} merged.")

        A_avg /= ((1.0 + self.alpha) * N)

        # --- Step 3: final SNMF ---
        print("### Final Factorization on Consensus Matrix...")
        objSNMF = SNMF(A_avg, self.rank, 
                       init=self.init, 
                       displ=self.displ)
        res_snmf = objSNMF.fit()

        return res_snmf
    
    
