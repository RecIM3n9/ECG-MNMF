import numpy as np

def normalize(X, axis=1, epsilon=1e-8):
    """L2 Normalization for arrays"""
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / (norms + epsilon)

def compute_infonce_gradient(H, view_list, labels, tau=0.5, neg_sample_size=256):
    """
    Robust InfoNCE Gradient Calculation (Numpy version)
    """
    # 1. ndarray
    H = np.asarray(H)
    N, K = H.shape
    
    # L2 Normalize
    H_norm = normalize(H)
    views_norm = [normalize(np.asarray(v)) for v in view_list]
    
    grad_H = np.zeros_like(H)
    total_loss = 0.0
    
    # 2. Negative Sampling
    if neg_sample_size >= N:
        candidate_indices = np.arange(N)
    else:
        candidate_indices = np.random.choice(N, neg_sample_size, replace=False)
    
    neg_feats = H_norm[candidate_indices]  # (B, K)
    neg_labels = labels[candidate_indices] # (B,)
    
    # 3. Mask (Debiased Negatives)
    # explicit broadcasting: (N, 1) == (1, B)
    label_mask = (labels[:, None] == neg_labels[None, :])
    
    # Self-mask
    # (N, 1) == (1, B) -> comparing indices
    self_mask = (np.arange(N)[:, None] == candidate_indices[None, :])
    
    final_mask = label_mask | self_mask

    # 4. calculate gradient
    for V_norm in views_norm:
        # --- Logits Calculation ---
        # Positive logits: (N, 1)
        logits_pos = np.sum(H_norm * V_norm, axis=1, keepdims=True) / tau
        
        # Negative logits: (N, B)
        logits_neg = (H_norm @ neg_feats.T) / tau
        
        # --- Numerical Stability (Log-Sum-Exp Trick) ---
        # 1. Apply mask (set to -inf)
        logits_neg[final_mask] = -1e9
        
        # 2. Subtract max for stability
        # (N, 1)
        max_logits = np.maximum(logits_pos, np.max(logits_neg, axis=1, keepdims=True))
        
        exp_pos = np.exp(logits_pos - max_logits)
        exp_neg = np.exp(logits_neg - max_logits)
        
        # Sum of negatives
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
        
        # Softmax Probability
        denominator = exp_pos + sum_exp_neg + 1e-10
        prob_pos = exp_pos / denominator       # (N, 1)
        prob_neg = exp_neg / denominator       # (N, B)
        
        # --- Loss Tracking ---
        # L = -log(P_pos)
        total_loss += -np.mean(np.log(prob_pos + 1e-10))
        
        # --- Gradient Calculation ---
        # Formula: 1/tau * ( sum(P_neg * h_neg) - (1-P_pos) * h_pos )
        
        coef_pos = -(1.0 - prob_pos) / tau     # (N, 1)
        coef_neg = prob_neg / tau              # (N, B)
        
        # Term 1: Pull towards positive view
        grad_pos_term = coef_pos * V_norm      # (N, K)
        
        # Term 2: Push away from negatives
        grad_neg_term = coef_neg @ neg_feats   # (N, K)
        
        grad_H += (grad_pos_term + grad_neg_term)

    # Average over views
    if len(view_list) > 0:
        grad_H /= len(view_list)
        total_loss /= len(view_list)
        
    return grad_H, total_loss