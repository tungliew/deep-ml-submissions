# https://www.deep-ml.com/problems/376
# KV Cache for Efficient Autoregressive Attention

import torch

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    # pass
    Q = torch.matmul(x_new, W_Q) # (d_k, )
    K = torch.matmul(x_new, W_K)
    V = torch.matmul(x_new, W_V)

    if cache is None:
        K_cache = K.unsqueeze(0) # (1, d_k)
        V_cache = V.unsqueeze(0) # (1, d_v)
    else:
        K_cache, V_cache = cache

        # Append new K and V to cache
        K_cache = torch.cat([K_cache, K.unsqueeze(0)], dim=0)
        V_cache = torch.cat([V_cache, V.unsqueeze(0)], dim=0)
        
    d_k = Q.shape[0]

    scores = torch.matmul(K_cache, Q) / torch.sqrt(torch.tensor(d_k))

    scores = scores - torch.max(scores)
    exp_scores = torch.exp(scores)
    attention_weights = exp_scores / torch.sum(exp_scores)
        
    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V_cache)  # (d_v,)
        
    updated_cache = (K_cache, V_cache)
        
    return output, updated_cache

    
