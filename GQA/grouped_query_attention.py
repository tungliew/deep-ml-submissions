# https://www.deep-ml.com/problems/391
# Implement Grouped Query Attention (GQA)

import torch
import torch.nn.functional as F

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    # pass
    assert num_heads % num_kv_heads == 0
    
    batch_size, seq_len, d_model = Q.shape

    head_dim = d_model // num_heads
    
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1,2)
    K = K.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1,2)
    V = V.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1,2)

    repeated = num_heads // num_kv_heads
    K = K.repeat_interleave(repeated, dim=1)
    V = V.repeat_interleave(repeated, dim=1)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(head_dim))

    max_value = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_value)
    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    output = torch.matmul(attention_weights, V)

    output = output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, num_heads * head_dim
    )

    return output
