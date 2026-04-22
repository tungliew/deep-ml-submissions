# LoRA: Low-Rank Adaptation Forward Pass
# https://www.deep-ml.com/problems/222

import torch

def lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute the LoRA forward pass using PyTorch.
    
    Args:
        x: Input tensor (batch_size x in_features)
        W: Frozen pretrained weights (in_features x out_features)
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output tensor (batch_size x out_features)
    """
    # Your code here
    # pass
    frozen = x @ W
    low_rank = x @ B @ A 
    output = frozen + (low_rank / alpha)
    return output


x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) 
W = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]) 
B = torch.tensor([[1.0], [2.0], [3.0]]) 
A = torch.tensor([[0.1, 0.2]]) 
result = lora_forward(x, W, A, B, alpha=1.0) 
print([[round(v, 4) for v in row.tolist()] for row in result])

‘’‘
Expected
[[1.1, 1.2], [1.2, 1.4], [1.3, 1.6]]
Your Output
Passed
[[1.1, 1.2], [1.2, 1.4], [1.3, 1.6]]
’‘’
