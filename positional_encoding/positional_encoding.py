# https://www.deep-ml.com/problems/85
# Positional Encoding Calculator

import torch

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """
    # Your code here
    # pass
    if position==0 or d_model<=0:
        return -1
    
    pos = torch.arange(position).unsqueeze(-1)
    i = torch.arange(d_model).unsqueeze(0)

    angle_rate = 1 / 10000**(2 * (i // 2) / d_model)
    angle = pos * angle_rate

    pe = torch.zeros((position, d_model))
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])

    return pe.to(torch.float16)
    
