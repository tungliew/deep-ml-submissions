# https://www.deep-ml.com/problems/416
# Compute Attention Memory Traffic and FLOPs

import torch

def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention using PyTorch tensors.

    The attention mechanism uses:
    - torch.bmm for Q @ K^T and P @ V matrix multiplications
    - torch.softmax for row-wise normalization

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    # Your code here
    # pass
    
    # Q, K, V shape
    # (B, h, N, d)

    # compute score flops
    # (B, h, N, d) @ (B, h, d, N) = (B, h, N, N)
    qk_flops = B * h * (2 * N * d * N )
    softmax_flops = B * h * (5 * N * N)
    # (B, h, N, N) *  (B, h, N, d) = (B, h, N, d)
    pv_flops = B * h * (2 * N * N* d )

    total_flops = qk_flops + softmax_flops + pv_flops


    # Step 1: read Q + read K + write S
    step1_elements = (B * h * N * d) + (B * h * N * d) + (B * h * N * N)

    # Step 2: read S + write P
    step2_elements = (B * h * N * N) + (B * h * N * N)

    # Step 3: read P + read V + write O
    step3_elements = (B * h * N * N) + (B * h * N * d) + (B * h * N * d)

    total_elements = step1_elements + step2_elements + step3_elements

    memory_bytes = total_elements * bytes_per_element

    # FLOPs per byte,
    arithmetic_intensity = round(total_flops / memory_bytes, 2)

    return {
        "qk_flops": qk_flops,
        "softmax_flops": softmax_flops,
        "pv_flops": pv_flops,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": arithmetic_intensity
    }
