# https://www.deep-ml.com/problems/427
# FP4 Quantization with Microscaling (MXFP4)

import torch
import torch.nn.functional as F

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling using PyTorch.

    Args:
        x: list of float values to quantize
        block_size: number of elements per scaling block

    Returns:
        dict with keys:
            'quantized': list of dequantized values (rounded to 4 decimals)
            'scales': list of per-block scale factors (rounded to 4 decimals)
    """
    # pass
    
    fp4_values = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
        0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )

    # convert list to tensor
    x_tensor = torch.tensor(x)

    # padding
    x_len = len(x)
    remainder = x_len % block_size
    if remainder!=0:
        pad_len = block_size - remainder
        pad_tensor = torch.zeros(pad_len, dtype=x_tensor.dtype)
        x_tensor = torch.cat((x_tensor, pad_tensor), dim=0)
    
    # reshape into blocks
    blocks = x_tensor.view(-1, block_size)


    quantized_blocks = []
    scales = []

    for block in blocks:
        # compute the block scale
        max_value = torch.max(torch.abs(block)).item()

        if max_value==0:
            scale = 1.0
        else:
            raw_scale = max_value / 6.0

            scale = 2 ** torch.ceil(torch.log2(torch.tensor(raw_scale))).item()
        
        scales.append(round(float(scale), 4))

        # scale the block
        normalized_block = block / scale

        dequantized_block = []

        # find the nearest fp4 value for every element
        # in the block
        for val in normalized_block:
            distances = torch.abs(fp4_values - val)

            # find the minimum distance
            min_dist = torch.min(distances)

            # Candidates with same minimum distance
            candidates = fp4_values[distances == min_dist]

            chosen = candidates[torch.argmin(torch.abs(candidates))]

            dequantized = chosen.item() * scale
            dequantized_block.append(round(float(dequantized), 4))

        quantized_blocks.extend(dequantized_block)
    
    quantized_blocks = quantized_blocks[:x_len]

    return {
        "quantized": quantized_blocks,
        "scales": scales
    }
