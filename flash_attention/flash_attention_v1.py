# https://www.deep-ml.com/problems/208
# Flash Attention v1 - Forward Pass

import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    # Your code here
    # pass
    seq_len, d_model = Q.shape
    scale = 1.0 / (d_model ** 0.5)

    output = torch.zeros_like(Q)

    for q_start in range(0, seq_len, block_size):
        Q_block = Q[q_start: q_start + block_size]

        running_max = torch.full((Q_block.shape[0],), -float("inf"))
        running_sum = torch.zeros(Q_block.shape[0])
        running_output = torch.zeros_like(Q_block)


        for kv_start in range(0, seq_len, block_size):
            K_block = K[kv_start: kv_start + block_size]
            V_block = V[kv_start: kv_start + block_size]

            scores = (Q_block @ K_block.T) * scale

            block_max = torch.max(scores, dim=1).values
            new_max = torch.maximum(running_max, block_max)

            # compute contributions
            exp_old = torch.exp(running_max - new_max)
            exp_new = torch.exp(scores - new_max[:, None])

            # update running_sum
            running_sum = exp_old * running_sum + torch.sum(exp_new, dim=1)

            running_output = (
                exp_old[:, None] * running_output + exp_new @ V_block
            )
            

            # update max
            running_max = new_max
        
        # normalization
        output[q_start: q_start + block_size] = running_output / running_sum[:, None]

    
    return output

'''
5/5 tests passed
'''
