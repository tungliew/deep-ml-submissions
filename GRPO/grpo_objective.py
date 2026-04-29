# https://www.deep-ml.com/problems/101?from=DeepSeek%20R1
# Implement the GRPO Objective Function

import torch

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> torch.Tensor:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities.
        pi_theta_ref: List of reference policy probabilities.
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value as a torch.Tensor.
    """
    # Your code here
    # pass
    rhos = torch.tensor(rhos)
    A = torch.tensor(A)
    pi_theta_old = torch.tensor(pi_theta_old)
    pi_theta_ref = torch.tensor(pi_theta_ref)
    
    pi_theta = rhos * pi_theta_old

    clipped_rhos = torch.clamp(rhos, min=1-epsilon, max=1+epsilon)
    
    surrogate_loss = torch.min(
        rhos * A, 
        clipped_rhos * A
    )
    
    r = pi_theta_ref / pi_theta
    kl_div = rhos * (r - torch.log(r) - 1)

    objective = torch.mean(surrogate_loss - beta * kl_div)

    return objective
