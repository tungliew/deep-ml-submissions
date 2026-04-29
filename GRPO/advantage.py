# https://www.deep-ml.com/problems/224
# Group Relative Advantage for GRPO

import torch

def compute_group_relative_advantage(rewards: torch.Tensor) -> torch.Tensor:
	"""
	Compute the Group Relative Advantage for GRPO using PyTorch.
	
	Args:
		rewards: 1D tensor of rewards for a group of outputs
		
	Returns:
		1D tensor of normalized advantages
	"""
	# Your code here
	# pass
	
	# compute advantage
	rewards_mean = torch.mean(rewards)
	rewards_std = torch.std(rewards, unbiased=False)

	if rewards_std == 0:
        return torch.zeros_like(rewards)

	advantage = (rewards - rewards_mean) / rewards_std

	return advantage
