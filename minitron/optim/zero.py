from typing import List

import torch
from torch.optim import Optimizer


class ZeroRedundancyOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optim: Optimizer,
        world_size: int
    ):
        defaults = dict()
        self.optim = optim
        self.world_size = world_size
        super().__init__(params, defaults)

    def _patrition_parameters(self) -> List[torch.Tensor]:
        num_param_per_ranks = [0 for _ in range(self.world_size)]
        params_per_ranks = [[] for _ in range(self.world_size)]

        for param in self.param_groups[0]['params']:
            next_device = num_param_per_ranks.index(min(num_param_per_ranks))
            params_per_ranks[next_device].append(param)
            num_param_per_ranks[next_device] += param.numel()

        return params_per_ranks
