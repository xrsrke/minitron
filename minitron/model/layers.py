import torch
from torch import nn
import torch.nn.functional as F

from .utils import get_model_parallel_world_size
from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region
)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()
        world_size = get_model_parallel_world_size()

        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // world_size

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device()
            )
        )
        self.bias = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device()
            )
        )

    def forward(self, input):
        input_parallel = copy_to_model_parallel_region(input)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = gather_from_model_parallel_region(output_parallel)
