import torch
from torch import nn
import torch.nn.functional as F

# from .utils import get_model_parallel_world_size
# from .mappings import (
#     copy_to_model_parallel_region,
#     gather_from_model_parallel_region
# )


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, num_partitions: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // num_partitions

        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size
        ))
        self.bias = nn.Parameter(torch.empty(
            self.output_size_per_partition
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_parallel = F.linear(input, self.weight, self.bias)
        world_size = torch.distributed.get_world_size()
        outputs = [torch.empty_like(output_parallel) for _ in range(world_size)]
        torch.distributed.all_gather(outputs, output_parallel)

        outputs = torch.cat(outputs, dim=-1).contiguous()
        return outputs


class RowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        world_size = torch.distributed.get_world_size()
        self.input_size_per_patrition = input_size // world_size
        self.weight = nn.Parameter(torch.empty(
            self.output_size,
            self.input_size_per_patrition
        ))
        self.bias = nn.Parameter(torch.empty(
            self.output_size
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # split the input
        last_dim_size = input.shape[-1]
        world_size = torch.distributed.get_world_size()
        dim_per_patrition = last_dim_size // world_size
        input_chunks = torch.split(input, dim_per_patrition, dim=-1)

        rank = torch.distributed.get_rank()
        input_parallel = input_chunks[rank]
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        torch.distributed.all_reduce(output_parallel)

        return output_parallel


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.vocab_start_idx, self.vocab_end_idx = self.extract_range(
            self.num_embeddings,
            rank,
            world_size
        )
        self.num_embedding_per_patrition = self.vocab_end_idx - self.vocab_start_idx

        self.weight = nn.Parameter(torch.empty(
            self.num_embedding_per_patrition,
            self.embedding_dim
        ))

    def extract_range(self, num_embeddings, rank, world_size):
        per_patrition_vocab_size = num_embeddings // world_size
        start_idx = rank * per_patrition_vocab_size
        end_idx = start_idx + per_patrition_vocab_size
        return start_idx, end_idx

    def forward(self, input):
        input_mask = (input < self.vocab_start_idx) | (input >= self.vocab_end_idx)
        masked_input = input.clone() - self.vocab_start_idx
        masked_input[input_mask] = 0

        output_parallel = F.embedding(masked_input, self.weight)
        masked_idxs = torch.where(input_mask == True)[1]
        output_parallel[:, masked_idxs, :] = 0.

        torch.distributed.all_reduce(
            output_parallel,
            op=torch.distributed.ReduceOp.SUM
        )

        return output_parallel