import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributed as dist


class Broadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(
            grad_output,
            op=dist.ReduceOp.SUM
        )
        return grad_output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        inputs = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(inputs, input)
        inputs = torch.cat(inputs, dim=-1)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dim_size = grad_output.shape[-1]
        dim_size_per_partition = dim_size // world_size
        grad_chunks = torch.split(grad_output, dim_size_per_partition, dim=-1)
        return grad_chunks[rank]


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        last_dim_size = input.shape[-1]
        n_chunks = last_dim_size // world_size
        input_chunks = torch.split(input, n_chunks, dim=-1)
        return input_chunks[rank]

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        grad_outputs = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_outputs, grad_output)
        grad_outputs = torch.cat(grad_outputs, dim=-1)
        return grad_outputs


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        if world_size == 1:
            return input
        dist.all_reduce(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        world_size = dist.get_world_size()
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // world_size

        self.weight = Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size,
        ))
        self.bias = Parameter(torch.empty(
            self.output_size_per_partition,
        ))

    def forward(self, input):
        input_parallel = Broadcast.apply(input)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        outputs = Gather.apply(output_parallel)
        return outputs


class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        world_size = dist.get_world_size()
        input_size_per_partition = input_size // world_size

        self.weight = nn.Parameter(torch.randn(
            output_size,
            input_size_per_partition
        ))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, input):
        input_parallel = Scatter.apply(input)
        output_parallel = F.linear(input_parallel, self.weight)
        outputs = Reduce.apply(output_parallel)
        return outputs + self.bias
